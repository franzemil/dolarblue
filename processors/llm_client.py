"""Clientes LLM (OpenAI-compatible) para extracción JSON con control de costo.

Proveedores (selección explícita con --provider o LLM_PROVIDER, default zai):
- zai:      https://api.z.ai/api/paas/v4/ · glm-5.3-flash · PDF crudo (bloque file)
- deepseek: https://api.deepseek.com      · deepseek-flash · sin PDF: páginas -> PNG

Ambos comparten chat_json (JSON mode + validación externa con Pydantic), costo
exacto por usage y tope acumulado LLM_MAX_COST_USD.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, OpenAI

DEFAULT_MAX_COST_USD = 2.0

_UNAVAILABLE_TOKENS = (
    "insufficient balance",
    "no resource package",
    "invalid api key",
    "authentication fail",
    "unauthorized",
)


class CostCapExceeded(RuntimeError):
    """El gasto acumulado alcanzó LLM_MAX_COST_USD."""


class LLMUnavailable(RuntimeError):
    """El proveedor rechazó la cuenta (sin saldo, key inválida): no reintentar."""


class BaseLLMClient:
    """Cliente OpenAI-compatible genérico: una llamada JSON por invocación."""

    provider = "base"
    api_key_env = "LLM_API_KEY"
    base_url_env = "LLM_BASE_URL"
    model_env = "LLM_MODEL"
    default_base_url = ""
    default_model = ""
    supports_pdf = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_cost_usd: float | None = None,
    ):
        self.api_key = api_key or os.environ.get(self.api_key_env)
        self.base_url = base_url or os.environ.get(self.base_url_env, self.default_base_url)
        self.model = model or os.environ.get(self.model_env, self.default_model)
        self.max_cost = float(
            max_cost_usd
            if max_cost_usd is not None
            else os.environ.get("LLM_MAX_COST_USD", DEFAULT_MAX_COST_USD)
        )
        self.spent = 0.0
        self._client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
            if self.api_key
            else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def extra_body(self) -> dict:
        return {}

    def _cost(self, usage) -> float:
        return 0.0

    def build_messages(self, system: str, user_text: str, pdf_path: Path) -> list[dict]:
        raise NotImplementedError

    def chat_json(self, messages: list[dict]) -> tuple[dict, dict]:
        """Una llamada al modelo; devuelve (json parseado, info de uso/costo).

        El reintento y la validación de esquema viven en el processor: acá solo
        se hace UNA llamada por invocación (cada intento cuesta).
        """
        if self._client is None:
            raise RuntimeError(f"falta {self.api_key_env} en .env")
        if self.spent >= self.max_cost:
            raise CostCapExceeded(
                f"gasto acumulado ${self.spent:.4f} >= tope ${self.max_cost:.2f}"
            )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=1.0,
                max_tokens=1024,
                extra_body=self.extra_body(),
            )
        except APIError as e:
            message = str(e).lower()
            if any(token in message for token in _UNAVAILABLE_TOKENS):
                raise LLMUnavailable(str(e)) from e
            if "response_format" not in message:
                raise
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                max_tokens=1024,
                extra_body=self.extra_body(),
            )
        usage = response.usage
        cost = self._cost(usage)
        self.spent += cost
        content = response.choices[0].message.content or ""
        return json.loads(content), {
            "provider": self.provider,
            "model": self.model,
            "usage": usage.model_dump() if usage else {},
            "cost_usd": cost,
            "raw_content": content,
        }


class ZaiLLMClient(BaseLLMClient):
    """Z.ai: el PDF viaja crudo como bloque ``file`` en base64."""

    provider = "zai"
    api_key_env = "LLM_API_KEY"
    base_url_env = "LLM_BASE_URL"
    model_env = "LLM_MODEL"
    default_base_url = "https://api.z.ai/api/paas/v4/"
    default_model = "glm-5.3-flash"
    supports_pdf = True

    PRICE_INPUT = 0.15 / 1_000_000
    PRICE_CACHED_INPUT = 0.03 / 1_000_000
    PRICE_OUTPUT = 0.50 / 1_000_000

    def extra_body(self) -> dict:
        return {"reasoning_effort": "low"}

    def _cost(self, usage) -> float:
        if usage is None:
            return 0.0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        prompt = (usage.prompt_tokens or 0) - cached
        completion = usage.completion_tokens or 0
        return prompt * self.PRICE_INPUT + cached * self.PRICE_CACHED_INPUT + completion * self.PRICE_OUTPUT

    def build_messages(self, system: str, user_text: str, pdf_path: Path) -> list[dict]:
        b64 = base64.b64encode(Path(pdf_path).read_bytes()).decode("ascii")
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/pdf;base64,{b64}",
                            "filename": Path(pdf_path).name,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]


class DeepSeekClient(BaseLLMClient):
    """DeepSeek (sin entrada PDF): renderiza páginas a PNG y las envía como imágenes."""

    provider = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url_env = "DEEPSEEK_BASE_URL"
    model_env = "DEEPSEEK_MODEL"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-flash"
    supports_pdf = False

    MAX_PAGES = 20
    RENDER_DPI = 150

    PRICE_OFFPEAK = {
        "miss": 0.15 / 1_000_000,
        "hit": 0.003 / 1_000_000,
        "out": 0.60 / 1_000_000,
    }

    def extra_body(self) -> dict:
        return {"thinking": {"type": "disabled"}}

    @staticmethod
    def _is_peak(now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        return 1 <= now.hour < 4 or 6 <= now.hour < 10

    def _cost(self, usage) -> float:
        if usage is None:
            return 0.0
        miss = getattr(usage, "prompt_cache_miss_tokens", None)
        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if miss is None or hit is None:
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            hit = cached
            miss = (usage.prompt_tokens or 0) - cached
        completion = usage.completion_tokens or 0
        price = self.PRICE_OFFPEAK
        multiplier = 2.0 if self._is_peak() else 1.0
        return (miss * price["miss"] + hit * price["hit"] + completion * price["out"]) * multiplier

    def build_messages(self, system: str, user_text: str, pdf_path: Path) -> list[dict]:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page}"}}
            for page in self._pdf_pages_png(Path(pdf_path))
        ]
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def _pdf_pages_png(self, pdf_path: Path) -> list[str]:
        import pymupdf

        pages: list[str] = []
        with pymupdf.open(pdf_path) as doc:
            for page in doc[: self.MAX_PAGES]:
                pixmap = page.get_pixmap(dpi=self.RENDER_DPI)
                pages.append(base64.b64encode(pixmap.tobytes("png")).decode("ascii"))
        return pages


def build_client(provider: str | None = None) -> BaseLLMClient:
    """Cliente según --provider o LLM_PROVIDER (default zai)."""
    name = (provider or os.environ.get("LLM_PROVIDER", "zai")).lower()
    clients = {"zai": ZaiLLMClient, "deepseek": DeepSeekClient}
    if name not in clients:
        raise ValueError(f"proveedor inválido: {name!r} (usar zai|deepseek)")
    return clients[name]()


__all__ = [
    "BaseLLMClient",
    "ZaiLLMClient",
    "DeepSeekClient",
    "build_client",
    "CostCapExceeded",
    "LLMUnavailable",
]
