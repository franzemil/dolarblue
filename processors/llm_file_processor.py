"""LLMFileProcessor: PDFs de Gaceta -> GacetaNorma tipadas.

- El cliente decide cómo enviar el documento (Z.ai: PDF crudo; DeepSeek:
  páginas renderizadas a PNG).
- response_format json_object + validación Pydantic contra GacetaNorma.
- 3 intentos con backoff (incluye schema_invalid); nunca aborta el batch.
- Cache = data/processed/llm_manifest.json keyed por sha256(pdf):provider:model.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from processors.llm_client import (
    BaseLLMClient,
    CostCapExceeded,
    LLMUnavailable,
    build_client,
)
from processors.llm_manifest import LLMManifest, sha256_file
from processors.results import GacetaNorma, ProcessResult
from processors.stage import Stage
from processors.taxonomy import TAGS_BOUNDARY_RULES, TAGS_FX

OUT_FIELDS = [
    "norma_id",
    "edicion",
    "instrumento",
    "entidad_emisora",
    "fecha_publicacion",
    "fecha_vigencia",
    "monto_usd",
    "moneda",
    "resumen_1l",
    "tags",
    "relevancia_fx",
    "confidence",
    "pdf_path",
    "model",
    "cost_usd",
    "status",
    "extracted_date",
]

SYSTEM_PROMPT = """Sos un extractor de metadatos de normas de la Gaceta Oficial de \
Bolivia para un estudio del tipo de cambio BOB/USD. Respondés ÚNICAMENTE con un \
objeto json válido y completo (sin markdown, sin texto extra)."""


def _user_prompt(norma_id: str, edicion: str) -> str:
    return f"""Extraé los campos de esta norma (norma_id={norma_id}, edicion={edicion}).

Devolvé un único objeto json válido con exactamente estas claves:
{{
  "norma_id": "{norma_id}",
  "edicion": "{edicion}",
  "instrumento": "Decreto Supremo | Resolución Ministerial | Ley | ... | null",
  "entidad_emisora": "entidad | null",
  "fecha_publicacion": "YYYY-MM-DD | null",
  "fecha_vigencia": "YYYY-MM-DD | null",
  "monto_usd": número o null,
  "moneda": "USD | BOB | null",
  "resumen_1l": "una línea | null",
  "tags": ["tags de la lista"],
  "relevancia_fx": "alta | media | baja | nula",
  "confidence": 0.0 a 1.0
}}

Reglas:
- monto_usd: SOLO montos explícitos en dólares; normalizá separadores bolivianos
  (15.000.000 -> 15000000); null si no hay monto o está en otra moneda.
- moneda: moneda del monto mencionado; null si no hay monto.
- tags: elegí SOLO de esta lista (podés usar varias): {", ".join(TAGS_FX)}.
- relevancia_fx: impacto esperado de la norma sobre el tipo de cambio boliviano.
{TAGS_BOUNDARY_RULES}
- confidence: certeza global de la extracción (texto claro=alto; scan dudoso=bajo)."""


class LLMFileProcessor(Stage):
    """Extrae GacetaNorma por PDF con manifest reanudable y tope de costo."""

    def __init__(
        self,
        data_dir: Path,
        client: BaseLLMClient | None = None,
        days: int | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        force: bool = False,
        attempts: int = 3,
        backoff: float = 2.0,
    ):
        self.data_dir = Path(data_dir)
        self.client = client or build_client()
        self.days = days
        self.since = since
        self.limit = limit
        self.dry_run = dry_run
        self.force = force
        self.attempts = attempts
        self.backoff = backoff
        self.processed_dir = self.data_dir / "processed"
        self.manifest = LLMManifest(self.processed_dir / "llm_manifest.json")
        self.out_csv = self.processed_dir / "gaceta_llm.csv"
        self.client_error: str | None = None

    def transform(self, items: list[dict]) -> list[ProcessResult]:
        results: list[ProcessResult] = []
        candidates = 0

        for item in items:
            pdf_sha = sha256_file(item["pdf_path"])
            hit = self.manifest.get(pdf_sha, self.client.provider, self.client.model)
            if hit is not None and not self.force:
                results.append(
                    self._result(item, self.manifest.path, "cache", 0.0, hit.get("record", {}))
                )
                continue
            if not self._in_window(item):
                results.append(self._result(item, None, "skipped", 0.0, {"motivo": "fuera de ventana"}))
                continue
            if self.client_error:
                results.append(
                    self._result(item, None, "llm_failed", 0.0, {"motivo": self.client_error})
                )
                continue
            if self.limit is not None and candidates >= self.limit:
                results.append(self._result(item, None, "skipped", 0.0, {"motivo": "limit"}))
                continue
            candidates += 1
            if self.dry_run:
                results.append(self._result(item, None, "skipped", 0.0, {"motivo": "dry_run"}))
                continue
            res, _entry = self._extract(item, pdf_sha)
            results.append(res)

        if not self.dry_run:
            self._write_csv(self.manifest.all_records())
        return results

    # ------------------------------ extracción -------------------------------
    def _extract(self, item: dict, pdf_sha: str) -> tuple[ProcessResult, dict | None]:
        messages = self.client.build_messages(
            SYSTEM_PROMPT,
            _user_prompt(item["norma_id"], item["edicion"]),
            item["pdf_path"],
        )
        cost = 0.0
        last_error: Exception | None = None

        for attempt in range(self.attempts):
            try:
                data, info = self.client.chat_json(messages)
                cost += info["cost_usd"]
                norma = GacetaNorma.model_validate(data)
            except CostCapExceeded as e:
                return self._result(item, None, "llm_cost_cap", cost, {"error": str(e)}), None
            except LLMUnavailable as e:
                self.client_error = f"proveedor LLM no disponible: {e}"
                return self._result(item, None, "llm_failed", cost, {"error": self.client_error}), None
            except Exception as e:  # noqa: BLE001 — un PDF malo nunca aborta el batch
                last_error = e
                if attempt < self.attempts - 1:
                    time.sleep(self.backoff * (attempt + 1))
                continue

            record = {
                **norma.model_dump(),
                "norma_id": item["norma_id"],
                "edicion": item["edicion"],
                "fecha_publicacion": (
                    item["fecha"].strftime("%Y-%m-%d")
                    if item.get("fecha")
                    else norma.fecha_publicacion
                ),
                "tags": "|".join(norma.tags),
                "pdf_path": str(item["pdf_path"].relative_to(self.data_dir)),
                "model": info["model"],
                "cost_usd": round(info["cost_usd"], 6),
                "status": "ok",
                "extracted_date": datetime.now(timezone.utc).isoformat(),
            }
            provider = info.get("provider", self.client.provider)
            entry = {
                "norma_id": item["norma_id"],
                "edicion": item["edicion"],
                "pdf_path": record["pdf_path"],
                "pdf_sha256": pdf_sha,
                "provider": provider,
                "model": info["model"],
                "record": record,
                "raw_content": info.get("raw_content", ""),
                "usage": info.get("usage", {}),
                "cost_usd": record["cost_usd"],
                "extracted_date": record["extracted_date"],
            }
            self.manifest.put(pdf_sha, provider, info["model"], entry)
            detail = {"norma_id": item["norma_id"], "edicion": item["edicion"], "provider": provider}
            return self._result(item, self.manifest.path, "ok", info["cost_usd"], detail), entry

        status = "schema_invalid" if isinstance(last_error, (ValidationError, ValueError)) else "llm_failed"
        return self._result(item, None, status, cost, {"error": str(last_error)[:500]}), None

    # ------------------------------ helpers ----------------------------------
    def _in_window(self, item: dict) -> bool:
        if self.since is None and self.days is None:
            return True
        fecha = item.get("fecha")
        if fecha is None:
            return False
        cutoff = self.since
        if self.days is not None:
            by_days = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=self.days)
            cutoff = max(filter(None, (cutoff.replace(tzinfo=None) if cutoff else None, by_days)))
        return fecha >= cutoff

    def _result(
        self,
        item: dict,
        output: Path | None,
        status: str,
        cost: float,
        detail: dict,
    ) -> ProcessResult:
        return ProcessResult(
            input_path=item["pdf_path"],
            output_path=output,
            status=status,
            extracted_date=datetime.now(timezone.utc),
            meta={
                "norma_id": item["norma_id"],
                "edicion": item["edicion"],
                "provider": self.client.provider,
                "model": self.client.model,
                **detail,
            },
            cost_usd=cost,
        )

    def _write_csv(self, records: list[dict]) -> None:
        if not records:
            return
        records = sorted(
            records, key=lambda r: (r.get("fecha_publicacion") or "", r.get("norma_id", ""))
        )
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
