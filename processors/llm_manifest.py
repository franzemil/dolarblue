"""Manifest de resultados LLM: única fuente de verdad del cache.

Un solo JSON con entradas keyed por ``sha256(pdf):provider:model``:
- el hash del contenido invalida el cache si el PDF cambia;
- provider/model lo invalidan si se cambia de proveedor o modelo.
Cada entrada guarda el registro validado y la respuesta cruda (auditoría),
así no hacen falta directorios llm_cache/ ni llm_raw/.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "llm_manifest.json"
CACHE_VERSION = "3"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 del contenido del archivo, en streaming."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class LLMManifest:
    """Registro persistente de extracciones; escritura atómica tras cada alta."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.updated_at: str | None = None
        self.entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.updated_at = payload.get("updated_at")
        raw = payload.get("entries", {})
        suffix = f":{CACHE_VERSION}"
        self.entries = {k: v for k, v in raw.items() if k.endswith(suffix)}
        if len(self.entries) != len(raw):
            self.save()

    @staticmethod
    def key(pdf_sha256: str, provider: str, model: str, version: str = CACHE_VERSION) -> str:
        return f"{pdf_sha256}:{provider}:{model}:{version}"

    def get(
        self, pdf_sha256: str, provider: str, model: str, version: str = CACHE_VERSION
    ) -> dict | None:
        return self.entries.get(self.key(pdf_sha256, provider, model, version))

    def put(
        self,
        pdf_sha256: str,
        provider: str,
        model: str,
        entry: dict,
        version: str = CACHE_VERSION,
    ) -> None:
        self.entries[self.key(pdf_sha256, provider, model, version)] = entry
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def all_records(self) -> list[dict]:
        """Un registro por norma_id: si hubo re-extracciones, gana la más reciente."""
        latest: dict[str, dict] = {}
        for entry in self.entries.values():
            record = entry.get("record", {})
            norma_id = str(record.get("norma_id") or entry.get("norma_id", ""))
            current = latest.get(norma_id)
            if current is None or record.get("extracted_date", "") >= current.get("extracted_date", ""):
                latest[norma_id] = record
        return list(latest.values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": self.updated_at or datetime.now(timezone.utc).isoformat(),
            "entries": self.entries,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
