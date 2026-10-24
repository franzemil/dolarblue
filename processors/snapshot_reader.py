"""SnapshotReader: entrada de cada rama desde lo ya persistido en disco.

- dolarblue: data/manifest.json (filas del manifiesto de corrida) +
  data/sources/<endpoint>.json (body crudo de cada endpoint ok).
- gaceta: data/gaceta/manifiesto.csv + PDFs norma_*.pdf (con fecha de edición).

Nunca invoca collectors ni descarga nada.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from collectors.endpoints import PRIVATE_ENDPOINTS, PUBLIC_ENDPOINTS, EndpointSpec
from processors.stage import Stage

SPECS: dict[str, EndpointSpec] = {
    spec.name: spec for spec in (*PUBLIC_ENDPOINTS, *PRIVATE_ENDPOINTS)
}


class SnapshotReader(Stage):
    """Emite los inputs de una rama a partir de los archivos persistidos."""

    def __init__(self, source: str, data_dir: Path):
        if source not in ("dolarblue", "gaceta"):
            raise ValueError(f"source inválido: {source!r}")
        self.source = source
        self.data_dir = Path(data_dir)

    def transform(self, inputs=None) -> list[dict]:
        return self._dolarblue() if self.source == "dolarblue" else self._gaceta()

    # ------------------------------ rama A ----------------------------------
    def _dolarblue(self) -> list[dict]:
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"falta {manifest_path}: corré `main.py update-dataset` al menos una vez"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items: list[dict] = []
        for row in manifest.get("results", []):
            if row.get("source") != "dolarblue" or row.get("status") != "ok":
                continue
            artifact = row.get("artifact", "")
            if not artifact.startswith("sources/"):
                continue
            name = Path(artifact).stem
            spec = SPECS.get(name)
            path = self.data_dir / artifact
            if spec is None or not path.exists():
                continue
            items.append(
                {
                    "name": name,
                    "familia": spec.family,
                    "url": row.get("url", ""),
                    "extracted_date": datetime.fromisoformat(row["extracted_date"]),
                    "path": path,
                    "payload": json.loads(path.read_text(encoding="utf-8")),
                }
            )
        return items

    # ------------------------------ rama B ----------------------------------
    def _gaceta(self) -> list[dict]:
        gaceta_dir = self.data_dir / "gaceta"
        manifest_path = gaceta_dir / "manifiesto.csv"
        items: list[dict] = []
        if manifest_path.exists():
            with open(manifest_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("estado") not in ("ok", "cache"):
                        continue
                    pdf = gaceta_dir / row.get("edicion", "") / row.get("archivo", "")
                    if not pdf.is_file():
                        continue
                    items.append(
                        {
                            "norma_id": row.get("norma_id", ""),
                            "edicion": row.get("edicion", ""),
                            "fecha": self._parse_fecha(row.get("fecha", "")),
                            "tipo": row.get("tipo", ""),
                            "pdf_path": pdf,
                        }
                    )
        else:
            for pdf in sorted(gaceta_dir.glob("*/norma_*.pdf")):
                items.append(
                    {
                        "norma_id": pdf.stem.removeprefix("norma_"),
                        "edicion": pdf.parent.name,
                        "fecha": None,
                        "tipo": "",
                        "pdf_path": pdf,
                    }
                )
        return items

    @staticmethod
    def _parse_fecha(value: str) -> datetime | None:
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        return None
