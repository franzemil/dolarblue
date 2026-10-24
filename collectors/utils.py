"""Utilidades compartidas de colecta.

Helpers de compatibilidad que no hacen red: solo copian/derivan artefactos
ya producidos por los collectors.
"""
from __future__ import annotations

import csv
from pathlib import Path

from collectors.models import TypedResult


def write_chart_6m_copy(out_dir: Path) -> Path | None:
    """data/normalized/chart_6m.csv -> data/chart_6m.csv (compatibilidad)."""
    src = out_dir / "normalized" / "chart_6m.csv"
    if not src.exists():
        return None
    dest = out_dir / "chart_6m.csv"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def write_gaceta_manifest(out_dir: Path, results: list[TypedResult]) -> Path:
    """Manifiesto derivado de los FileResults: el link extraído <-> datos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifiesto.csv"
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["edicion", "fecha", "tipo", "norma_id", "url", "archivo",
             "bytes", "estado", "extracted_date"]
        )
        for res in results:
            meta = res.meta
            w.writerow(
                [
                    meta.get("edicion", ""),
                    res.published_date.strftime("%d-%m-%Y")
                    if res.published_date else "",
                    meta.get("tipo", ""),
                    meta.get("norma_id", ""),
                    res.url,
                    res.path.name if res.path else "",
                    res.path.stat().st_size
                    if res.path and res.path.exists() else 0,
                    res.status,
                    res.extracted_date.isoformat(),
                ]
            )
    return man_path
