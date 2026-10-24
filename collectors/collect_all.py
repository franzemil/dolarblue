"""Orquestador de colecta: corre TODOS los collectors del proyecto en una pasada.

Orquesta las dos fuentes concretas del paquete collectors/:
    - collectors/dolarblue.py -> DolarBlueCollector (ApiCollector por URL:
      endpoints públicos + privados con X-API-Key de DBB_API_KEY)
    - collectors/gaceta.py    -> GacetaCollector    (FileCollector: PDFs
      reanudables de la Gaceta Oficial de Bolivia)

Reutiliza los helpers de collectors/utils.py y agrega el manifiesto único
de corrida (data/manifest.json) con TODOS los TypedResult/FileResult de
ambas fuentes.

Uso:
    from collectors.collect_all import run_all
    # o desde la CLI: uv run python main.py update-dataset

Salidas:
    (dolarblue) data/raw/, data/sources/, data/normalized/, chart_6m.csv, history_full.csv
    (gaceta)    data/gaceta/<edicion>/norma_<id>.pdf + manifiesto.csv
    (corrida)   data/manifest.json — updated_at + conteos + filas por resultado
                (lo lee main.py update-dataset para decidir frescura)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from collectors.dolarblue import DolarBlueCollector
from collectors.gaceta import GacetaCollector
from collectors.models import TypedResult
from collectors.utils import write_chart_6m_copy, write_gaceta_manifest


def run_dolarblue(data_dir: Path) -> list[TypedResult]:
    """Corre DolarBlueCollector y deja las salidas de compatibilidad."""
    print("== dolarblue: API Dolar Blue Bolivia ==")
    results = DolarBlueCollector(data_dir).collect()

    print(f"{'endpoint':<34}{'status':<12}{'published_date'}")
    for res in results:
        published = res.published_date.date() if res.published_date else "-"
        print(f"{res.meta['endpoint']:<34}{res.status:<12}{published}")

    if chart := write_chart_6m_copy(data_dir):
        print(f"chart_6m.csv: {len(chart.read_text().splitlines()) - 1} rows")

    norms = sorted((data_dir / "normalized").glob("*.csv"))
    if norms:
        print("normalized/:")
        for f in norms:
            print(f"  {f.name}: {len(f.read_text().splitlines()) - 1} rows")
    return results


def run_gaceta(data_dir: Path, tipo: int, days: int, max_pages: int) -> list[TypedResult]:
    """Corre GacetaCollector (cache reanudable) y escribe su manifiesto de PDFs."""
    print("\n== gaceta: Gaceta Oficial de Bolivia ==")
    gaceta_dir = data_dir / "gaceta"
    results: list[TypedResult] = GacetaCollector(
        tipo=tipo, days=days, max_pages=max_pages, out_dir=gaceta_dir
    ).collect()
    write_gaceta_manifest(gaceta_dir, results)
    ok = sum(1 for r in results if r.status == "ok")
    cache = sum(1 for r in results if r.status == "cache")
    print(f"{len(results)} normas ({ok} nuevos, {cache} en cache)")
    return results


def _manifest_row(source: str, res: TypedResult, data_dir: Path) -> dict:
    """Una fila de manifiesto por TypedResult (api) o FileResult (file)."""
    if source == "gaceta":
        artifact = ""
        if res.path:
            try:
                artifact = str(res.path.relative_to(data_dir))
            except ValueError:
                artifact = str(res.path)
        detail = "/".join(
            str(m)
            for m in (res.meta.get("tipo"), res.meta.get("edicion"), res.meta.get("norma_id"))
            if m
        )
    else:
        artifact = f"sources/{res.meta['endpoint']}.json" if res.status == "ok" else ""
        detail = str(res.meta.get("familia", ""))
        for err_key in ("norm_error", "_error"):
            if res.meta.get(err_key):
                detail = f"{detail} | {res.meta[err_key]}" if detail else str(res.meta[err_key])
    return {
        "source": source,
        "kind": "file" if source == "gaceta" else "api",
        "url": res.url,
        "status": res.status,
        "published_date": res.published_date.isoformat() if res.published_date else "",
        "extracted_date": res.extracted_date.isoformat(),
        "artifact": artifact,
        "detail": detail,
    }


def write_manifest(data_dir: Path, groups: dict[str, list[TypedResult]]) -> Path:
    """data/manifest.json: marca de frescura + bitácora de la corrida.

    updated_at + conteos por fuente + una fila por TypedResult/FileResult
    (url, estado, fechas, artefacto). Lo lee main.py update-dataset para
    decidir si toca re-descargar (> --max-age horas desde updated_at).
    """
    rows = [
        _manifest_row(source, res, data_dir)
        for source, results in groups.items()
        for res in results
    ]
    sources = {}
    for name, results in groups.items():
        counts: dict[str, int] = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        sources[name] = {"results": len(results), **counts}
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "results": rows,
    }
    out = data_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def print_summary(groups: dict[str, list[TypedResult]]) -> None:
    print("\n== resumen de corrida ==")
    for source, results in groups.items():
        counts: dict[str, int] = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        breakdown = ", ".join(f"{s} {n}" for s, n in sorted(counts.items())) or "sin resultados"
        print(f"{source}: {len(results)} ({breakdown})")


def run_all(
    data_dir: Path,
    *,
    skip_dolarblue: bool = False,
    skip_gaceta: bool = False,
    gaceta_tipo: int = 13,
    gaceta_days: int = 122,
    gaceta_max_pages: int = 25,
) -> dict[str, list[TypedResult]] | None:
    """Colecta las fuentes pedidas y escribe data/manifest.json.

    Punto de entrada reutilizable (lo llama main.py):
    corre DolarBlueCollector/GacetaCollector según flags y deja el manifiesto
    de corrida (updated_at + conteos + filas por resultado). Devuelve los
    grupos colectados, o None si se saltaron todas las fuentes.
    """
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    groups: dict[str, list[TypedResult]] = {}
    if not skip_dolarblue:
        groups["dolarblue"] = run_dolarblue(data_dir)
    if not skip_gaceta:
        groups["gaceta"] = run_gaceta(data_dir, gaceta_tipo, gaceta_days, gaceta_max_pages)

    if not groups:
        print("nada que colectar: todas las fuentes saltadas")
        return None

    manifest = write_manifest(data_dir, groups)
    print_summary(groups)
    print(f"manifiesto de corrida: {manifest}")
    return groups
