"""CLI principal del pipeline (typer).

`update-dataset` mantiene data/ al día usando el manifiesto de corrida
(data/manifest.json, escrito por collectors.collect_all.run_all en cada corrida):

    - no existe manifiesto (clon limpio: data/ es gitignored) -> descarga TODO
    - manifiesto con más de --max-age horas (default 24)      -> descarga TODO
    - manifiesto fresco                                       -> no llama a la API
                                                                (salvo --force)

Uso (desde 01_intro/):
    uv run python final_project/main.py update-dataset
    uv run python final_project/main.py update-dataset --dry-run
    uv run python final_project/main.py update-dataset --force --skip-gaceta
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

from collectors.collect_all import run_all

app = typer.Typer(add_completion=False, help="Pipeline de datos: dólar blue Bolivia.")


@app.callback()
def _callback() -> None:
    """Pipeline de datos del proyecto (fuentes: dolarblue API + Gaceta)."""

DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_MAX_AGE_HOURS = 24.0


def _read_updated_at(manifest: Path) -> datetime | None:
    """updated_at del manifiesto en UTC, o None si falta o está corrupto."""
    if not manifest.exists():
        return None
    try:
        updated_at = datetime.fromisoformat(
            json.loads(manifest.read_text(encoding="utf-8"))["updated_at"]
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return updated_at


def _fmt_age(age: timedelta) -> str:
    hours = age.total_seconds() / 3600
    return f"{hours:.1f} h" if hours < 48 else f"{age.days} d"


@app.command("update-dataset")
def update_dataset(
    max_age: float = typer.Option(
        DEFAULT_MAX_AGE_HOURS, "--max-age", help="Horas de frescura del manifiesto antes de re-descargar."
    ),
    force: bool = typer.Option(False, "--force", help="Descargar aunque el manifiesto esté fresco."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Solo mostrar la decisión; no toca la API."),
    out: Path = typer.Option(DATA_DIR, "--out", help="Directorio data/."),
    skip_dolarblue: bool = typer.Option(False, "--skip-dolarblue", help="No colectar la API dolarblue."),
    skip_gaceta: bool = typer.Option(False, "--skip-gaceta", help="No colectar la Gaceta Oficial."),
    gaceta_tipo: int = typer.Option(13, "--gaceta-tipo", help="10 especial, 11 prop. intelectual, 13 normal."),
    gaceta_days: int = typer.Option(122, "--gaceta-days", help="Ventana de ediciones en días."),
    gaceta_max_pages: int = typer.Option(25, "--gaceta-max-pages", help="Tope de páginas del listado."),
) -> None:
    """Descarga todas las fuentes si no hay manifiesto fresco (< --max-age horas)."""
    load_dotenv(ENV_FILE)
    manifest = out / "manifest.json"
    updated_at = _read_updated_at(manifest)
    now = datetime.now(timezone.utc)

    if updated_at is None:
        download, reason = True, f"no existe {manifest} (clon limpio o manifiesto corrupto)"
    else:
        age = now - updated_at
        if force:
            download, reason = True, "--force"
        elif age <= timedelta(hours=max_age):
            download, reason = False, f"manifiesto fresco ({_fmt_age(age)} <= {max_age:g} h)"
        else:
            download, reason = True, f"manifiesto viejo ({_fmt_age(age)} > {max_age:g} h)"

    print(f"decisión: {'DESCARGAR' if download else 'OMITIR'} — {reason}")
    if dry_run:
        print("dry-run: sin cambios, sin llamadas a la API")
        raise typer.Exit(0)
    if not download:
        print("usá --force para re-descargar de todos modos")
        raise typer.Exit(0)

    groups = run_all(
        out,
        skip_dolarblue=skip_dolarblue,
        skip_gaceta=skip_gaceta,
        gaceta_tipo=gaceta_tipo,
        gaceta_days=gaceta_days,
        gaceta_max_pages=gaceta_max_pages,
    )
    if groups is None:
        raise typer.Exit(0)


@app.command("process")
def process_dataset(
    skip_dolarblue: bool = typer.Option(False, "--skip-dolarblue", help="No procesar la rama dolarblue."),
    skip_gaceta: bool = typer.Option(False, "--skip-gaceta", help="No procesar la rama Gaceta (LLM)."),
    only: str | None = typer.Option(None, "--only", help="Familias JSON a replayear (coma-separadas)."),
    all_families: bool = typer.Option(False, "--all-families", help="Replayear todas las familias JSON."),
    gaceta_backend: str = typer.Option("llm", "--gaceta-backend", help="llm | keywords (legacy)."),
    provider: str | None = typer.Option(None, "--provider", help="Proveedor LLM: zai | deepseek (default LLM_PROVIDER)."),
    days: int | None = typer.Option(None, "--days", help="Limitar extracción LLM a las últimas N jornadas."),
    since: str | None = typer.Option(None, "--since", help="Limitar extracción LLM desde YYYY-MM-DD."),
    limit: int | None = typer.Option(None, "--limit", help="Tope de PDFs nuevos a extraer."),
    dry_run: bool = typer.Option(False, "--dry-run", help="No escribe ni llama a la API."),
    force: bool = typer.Option(False, "--force", help="Ignora el cache LLM (re-extrae)."),
    out: Path = typer.Option(DATA_DIR, "--out", help="Directorio data/."),
) -> None:
    """Procesa snapshots en disco: replay JSON + LLM de la Gaceta (sin re-descargar).

    Salidas: data/processed/market_hourly.csv (horaria, dolarblue),
    data/processed/gaceta_daily.csv (diaria, Gaceta) y proc_manifest.json.
    """
    from processors.indexers import (
        EventWindowIndexer,
        GacetaDailyIndexer,
        MarketHourlyIndexer,
    )
    from processors.json_processor import ALL_FAMILIES, CORE_FAMILIES, JsonProcessor
    from processors.llm_client import build_client
    from processors.llm_file_processor import LLMFileProcessor
    from processors.results import ProcessResult
    from processors.snapshot_reader import SnapshotReader
    from processors.stage import Pipeline

    load_dotenv(ENV_FILE)

    if only:
        families = tuple(f.strip() for f in only.split(",") if f.strip())
    elif all_families:
        families = ALL_FAMILIES
    else:
        families = CORE_FAMILIES

    since_dt = datetime.fromisoformat(since) if since else None
    results: list[ProcessResult] = []
    stages: dict[str, dict[str, int]] = {}

    def run_branch(pipeline: Pipeline) -> None:
        data = None
        for name, stage in pipeline.steps:
            data = stage.fit_transform(data)
            produced = data if isinstance(data, list) else [data]
            for res in produced:
                if not isinstance(res, ProcessResult):
                    continue
                res.meta["stage"] = name
                results.append(res)
                counts = stages.setdefault(name, {})
                counts[res.status] = counts.get(res.status, 0) + 1

    if not skip_dolarblue:
        run_branch(
            Pipeline(
                [
                    ("read", SnapshotReader("dolarblue", out)),
                    ("normalize", JsonProcessor(out, families=families, dry_run=dry_run)),
                    ("index", MarketHourlyIndexer(out, dry_run=dry_run)),
                ]
            )
        )

    if not skip_gaceta:
        client = build_client(provider)
        if not dry_run and not client.available:
            raise typer.BadParameter(
                f"falta {client.api_key_env} en .env (usá --dry-run o --skip-gaceta)"
            )
        print(f"proveedor LLM: {client.provider} ({client.model})")
        run_branch(
            Pipeline(
                [
                    ("read", SnapshotReader("gaceta", out)),
                    (
                        "extract",
                        LLMFileProcessor(
                            out,
                            client=client,
                            days=days,
                            since=since_dt,
                            limit=limit,
                            dry_run=dry_run,
                            force=force,
                        ),
                    ),
                    ("index", GacetaDailyIndexer(out, backend=gaceta_backend, dry_run=dry_run)),
                ]
            )
        )

    if not skip_dolarblue and not skip_gaceta:
        run_branch(
            Pipeline([("event_windows", EventWindowIndexer(out, dry_run=dry_run))])
        )

    total_cost = sum(r.cost_usd for r in results)
    for name, counts in stages.items():
        breakdown = ", ".join(f"{s} {n}" for s, n in sorted(counts.items()))
        print(f"{name}: {breakdown}")
    print(f"costo LLM: ${total_cost:.4f}")
    if dry_run:
        print("dry-run: sin escrituras ni llamadas a la API")
        return

    processed = out / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": round(total_cost, 6),
        "stages": stages,
        "results": [
            {
                "stage": r.meta.get("stage", ""),
                "status": r.status,
                "input_path": str(r.input_path) if r.input_path else "",
                "output_path": str(r.output_path) if r.output_path else "",
                "extracted_date": r.extracted_date.isoformat(),
                "cost_usd": round(r.cost_usd, 6),
                "meta": r.meta,
            }
            for r in results
        ],
    }
    manifest = processed / "proc_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifiesto de procesamiento: {manifest}")
    print(
        f"salidas: {processed / 'market_hourly.csv'} | "
        f"{processed / 'gaceta_daily.csv'} | {processed / 'gaceta_event_windows.csv'}"
    )


if __name__ == "__main__":
    app()
