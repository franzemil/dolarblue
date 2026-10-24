"""JsonProcessor: replay de data/sources/*.json -> data/normalized/*.csv.

No hace red: desenrolla el body guardado por el collector, normaliza con
collectors/normalizers.py y escribe las tablas con el mismo writer del
collector (byte-idénticas si el snapshot no cambió).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from collectors.normalizers import NormalizeError, family_records, normalize, unwrap_envelope
from collectors.writers import write_tables
from processors.results import ProcessResult
from processors.stage import Stage

CORE_FAMILIES: tuple[str, ...] = ("chart_6m", "bancos", "velas")
ALL_FAMILIES: tuple[str, ...] = (
    "quote",
    "chart_6m",
    "plataformas",
    "bancos",
    "velas",
    "orders",
    "snapshots",
)


class JsonProcessor(Stage):
    """Replay de respuestas guardadas; familias core por defecto."""

    def __init__(
        self,
        data_dir: Path,
        families: tuple[str, ...] = CORE_FAMILIES,
        dry_run: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.families = families
        self.dry_run = dry_run

    def transform(self, items: list[dict]) -> list[ProcessResult]:
        tables: dict[str, list[dict]] = {}
        results: list[ProcessResult] = []
        for item in items:
            familia = item["familia"]
            if familia not in self.families:
                continue
            extracted = datetime.now(timezone.utc)
            try:
                inner, _published = unwrap_envelope(item["payload"])
                payload = normalize(familia, inner, item["name"])
                table, rows = family_records(
                    familia, payload, item["url"], item["extracted_date"]
                )
                if table:
                    tables.setdefault(table, []).extend(rows)
                results.append(
                    ProcessResult(
                        input_path=item["path"],
                        status="ok",
                        extracted_date=extracted,
                        meta={"endpoint": item["name"], "familia": familia, "rows": len(rows)},
                    )
                )
            except (NormalizeError, ValueError, ValidationError) as e:
                results.append(
                    ProcessResult(
                        input_path=item["path"],
                        status="error",
                        extracted_date=extracted,
                        meta={
                            "endpoint": item["name"],
                            "familia": familia,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                )
        if not self.dry_run and tables:
            files = {f.stem: f for f in write_tables(tables, self.data_dir / "normalized")}
            for res in results:
                table = res.meta.get("familia")
                if res.status == "ok" and table in files:
                    res.output_path = files[table]
        return results
