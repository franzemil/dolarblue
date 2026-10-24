"""Writer de tablas normalizadas: filas tipadas -> CSV con procedencia.

Compartido por el collector (escritura en vivo) y por JsonProcessor (replay
desde data/sources/) para garantizar CSVs byte-idénticos entre ambos caminos.
"""

from __future__ import annotations

from pathlib import Path


def write_tables(tables: dict[str, list[dict]], norm_dir: Path) -> list[Path]:
    """Escribe data/normalized/<tabla>.csv respetando el orden de aparición."""
    norm_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for table, rows in tables.items():
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        lines = [",".join(keys)]
        lines += [",".join(str(row.get(k, "")) for k in keys) for row in rows]
        out = norm_dir / f"{table}.csv"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(out)
    return files
