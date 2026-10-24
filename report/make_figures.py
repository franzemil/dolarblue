"""make_figures.py — CSVs vigentes → figures/*.pdf + tables/*.tex.

Regenera TODO el material gráfico y tabular del informe desde los datos
locales más recientes (data/), sin llamar a la API. Parte del pipeline
`make render` (ver Makefile).

Uso (desde 01_intro/ o desde final_project/):
    uv run python class_one/final_project/report/make_figures.py

Salidas:
    figures/fig_serie_blue_oficial.pdf  — serie horaria blue vs oficial (6m)
    figures/fig_brecha.pdf              — brecha blue−oficial (history_full)
    tables/tab_resumen.tex              — estadísticas de la serie y la brecha
    tables/data_stamp.tex               — "datos al <última extracción UTC>"
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display; genera PDFs directamente

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Rutas (independientes del cwd)
# ----------------------------------------------------------------------------
REPORT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPORT_DIR.parent / "data"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

# Estilo compartido del proyecto (charts/): misma paleta y rcParams
sys.path.insert(0, str(REPORT_DIR.parent))
from charts.estilo import (BLUE, GRAY, GREEN, ORANGE, RED,  # noqa: E402
                           aplicar_estilo, finish, fmt_es, tex)

aplicar_estilo()

# ---------------------------------------------------------------- utilidades

_fmt_es = fmt_es   # alias: las tablas usan el nombre con guion bajo
_tex = tex


EXCLUDED_PLATFORMS = ("saldoar", "p2p_mexc", "p2p_okx")


def _platform_visible(name: str) -> bool:
    """Plataformas fuera del informe (OKX, MEXC, SaldoAR)."""
    lowered = name.lower()
    return not any(token in lowered for token in EXCLUDED_PLATFORMS)


def _load(rel: str, **kw) -> pd.DataFrame | None:
    path = DATA_DIR / rel
    if not path.exists():
        print(f"  [aviso] {rel} no existe — se omite", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as exc:  # noqa: BLE001 — un CSV corrupto no debe romper el render
        print(f"  [aviso] {rel} ilegible ({exc}) — se omite", file=sys.stderr)
        return None


def _finish(fig, name: str) -> None:
    finish(fig, FIG_DIR / f"{name}.pdf", label=f"  figures/{name}.pdf")


# ------------------------------------------------------------------ figuras


def fig_serie_blue_oficial() -> None:
    df = _load("chart_6m.csv")
    if df is None or df.empty:
        return
    ts = pd.to_datetime(df["t"], unit="s", utc=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(ts, df["bb"], color=BLUE, lw=1.2, label="Blue compra (bb)")
    ax.plot(ts, df["bs"], color=BLUE, lw=1.2, alpha=0.55, ls="--", label="Blue venta (bs)")
    ax.plot(ts, df["ob"], color=ORANGE, lw=1.2, label="Oficial compra (ob)")
    ax.plot(ts, df["os"], color=ORANGE, lw=1.2, alpha=0.55, ls="--", label="Oficial venta (os)")
    ax.set_ylabel("Bs / USD")
    ax.set_title("Serie horaria blue vs oficial (últimos ~6 meses)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y", tz=None))
    _finish(fig, "fig_serie_blue_oficial")


def fig_brecha() -> pd.DataFrame | None:
    df = _load("history_full.csv")
    if df is None or df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    blue_mid = (df["blue_buy"] + df["blue_sell"]) / 2
    ofi_mid = (df["official_buy"] + df["official_sell"]) / 2
    gap = blue_mid - ofi_mid
    serie = pd.Series(gap.values, index=ts).resample("1D").mean().dropna()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(serie.index, serie.values, color=BLUE, lw=1.4)
    ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    ax.fill_between(serie.index, serie.values, 0, color=BLUE, alpha=0.12)
    ax.set_ylabel("Brecha (Bs/USD)")
    ax.set_title("Brecha diaria blue − oficial (promedio del día, history_full)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y", tz=None))
    stats = {
        "gap_mean": float(serie.mean()),
        "gap_max": float(serie.max()),
        "n_days": int(serie.size),
        "start": serie.index.min(),
        "end": serie.index.max(),
    }
    _finish(fig, "fig_brecha")
    return stats



# ------------------------------------------------------------------- tablas


def tab_resumen(gap_stats: dict | None) -> None:
    df = _load("chart_6m.csv")
    if df is None or df.empty:
        return
    blue_mid = (df["bb"] + df["bs"]) / 2
    ofi_mid = (df["ob"] + df["os"]) / 2
    rows = [
        ("Puntos horarios", f"{len(df):,}".replace(",", "."), ""),
        ("Blue medio", _fmt_es(blue_mid.mean()), "Bs/USD"),
        ("Oficial medio", _fmt_es(ofi_mid.mean()), "Bs/USD"),
        ("Spread blue medio (bs$-$bb)", _fmt_es((df['bs'] - df['bb']).mean()), "Bs/USD"),
        ("Volatilidad diaria blue ($\\sigma$ de retornos)",
         _fmt_es(blue_mid.pct_change().std() * 100), r"\%"),
    ]
    if gap_stats:
        rows += [
            ("Días de brecha (history_full)", f"{gap_stats['n_days']}", ""),
            ("Brecha media", _fmt_es(gap_stats["gap_mean"]), "Bs/USD"),
            ("Brecha máxima", _fmt_es(gap_stats["gap_max"]), "Bs/USD"),
        ]
    body = "\n".join(f"{_tex(n)} & {v} & {u} \\\\" for n, v, u in rows)
    (TAB_DIR / "tab_resumen.tex").write_text(
        f"% Generado por make_figures.py — no editar a mano\n"
        f"\\begin{{tabular}}{{@{{}}lll@{{}}}}\n\\toprule\n"
        f"Métrica & Valor & Unidad \\\\\n\\midrule\n{body}\n\\bottomrule\n"
        f"\\end{{tabular}}\n",
        encoding="utf-8",
    )
    print("  tables/tab_resumen.tex")


def data_stamp() -> None:
    """Fecha de la última extracción: máx extracted_date entre los insumos."""
    stamps: list[str] = []
    for rel, col in [
        ("chart_6m.csv", "extracted_date"),
        ("history_full.csv", None),
    ]:
        df = _load(rel)
        if df is None:
            continue
        if col and col in df.columns:
            stamps.extend(df[col].dropna().astype(str))
        elif "timestamp" in df.columns:  # history_full (epoch)
            stamps.append(
                datetime.fromtimestamp(
                    float(df["timestamp"].max()), tz=timezone.utc
                ).isoformat()
            )
    if not stamps:
        texto = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC (sin datos)")
    else:
        latest = pd.to_datetime(max(stamps))
        texto = latest.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    (TAB_DIR / "data_stamp.tex").write_text(texto + "\n", encoding="utf-8")
    print(f"  tables/data_stamp.tex  ({texto})")


# --------------------------------------------------------------------- main

def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    TAB_DIR.mkdir(exist_ok=True)
    print(f"Datos: {DATA_DIR}")
    gap_stats = fig_brecha()
    fig_serie_blue_oficial()
    tab_resumen(gap_stats)
    data_stamp()


if __name__ == "__main__":
    main()
