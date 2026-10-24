"""make_animation.py — animación del pipeline de datos (dólar blue Bolivia).

Recorre, en ocho escenas, el camino real de los datos:

    1. fuentes (API dolarblue + Gaceta en PDF)
    2. ingesta con manifiesto de 24 h y cache de archivos
    3. extracción LLM con cache sha256 y reintentos acotados
    4. procesadores que hacen replay desde disco (sin red)
    5. MarketHourlyIndexer: features y NaN por diseño
    6. GacetaDailyIndexer: calendario completo y ceros estructurales
    7. EDA y reglas explícitas para datos faltantes
    8. resumen de garantías

Todo se calcula desde `data/` en tiempo de render: no llama a la API. Escribe
solo en `report/anim/`, `report/figures/` y, con `--export-scenes`, en el
directorio que se indique (por defecto el dashboard usa
`dashboard/assets/pipeline/`).

Uso:
    uv run python report/make_animation.py              # MP4 + GIF + póster
    uv run python report/make_animation.py --poster-only
    uv run python report/make_animation.py --preview 6  # MP4 corto de prueba
    uv run python report/make_animation.py --export-scenes dashboard/assets/pipeline
    uv run python report/make_animation.py --export-clips report/anim/scenes
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# En un notebook no forzamos Agg: rompería el backend inline y `plt.show()`.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# --------------------------------------------------------------------------- #
# Rutas, formato y paleta
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG_DIR = REPORT / "figures"
ANIM_DIR = REPORT / "anim"

W, H = 16.0, 9.0
FPS = 30


# --------------------------------------------------------------------------- #
# Guion: una ficha por escena (video, notebook y dashboard)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneSpec:
    """Contenido y explicación de una escena.

    Es la fuente única para el video, el póster, las celdas del notebook y los
    frames que consume el dashboard. Los conteos se completan con `Stats` en
    tiempo de render; el texto no repite cifras fijas.
    """

    key: str
    number: int
    short: str
    title: str
    subtitle: str
    duration: float
    lead: str
    bullets: tuple[str, ...]
    rule: str
    source: str


SCENE_SPECS: list[SceneSpec] = [
    SceneSpec(
        key="fuentes",
        number=1,
        short="Fuentes",
        title="Dos fuentes, dos ritmos",
        subtitle="cotizaciones de mercado y normas publicadas en PDF",
        duration=6.0,
        lead=(
            "El pipeline tiene dos entradas que no se parecen: snapshots JSON "
            "de cotizaciones y PDFs de normas. El colector no interpreta el "
            "contenido; persiste el original y deja la normalización para la "
            "etapa siguiente."
        ),
        bullets=(
            "API: endpoints públicos y privados, `X-API-Key` y ventana móvil de 6 meses.",
            "Gaceta: listado de ediciones tipo 13, ventana de 122 días y un PDF por norma.",
            "Manifiesto de corrida: si `updated_at` tiene menos de 24 h, no se descarga nada.",
        ),
        rule="La descarga es la excepción: el manifiesto decide si la corrida está fresca.",
        source="collectors/dolarblue.py · collectors/gaceta.py · main.py update-dataset",
    ),
    SceneSpec(
        key="ingesta",
        number=2,
        short="Ingesta",
        title="Ingesta: descargar una sola vez",
        subtitle="manifiesto de 24 h + cache de archivo antes de tocar la red",
        duration=9.0,
        lead=(
            "La API se normaliza con Pydantic y se escribe como CSV. Los PDFs "
            "se buscan primero en disco: si el archivo existe y supera 1 KB, "
            "se reutiliza sin hacer ningún request."
        ),
        bullets=(
            "Snapshot crudo → `raw/`; normalización → `normalized/*.csv` + `history_full.csv`.",
            "PDF ausente → `GET` + validación `%PDF`; la descarga tiene 3 intentos con backoff.",
            "Límite conocido: los PDFs ya cacheados se validan por tamaño, no por firma; `%PDF` se exige solo en descargas nuevas.",
        ),
        rule=(
            "El cache de archivo evita pedir dos veces el mismo PDF; el "
            "manifiesto evita repetir la corrida completa."
        ),
        source="collectors/base.py::download_file · collectors/gaceta.py",
    ),
    SceneSpec(
        key="llm",
        number=3,
        short="LLM",
        title="Extracción LLM: pagar una sola vez",
        subtitle="hash del PDF, cache por proveedor/modelo y reintentos acotados",
        duration=9.0,
        lead=(
            "Cada PDF se identifica por el SHA-256 de su contenido. La clave "
            "de cache combina hash, proveedor, modelo y versión; un hit no "
            "llama al LLM."
        ),
        bullets=(
            "Miss: `deepseek/deepseek-flash` → JSON → Pydantic → `cache.put`.",
            "Fallo de red, LLM o esquema: 3 intentos con backoff; si persiste, `schema_invalid` o `llm_failed`.",
            "`relevancia_fx` es una salida: `nula` se conserva con peso 0, se cachea y **no** se reintenta.",
        ),
        rule="Solo se reintenta un fallo técnico; una etiqueta baja o nula no es un fallo.",
        source="processors/llm_file_processor.py · processors/llm_manifest.py",
    ),
    SceneSpec(
        key="replay",
        number=4,
        short="Replay",
        title="Procesadores: replay desde disco",
        subtitle="sin red: se reconstruye todo desde lo ya persistido",
        duration=6.0,
        lead=(
            "Los procesadores no descargan: leen lo persistido y reconstruyen "
            "las series. `SnapshotReader` alimenta dos indexers independientes."
        ),
        bullets=(
            "`MarketHourlyIndexer` → `market_hourly.csv` (horas observadas, 0 duplicados).",
            "`GacetaDailyIndexer` → `gaceta_daily.csv` (calendario diario, días con y sin normas).",
            "Cada serie conserva su resolución; el cruce horario × diario se hace al modelar con `merge_asof`.",
        ),
        rule="Reprocesar es barato y determinista: no hay red ni LLM en esta etapa.",
        source="processors/indexers.py · main.py process",
    ),
    SceneSpec(
        key="indexer_h",
        number=5,
        short="Indexer H",
        title="Indexer horario: una fila por hora",
        subtitle="features derivadas y NaN de warm-up / horizonte futuro",
        duration=9.0,
        lead=(
            "El indexer ordena y deduplica por timestamp, calcula precios "
            "medios y deriva las features. No imputa: el calentamiento y el "
            "futuro no observado quedan `NaN`."
        ),
        bullets=(
            "`blue_mid`, `oficial_mid`, `spread_blue`, `brecha_pct`, `ret_blue_1h`, `vol_24h`.",
            "`fwd_h = blue_mid.shift(-h) / blue_mid - 1` para h = 1, 6, 24, 72, 168.",
            "Las primeras 24 h de `vol_24h` y el tramo final de cada `fwd_h` son `NaN` por diseño.",
        ),
        rule=(
            "El indexer declara la ausencia; la decisión de imputar o "
            "descartar es del modelado."
        ),
        source="processors/indexers.py::MarketHourlyIndexer · models/regresion.py::grilla_horaria",
    ),
    SceneSpec(
        key="indexer_d",
        number=6,
        short="Indexer D",
        title="Indexer diario: calendario completo",
        subtitle="cada día del rango existe; sin publicación → cero estructural",
        duration=9.0,
        lead=(
            "El indexer diario construye un calendario entre la primera y la "
            "última fecha. Cada día existe, haya o no publicación."
        ),
        bullets=(
            "Día con norma: conteos, `monto_usd_sum`, `relevancia_max` y `confidence_mean`.",
            "Día sin norma: conteos, montos y relevancia se rellenan con 0; `confidence_mean` queda `NaN`.",
            "`gaceta_daily.csv` cubre todo el calendario; el panel de modelado recorta los primeros días sin retorno previo.",
        ),
        rule=(
            "El 0 es ausencia de publicación, no un dato faltante; "
            "`fillna(0)` no inventa confianza."
        ),
        source="processors/indexers.py::GacetaDailyIndexer · modelado/prepare_datos.py",
    ),
    SceneSpec(
        key="eda_na",
        number=7,
        short="EDA y NA",
        title="EDA y faltantes: decidir, no rellenar a ciegas",
        subtitle="interpolación acotada, ceros estructurales y NaN explícitos",
        duration=7.0,
        lead=(
            "El EDA separa tres ausencias: el precio cero del crudo, el hueco "
            "horario y el día sin Gaceta. Antes de imputar hay que entender "
            "por qué falta cada dato: un `NaN` es una pregunta, no un cero."
        ),
        bullets=(
            "Precio 0: snapshot inválido → se descarta; no se imputa.",
            "Hueco ≤ 3 h: `interpolate(limit=3)`; un hueco mayor queda `NaN`.",
            "Día sin Gaceta: conteos, montos y relevancia = 0; `confidence_mean` = `NaN`.",
            "`vol_24h` (warm-up) y `fwd_*` (futuro): `NaN` por diseño, no 0.",
            "Antes de imputar: cuantificar, entender el mecanismo y documentar la regla.",
        ),
        rule=(
            "Un `NaN` es una pregunta, no un cero: primero se interpreta y "
            "después se aplica una regla explícita. Rellenar a ciegas inventa "
            "datos y sesga el modelo."
        ),
        source="models/regresion.py · processors/indexers.py",
    ),
    SceneSpec(
        key="resumen",
        number=8,
        short="Resumen",
        title="Resumen: cuatro garantías",
        subtitle="lo que el pipeline evita repetir y lo que declara como faltante",
        duration=5.0,
        lead=(
            "El pipeline evita repetir trabajo y declara lo que falta. Estas "
            "cuatro garantías son la lectura operativa del recorrido."
        ),
        bullets=(
            "No re-descarga: manifiesto de 24 h + cache de archivo.",
            "No re-extracción: SHA-256 + proveedor + modelo → todos los hits, costo marginal $0.",
            "Reintento acotado: 3 intentos; el fallo no se cachea.",
            "Faltantes explícitos: huecos ≤ 3 h interpolados, días sin norma en 0, warm-up declarado.",
        ),
        rule="Hacia afuera es idempotente; hacia adentro es explícito.",
        source="report/make_animation.py · data/processed/llm_manifest.json",
    ),
]

SCENE_BY_KEY: dict[str, SceneSpec] = {spec.key: spec for spec in SCENE_SPECS}
SCENES: list[tuple[str, float]] = [(spec.short, spec.duration) for spec in SCENE_SPECS]
DURATION = sum(spec.duration for spec in SCENE_SPECS)
POSTER_T = DURATION - 1.2


BLUE = "#2563eb"
ORANGE = "#f59e0b"
GREEN = "#10b981"
RED = "#ef4444"
GRAY = "#6b7280"
DARK = "#0f172a"
MID = "#334155"
SOFT = "#94a3b8"
BORDER = "#e2e8f0"
CANVAS = "#ffffff"
PANEL = "#f8fafc"
BLUE_BG = "#eff6ff"
ORANGE_BG = "#fffbeb"
GREEN_BG = "#ecfdf5"
RED_BG = "#fef2f2"
GRAY_BG = "#f1f5f9"

ESTILO = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": DARK,
    "figure.facecolor": CANVAS,
    "savefig.facecolor": CANVAS,
}

MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    snapshots: int = 0
    zero_price: int = 0
    hours: int = 0
    hours_expected: int = 0
    market_cols: int = 0
    duplicates: int = 0
    warmup_nan: int = 0
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    daily_blue: pd.Series | None = None
    zoom: pd.DataFrame | None = None
    chart_rows: int = 0
    chart_dups: int = 0
    first_date: str = "—"
    last_date: str = "—"
    first_blue: float = float("nan")
    first_brecha: float = float("nan")
    first_vol: float = float("nan")
    first_fwd24: float = float("nan")
    last_blue: float = float("nan")
    last_brecha: float = float("nan")
    last_vol: float = float("nan")
    last_fwd24: float = float("nan")
    pdfs: int = 0
    extracted: int = 0
    cache_entries: int = 0
    provider: str = "—"
    model: str = "—"
    relevance: dict[str, int] = field(default_factory=dict)
    days_total: int = 0
    days_with_docs: int = 0
    docs_by_day: pd.Series | None = None
    monto_null: int = 0
    confidence_mean: float = float("nan")
    updated: str = "—"

    @property
    def gap_hours(self) -> int:
        return self.gaps[0][2] if self.gaps else 0

    @property
    def gap_start(self) -> pd.Timestamp | None:
        return self.gaps[0][0] if self.gaps else None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [aviso] no existe {path}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 — un CSV roto no debe romper el render
        print(f"  [aviso] ilegible {path}: {exc}", file=sys.stderr)
        return None


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _fmt_num(value: float, nd: int = 2) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{value:.{nd}f}".replace(".", ",")


def load_stats() -> Stats:
    stats = Stats()

    market = _read_csv(DATA / "processed" / "market_hourly.csv", parse_dates=["fecha"])
    if market is None or market.empty:
        raise SystemExit(
            "falta data/processed/market_hourly.csv: corré `main.py process` primero"
        )
    market = market.sort_values("fecha").reset_index(drop=True)
    stats.hours = len(market)
    stats.market_cols = len(market.columns)
    stats.duplicates = int(market["fecha"].duplicated().sum())
    stats.warmup_nan = int(market["vol_24h"].isna().sum()) if "vol_24h" in market else 0
    stats.daily_blue = (
        market.set_index("fecha")["blue_mid"].resample("D").mean()
    )

    first, last = market.iloc[0], market.iloc[-1]
    stats.first_date = pd.Timestamp(first["fecha"]).strftime("%Y-%m-%d %H:%M")
    stats.last_date = pd.Timestamp(last["fecha"]).strftime("%Y-%m-%d %H:%M")
    stats.first_blue = float(first["blue_mid"])
    stats.first_brecha = float(first["brecha_pct"])
    stats.first_vol = float(first["vol_24h"]) if pd.notna(first["vol_24h"]) else float("nan")
    stats.first_fwd24 = float(first["fwd_24h"]) if pd.notna(first["fwd_24h"]) else float("nan")
    stats.last_blue = float(last["blue_mid"])
    stats.last_brecha = float(last["brecha_pct"])
    stats.last_vol = float(last["vol_24h"]) if pd.notna(last["vol_24h"]) else float("nan")
    stats.last_fwd24 = float(last["fwd_24h"]) if pd.notna(last["fwd_24h"]) else float("nan")

    chart = _read_csv(DATA / "normalized" / "chart_6m.csv")
    if chart is None:
        chart = _read_csv(DATA / "chart_6m.csv")
    if chart is not None and not chart.empty:
        stats.chart_rows = len(chart)
        stats.chart_dups = int(chart["t"].duplicated().sum()) if "t" in chart else 0

    idx = pd.DatetimeIndex(market["fecha"])
    full = pd.date_range(idx.min(), idx.max(), freq="h")
    stats.hours_expected = len(full)
    missing = full.difference(idx)
    if len(missing):
        blocks = (missing.to_series().diff() != pd.Timedelta(hours=1)).cumsum()
        for _, block in missing.to_series().groupby(blocks):
            stats.gaps.append((block.min(), block.max(), len(block)))

    if stats.gaps:
        start, end, _ = stats.gaps[0]
        zoom = market[
            (market["fecha"] >= start - pd.Timedelta(hours=8))
            & (market["fecha"] <= end + pd.Timedelta(hours=8))
        ][["fecha", "blue_mid"]]
        stats.zoom = zoom.reset_index(drop=True)

    history = _read_csv(DATA / "history_full.csv", usecols=["blue_sell"])
    if history is not None:
        stats.snapshots = len(history)
        stats.zero_price = int((history["blue_sell"] == 0).sum())

    gaceta = _read_csv(DATA / "processed" / "gaceta_llm.csv", dtype={"norma_id": str})
    if gaceta is not None and not gaceta.empty:
        stats.extracted = len(gaceta)
        rel = gaceta["relevancia_fx"].fillna("nula").value_counts()
        stats.relevance = {
            k: int(rel.get(k, 0)) for k in ("alta", "media", "baja", "nula")
        }
        stats.monto_null = int(gaceta["monto_usd"].isna().sum())
        if "confidence" in gaceta:
            stats.confidence_mean = float(gaceta["confidence"].mean())

    gaceta_daily = _read_csv(
        DATA / "processed" / "gaceta_daily.csv", parse_dates=["fecha"]
    )
    if gaceta_daily is not None and not gaceta_daily.empty:
        stats.days_total = len(gaceta_daily)
        stats.days_with_docs = int((gaceta_daily["n_docs_total"] > 0).sum())
        stats.docs_by_day = (
            gaceta_daily.set_index("fecha")["n_docs_total"].sort_index()
        )

    stats.pdfs = len(list((DATA / "gaceta").glob("*/norma_*.pdf")))

    manifest_path = DATA / "processed" / "llm_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("entries", {})
            stats.cache_entries = len(entries)
            providers = Counter(
                (e.get("provider", "—"), e.get("model", "—"))
                for e in entries.values()
            )
            if providers:
                (provider, model), _ = providers.most_common(1)[0]
                stats.provider, stats.model = provider, model
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [aviso] llm_manifest ilegible: {exc}", file=sys.stderr)

    chart = _read_csv(DATA / "chart_6m.csv")
    if chart is not None and "extracted_date" in chart.columns:
        stamps = chart["extracted_date"].dropna().astype(str)
        if not stamps.empty:
            try:
                stamp = pd.Timestamp(stamps.max())
                stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
                stats.updated = stamp.strftime("%Y-%m-%d %H:%M UTC")
            except (TypeError, ValueError):
                pass

    run_manifest = DATA / "manifest.json"
    if stats.updated == "—" and run_manifest.exists():
        try:
            payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            stamp = payload.get("updated_at")
            if stamp:
                dt = datetime.fromisoformat(stamp)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                stats.updated = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    if stats.updated == "—":
        stats.updated = pd.Timestamp(idx.max()).strftime("%Y-%m-%d %H:%M UTC")

    return stats


# --------------------------------------------------------------------------- #
# Primitivas de dibujo
# --------------------------------------------------------------------------- #
def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ease(p: float) -> float:
    p = _clip01(p)
    return p * p * (3 - 2 * p)


def _appear(p: float, start: float = 0.0, dur: float = 0.08) -> float:
    return _ease(_clip01((p - start) / dur)) if dur > 0 else 1.0


def txt(
    ax,
    x: float,
    y: float,
    s: str,
    fs: float = 10,
    color: str = DARK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    alpha: float = 1.0,
    family: str | None = None,
    linespacing: float = 1.2,
    zorder: int = 5,
) -> None:
    if alpha <= 0.01 or not s:
        return
    kwargs = {}
    if family:
        kwargs["fontfamily"] = family
    ax.text(
        x, y, s,
        fontsize=fs, color=color, fontweight=weight,
        ha=ha, va=va, alpha=alpha, linespacing=linespacing,
        clip_on=False, zorder=zorder, **kwargs,
    )


def card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str = "white",
    edge: str = BORDER,
    lw: float = 1.1,
    alpha: float = 1.0,
    radius: float = 0.12,
    shadow: bool = True,
    zorder: int = 2,
) -> None:
    if alpha <= 0.01:
        return
    style = f"round,pad=0,rounding_size={radius}"
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.035, y - 0.035), w, h,
                boxstyle=style, facecolor=DARK, edgecolor="none",
                alpha=0.05 * alpha, zorder=zorder - 1,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=style, facecolor=face, edgecolor=edge,
            linewidth=lw, alpha=alpha, zorder=zorder,
        )
    )


def node(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    sub: str | None = None,
    face: str = "white",
    edge: str = BORDER,
    title_color: str = DARK,
    sub_color: str = GRAY,
    title_fs: float = 10.5,
    sub_fs: float = 8.4,
    alpha: float = 1.0,
    align: str = "center",
    zorder: int = 3,
) -> None:
    card(ax, x, y, w, h, face=face, edge=edge, alpha=alpha, zorder=zorder)
    tx = x + w / 2 if align == "center" else x + 0.18
    ha = "center" if align == "center" else "left"
    if sub:
        txt(ax, tx, y + h * 0.63, title, fs=title_fs, color=title_color,
            weight="bold", ha=ha, alpha=alpha, linespacing=1.15, zorder=zorder + 1)
        txt(ax, tx, y + h * 0.29, sub, fs=sub_fs, color=sub_color,
            ha=ha, alpha=alpha, linespacing=1.15, zorder=zorder + 1)
    else:
        txt(ax, tx, y + h / 2, title, fs=title_fs, color=title_color,
            weight="bold", ha=ha, alpha=alpha, zorder=zorder + 1)


def pill(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    face: str = BLUE_BG,
    edge: str = BLUE,
    color: str = BLUE,
    fs: float = 8.6,
    alpha: float = 1.0,
    weight: str = "bold",
) -> None:
    card(ax, x, y, w, h, face=face, edge=edge, alpha=alpha,
         radius=h / 2, shadow=False, zorder=4)
    txt(ax, x + w / 2, y + h / 2, label, fs=fs, color=color,
        weight=weight, ha="center", alpha=alpha, zorder=5)


def diamond(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label: str,
    face: str = "white",
    edge: str = BLUE,
    color: str = DARK,
    fs: float = 8.6,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=face, edgecolor=edge,
                linewidth=1.2, alpha=alpha, zorder=3)
    )
    txt(ax, cx, cy, label, fs=fs, color=color, weight="bold",
        ha="center", va="center", alpha=alpha, linespacing=1.1, zorder=4)


def arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = SOFT,
    lw: float = 1.5,
    alpha: float = 1.0,
    ls: str = "-",
    rad: float = 0.0,
    zorder: int = 2,
) -> None:
    if alpha <= 0.01:
        return
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=lw, color=color, alpha=alpha, linestyle=ls,
            shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={rad}", zorder=zorder,
        )
    )


def path_arrow(
    ax,
    pts: list[tuple[float, float]],
    color: str = SOFT,
    lw: float = 1.5,
    alpha: float = 1.0,
    ls: str = "-",
    head: bool = True,
    zorder: int = 2,
) -> None:
    if alpha <= 0.01 or len(pts) < 2:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha,
            solid_capstyle="round", zorder=zorder)
    if head:
        ax.annotate(
            "", xy=pts[-1], xytext=pts[-2],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                            alpha=alpha, shrinkA=0, shrinkB=0),
            zorder=zorder,
        )


def point_on_path(pts: list[tuple[float, float]], p: float) -> tuple[float, float]:
    p = _clip01(p)
    lengths = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
    total = sum(lengths)
    if total == 0:
        return pts[0]
    target = p * total
    walked = 0.0
    for (x1, y1), (x2, y2), seg in zip(pts, pts[1:], lengths):
        if walked + seg >= target:
            q = 0.0 if seg == 0 else (target - walked) / seg
            return x1 + (x2 - x1) * q, y1 + (y2 - y1) * q
        walked += seg
    return pts[-1]


def dot(
    ax,
    x: float,
    y: float,
    color: str = BLUE,
    r: float = 0.07,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    ax.add_patch(
        Circle((x, y), r, facecolor=color, edgecolor="white",
               linewidth=0.8, alpha=alpha, zorder=7)
    )


def scene_header(
    ax,
    number: int,
    title: str,
    subtitle: str | None = None,
    color: str = BLUE,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    ax.add_patch(Circle((0.95, 7.95), 0.27, facecolor=color,
                        edgecolor="none", alpha=alpha, zorder=3))
    txt(ax, 0.95, 7.94, str(number), fs=11, color="white",
        weight="bold", ha="center", alpha=alpha, zorder=4)
    txt(ax, 1.42, 7.98, title, fs=19, weight="bold", alpha=alpha)
    if subtitle:
        txt(ax, 1.42, 7.53, subtitle, fs=9.8, color=GRAY, alpha=alpha)


# --------------------------------------------------------------------------- #
# Cromo común (marca, progreso, pie)
# --------------------------------------------------------------------------- #
def draw_chrome(ax, t: float, stats: Stats) -> None:
    txt(ax, 0.6, 8.68, "DÓLAR BLUE · BOLIVIA", fs=8.8, color=GRAY, weight="bold")
    txt(ax, 15.4, 8.68, "Pipeline de datos · CRISP-DM", fs=8.2,
        color=GRAY, ha="right")
    ax.plot([0.6, 15.4], [8.48, 8.48], color=BORDER, lw=0.8, zorder=1)

    x0, total_w, y, h = 0.6, 14.8, 0.56, 0.065
    current, _ = scene_state(t)
    acc = 0.0
    for i, (name, duration) in enumerate(SCENES):
        left = x0 + total_w * acc / DURATION
        width = total_w * duration / DURATION
        active = i == current
        ax.add_patch(
            Rectangle(
                (left, y), max(width - 0.04, 0.02), h,
                facecolor=BLUE if active else BORDER,
                edgecolor="none", alpha=1.0 if active else 0.9, zorder=2,
            )
        )
        short = {"LLM + caché": "LLM"}.get(name, name)
        txt(ax, left + 0.02, 0.39, f"{i + 1}  {short}",
            fs=7.2, color=BLUE if active else SOFT,
            weight="bold" if active else "normal")
        acc += duration

    marker_x = x0 + total_w * _clip01(t / DURATION)
    ax.plot([marker_x, marker_x], [y - 0.05, y + h + 0.05],
            color=DARK, lw=1.0, alpha=0.55, zorder=3)

    txt(ax, 0.6, 0.2,
        f"report/make_animation.py · datos al {stats.updated}",
        fs=7.0, color=SOFT)
    txt(ax, 15.4, 0.2, "60 s · 8 escenas", fs=7.0, color=SOFT, ha="right")


def scene_state(t: float) -> tuple[int, float]:
    acc = 0.0
    for i, (_, duration) in enumerate(SCENES):
        if t < acc + duration or i == len(SCENES) - 1:
            return i, _clip01((t - acc) / duration)
        acc += duration
    return len(SCENES) - 1, 1.0


# --------------------------------------------------------------------------- #
# Escena 1 — fuentes
# --------------------------------------------------------------------------- #
def scene_sources(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 1, "Dos fuentes, dos ritmos",
        "cotizaciones de mercado y normas publicadas en PDF", BLUE, a,
    )

    # ----- dolarblue ----------------------------------------------------- #
    a1 = _appear(p, 0.06)
    card(ax, 0.8, 2.8, 6.6, 4.3, face=BLUE_BG, edge="#bfdbfe", alpha=a1)
    txt(ax, 1.15, 6.67, "dolarblue · API", fs=14, color=BLUE, weight="bold", alpha=a1)
    txt(ax, 1.15, 6.31, "cotizaciones blue / oficial", fs=9.2, color=MID, alpha=a1)
    bullets_l = [
        "endpoints públicos + privados",
        "X-API-Key en cada request",
        "un snapshot JSON por endpoint",
        "ventana móvil de 6 meses",
    ]
    for i, b in enumerate(bullets_l):
        ab = _appear(p, 0.18 + i * 0.09)
        ax.add_patch(Circle((1.18, 5.7 - i * 0.55), 0.055,
                            facecolor=BLUE, edgecolor="none", alpha=ab))
        txt(ax, 1.36, 5.7 - i * 0.55, b, fs=9.6, alpha=ab)
    ab = _appear(p, 0.58)
    pill(ax, 1.15, 3.12, 2.55, 0.48,
         f"{_fmt_int(s.snapshots)} snapshots", BLUE_BG, BLUE, BLUE, 8.5, ab)

    # ----- Gaceta -------------------------------------------------------- #
    a2 = _appear(p, 0.14)
    card(ax, 8.6, 2.8, 6.6, 4.3, face=ORANGE_BG, edge="#fde68a", alpha=a2)
    txt(ax, 8.95, 6.67, "Gaceta Oficial · PDFs", fs=14,
        color=ORANGE, weight="bold", alpha=a2)
    txt(ax, 8.95, 6.31, "normas publicadas", fs=9.2, color=MID, alpha=a2)
    bullets_r = [
        "listado de ediciones (tipo 13)",
        "ventana de 122 días",
        "una norma = un PDF",
        "descarga reanudable por archivo",
    ]
    for i, b in enumerate(bullets_r):
        ab = _appear(p, 0.26 + i * 0.09)
        ax.add_patch(Circle((8.98, 5.7 - i * 0.55), 0.055,
                            facecolor=ORANGE, edgecolor="none", alpha=ab))
        txt(ax, 9.16, 5.7 - i * 0.55, b, fs=9.6, alpha=ab)
    ab = _appear(p, 0.64)
    pill(ax, 8.95, 3.12, 2.25, 0.48,
         f"{s.pdfs} PDFs", ORANGE_BG, ORANGE, "#b45309", 8.5, ab)

    # ----- compuerta de frescura ----------------------------------------- #
    ag = _appear(p, 0.52)
    card(ax, 2.2, 1.15, 11.8, 0.92, face=PANEL, edge=BORDER, alpha=ag)
    diamond(ax, 3.25, 1.61, 1.55, 0.64, "¿fresco?", "white", BLUE, DARK, 8.2, ag)
    txt(ax, 4.25, 1.74, "Manifiesto de corrida", fs=9.6,
        weight="bold", alpha=ag)
    txt(ax, 4.25, 1.43,
        "fresco (< 24 h) → se omite la descarga   ·   viejo → descarga completa",
        fs=8.2, color=GRAY, alpha=ag)

    arrow(ax, 4.1, 2.8, 4.1, 2.12, BLUE, 1.4, ag)
    arrow(ax, 11.9, 2.8, 11.9, 2.12, ORANGE, 1.4, ag)
    q = _ease(_clip01((p - 0.68) / 0.28))
    if q > 0:
        dot(ax, 4.1, 2.8 + (2.12 - 2.8) * q, BLUE, 0.065, ag)
        dot(ax, 11.9, 2.8 + (2.12 - 2.8) * q, ORANGE, 0.065, ag)


# --------------------------------------------------------------------------- #
# Escena 2 — ingesta y cache de archivos
# --------------------------------------------------------------------------- #
def scene_ingest(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 2, "Ingesta: descargar una sola vez",
        "manifiesto de 24 h + cache de archivo antes de tocar la red", BLUE, a,
    )

    txt(ax, 0.9, 7.03, "Mercado · API dolarblue", fs=9.8,
        color=BLUE, weight="bold", alpha=a)
    txt(ax, 8.7, 7.03, "Gaceta Oficial · PDFs", fs=9.8,
        color=ORANGE, weight="bold", alpha=a)

    # ----- columna izquierda --------------------------------------------- #
    left = [
        (6.1, "GET /v1/chart/6m", "endpoints públicos + privados"),
        (5.05, "raw/snapshot_*.json", "respuesta cruda por endpoint"),
        (4.0, "normalización Pydantic", "valida y tipa cada familia"),
        (2.95, "normalized/*.csv + history_full.csv", f"{_fmt_int(s.snapshots)} snapshots"),
    ]
    for i, (y, title, sub) in enumerate(left):
        ai = _appear(p, 0.04 + i * 0.09)
        node(ax, 0.9, y, 6.4, 0.75, title, sub,
             face="white", edge="#bfdbfe", title_color=BLUE,
             title_fs=10.2, sub_fs=8.2, alpha=ai)
        if i < len(left) - 1:
            arrow(ax, 4.1, y, 4.1, y - 0.3, BLUE, 1.3, _appear(p, 0.10 + i * 0.09))
            q = _ease(_clip01((p - 0.16 - i * 0.10) / 0.16))
            if q > 0:
                dot(ax, 4.1, y - 0.3 * q, BLUE, 0.06,
                    _appear(p, 0.10 + i * 0.09))

    # ----- columna derecha ----------------------------------------------- #
    al = _appear(p, 0.12)
    node(ax, 8.7, 6.1, 6.3, 0.75, "listado de ediciones",
         "tipo 13 · ventana de 122 días", face="white", edge="#fde68a",
         title_color="#b45309", title_fs=10.2, sub_fs=8.2, alpha=al)

    ad = _appear(p, 0.20)
    diamond(ax, 11.85, 5.25, 3.9, 1.15, "¿PDF en disco\ny > 1 KB?",
            "white", ORANGE, DARK, 8.4, ad)
    arrow(ax, 11.85, 6.1, 11.85, 5.83, ORANGE, 1.3, ad)
    q = _ease(_clip01((p - 0.30) / 0.15))
    if q > 0:
        dot(ax, 11.85, 6.1 + (5.83 - 6.1) * q, ORANGE, 0.06, ad)

    ac = _appear(p, 0.34)
    node(ax, 8.7, 3.95, 2.9, 0.7, "sí → cache", "sin request",
         face=GREEN_BG, edge=GREEN, title_color="#047857",
         title_fs=9.6, sub_fs=8.0, alpha=ac)
    an = _appear(p, 0.40)
    node(ax, 12.1, 3.95, 2.9, 0.7, "no → GET + %PDF", "3 intentos · backoff",
         face=ORANGE_BG, edge=ORANGE, title_color="#b45309",
         title_fs=9.4, sub_fs=8.0, alpha=an)

    arrow(ax, 11.05, 4.72, 10.15, 4.65, GREEN, 1.2, ac)
    arrow(ax, 12.65, 4.72, 13.55, 4.65, ORANGE, 1.2, an)

    ao = _appear(p, 0.50)
    sin_detalle = max(s.pdfs - s.extracted, 0)
    node(ax, 8.7, 2.95, 6.3, 0.75, "norma_*.pdf",
         f"{s.pdfs} PDFs · {sin_detalle} sin extracción",
         face="white", edge="#fde68a", title_color="#b45309",
         title_fs=10.2, sub_fs=8.2, alpha=ao)

    arrow(ax, 10.15, 3.95, 10.15, 3.70, GREEN, 1.3, ao)
    arrow(ax, 13.55, 3.95, 13.55, 3.70, ORANGE, 1.3, ao)

    q = _ease(_clip01((p - 0.46) / 0.16))
    if q > 0:
        dot(ax, 11.05 + (10.15 - 11.05) * q, 4.72 + (4.65 - 4.72) * q,
            GREEN, 0.06, ac)
        dot(ax, 12.65 + (13.55 - 12.65) * q, 4.72 + (4.65 - 4.72) * q,
            ORANGE, 0.06, an)

    q2 = _ease(_clip01((p - 0.58) / 0.16))
    if q2 > 0:
        dot(ax, 10.15, 3.95 + (3.70 - 3.95) * q2, GREEN, 0.06, ao)
        dot(ax, 13.55, 3.95 + (3.70 - 3.95) * q2, ORANGE, 0.06, ao)

    ab = _appear(p, 0.66)
    card(ax, 0.9, 1.35, 14.1, 0.72, face=PANEL, edge=BORDER, alpha=ab)
    txt(ax, 1.15, 1.71,
        "El manifiesto de 24 h evita repetir la corrida completa; "
        "el cache de archivo evita pedir un PDF que ya está en disco.",
        fs=8.6, color=MID, alpha=ab)


# --------------------------------------------------------------------------- #
# Escena 3 — extracción LLM, cache y reintentos
# --------------------------------------------------------------------------- #
def scene_llm(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 3, "Extracción LLM: pagar una sola vez",
        "hash del PDF, cache por proveedor/modelo y reintentos acotados", BLUE, a,
    )

    a1 = _appear(p, 0.05)
    node(ax, 0.8, 5.95, 1.5, 0.8, "PDF", "norma_*.pdf",
         face=ORANGE_BG, edge=ORANGE, title_color="#b45309",
         title_fs=9.4, sub_fs=7.8, alpha=a1)
    node(ax, 2.55, 5.95, 1.8, 0.8, "SHA-256", "hash del contenido",
         face="white", edge=BORDER, title_fs=9.4, sub_fs=7.8, alpha=a1)
    diamond(ax, 5.35, 6.35, 2.3, 1.2, "¿en cache?",
            "white", BLUE, DARK, 8.4, a1)
    node(ax, 6.95, 5.95, 2.0, 0.8, "cache hit",
         f"{s.cache_entries}/{s.extracted} · $0",
         face=GREEN_BG, edge=GREEN, title_color="#047857",
         title_fs=9.4, sub_fs=7.8, alpha=a1)
    node(ax, 9.25, 5.95, 2.4, 0.8, "registro tipado", "status = ok",
         face="white", edge=BORDER, title_fs=9.4, sub_fs=7.8, alpha=a1)

    arrow(ax, 2.30, 6.35, 2.55, 6.35, ORANGE, 1.3, a1)
    arrow(ax, 4.35, 6.35, 4.20, 6.35, BLUE, 1.3, a1)
    arrow(ax, 6.50, 6.35, 6.95, 6.35, GREEN, 1.3, a1)
    arrow(ax, 8.95, 6.35, 9.25, 6.35, GREEN, 1.3, a1)

    # ----- rama miss (hipotética) ---------------------------------------- #
    am = _appear(p, 0.28)
    node(ax, 4.45, 4.15, 1.8, 0.8, "LLM", f"{s.provider} / {s.model}",
         face=BLUE_BG, edge=BLUE, title_color=BLUE,
         title_fs=9.4, sub_fs=7.6, alpha=am)
    node(ax, 6.65, 4.15, 2.4, 0.8, "JSON + Pydantic", "valida el esquema",
         face="white", edge="#bfdbfe", title_color=BLUE,
         title_fs=9.2, sub_fs=7.8, alpha=am)
    node(ax, 9.35, 4.15, 1.6, 0.8, "ok", "se guarda",
         face=GREEN_BG, edge=GREEN, title_color="#047857",
         title_fs=9.4, sub_fs=7.8, alpha=am)

    arrow(ax, 5.35, 5.75, 5.35, 4.95, ORANGE, 1.3, am, ls=(0, (4, 2)))
    txt(ax, 5.45, 5.38, "miss", fs=7.6, color=ORANGE, weight="bold", alpha=am)
    arrow(ax, 6.25, 4.55, 6.65, 4.55, BLUE, 1.3, am)
    arrow(ax, 9.05, 4.55, 9.35, 4.55, GREEN, 1.3, am)

    ac = _appear(p, 0.42)
    node(ax, 9.15, 3.05, 2.4, 0.7, "cache.put", "sha256:provider:model",
         face=GREEN_BG, edge=GREEN, title_color="#047857",
         title_fs=8.8, sub_fs=6.9, alpha=ac)
    arrow(ax, 10.15, 4.15, 10.15, 3.75, GREEN, 1.3, ac)

    # ----- reintento y fallo --------------------------------------------- #
    ar = _appear(p, 0.55)
    retry = [(7.85, 4.15), (7.85, 3.55), (5.35, 3.55), (5.35, 4.15)]
    path_arrow(ax, retry, RED, 1.3, ar, ls=(0, (4, 2)), head=True)
    txt(ax, 6.05, 3.42, "si falla: 3 intentos · backoff",
        fs=7.8, color=RED, weight="bold", alpha=ar)
    attempt = 1 + int(max(0.0, (p - 0.58) / 0.08) % 3) if p > 0.58 else 1
    txt(ax, 8.1, 3.72, f"intento {attempt} / 3",
        fs=7.6, color=RED, alpha=ar)

    af = _appear(p, 0.72)
    node(ax, 6.65, 2.2, 4.4, 0.65,
         "3 fallos → schema_invalid / llm_failed",
         "no se cachea · se reintenta en la próxima corrida",
         face=RED_BG, edge=RED, title_color="#b91c1c",
         title_fs=8.3, sub_fs=7.3, alpha=af)
    arrow(ax, 7.85, 3.55, 7.85, 2.85, RED, 1.2, af, ls=(0, (4, 2)))

    # ----- panel de relevancia ------------------------------------------- #
    arv = _appear(p, 0.30)
    card(ax, 12.0, 4.95, 3.2, 2.35, face="white", edge=BORDER, alpha=arv)
    txt(ax, 12.2, 6.98, "relevancia_fx", fs=9.6, weight="bold", alpha=arv)
    rows = [
        ("alta", s.relevance.get("alta", 0), "#047857", GREEN_BG),
        ("media", s.relevance.get("media", 0), "#b45309", ORANGE_BG),
        ("baja", s.relevance.get("baja", 0), "#475569", GRAY_BG),
        ("nula", s.relevance.get("nula", 0), "#b91c1c", RED_BG),
    ]
    for i, (label, value, color, bg) in enumerate(rows):
        y = 6.58 - i * 0.36
        pill(ax, 12.2, y - 0.13, 1.15, 0.26, label, bg, color, color, 7.4, arv)
        txt(ax, 13.55, y, str(value), fs=8.6, color=color,
            weight="bold", ha="right", alpha=arv)
    txt(ax, 12.2, 5.10,
        "nula se conserva · peso 0\nse cachea igual · no se reintenta",
        fs=7.4, color=GRAY, alpha=arv, linespacing=1.25)

    # ----- panel de cache ------------------------------------------------ #
    ac2 = _appear(p, 0.52)
    card(ax, 12.0, 3.0, 3.2, 1.55, face=GREEN_BG, edge="#a7f3d0", alpha=ac2)
    txt(ax, 12.2, 4.29, "cache", fs=9.6, color="#047857",
        weight="bold", alpha=ac2)
    txt(ax, 12.2, 3.93, f"{s.cache_entries} / {s.extracted} hits", fs=11,
        color="#047857", weight="bold", alpha=ac2)
    txt(ax, 12.2, 3.50, "costo marginal $0\nen una re-corrida",
        fs=7.5, color="#047857", alpha=ac2, linespacing=1.25)

    # ----- paquetes animados --------------------------------------------- #
    q = _ease(_clip01((p - 0.10) / 0.32))
    if q > 0:
        path = [(2.30, 6.35), (2.55, 6.35), (4.20, 6.35),
                (6.95, 6.35), (9.25, 6.35)]
        x, y = point_on_path(path, q)
        dot(ax, x, y, GREEN, 0.065, a1)

    q2 = _ease(_clip01((p - 0.36) / 0.25))
    if q2 > 0:
        path = [(5.35, 5.75), (5.35, 4.95), (6.25, 4.55),
                (9.05, 4.55), (10.15, 4.15)]
        x, y = point_on_path(path, q2)
        dot(ax, x, y, ORANGE, 0.065, am)

    if p > 0.62:
        cycle = ((p - 0.62) / 0.12) % 1.0
        x, y = point_on_path(retry, _ease(cycle))
        dot(ax, x, y, RED, 0.065, ar)

    ab = _appear(p, 0.78)
    txt(ax, 0.8, 1.72,
        "clave = sha256(pdf) : provider : model : version",
        fs=8.0, color=GRAY, family="DejaVu Sans Mono", alpha=ab)
    card(ax, 0.8, 1.15, 14.4, 0.48, face=PANEL, edge=BORDER, alpha=ab)
    txt(ax, 1.05, 1.39,
        "Cache hit: no se llama al LLM. Fallo: no se cachea y se reintenta la próxima corrida.",
        fs=8.0, color=MID, alpha=ab)


# --------------------------------------------------------------------------- #
# Escena 4 — procesadores y replay
# --------------------------------------------------------------------------- #
def scene_processors(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 4, "Procesadores: replay desde disco",
        "sin red: se reconstruye todo desde lo ya persistido", BLUE, a,
    )

    a0 = _appear(p, 0.05)
    node(ax, 0.8, 4.15, 2.5, 1.3, "data/ en disco", "replay · sin red",
         face=GRAY_BG, edge="#cbd5e1", title_color=MID,
         title_fs=10.2, sub_fs=8.2, alpha=a0)
    node(ax, 3.9, 4.3, 2.2, 1.0, "SnapshotReader", "lee lo persistido",
         face="white", edge="#bfdbfe", title_color=BLUE,
         title_fs=9.3, sub_fs=7.8, alpha=a0)
    arrow(ax, 3.3, 4.8, 3.9, 4.8, BLUE, 1.4, a0)
    q = _ease(_clip01((p - 0.12) / 0.18))
    if q > 0:
        dot(ax, 3.3 + 0.6 * q, 4.8, BLUE, 0.065, a0)

    a1 = _appear(p, 0.18)
    node(ax, 6.8, 5.55, 3.0, 0.9, "MarketHourlyIndexer", "grilla horaria",
         face=BLUE_BG, edge=BLUE, title_color=BLUE,
         title_fs=9.0, sub_fs=7.8, alpha=a1)
    node(ax, 6.8, 2.95, 3.0, 0.9, "GacetaDailyIndexer", "calendario diario",
         face=ORANGE_BG, edge=ORANGE, title_color="#b45309",
         title_fs=9.0, sub_fs=7.8, alpha=a1)

    path_up = [(6.1, 4.8), (6.45, 4.8), (6.45, 6.0), (6.8, 6.0)]
    path_down = [(6.1, 4.8), (6.45, 4.8), (6.45, 3.4), (6.8, 3.4)]
    path_arrow(ax, path_up, BLUE, 1.4, a1)
    path_arrow(ax, path_down, ORANGE, 1.4, a1)

    a2 = _appear(p, 0.32)
    node(ax, 10.5, 5.55, 4.0, 0.9, "market_hourly.csv",
         f"{_fmt_int(s.hours)} h · {s.duplicates} duplicados",
         face="white", edge="#bfdbfe", title_color=BLUE,
         title_fs=9.2, sub_fs=7.8, alpha=a2)
    node(ax, 10.5, 2.95, 4.0, 0.9, "gaceta_daily.csv",
         f"{s.days_total} días · {s.days_with_docs} con normas",
         face="white", edge="#fde68a", title_color="#b45309",
         title_fs=9.2, sub_fs=7.8, alpha=a2)
    arrow(ax, 9.8, 6.0, 10.5, 6.0, BLUE, 1.4, a2)
    arrow(ax, 9.8, 3.4, 10.5, 3.4, ORANGE, 1.4, a2)

    q1 = _ease(_clip01((p - 0.30) / 0.18))
    if q1 > 0:
        x, y = point_on_path(path_up, q1)
        dot(ax, x, y, BLUE, 0.065, a1)
        x, y = point_on_path(path_down, q1)
        dot(ax, x, y, ORANGE, 0.065, a1)

    q2 = _ease(_clip01((p - 0.48) / 0.18))
    if q2 > 0:
        dot(ax, 9.8 + 0.7 * q2, 6.0, BLUE, 0.065, a2)
        dot(ax, 9.8 + 0.7 * q2, 3.4, ORANGE, 0.065, a2)

    am = _appear(p, 0.58)
    pill(ax, 6.8, 1.78, 7.7, 0.48,
         "el cruce horario × diario se hace recién al modelar (merge_asof)",
         BLUE_BG, BLUE, BLUE, 8.2, am)
    ab = _appear(p, 0.68)
    card(ax, 0.8, 1.15, 14.4, 0.48, face=PANEL, edge=BORDER, alpha=ab)
    txt(ax, 1.05, 1.39,
        "Cada fuente conserva su resolución nativa: horaria y diaria no se remuestrean al recolectar.",
        fs=8.0, color=MID, alpha=ab)


# --------------------------------------------------------------------------- #
# Escena 5 — MarketHourlyIndexer
# --------------------------------------------------------------------------- #
def scene_indexer_market(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 5, "Indexer horario: una fila por hora",
        "features derivadas y NaN de warm-up / horizonte futuro", BLUE, a,
    )

    ai = _appear(p, 0.02)
    card(ax, 0.8, 4.8, 3.0, 1.8, face=BLUE_BG, edge="#bfdbfe", alpha=ai)
    txt(ax, 1.05, 6.25, "chart_6m.csv", fs=11.5, color=BLUE,
        weight="bold", alpha=ai)
    txt(ax, 1.05, 5.95,
        f"{_fmt_int(s.chart_rows)} filas · {s.chart_dups} duplicados",
        fs=8.2, color=MID, alpha=ai)
    txt(ax, 1.05, 5.55, "t · bb · bs · ob · os", fs=7.8,
        color=GRAY, family="DejaVu Sans Mono", alpha=ai)
    txt(ax, 1.05, 5.18,
        f"salida: {_fmt_int(s.hours)} filas · {s.market_cols} columnas", fs=7.8,
        color=BLUE, weight="bold", alpha=ai)

    steps = [
        (6.0, "1 · ordenar + deduplicar",
         "sort_values('t') · drop_duplicates('t')"),
        (4.9, "2 · timestamp UTC",
         "to_datetime(t, unit='s', utc=True)"),
        (3.8, "3 · precios medios",
         "blue_mid · oficial_mid"),
        (2.7, "4 · derivadas",
         "spread · brecha · ret · vol · fwd"),
    ]
    for i, (y, title, sub) in enumerate(steps):
        ai = _appear(p, 0.08 + i * 0.07)
        node(ax, 4.1, y, 3.2, 0.8, title, sub,
             face="white", edge="#bfdbfe", title_color=BLUE,
             title_fs=9.0, sub_fs=6.9, alpha=ai)
        if i < len(steps) - 1:
            arrow(ax, 5.7, y, 5.7, y - 0.3, BLUE, 1.3,
                  _appear(p, 0.12 + i * 0.07))
            q = _ease(_clip01((p - 0.16 - i * 0.08) / 0.14))
            if q > 0:
                dot(ax, 5.7, y - 0.3 * q, BLUE, 0.055,
                    _appear(p, 0.12 + i * 0.07))

    ar = _appear(p, 0.12)
    card(ax, 7.7, 1.85, 7.5, 5.35, face="white", edge=BORDER, alpha=ar)
    txt(ax, 7.95, 6.95, "columnas que produce", fs=10,
        weight="bold", alpha=ar)

    formulas = [
        (7.95, 6.48, 3.35, "blue_mid = (bb + bs) / 2"),
        (11.55, 6.48, 3.35, "oficial_mid = (ob + os) / 2"),
        (7.95, 5.88, 3.35, "spread_blue = bs - bb"),
        (11.55, 5.88, 3.35, "brecha_pct = (blue_mid / oficial_mid - 1) × 100"),
        (7.95, 5.28, 3.35, "ret_blue_1h = blue_mid.pct_change() × 100"),
        (11.55, 5.28, 3.35, "vol_24h = ret_blue_1h.rolling(24).std()"),
    ]
    for i, (x, y, w, label) in enumerate(formulas):
        aa = _appear(p, 0.18 + i * 0.04)
        card(ax, x, y - 0.22, w, 0.44, face=PANEL, edge=BORDER,
             alpha=aa, shadow=False, radius=0.08)
        txt(ax, x + 0.12, y, label, fs=7.2, color=MID,
            family="DejaVu Sans Mono", alpha=aa)

    af = _appear(p, 0.38)
    card(ax, 7.95, 4.50, 6.95, 0.44, face=PANEL, edge=BORDER,
         alpha=af, shadow=False, radius=0.08)
    txt(ax, 8.07, 4.72,
        "fwd_h = blue_mid.shift(-h) / blue_mid - 1   ·   h = 1, 6, 24, 72, 168",
        fs=7.2, color=MID, family="DejaVu Sans Mono", alpha=af)

    at = _appear(p, 0.42)
    cols = [
        ("fecha", 7.95, 1.65),
        ("blue_mid", 9.75, 0.95),
        ("brecha_pct", 10.85, 1.05),
        ("vol_24h", 12.05, 0.95),
        ("fwd_24h", 13.20, 0.95),
    ]
    for label, x, _ in cols:
        txt(ax, x, 4.05, label, fs=7.2, color=SOFT,
            weight="bold", family="DejaVu Sans Mono", alpha=at)
    ax.plot([7.95, 15.05], [3.93, 3.93], color=BORDER, lw=0.8, alpha=at)

    rows = [
        (3.60, s.first_date, s.first_blue, s.first_brecha, s.first_vol, s.first_fwd24),
        (3.19, s.last_date, s.last_blue, s.last_brecha, s.last_vol, s.last_fwd24),
    ]
    for i, (y, date, blue, brecha, vol, fwd) in enumerate(rows):
        aa = _appear(p, 0.46 + i * 0.06)
        vals = [
            date,
            _fmt_num(blue, 2),
            _fmt_num(brecha, 2),
            _fmt_num(vol, 3),
            _fmt_num(fwd, 4),
        ]
        for (label, x, _), val in zip(cols, vals):
            if val == "NaN":
                continue
            txt(ax, x, y, val, fs=7.2, color=MID,
                family=None if label == "fecha" else "DejaVu Sans Mono",
                alpha=aa)
        if pd.isna(vol):
            card(ax, 11.98, y - 0.15, 0.92, 0.30, face=ORANGE_BG,
                 edge=ORANGE, lw=0.7, alpha=aa * 0.8, shadow=False,
                 radius=0.06, zorder=4)
            txt(ax, 12.44, y, "NaN", fs=7.2, color=RED,
                weight="bold", ha="center", family="DejaVu Sans Mono", alpha=aa)
        if pd.isna(fwd):
            card(ax, 13.12, y - 0.15, 0.92, 0.30, face=ORANGE_BG,
                 edge=ORANGE, lw=0.7, alpha=aa * 0.8, shadow=False,
                 radius=0.06, zorder=4)
            txt(ax, 13.58, y, "NaN", fs=7.2, color=RED,
                weight="bold", ha="center", family="DejaVu Sans Mono", alpha=aa)

    an = _appear(p, 0.50)
    txt(ax, 7.95, 2.45,
        "El indexer no imputa: el warm-up (24 h) y los horizontes futuros quedan NaN.",
        fs=7.6, color=GRAY, alpha=an)
    txt(ax, 7.95, 2.15,
        "El modelado decide después: interpola huecos ≤ 3 h o descarta la fila.",
        fs=7.6, color=GRAY, alpha=an)


# --------------------------------------------------------------------------- #
# Escena 6 — GacetaDailyIndexer
# --------------------------------------------------------------------------- #
def scene_indexer_gaceta(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 6, "Indexer diario: calendario completo",
        "cada día del rango existe; sin publicación → cero estructural", ORANGE, a,
    )

    ai = _appear(p, 0.02)
    card(ax, 0.8, 4.7, 2.9, 1.5, face=ORANGE_BG, edge="#fde68a", alpha=ai)
    txt(ax, 1.05, 5.90, "gaceta_llm.csv", fs=10.6, color="#b45309",
        weight="bold", alpha=ai)
    txt(ax, 1.05, 5.55, f"{s.extracted} normas · status ok/cache",
        fs=8.0, color=MID, alpha=ai)
    txt(ax, 1.05, 5.20, "fecha_publicacion", fs=7.8,
        color=GRAY, family="DejaVu Sans Mono", alpha=ai)
    arrow(ax, 3.7, 5.45, 4.15, 5.45, ORANGE, 1.4, ai)

    ac = _appear(p, 0.10)
    card(ax, 4.15, 3.30, 3.30, 3.20, face="white", edge="#fde68a", alpha=ac)
    txt(ax, 4.40, 6.20, "calendario completo", fs=10,
        weight="bold", alpha=ac)
    txt(ax, 4.40, 5.92,
        f"{s.days_total} días · date_range(min, max, 'D')",
        fs=6.8, color=GRAY, family="DejaVu Sans Mono", alpha=ac)

    if s.docs_by_day is not None and not s.docs_by_day.empty:
        vals = s.docs_by_day
        n = min(len(vals), 182)
        cell, gap, step = 0.16, 0.02, 0.18
        x0, y0 = 4.30, 3.50
        scan = _ease(_clip01((p - 0.15) / 0.55)) * 13
        for i, (_, count) in enumerate(vals.iloc[:n].items()):
            row, col = divmod(i, 14)
            aa = ac if row <= scan else ac * 0.18
            x = x0 + col * step
            y = y0 + (12 - row) * step
            face = BLUE if count > 0 else "#e2e8f0"
            ax.add_patch(
                Rectangle((x, y), cell, cell, facecolor=face,
                          edgecolor="white", linewidth=0.5,
                          alpha=aa, zorder=4)
            )
        ax.add_patch(Circle((4.30, 3.38), 0.05, facecolor=BLUE,
                            edgecolor="none", alpha=ac, zorder=5))
        txt(ax, 4.41, 3.38, f"con norma {s.days_with_docs}", fs=6.6,
            color=MID, alpha=ac)
        ax.add_patch(Circle((5.55, 3.38), 0.05, facecolor="#cbd5e1",
                            edgecolor="none", alpha=ac, zorder=5))
        txt(ax, 5.66, 3.38,
            f"sin norma {s.days_total - s.days_with_docs} → 0",
            fs=6.6, color=MID, alpha=ac)

    ap = _appear(p, 0.26)
    card(ax, 7.7, 1.85, 7.5, 5.35, face="white", edge=BORDER, alpha=ap)
    txt(ax, 7.95, 6.95, "qué se agrega por día", fs=10,
        weight="bold", alpha=ap)

    headers = [
        ("regla", 7.95),
        ("día con norma", 11.10),
        ("día sin norma", 13.25),
    ]
    for label, x in headers:
        txt(ax, x, 6.45, label, fs=7.4, color=SOFT,
            weight="bold", alpha=ap)
    ax.plot([7.95, 15.05], [6.28, 6.28], color=BORDER, lw=0.8, alpha=ap)

    rows = [
        ("n_docs_total", "conteo", "0", None),
        ("n_docs_*", "conteo", "0", None),
        ("monto_usd_sum", "suma", "0", None),
        ("relevancia_max", "máx", "0", None),
        ("confidence_mean", _fmt_num(s.confidence_mean, 2), "NaN", "nan"),
    ]
    for i, (label, with_doc, without_doc, flag) in enumerate(rows):
        y = 5.82 - i * 0.52
        aa = _appear(p, 0.30 + i * 0.05)
        txt(ax, 7.95, y, label, fs=7.6, color=MID,
            family="DejaVu Sans Mono", alpha=aa)
        txt(ax, 11.10, y, with_doc, fs=8.0, color=MID,
            weight="bold", alpha=aa)
        if flag == "nan":
            card(ax, 13.12, y - 0.16, 0.98, 0.32, face=ORANGE_BG,
                 edge=ORANGE, lw=0.7, alpha=aa * 0.85, shadow=False,
                 radius=0.06, zorder=4)
            txt(ax, 13.61, y, without_doc, fs=8.0, color=RED,
                weight="bold", ha="center", alpha=aa)
        else:
            txt(ax, 13.25, y, without_doc, fs=8.0, color=GREEN,
                weight="bold", alpha=aa)

    acode = _appear(p, 0.42)
    card(ax, 7.95, 3.10, 6.95, 0.42, face=PANEL, edge=BORDER,
         alpha=acode, shadow=False, radius=0.08)
    txt(ax, 8.07, 3.31, "daily.fillna(0)   # conteos, montos y relevancia",
        fs=7.0, color=MID, family="DejaVu Sans Mono", alpha=acode)
    txt(ax, 7.95, 2.88,
        "confidence_mean queda NaN: no hay documento que promediar.",
        fs=7.4, color=GRAY, alpha=acode)

    acall = _appear(p, 0.50)
    card(ax, 7.95, 2.10, 6.95, 0.65, face=BLUE_BG, edge=BLUE,
         alpha=acall, shadow=False, radius=0.10)
    txt(ax, 8.10, 2.52, "El día existe igual", fs=8.4,
        color=BLUE, weight="bold", alpha=acall)
    txt(ax, 8.10, 2.27,
        "0 no es un dato faltante: es ausencia de publicación.",
        fs=7.6, color=MID, alpha=acall)

    ab = _appear(p, 0.62)
    card(ax, 0.8, 1.15, 14.4, 0.55, face=PANEL, edge=BORDER, alpha=ab)
    txt(ax, 1.05, 1.42,
        f"La grilla conserva los {s.days_total} días; el panel luego cruza "
        "mercado × Gaceta con merge_asof.",
        fs=8.0, color=MID, alpha=ab)


# --------------------------------------------------------------------------- #
# Escena 7 — EDA y faltantes
# --------------------------------------------------------------------------- #
def _map_series(
    values: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    if vmax == vmin:
        vmax = vmin + 1.0
    xs = np.linspace(x0, x1, len(values))
    ys = y0 + (values - vmin) / (vmax - vmin) * (y1 - y0)
    return xs, ys


def _draw_series(ax, s: Stats, alpha: float) -> None:
    if s.daily_blue is None:
        return
    serie = s.daily_blue.dropna()
    if serie.empty:
        return

    x0, x1, y0, y1 = 1.25, 7.95, 4.55, 6.25
    vmin, vmax = float(serie.min()), float(serie.max())
    pad = (vmax - vmin) * 0.08 or 1.0
    vmin -= pad
    vmax += pad

    card(ax, 1.0, 4.35, 7.15, 2.15, face=PANEL, edge=BORDER, alpha=alpha, shadow=False)
    for frac in (0.0, 0.5, 1.0):
        y = y0 + frac * (y1 - y0)
        ax.plot([x0, x1], [y, y], color=BORDER, lw=0.7,
                alpha=alpha * 0.9, zorder=2)
    xs, ys = _map_series(serie.values, x0, x1, y0, y1, vmin, vmax)
    ax.fill_between(xs, y0, ys, color=BLUE, alpha=0.09 * alpha, zorder=3)
    ax.plot(xs, ys, color=BLUE, lw=1.6, alpha=alpha, zorder=4)

    txt(ax, x0, y1 + 0.18, "Serie diaria del blue", fs=9.6,
        weight="bold", alpha=alpha)
    n_gaps = len(s.gaps)
    gaps_txt = f"{n_gaps} hueco" if n_gaps == 1 else f"{n_gaps} huecos"
    txt(ax, x1, y1 + 0.18,
        f"{_fmt_int(s.hours)} h · {gaps_txt}",
        fs=8.2, color=GRAY, ha="right", alpha=alpha)

    for frac, val in ((0.0, vmin), (0.5, (vmin + vmax) / 2), (1.0, vmax)):
        y = y0 + frac * (y1 - y0)
        txt(ax, x0 - 0.1, y, f"{val:.2f}", fs=7.0, color=SOFT,
            ha="right", alpha=alpha)

    first, last = serie.index.min(), serie.index.max()
    for ts in (first, first + (last - first) / 2, last):
        x = x0 + (ts - first).total_seconds() / (last - first).total_seconds() * (x1 - x0)
        txt(ax, x, y0 - 0.16, MESES[ts.month - 1], fs=7.0,
            color=SOFT, ha="center", alpha=alpha)

    if s.gap_start is not None:
        xg = x0 + (s.gap_start - first).total_seconds() / (last - first).total_seconds() * (x1 - x0)
        xg = max(xg, x0 + 0.03)
        ax.add_patch(
            Rectangle((xg - 0.03, y0), 0.06, y1 - y0,
                      facecolor=RED, edgecolor="none", alpha=0.22 * alpha, zorder=5)
        )
        txt(ax, xg + 0.10, y1 - 0.16,
            f"{s.gap_hours} h", fs=7.4, color=RED, weight="bold", alpha=alpha)


def _draw_zoom(ax, s: Stats, alpha: float, progress: float = 1.0) -> None:
    if s.zoom is None or s.zoom.empty:
        txt(ax, 1.25, 3.0, "sin huecos que interpolar", fs=8.0,
            color=GRAY, alpha=alpha)
        return

    zoom = s.zoom
    x0, x1, y0, y1 = 1.25, 7.95, 2.25, 3.45
    vmin, vmax = float(zoom["blue_mid"].min()), float(zoom["blue_mid"].max())
    pad = (vmax - vmin) * 0.15 or 0.01
    vmin -= pad
    vmax += pad

    card(ax, 1.0, 2.05, 7.15, 1.75, face="white", edge=BORDER, alpha=alpha, shadow=False)
    txt(ax, x0, y1 + 0.16,
        f"zoom {s.gap_start:%d-%m-%Y} · interpolación ≤ 3 h",
        fs=8.8, weight="bold", alpha=alpha)

    t0, t1 = zoom["fecha"].min(), zoom["fecha"].max()
    span = (t1 - t0).total_seconds() or 1.0
    xs = x0 + (zoom["fecha"] - t0).dt.total_seconds() / span * (x1 - x0)
    ys = y0 + (zoom["blue_mid"].values - vmin) / (vmax - vmin) * (y1 - y0)

    # banda del hueco
    if s.gap_start is not None and s.gaps:
        gs, ge, _ = s.gaps[0]
        xa = x0 + (gs - pd.Timedelta(hours=1) - t0).total_seconds() / span * (x1 - x0)
        xb = x0 + (ge + pd.Timedelta(hours=1) - t0).total_seconds() / span * (x1 - x0)
        ax.add_patch(
            Rectangle((xa, y0), max(xb - xa, 0.05), y1 - y0,
                      facecolor=RED, edgecolor="none", alpha=0.10 * alpha, zorder=2)
        )

    ax.plot(xs, ys, color=BLUE, lw=1.4, alpha=alpha, zorder=4)
    ax.scatter(xs, ys, s=14, facecolor=BLUE, edgecolor="white",
               linewidth=0.8, alpha=alpha, zorder=5)

    for frac in (0.0, 0.5, 1.0):
        y = y0 + frac * (y1 - y0)
        ax.plot([x0, x1], [y, y], color=BORDER, lw=0.6,
                alpha=alpha * 0.8, zorder=1)

    if s.gap_start is not None and s.gaps:
        gs, ge, _ = s.gaps[0]
        before = zoom[zoom["fecha"] < gs]
        after = zoom[zoom["fecha"] > ge]
        if not before.empty and not after.empty:
            fa = before.iloc[-1]
            fb = after.iloc[0]
            xa = x0 + (fa["fecha"] - t0).total_seconds() / span * (x1 - x0)
            xb = x0 + (fb["fecha"] - t0).total_seconds() / span * (x1 - x0)
            ya = y0 + (fa["blue_mid"] - vmin) / (vmax - vmin) * (y1 - y0)
            yb = y0 + (fb["blue_mid"] - vmin) / (vmax - vmin) * (y1 - y0)
            qfill = _ease(_clip01((progress - 0.24) / 0.48))
            ax.plot([xa, xb], [ya, yb], color=ORANGE, lw=1.5,
                    ls=(0, (3, 2)), alpha=alpha * min(1.0, qfill * 2.0), zorder=6)
            # puntos interpolados: aparecen uno por uno
            n_fill = int(round(qfill * s.gap_hours))
            for i in range(1, n_fill + 1):
                q = i / (s.gap_hours + 1)
                ax.scatter([xa + (xb - xa) * q], [ya + (yb - ya) * q],
                           s=24, facecolor="white", edgecolor=ORANGE,
                           linewidth=1.2, alpha=alpha, zorder=7)
            if qfill > 0:
                txt(ax, (xa + xb) / 2, y1 - 0.12,
                    f"{n_fill} / {s.gap_hours} h interpoladas", fs=7.2,
                    color=ORANGE, weight="bold", ha="center", alpha=alpha)
                txt(ax, x0 + 0.05, y0 + 0.10, "interpolate(limit=3)",
                    fs=6.8, color=ORANGE, family="DejaVu Sans Mono",
                    alpha=alpha * qfill)

    for ts in (t0, t0 + (t1 - t0) / 2, t1):
        x = x0 + (ts - t0).total_seconds() / span * (x1 - x0)
        txt(ax, x, y0 - 0.14, ts.strftime("%H:%M"), fs=6.6,
            color=SOFT, ha="center", alpha=alpha)


def scene_missing(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 7, "EDA y faltantes: decidir, no rellenar a ciegas",
        "interpolación acotada, ceros estructurales y NaN explícitos", BLUE, a,
    )

    ap = _appear(p, 0.08)
    _draw_series(ax, s, ap)
    _draw_zoom(ax, s, _appear(p, 0.22), p)

    items = [
        ("Precio 0", f"{s.zero_price} snapshot descartado", RED),
        ("Duplicados", f"{s.duplicates} marcas repetidas", GREEN),
        ("Hueco ≤ 3 h", f"{s.gap_hours} h interpoladas", ORANGE),
        ("Hueco > 3 h", "queda NaN, sin inventar", GRAY),
        ("vol_24h", f"{s.warmup_nan} h de warm-up", GRAY),
        ("Gaceta sin norma",
         f"{s.days_total - s.days_with_docs} / {s.days_total} días → 0", BLUE),
        ("monto_usd", f"{s.monto_null} / {s.extracted} no declarado", GRAY),
    ]
    for i, (title, value, color) in enumerate(items):
        ai = _appear(p, 0.12 + i * 0.075)
        y = 6.6 - i * 0.70
        card(ax, 8.7, y, 6.4, 0.58, face="white", edge=BORDER, alpha=ai, shadow=False)
        ax.add_patch(
            Rectangle((8.7, y), 0.08, 0.58, facecolor=color,
                      edgecolor="none", alpha=ai, zorder=4)
        )
        txt(ax, 8.95, y + 0.37, title, fs=9.0, weight="bold", alpha=ai)
        txt(ax, 15.0, y + 0.37, value, fs=7.9, color=color,
            ha="right", alpha=ai)

    ab = _appear(p, 0.72)
    card(ax, 0.8, 1.15, 14.4, 0.55, face=PANEL, edge=BORDER, alpha=ab)
    txt(ax, 1.05, 1.42,
        "Descarte, interpolación acotada, cero estructural o NaN explícito: nada se imputa en silencio.",
        fs=8.2, color=MID, alpha=ab)


# --------------------------------------------------------------------------- #
# Escena 6 — resumen
# --------------------------------------------------------------------------- #
def scene_summary(ax, p: float, s: Stats) -> None:
    a = 1.0
    scene_header(
        ax, 8, "Resumen: cuatro garantías",
        "lo que el pipeline evita repetir y lo que declara como faltante", BLUE, a,
    )

    total_rel = sum(s.relevance.values())
    media_alta = s.relevance.get("media", 0) + s.relevance.get("alta", 0)
    cards = [
        ("↓", "No re-descarga", "manifiesto 24 h\n+ cache de archivo",
         f"{s.pdfs} PDFs", BLUE, BLUE_BG),
        ("≡", "No re-extracción", "sha256 + provider + model",
         f"{s.cache_entries} / {s.extracted} hits", GREEN, GREEN_BG),
        ("↻", "Reintento acotado", "3 intentos · backoff\nfallo no cacheado",
         "si falla", ORANGE, ORANGE_BG),
        ("!", "Faltantes explícitos", "regla por tipo\nsin imputar a ciegas",
         f"{s.gap_hours} h · {s.days_total - s.days_with_docs} días", GRAY, GRAY_BG),
    ]
    for i, (icon, title, body, stat, color, bg) in enumerate(cards):
        ai = _appear(p, 0.05 + i * 0.10)
        x = 0.8 + i * 3.8
        dy = (1 - ai) * 0.16
        card(ax, x, 4.15 + dy, 3.5, 2.45, face=bg, edge=color,
             alpha=ai, shadow=True)
        txt(ax, x + 0.25, 6.23 + dy, icon, fs=24, color=color,
            weight="bold", alpha=ai)
        txt(ax, x + 0.25, 5.63 + dy, title, fs=16.5, color=color,
            weight="bold", alpha=ai)
        txt(ax, x + 0.25, 5.02 + dy, body, fs=10.5, color=MID,
            alpha=ai, linespacing=1.25)
        txt(ax, x + 0.25, 4.45 + dy, stat, fs=14.5, color=color,
            weight="bold", alpha=ai)

    chips = [
        (0.8, f"{_fmt_int(s.snapshots)} snapshots · {_fmt_int(s.hours)} h"),
        (5.75, f"{s.pdfs} PDFs · {s.extracted} extracciones"),
        (10.7, f"{s.relevance.get('nula', 0)} nula · {media_alta} media/alta"),
    ]
    for i, (x, label) in enumerate(chips):
        ac = _appear(p, 0.45 + i * 0.08)
        pill(ax, x, 3.05, 4.5, 0.62, label, "white", BORDER, MID, 9.0, ac)

    af = _appear(p, 0.62)
    txt(ax, 8.0, 2.20,
        "fuentes  →  ingesta  →  LLM + caché  →  procesadores  →  EDA",
        fs=9.4, color=SOFT, ha="center", alpha=af)
    txt(ax, 8.0, 1.62,
        f"Datos al {s.updated} · animación generada desde data/ sin llamadas a la API",
        fs=8.6, color=GRAY, ha="center", alpha=af)
    txt(ax, 8.0, 1.25,
        f"{total_rel} normas tipadas · {s.provider} / {s.model}",
        fs=8.2, color=SOFT, ha="center", alpha=af)


# --------------------------------------------------------------------------- #
# Composición y render
# --------------------------------------------------------------------------- #
def scene_start(index: int) -> float:
    """Segundo de inicio de la escena `index` (0-based)."""
    return float(sum(spec.duration for spec in SCENE_SPECS[:index]))


def scene_time(key: str, p: float = 1.0) -> float:
    """Tiempo global para el progreso `p` de una escena.

    `p=1.0` se recorta a 0.999 para que `scene_state` no salte a la escena
    siguiente justo en el borde.
    """
    spec = SCENE_BY_KEY[key]
    return scene_start(spec.number - 1) + min(max(p, 0.0), 0.999) * spec.duration


def draw_scene(ax, index: int, p: float, stats: Stats) -> None:
    if index == 0:
        scene_sources(ax, p, stats)
    elif index == 1:
        scene_ingest(ax, p, stats)
    elif index == 2:
        scene_llm(ax, p, stats)
    elif index == 3:
        scene_processors(ax, p, stats)
    elif index == 4:
        scene_indexer_market(ax, p, stats)
    elif index == 5:
        scene_indexer_gaceta(ax, p, stats)
    elif index == 6:
        scene_missing(ax, p, stats)
    else:
        scene_summary(ax, p, stats)


def draw_frame(ax, t: float, stats: Stats) -> None:
    ax.clear()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(CANVAS)
    draw_chrome(ax, t, stats)

    idx, p = scene_state(t)
    draw_scene(ax, idx, p, stats)


def scene_figure(
    key: str,
    p: float = 1.0,
    stats: Stats | None = None,
    dpi: int = 120,
):
    """Figura matplotlib con el frame final (o el progreso `p`) de una escena.

    No toca los rcParams globales: el estilo se aplica en un `rc_context`, de
    modo que importar este módulo desde el notebook no cambia las figuras
    posteriores.
    """
    stats = stats or load_stats()
    with plt.rc_context(ESTILO):
        fig = plt.figure(figsize=(W, H), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_facecolor(CANVAS)
        draw_frame(ax, scene_time(key, p), stats)
    return fig


def render_scene_png(
    key: str,
    path: Path,
    p: float = 1.0,
    stats: Stats | None = None,
    dpi: int = 120,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = scene_figure(key, p=p, stats=stats, dpi=dpi)
    fig.savefig(path, dpi=dpi, facecolor=CANVAS)
    plt.close(fig)
    return path


def _stats_payload(s: Stats) -> dict:
    """Subconjunto serializable de `Stats` para el dashboard."""
    def num(value: float):
        return None if pd.isna(value) else float(value)

    return {
        "snapshots": s.snapshots,
        "zero_price": s.zero_price,
        "hours": s.hours,
        "hours_expected": s.hours_expected,
        "market_cols": s.market_cols,
        "duplicates": s.duplicates,
        "warmup_nan": s.warmup_nan,
        "gaps": [[str(a), str(b), n] for a, b, n in s.gaps],
        "chart_rows": s.chart_rows,
        "chart_dups": s.chart_dups,
        "pdfs": s.pdfs,
        "extracted": s.extracted,
        "cache_entries": s.cache_entries,
        "provider": s.provider,
        "model": s.model,
        "relevance": s.relevance,
        "days_total": s.days_total,
        "days_with_docs": s.days_with_docs,
        "monto_null": s.monto_null,
        "confidence_mean": num(s.confidence_mean),
        "updated": s.updated,
    }


def export_scenes(stats: Stats, out_dir: Path, dpi: int = 120) -> Path:
    """Exporta los ocho frames y el guion que consume el dashboard."""
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = []
    for spec in SCENE_SPECS:
        image = f"scene_{spec.number:02d}_{spec.key}.png"
        path = out_dir / image
        fig = scene_figure(spec.key, p=1.0, stats=stats, dpi=dpi)
        fig.savefig(path, dpi=dpi, facecolor=CANVAS)
        plt.close(fig)
        start = scene_start(spec.number - 1)
        scenes.append({
            "n": spec.number,
            "key": spec.key,
            "short": spec.short,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "start": round(start, 2),
            "end": round(start + spec.duration, 2),
            "duration": spec.duration,
            "lead": spec.lead,
            "bullets": list(spec.bullets),
            "rule": spec.rule,
            "source": spec.source,
            "image": image,
            "clip": f"scene_{spec.number:02d}_{spec.key}.gif",
        })

    payload = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration": DURATION,
        "fps": FPS,
        "video": "report/anim/pipeline.mp4",
        "gif": "report/anim/pipeline.gif",
        "stats": _stats_payload(stats),
        "scenes": scenes,
    }
    meta = out_dir / "scenes.json"
    meta.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        shown = out_dir.relative_to(ROOT)
    except ValueError:
        shown = out_dir
    print(f"  {shown}/scene_*.png ({len(scenes)} frames)")
    print(f"  {shown}/scenes.json")
    return meta


def export_scene_clips(
    out_dir: Path,
    mp4: Path | None = None,
    fps: int = 12,
    width: int = 960,
    colors: int = 192,
) -> list[Path]:
    """Corta el MP4 maestro en un GIF animado por escena.

    Los clips se generan desde `report/anim/pipeline.mp4` y se guardan con el
    mismo nombre base que los frames (`scene_XX_key.gif`). El dashboard y el
    notebook los usan si existen; si no, caen al PNG estático.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("falta ffmpeg para generar los clips por escena")

    mp4 = mp4 or (ANIM_DIR / "pipeline.mp4")
    if not mp4.exists():
        raise SystemExit(
            f"falta {mp4}: generá primero el video maestro con `make anim`"
        )

    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        f"split[s0][s1];[s0]palettegen=max_colors={colors}[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )

    clips: list[Path] = []
    print(f"  generando {len(SCENE_SPECS)} clips de escena ...")
    for spec in SCENE_SPECS:
        start = scene_start(spec.number - 1)
        out = out_dir / f"scene_{spec.number:02d}_{spec.key}.gif"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", str(mp4),
                "-t", f"{spec.duration:.3f}",
                "-vf", vf, "-loop", "0", str(out),
            ],
            check=True,
        )
        clips.append(out)
        print(f"    {out.name}")

    try:
        shown = out_dir.relative_to(ROOT)
    except ValueError:
        shown = out_dir
    print(f"  {shown}/scene_*.gif ({len(clips)} clips)")
    return clips


def render_poster(stats: Stats) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(ESTILO):
        fig = plt.figure(figsize=(W, H), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        draw_frame(ax, POSTER_T, stats)
        pdf = FIG_DIR / "fig_pipeline_poster.pdf"
        png = FIG_DIR / "fig_pipeline_poster.png"
        fig.savefig(pdf, facecolor=CANVAS)
        fig.savefig(png, dpi=160, facecolor=CANVAS)
        plt.close(fig)
    print(f"  {pdf.relative_to(ROOT)}")
    print(f"  {png.relative_to(ROOT)}")
    return pdf


def render_video(
    stats: Stats,
    mp4: Path,
    duration: float = DURATION,
    fps: int = FPS,
    dpi: int = 120,
) -> Path:
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(duration * fps))
    with plt.rc_context(ESTILO):
        fig = plt.figure(figsize=(W, H), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_facecolor(CANVAS)

        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=4000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        print(f"  renderizando {n_frames} frames a {fps} fps ...")
        with writer.saving(fig, str(mp4), dpi=dpi):
            for frame in range(n_frames):
                draw_frame(ax, frame / fps, stats)
                writer.grab_frame()
                if frame and frame % fps == 0:
                    print(f"    {frame / fps:4.0f} s / {duration:.0f} s")
        plt.close(fig)
    print(f"  {mp4.relative_to(ROOT)}")
    return mp4


def make_gif(mp4: Path, gif: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("falta ffmpeg para generar el GIF")
    vf = (
        "fps=12,scale=960:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=192[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    print("  convirtiendo a GIF ...")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", vf, "-loop", "0", str(gif)],
        check=True,
    )
    print(f"  {gif.relative_to(ROOT)}")
    return gif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poster-only", action="store_true",
                        help="solo genera el póster vectorial para el informe")
    parser.add_argument("--preview", type=float, default=None,
                        help="renderiza solo los primeros N segundos (prueba)")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--no-gif", action="store_true",
                        help="no genera el GIF")
    parser.add_argument("--export-scenes", type=Path, default=None,
                        metavar="DIR",
                        help="exporta los 8 frames + scenes.json para el dashboard")
    parser.add_argument("--export-clips", type=Path, default=None,
                        metavar="DIR",
                        help="corta el MP4 maestro en 8 GIF animados por escena")
    args = parser.parse_args()

    print(f"Datos: {DATA}")
    stats = load_stats()
    print(
        f"  {_fmt_int(stats.snapshots)} snapshots · {_fmt_int(stats.hours)} h · "
        f"{stats.pdfs} PDFs · {stats.extracted} extracciones · "
        f"{stats.cache_entries} en cache"
    )

    if args.export_scenes:
        export_scenes(stats, args.export_scenes, dpi=args.dpi)
        return

    if args.export_clips:
        export_scene_clips(args.export_clips)
        return

    if shutil.which("ffmpeg") is None and not args.poster_only:
        raise SystemExit("falta ffmpeg: instalalo para generar el MP4")

    render_poster(stats)
    if args.poster_only:
        return

    if args.preview:
        mp4 = ANIM_DIR / "preview.mp4"
        render_video(stats, mp4, duration=args.preview, fps=args.fps, dpi=args.dpi)
        return

    mp4 = ANIM_DIR / "pipeline.mp4"
    render_video(stats, mp4, duration=DURATION, fps=args.fps, dpi=args.dpi)
    if not args.no_gif:
        make_gif(mp4, ANIM_DIR / "pipeline.gif")


if __name__ == "__main__":
    main()
