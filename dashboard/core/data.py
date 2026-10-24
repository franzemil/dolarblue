"""Capa de datos del dashboard: lectura cacheada de artefactos.

Regla del proyecto: la app **lee**, no re-entrena. Los parquet de
`dashboard/assets/` se generan con `dashboard/scripts/export_assets.py`.
Si un asset falta, se intenta el CSV del pipeline; si tampoco existe, el
llamador decide cómo degradar.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import ASSETS, DATA, MODELADO, OUTPUTS, ROOT


class DataFaltante(RuntimeError):
    """Falta un artefacto imprescindible para renderizar el dashboard."""


def _asset(nombre: str, fallback: Path | None = None,
           parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    ruta = ASSETS / nombre
    if ruta.exists():
        df = pd.read_parquet(ruta)
    elif fallback is not None and fallback.exists():
        df = pd.read_csv(fallback)
    else:
        return None
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def metricas() -> dict:
    ruta = MODELADO / "metricas_reporte.json"
    if not ruta.exists():
        raise DataFaltante(
            "Falta modelado/metricas_reporte.json. Generalo con "
            "`make dashboard-assets`."
        )
    return json.loads(ruta.read_text())


@st.cache_data(show_spinner=False)
def mercado() -> pd.DataFrame:
    df = _asset("mercado.parquet", DATA / "market_hourly.csv",
                parse_dates=["fecha"])
    if df is None:
        raise DataFaltante(
            "Falta dashboard/assets/mercado.parquet. Generalo con "
            "`make dashboard-assets`."
        )
    return df.sort_values("fecha").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def panel() -> pd.DataFrame:
    df = _asset("panel.parquet", MODELADO / "panel_diario.csv",
                parse_dates=["fecha_d"])
    if df is None:
        raise DataFaltante("Falta dashboard/assets/panel.parquet.")
    return df.sort_values("fecha_d").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def gaceta() -> pd.DataFrame:
    df = _asset("gaceta.parquet", DATA / "gaceta_llm.csv",
                parse_dates=["fecha_publicacion"])
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def crudo() -> pd.DataFrame:
    df = _asset("crudo.parquet", parse_dates=["fecha"])
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def predicciones(horizonte: int) -> pd.DataFrame:
    df = _asset(f"predicciones_h{int(horizonte)}.parquet")
    if df is None:
        raise DataFaltante(
            f"Falta dashboard/assets/predicciones_h{horizonte}.parquet. "
            "Generalo con `make dashboard-assets`."
        )
    df["fecha"] = pd.to_datetime(df["fecha"])
    if df["fecha"].dt.tz is not None:
        df["fecha"] = df["fecha"].dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def roc() -> pd.DataFrame:
    df = _asset("roc_curvas.parquet")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def ols_hac() -> pd.DataFrame:
    df = _asset("ols_hac.parquet")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def direccion() -> pd.DataFrame:
    df = _asset("direccion.parquet", OUTPUTS / "informe_direccion.csv")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def grid_arima() -> pd.DataFrame:
    """Grilla de 16 candidatos de tab:arima_grid (p, q <= 2; d en {0,1}).

    Se usa la tabla del informe —no el CSV regenerado— porque el informe
    documenta explícitamente esos 16 candidatos y su AIC. El CSV del pipeline
    puede ampliar la grilla y desalinearse de la tabla publicada.
    """
    filas = [
        ((0, 0, 1), "c", -916.9, 8909.8), ((0, 0, 2), "c", -3121.2, 7241.5),
        ((1, 0, 0), "c", -9210.3, 2944.7), ((1, 0, 1), "c", -9299.4, 2946.4),
        ((1, 0, 2), "c", -9299.0, 2948.0), ((2, 0, 0), "c", -9291.8, 2946.4),
        ((2, 0, 1), "c", -9302.6, 2931.6), ((2, 0, 2), "c", -9297.5, 2929.7),
        ((0, 1, 1), "n", -9303.7, 2948.2), ((0, 1, 2), "n", -9304.2, 2949.5),
        ((1, 1, 0), "n", -9296.4, 2948.2), ((1, 1, 1), "n", -9304.6, 2933.8),
        ((1, 1, 2), "n", -9304.5, 2931.0), ((2, 1, 0), "n", -9299.6, 2949.8),
        ((2, 1, 1), "n", -9305.0, 2930.1), ((2, 1, 2), "n", -9303.7, 2926.5),
    ]
    return pd.DataFrame(
        [{"p": p, "d": d, "q": q, "trend": t, "blue_sell": a, "brecha_pct": b}
         for (p, d, q), t, a, b in filas]
    )


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    ruta = ASSETS / "manifest.json"
    return json.loads(ruta.read_text()) if ruta.exists() else {}


@st.cache_data(show_spinner=False)
def grilla() -> pd.DataFrame:
    from models.regresion import grilla_horaria
    return grilla_horaria(mercado())


@st.cache_data(show_spinner=False)
def serie(target: str) -> pd.Series:
    return grilla()[target].dropna()


@st.cache_data(show_spinner=False)
def features(horizonte: int):
    from models.regresion import construir_features_horarias
    return construir_features_horarias(mercado(), panel(), horizonte=horizonte)


@st.cache_data(show_spinner=False)
def glosario() -> pd.DataFrame:
    from diccionario_datos import tabla_glosario
    return tabla_glosario()


def estado_assets() -> list[dict]:
    filas = [
        ("metricas_reporte.json", MODELADO / "metricas_reporte.json",
         "métricas y tablas"),
        ("mercado.parquet", ASSETS / "mercado.parquet", "serie horaria"),
        ("panel.parquet", ASSETS / "panel.parquet", "Gaceta diaria"),
        ("gaceta.parquet", ASSETS / "gaceta.parquet", "normas etiquetadas"),
        ("crudo.parquet", ASSETS / "crudo.parquet", "snapshots crudos"),
        ("predicciones_h1.parquet", ASSETS / "predicciones_h1.parquet",
         "walk-forward 1 h"),
        ("predicciones_h24.parquet", ASSETS / "predicciones_h24.parquet",
         "walk-forward 24 h"),
        ("roc_curvas.parquet", ASSETS / "roc_curvas.parquet", "ROC / AUC"),
        ("ols_hac.parquet", ASSETS / "ols_hac.parquet", "coeficientes HAC"),
        ("direccion.parquet", ASSETS / "direccion.parquet",
         "matriz de confusión"),
    ]
    out = []
    for nombre, ruta, uso in filas:
        out.append({
            "archivo": nombre,
            "uso": uso,
            "existe": ruta.exists(),
            "bytes": ruta.stat().st_size if ruta.exists() else 0,
            "ruta": str(ruta.relative_to(ROOT)),
        })
    return out


def cobertura() -> dict:
    m, p, g = mercado(), panel(), gaceta()
    return {
        "mercado_filas": len(m),
        "mercado_desde": m.fecha.min(),
        "mercado_hasta": m.fecha.max(),
        "panel_filas": len(p),
        "panel_desde": p.fecha_d.min(),
        "panel_hasta": p.fecha_d.max(),
        "gaceta_filas": len(g),
        "gaceta_desde": g.fecha_publicacion.min() if len(g) else None,
        "gaceta_hasta": g.fecha_publicacion.max() if len(g) else None,
    }
