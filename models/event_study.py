"""Event study Gaceta <-> blue: ancla horaria y retornos por horizonte.

Funciones puras extraídas de `modelado/reporte_modelado.py` y del notebook —
mismas operaciones, sin ejecución al importar (la serie de precios entra por
parámetro en vez de vivir como global del módulo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONTES = {"5 h": 5, "1 d": 24, "3 d": 72, "5 d": 120}
HORA_REF_UTC = 12


def preparar_serie_precio(mkt: pd.DataFrame) -> pd.DataFrame:
    """Serie horaria ordenada [fecha, blue_mid] para anclar y medir retornos."""
    return mkt[["fecha", "blue_mid"]].dropna().sort_values("fecha").reset_index(drop=True)


def ancla_precio(mkt: pd.DataFrame, fecha: pd.Timestamp, hora_ref: int = HORA_REF_UTC):
    """(fecha, precio) de la hora más cercana a `hora_ref` UTC del día dado.

    Devuelve None si el día no tiene datos horarios.
    """
    dia = mkt[mkt.fecha.dt.date == fecha.date()]
    if dia.empty:
        return None
    i0 = (dia.fecha.dt.hour - hora_ref).abs().idxmin()
    return mkt.at[i0, "fecha"], mkt.at[i0, "blue_mid"]


def retorno_horizonte(mkt: pd.DataFrame, t0, p0, horas: int) -> float:
    """Variación % desde el ancla hasta t0 + horas (hora más cercana)."""
    i1 = (mkt.fecha - (t0 + pd.Timedelta(hours=horas))).abs().idxmin()
    p1 = mkt.at[i1, "blue_mid"]
    return (p1 - p0) / p0 * 100 if p0 and not pd.isna(p1) else np.nan
