"""Formato numérico y de fechas en español (coma decimal, punto de miles)."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

VACIO = "—"


def _es(v: float, nd: int) -> str:
    s = f"{v:,.{nd}f}"
    return s.replace(",", "\u00a0").replace(".", ",").replace("\u00a0", ".")


def num(v, nd: int = 2) -> str:
    """Número con separador de miles y coma decimal: 1234.5 -> '1.234,5'."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return VACIO
    return _es(float(v), nd)


def sig(v, nd: int = 4) -> str:
    return num(v, nd)


def bs(v, nd: int = 4) -> str:
    return f"{num(v, nd)} Bs/USD"


def pp(v, nd: int = 4) -> str:
    return f"{num(v, nd)} pp"


def pct(v, nd: int = 1, decimal: bool = True) -> str:
    if v is None:
        return VACIO
    valor = float(v) if decimal else float(v) * 100
    return f"{num(valor, nd)} %"


def miles(v, nd: int = 0) -> str:
    return num(v, nd)


def fecha(v, con_hora: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return VACIO
    if isinstance(v, str):
        v = pd.to_datetime(v, errors="coerce")
    if v is None or pd.isna(v):
        return VACIO
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.strftime("%d/%m/%Y %H:%M" if con_hora else "%d/%m/%Y")
    if isinstance(v, date):
        return v.strftime("%d/%m/%Y")
    return str(v)


def fecha_hora(v) -> str:
    return fecha(v, con_hora=True)


def unidad(target: str) -> str:
    return "Bs/USD" if target == "blue_mid" else "pp"


def metricas_es(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            nd = 4 if out[c].abs().max(skipna=True) < 10 else 2
            out[c] = out[c].map(lambda v: num(v, nd))
    return out


def pvalor(p) -> str:
    if p is None:
        return VACIO
    p = float(p)
    if p < 0.001:
        return "< 0,001"
    return num(p, 4)


def cobertura(v) -> str:
    """Cobertura empírica expresada como porcentaje."""
    if v is None:
        return VACIO
    return pct(float(v) * 100, 1)
