"""Normalización de respuestas de la API: envelope crudo -> payload tipado.

La API tiene 4 variantes de envelope y ~10 formas de payload internas. Este
módulo centraliza ese conocimiento para que los collectors solo traten con
modelos Pydantic (collectors/models.py).

- ``unwrap_envelope``: detecta la variante (A cached-quote, B bare wrapper,
  C sin wrapper) y devuelve el payload interno + la fecha de publicación.
- ``normalize[familia]``: convierte el payload interno en modelos tipados.
- ``family_records``: aplana el payload tipado a filas CSV con columnas de
  procedencia (source_url, extracted_date) para data/normalized/.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ValidationError

from collectors.models import (
    BancosHistorial,
    ChartPoint,
    ExchangeSnapshot,
    PlatformPoint,
    P2POrders,
    Quote,
    RawDict,
    Vela,
)


class NormalizeError(ValueError):
    """El payload no calzó con el modelo de su familia."""


def _parse_dt(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def unwrap_envelope(body) -> tuple[object, datetime | None]:
    """Desenrolla el envelope y devuelve (payload_interno, published_date).

    Variante A (cached-quote): {data, cached, fetched_at, updated_at, cache_key}
    Variante B (bare wrapper): {data} — a veces con count/window al lado
    Variante C (sin wrapper):  el body ya ES el payload
    """
    if not isinstance(body, dict):
        return body, None
    if "data" in body:
        published = _parse_dt(body.get("fetched_at")) or _parse_dt(body.get("updated_at"))
        inner = body["data"]
        if isinstance(inner, dict):
            published = published or _parse_dt(inner.get("fetched_at"))
        return inner, published
    return body, _parse_dt(body.get("fetched_at"))


# --------------------- Normalización por familia ------------------------------
# Cada función recibe el payload interno desenrollado y devuelve modelos tipados.


def normalize_quote(payload, spec_name: str) -> list[Quote]:
    """Familia quote: cotizaciones sueltas. Normaliza las 3 variantes de campos
    (buy/sell, addValue/withdrawValue, blue/official anidados) a una lista."""
    if not isinstance(payload, dict):
        raise NormalizeError(f"{spec_name}: quote sin dict (llegó {type(payload).__name__})")
    quotes: list[Quote] = []
    fetched = _parse_dt(payload.get("fetched_at"))
    if "blue" in payload or "official" in payload:
        pair = payload.get("pair")
        if payload.get("blue"):
            quotes.append(
                Quote(platform="dolar_blue_p2p", pair=pair, buy=payload["blue"].get("buy"),
                      sell=payload["blue"].get("sell"), fetched_at=fetched)
            )
        if payload.get("official"):
            quotes.append(
                Quote(platform="oficial_bcb", pair=pair, buy=payload["official"].get("buy"),
                      sell=payload["official"].get("sell"), fetched_at=fetched)
            )
        return quotes
    buy = payload.get("buy")
    sell = payload.get("sell")
    if buy is None and "withdrawValue" in payload:  # airtm
        buy, sell = payload.get("withdrawValue"), payload.get("addValue")
    quotes.append(
        Quote(platform=spec_name, pair=payload.get("pair"), buy=buy, sell=sell, fetched_at=fetched)
    )
    return quotes


def normalize_chart_6m(payload, spec_name: str) -> list[ChartPoint]:
    if not isinstance(payload, list):
        raise NormalizeError(f"{spec_name}: chart sin lista (llegó {type(payload).__name__})")
    return [ChartPoint.model_validate(p) for p in payload]


def normalize_plataformas(payload, spec_name: str, lado: str | None = None) -> list[PlatformPoint]:
    """Familia plataformas: series por plataforma -> formato largo tipado."""
    if isinstance(payload, dict) and "lado" in payload:  # variante C sin wrapper
        lado = payload.get("lado")
        series = payload.get("plataformas", [])
    elif isinstance(payload, dict):
        series = payload.get("plataformas", [])
    else:
        raise NormalizeError(f"{spec_name}: plataformas sin dict")
    points = []
    for platform in series:
        for t, rate in platform.get("puntos", []):
            points.append(PlatformPoint(lado=lado or "?", platform=platform["id"], t=t, rate=rate))
    return points


def normalize_bancos(payload, spec_name: str) -> BancosHistorial:
    """Familia bancos: {pizarra:{bank:[...]}, real:{...}, tco:[...]} aplanado."""
    if not isinstance(payload, dict):
        raise NormalizeError(f"{spec_name}: bancos sin dict")
    pizarra = [
        {"banco": bank, **p} for bank, series in (payload.get("pizarra") or {}).items() for p in series
    ]
    real = [
        {"banco": bank, **p} for bank, series in (payload.get("real") or {}).items() for p in series
    ]
    return BancosHistorial.model_validate({"pizarra": pizarra, "real": real, "tco": payload.get("tco", [])})


def normalize_velas(payload, spec_name: str) -> list[Vela]:
    if isinstance(payload, dict):  # por si envuelve en {"velas": [...]}
        payload = payload.get("velas", [])
    if not isinstance(payload, list):
        raise NormalizeError(f"{spec_name}: velas sin lista")
    return [Vela.model_validate(v) for v in payload]


def normalize_orders(payload, spec_name: str) -> P2POrders:
    if not isinstance(payload, dict):
        raise NormalizeError(f"{spec_name}: orders sin dict")
    return P2POrders.model_validate(
        {
            "source": payload.get("source", spec_name),
            "pair": payload.get("pair"),
            "side": payload.get("side"),
            "count": payload.get("count", len(payload.get("orders", []))),
            "fetched_at": payload.get("fetched_at"),
            "orders": payload.get("orders", []),
        }
    )


def normalize_snapshots(payload, spec_name: str) -> list[ExchangeSnapshot]:
    if not isinstance(payload, list):
        raise NormalizeError(f"{spec_name}: snapshots sin lista")
    return [ExchangeSnapshot.model_validate(s) for s in payload]


NORMALIZERS: dict[str, Callable] = {
    "quote": normalize_quote,
    "chart_6m": normalize_chart_6m,
    "plataformas": normalize_plataformas,
    "bancos": normalize_bancos,
    "velas": normalize_velas,
    "orders": normalize_orders,
    "snapshots": normalize_snapshots,
}


def normalize(family: str, payload, spec_name: str, **kw) -> object:
    """Punto de entrada: familia -> payload tipado. Familia 'raw' se envuelve
    en RawDict (extra=allow) para que toda respuesta quede tipada."""
    if family == "raw":
        return RawDict(data=payload)
    if family == "plataformas":
        return normalize_plataformas(payload, spec_name, **kw)
    return NORMALIZERS[family](payload, spec_name)


# -------------------- Aplanado a CSV de data/normalized -----------------------
# Cada entrada: (nombre de tabla, columnas, extractor de filas desde el payload tipado).


def family_records(family: str, payload, source_url: str, extracted_date: datetime) -> tuple[str, list[dict]]:
    """Devuelve (tabla, filas) con columnas de procedencia inyectadas."""
    provenance = {"source_url": source_url, "extracted_date": extracted_date.isoformat()}

    if family == "quote":
        rows = [q.model_dump() for q in payload]
    elif family == "chart_6m":
        rows = [p.model_dump() for p in payload]
    elif family == "plataformas":
        rows = [p.model_dump() for p in payload]
    elif family == "bancos":
        rows = [p.model_dump() for p in payload.pizarra + payload.real + payload.tco]
    elif family == "velas":
        rows = [v.model_dump() for v in payload]
    elif family == "orders":
        o = payload.model_dump()
        rows = [{**order, "count": o["count"]} for order in o.pop("orders", [])]
    elif family == "snapshots":
        rows = [s.model_dump() for s in payload]
    else:  # raw: sin tabla normalizada
        return "", []

    # objetos anidados (fechas) a string
    for row in rows:
        for k, v in list(row.items()):
            if isinstance(v, datetime):
                row[k] = v.isoformat()
        row.update(provenance)
    return family, rows
