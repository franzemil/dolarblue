"""Modelos Pydantic del proyecto de colecta.

Dos capas:
- ``TypedResult[T]``: resultado de una colecta (procedencia + payload tipado).
  Es el "link" entre lo extraído (url, fechas, path) y los datos normalizados.
- Payloads por familia de endpoint (``Quote``, ``ChartPoint``, ``Vela``, ...):
  la forma normalizada que se persiste en data/normalized/.

Todo campo lleva ``description`` para que el modelo sea su propia
documentación y para poder exportar el JSON Schema completo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class TypedResult(BaseModel, Generic[T]):
    """Resultado tipado de una colecta: procedencia + payload normalizado."""

    url: str = Field(description="URL fuente: endpoint de la API o página de origen")
    extracted_date: datetime = Field(description="Momento (UTC) en que se extrajo el dato")
    published_date: datetime | None = Field(
        None,
        description="Fecha de publicación en la fuente (fetched_at / fecha de edición)",
    )
    path: Path | None = Field(
        None, description="Ruta del archivo crudo local (snapshot JSON / PDF)"
    )
    status: str = Field("ok", description="ok | cache | error | detalle_vacio | descarga_fallida")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Metadatos extra: {edicion, norma_id, endpoint, familia...}"
    )
    payload: T | None = Field(None, description="Datos normalizados y validados con el modelo T")


class FileResult(TypedResult[None]):
    """Resultado de una colecta de archivos (PDFs, CSVs): hijo de TypedResult
    sin payload normalizado — el dato crudo queda en ``path`` y la procedencia
    en los campos heredados."""

    payload: None = Field(
        None,
        description="Siempre None: las colectas de archivos no normalizan contenido, guardan el archivo en path",
    )


# --------------------------- Payloads normalizados ----------------------------


class Quote(BaseModel):
    """Cotización puntual de una plataforma (compra/venta en BOB por USD/USDT)."""

    model_config = ConfigDict(extra="allow")

    platform: str = Field(description="Identificador de la plataforma (airtm, binance_p2p, oficial_bcb...)")
    pair: str | None = Field(None, description="Par cotizado, p. ej. BOB/USDT")
    buy: float | None = Field(None, description="Precio de compra en BOB por USD/USDT")
    sell: float | None = Field(None, description="Precio de venta en BOB por USD/USDT")
    fetched_at: datetime | None = Field(None, description="Fecha de publicación en la fuente")


class ChartPoint(BaseModel):
    """Punto horario de la serie 6m blue vs oficial (endpoint chart/6m)."""

    t: int = Field(description="Timestamp UNIX (epoch seconds) del bucket horario")
    bb: float | None = Field(None, description="Blue compra (BOB/USD)")
    bs: float | None = Field(None, description="Blue venta (BOB/USD)")
    ob: float | None = Field(None, description="Oficial compra (BOB/USD)")
    os: float | None = Field(None, description="Oficial venta (BOB/USD)")


class PlatformPoint(BaseModel):
    """Punto de la serie de una plataforma (chart/plataformas, formato largo)."""

    lado: str = Field(description="Lado del mercado: compra | venta")
    platform: str = Field(description="Identificador de la plataforma")
    t: int = Field(description="Timestamp UNIX del punto")
    rate: float = Field(description="Cotización BOB/USD del punto")


class BancoPizarraPoint(BaseModel):
    """Pizarra diaria de un banco (precio mostrado en ventana)."""

    banco: str = Field(description="Código del banco (bcp, bancosol...)")
    fecha: str = Field(description="Fecha del corte (YYYY-MM-DD)")
    compra: float | None = Field(None, description="Precio de compra (BOB/USD)")
    venta: float | None = Field(None, description="Precio de venta (BOB/USD)")


class BancoRealPoint(BaseModel):
    """Operación real reportada por un banco (con volumen)."""

    banco: str = Field(description="Código del banco")
    fecha: str = Field(description="Fecha de la operación (YYYY-MM-DD)")
    compra: float | None = Field(None, description="Precio de compra (BOB/USD)")
    monto_usd: float | None = Field(None, description="Volumen operado en USD")
    operaciones: int | None = Field(None, description="Número de operaciones del día")


class TcoPoint(BaseModel):
    """Tipo de cambio compuesto (TCO) diario."""

    fecha: str = Field(description="Fecha (YYYY-MM-DD)")
    valor: float = Field(description="Valor TCO (BOB/USD)")


class BancosHistorial(BaseModel):
    """Historial bancario completo (endpoint bancos/historial)."""

    pizarra: list[BancoPizarraPoint] = Field(default_factory=list, description="Serie de pizarra por banco")
    real: list[BancoRealPoint] = Field(default_factory=list, description="Operaciones reales por banco")
    tco: list[TcoPoint] = Field(default_factory=list, description="Serie TCO diaria")


class Vela(BaseModel):
    """Vela OHLC del libro P2P (endpoint pro/velas)."""

    t: int = Field(description="Timestamp UNIX de apertura de la vela")
    o: float = Field(description="Precio de apertura (BOB/USDT)")
    h: float = Field(description="Precio máximo")
    l: float = Field(description="Precio mínimo")
    c: float = Field(description="Precio de cierre")
    vol_compra: float = Field(description="Volumen estimado de compra (USDT)")
    vol_venta: float = Field(description="Volumen estimado de venta (USDT)")
    trades: int = Field(description="Número estimado de operaciones en la vela")


class P2POrder(BaseModel):
    """Orden individual del libro P2P de un exchange."""

    rank: int | None = Field(None, description="Posición en el orden del libro")
    rate: float = Field(description="Precio de la orden (BOB/USDT)")
    platform: str | None = Field(None, description="Exchange de origen (binance, bybit, bitget)")
    side: str | None = Field(None, description="buy | sell")
    available_usdt: float | None = Field(None, description="Monto disponible en USDT")
    min_bob: float | None = Field(None, description="Mínimo aceptado en BOB")
    max_bob: float | None = Field(None, description="Máximo aceptado en BOB")
    completed_orders: int | None = Field(None, description="Órdenes completadas históricas del trader")
    completion_rate: float | None = Field(None, description="Tasa de finalización del trader (0-1)")


class P2POrders(BaseModel):
    """Libro de órdenes P2P (endpoints priv_p2p_*)."""

    source: str | None = Field(None, description="Exchange de origen")
    pair: str | None = Field(None, description="Par, p. ej. BOB/USDT")
    side: str | None = Field(None, description="buy | sell")
    count: int | None = Field(None, description="Número de órdenes en la respuesta")
    fetched_at: datetime | None = Field(None, description="Fecha de publicación en la fuente")
    orders: list[P2POrder] = Field(default_factory=list, description="Órdenes del libro")


class ExchangeSnapshot(BaseModel):
    """Snapshot de cotización de un exchange en el historial privado."""

    timestamp: int = Field(description="Timestamp UNIX del snapshot")
    exchange: str = Field(description="Identificador del exchange")
    buy: float | None = Field(None, description="Precio de compra (BOB/USD)")
    sell: float | None = Field(None, description="Precio de venta (BOB/USD)")
    created_at_ts: int | None = Field(None, description="Timestamp de creación del registro")


class RawDict(BaseModel):
    """Payload sin esquema cerrado (familias system, pro misc, stats)."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] | list[Any] | None = Field(
        None, description="Contenido crudo del endpoint tal como llegó"
    )


# Mapa familia -> anotación del payload (documenta el contrato de cada familia).
# Las familias que devuelven series van como list[Modelo]; las únicas, como modelo.
FAMILY_MODELS: dict[str, Any] = {
    "quote": list[Quote],
    "chart_6m": list[ChartPoint],
    "plataformas": list[PlatformPoint],
    "bancos": BancosHistorial,
    "velas": list[Vela],
    "orders": P2POrders,
    "snapshots": list[ExchangeSnapshot],
    "raw": RawDict,
}
