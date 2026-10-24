"""Registro de URLs de la API Dolar Blue Bolivia (un collector por URL).

Docs: https://www.dolarbluebolivia.click/devs · Base: https://api.dolarbluebolivia.click
Cada ``EndpointSpec`` declara la familia de normalización que le corresponde
(ver FAMILY_MODELS en collectors/models.py) y si requiere la X-API-Key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointSpec:
    """Una URL de la API.

    name:   nombre lógico del endpoint en los snapshots
    path:   ruta relativa al BASE_URL, con query incluida
    family: familia de normalización (FAMILY_MODELS en collectors/models.py)
    auth:   requiere header X-API-Key
    window: añade from_ts/to_ts de WINDOW_DAYS días
    export: streaming CSV/JSON (no pasa por envelope)
    """

    name: str
    path: str
    family: str
    auth: bool = False
    window: bool = False
    export: bool = False


BASE_URL = "https://api.dolarbluebolivia.click"

# Solo los endpoints con consumo downstream (modelos, notebook, informe):
#   chart_6m          -> normalized/chart_6m.csv -> market_hourly.csv (modelos horarios, fig_serie_blue_oficial)
#   bancos_historial  -> normalized/bancos.csv   -> fig_bancos (informe 04_analisis)
#   priv_history_export -> data/history_full.csv -> serie_diaria_blue.csv (modelos diarios, fig_brecha)
# Archivados (recuperables de git; sin consumo downstream: 40 llamadas/corrida):
#   health, official_rate, official_unificado, airtm, wallbit, takenos, eldorado,
#   saldoar, p2p_{bitget,bybit,mexc,okx}, chart_plataformas_{compra,venta},
#   pro_{resumen,libro,ejecucion_comprar_1000,profundidad_comprar,velas_1h,trades_100},
#   priv_{rates_official,history_data_quality,history_percentiles,history_moving_average,
#   history_rolling_returns,system_freshness,system_health,p2p_binance_sellers,
#   p2p_binance_buyers,p2p_bybit_sellers,p2p_bybit_buyers,p2p_bitget_sellers,
#   p2p_bitget_buyers,snapshots_exchanges_list,snapshots_exchanges_latest,
#   snapshots_exchanges_export}
PUBLIC_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("chart_6m", "/v1/chart/6m", "chart_6m"),
    EndpointSpec("bancos_historial", "/v1/bancos/historial", "bancos"),
)

PRIVATE_ENDPOINTS: tuple[EndpointSpec, ...] = ()

EXPORT_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec(
        "priv_history_export",
        "/private/v1/history/export",
        "raw",
        auth=True,
        window=True,
        export=True,
    ),
)

WINDOW_DAYS = 365
