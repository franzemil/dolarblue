"""Modelos del proyecto: regresión lineal (RL múltiple / OLS-HAC), ARIMA y
event study de la Gaceta.

Funciones puras extraídas de `modelado/reporte_modelado.py`,
`modelado/arima_modelo.py` y del notebook `analisis_gaceta_fx.ipynb` —
mismas operaciones, sin ejecución al importar.
"""
from models.arima import arima_holdout, seleccionar_arima  # noqa: F401
from models.event_study import (ancla_precio, retorno_horizonte,  # noqa: F401
                                preparar_serie_precio)
from models.regresion import ajustar_ols_hac, ajustar_rl_multiple  # noqa: F401

__all__ = ["ajustar_rl_multiple", "ajustar_ols_hac",
           "seleccionar_arima", "arima_holdout", "preparar_serie_precio",
           "ancla_precio", "retorno_horizonte"]
