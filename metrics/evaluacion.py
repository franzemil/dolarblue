"""RMSE, MAE, baseline naïve y tablas de evaluación.

Envoltorios de sklearn + baseline naïve. Contenido migrado verbatim desde
`modelado/metricas.py` (sin cambios de lógica).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Raíz del error cuadrático medio (sklearn)."""
    return float(root_mean_squared_error(np.asarray(y_true, float),
                                         np.asarray(y_pred, float)))


def mae(y_true, y_pred) -> float:
    """Error absoluto medio (sklearn)."""
    return float(mean_absolute_error(np.asarray(y_true, float),
                                     np.asarray(y_pred, float)))


def naive_ultimo(y_train, n: int, index=None) -> pd.Series:
    """Baseline naïve con sklearn: repite el último valor de entrenamiento.

    `DummyRegressor(strategy="constant")` aprende una constante y la predice;
    equivale al paseo aleatorio de un paso.
    """
    y_train = np.asarray(y_train, float)
    dummy = DummyRegressor(strategy="constant", constant=y_train[-1])
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return pd.Series(dummy.predict(np.zeros((n, 1))), index=index)


def evaluar_simple(nombre: str, y_true, y_pred) -> dict:
    """Versión sin baseline: solo RMSE y MAE del modelo."""
    return {
        "modelo": nombre,
        "n_test": len(np.asarray(y_true)),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
    }


def evaluar(nombre: str, y_true, y_pred, y_naive) -> dict:
    """RMSE/MAE del modelo y del naïve, más la mejora relativa (%)."""
    r, m = rmse(y_true, y_pred), mae(y_true, y_pred)
    rn, mn = rmse(y_true, y_naive), mae(y_true, y_naive)
    return {
        "modelo": nombre,
        "n_test": len(np.asarray(y_true)),
        "rmse": round(r, 4),
        "mae": round(m, 4),
        "rmse_naive": round(rn, 4),
        "mae_naive": round(mn, 4),
        "mejora_rmse_pct": round(100 * (1 - r / rn), 2) if rn else None,
        "mejora_mae_pct": round(100 * (1 - m / mn), 2) if mn else None,
    }
