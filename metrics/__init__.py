"""Métricas de evaluación del proyecto.

- RMSE/MAE: envoltorios de sklearn (`root_mean_squared_error`,
  `mean_absolute_error`).
- Naïve: `DummyRegressor(strategy="constant")` — repite el último valor
  observado en entrenamiento (paseo aleatorio de un paso).

Migrado desde `modelado/metricas.py` sin cambios de lógica.
"""
from metrics.evaluacion import (evaluar, evaluar_simple, mae,  # noqa: F401
                                naive_ultimo, rmse)

__all__ = ["rmse", "mae", "naive_ultimo", "evaluar", "evaluar_simple"]
