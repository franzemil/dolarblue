"""ARIMA: selección (p,d,q) por AIC y evaluación en hold-out.

Funciones puras extraídas de `modelado/reporte_modelado.py` (arima_7d) —
mismas operaciones, sin ejecución al importar.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from metrics import mae, naive_ultimo, rmse


def seleccionar_arima(train, max_pq=3):
    """Grilla (p,d,q) x término de deriva por AIC.

    Devuelve (aic, p, d, q, trend) del mejor modelo (menor AIC).
    """
    mejor = None
    for p, d, q in product(range(max_pq + 1), range(2), range(max_pq + 1)):
        if p == 0 and q == 0:
            continue
        for trend in (["n", "c"] if d >= 1 else ["c"]):
            try:
                fit = ARIMA(train, order=(p, d, q), trend=trend).fit()
                if mejor is None or fit.aic < mejor[0]:
                    mejor = (fit.aic, p, d, q, trend)
            except Exception:
                continue
    return mejor


def adf_pvalores(train):
    """p-valores ADF del nivel y de la primera diferencia."""
    return (round(float(adfuller(train)[1]), 4),
            round(float(adfuller(train.diff().dropna())[1]), 4))


def arima_holdout(y, n_test=7, max_pq=3):
    """Ajusta ARIMA en train, pronostica el hold-out y evalúa vs naïve.

    Devuelve un dict de métricas (idéntico a `arima_7d` de
    modelado/reporte_modelado.py) más las series `_pred` y `_naive`.
    """
    train, test = y.iloc[:-n_test], y.iloc[-n_test:]
    aic, bp, bd, bq, bt = seleccionar_arima(train, max_pq)
    mod = ARIMA(train, order=(bp, bd, bq), trend=bt).fit()
    pred = mod.get_forecast(steps=n_test).predicted_mean
    nv = naive_ultimo(train, n_test, test.index)

    # DM simple (sin corrección HLN), igual que reporte_modelado.py
    e1, e2 = test.values - pred.values, test.values - nv.values
    d = e1 ** 2 - e2 ** 2
    gamma0 = np.var(d, ddof=0)
    dm = d.mean() / np.sqrt(gamma0 / len(d)) if gamma0 > 0 else np.nan

    adf_nivel, adf_diff = adf_pvalores(train)
    r = rmse(test, pred)
    return {
        "modelo": f"ARIMA({bp},{bd},{bq})",
        "aic": round(float(aic), 1),
        "n_train": int(len(train)),
        "test_desde": str(test.index[0].date()),
        "test_hasta": str(test.index[-1].date()),
        "rmse": round(r, 3),
        "mae": round(mae(test, pred), 3),
        "rmse_naive": round(rmse(test, nv), 3),
        "mejora_pct": round(100 * (1 - r / rmse(test, nv)), 1),
        "dm": round(float(dm), 2),
        "adf_p_nivel": adf_nivel,
        "adf_p_diff": adf_diff,
        "_pred": pred,
        "_naive": nv,
    }


def arima_holdout_detalle(y, n_test=168, max_pq=2):
    """Como `arima_holdout`, más criterios de información y parámetros.

    Pensado para frecuencia horaria: `y` es la serie ya recortada a la ventana
    de entrenamiento + test, y `n_test` son las horas del hold-out.
    """
    out = arima_holdout(y, n_test=n_test, max_pq=max_pq)
    train = y.iloc[:-n_test]
    _, p, d, q, trend = seleccionar_arima(train, max_pq)
    mod = ARIMA(train, order=(p, d, q), trend=trend).fit()

    k, n = len(mod.params), int(mod.nobs)
    ci = mod.conf_int()
    out["criterios"] = {
        "k": k, "n": n, "logL": round(float(mod.llf), 1),
        "AIC": round(float(mod.aic), 1), "BIC": round(float(mod.bic), 1),
        "AICc": round(float(mod.aic + 2 * k * (k + 1) / (n - k - 1)), 1),
        "HQIC": round(float(mod.hqic), 1),
    }
    out["parametros"] = [
        {"parametro": str(nm), "coef": round(float(mod.params.iloc[i]), 4),
         "ee": round(float(mod.bse.iloc[i]), 4),
         "z": round(float(mod.tvalues.iloc[i]), 2),
         "p": float(mod.pvalues.iloc[i]),
         "ic_inf": round(float(ci.iloc[i, 0]), 4),
         "ic_sup": round(float(ci.iloc[i, 1]), 4)}
        for i, nm in enumerate(mod.params.index)
    ]
    out["mae_naive"] = round(float(mae(y.iloc[-n_test:], out["_naive"])), 3)
    out.pop("_pred", None)
    out.pop("_naive", None)
    return out

def arima_walkforward(y, split, horizonte, order, trend="n"):
    """Pronóstico de `horizonte` pasos con origen móvil (walk-forward).

    Para cada hora objetivo `t` de la ventana de prueba se usa el origen
    `t - horizonte`: el modelo se ajusta con los datos hasta el primer origen y
    luego se actualiza con los valores reales (`refit=False`), de modo que cada
    pronóstico solo usa información disponible en su origen.

    Devuelve (pred, naive, lo, hi): cuatro Series indexadas por la hora
    objetivo. `naive` es el último valor observado en el origen,
    `y(t - horizonte)`; `lo`/`hi` son los extremos del IC95 del pronóstico.
    """
    objetivos = y.loc[y.index > split].index
    origen0 = objetivos[0] - pd.Timedelta(hours=horizonte)
    res = ARIMA(y.loc[:origen0], order=order, trend=trend).fit()
    ultimo, preds, los, his = origen0, {}, {}, {}
    for t in objetivos:
        origen = t - pd.Timedelta(hours=horizonte)
        while ultimo < origen:
            ultimo = ultimo + pd.Timedelta(hours=1)
            if ultimo in y.index:
                res = res.append(y.loc[[ultimo]], refit=False)
        fc = res.get_forecast(steps=horizonte)
        ci = fc.conf_int(alpha=0.05).values[-1]
        preds[t] = float(fc.predicted_mean.iloc[-1])
        los[t], his[t] = float(ci[0]), float(ci[1])
    pred = pd.Series(preds)
    naive = y.shift(horizonte).loc[pred.index]
    return pred, naive, pd.Series(los), pd.Series(his)

def seleccionar_por_horizonte(y, split, horizonte, ventana_h=2160, max_pq=2):
    """Elige (p, d, q) por AIC usando la ventana de entrenamiento del horizonte.

    El origen inicial de un pronóstico a `horizonte` pasos es `primer objetivo -
    horizonte`, así que la ventana de entrenamiento se desplaza con el
    horizonte. Devuelve un dict con order, trend, aic, n_train y origen0.
    """
    objetivos = y.loc[y.index > split].index
    origen0 = objetivos[0] - pd.Timedelta(hours=horizonte)
    tr = y.loc[:origen0].iloc[-ventana_h:]
    aic, p, d, q, trend = seleccionar_arima(tr, max_pq=max_pq)
    return {"order": (int(p), int(d), int(q)), "trend": trend,
            "aic": round(float(aic), 1), "n_train": int(len(tr)),
            "origen0": origen0,
            "modelo": f"ARIMA({p},{d},{q})"}
