"""Regresión lineal: features horarias, RL múltiple (sklearn) y OLS-HAC.

Funciones puras extraídas de `modelado/reporte_modelado.py` y del notebook —
mismas operaciones, sin ejecución al importar.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from metrics import naive_ultimo

CANALES_GACETA = ["n_docs_fx", "n_docs_fx_alta", "n_docs_deuda",
                  "n_docs_comercio", "n_docs_mineria"]

# Canales de la Gaceta que se incorporan a la grilla horaria (broadcast diario)
CANALES_HORARIO = ["n_docs_fx", "n_docs_fx_alta"]


def grilla_horaria(mercado: pd.DataFrame) -> pd.DataFrame:
    """Grilla horaria regular con huecos cortos interpolados.

    Es la base común del notebook de análisis y del reporte: así ambos
    ajustan el ARIMA sobre exactamente la misma serie.
    """
    hor = mercado[["fecha", "blue_mid", "oficial_mid", "brecha_pct",
                   "vol_24h"]].copy()
    hor["fecha"] = pd.to_datetime(hor["fecha"]).dt.tz_localize(None)
    hor = hor.set_index("fecha").sort_index().asfreq("h")
    cols_num = ["blue_mid", "oficial_mid", "brecha_pct", "vol_24h"]
    hor[cols_num] = hor[cols_num].interpolate(limit=3)
    return hor.dropna(subset=["blue_mid"])


def construir_features_horarias(mercado: pd.DataFrame, panel: pd.DataFrame,
                                horizonte: int = 1):
    """Grilla horaria + rezagos + canales de la Gaceta, para un horizonte dado.

    Cada fila es una **hora objetivo** `t`: el objetivo es `y(t)` y todos los
    predictores son los que ya se conocían en `t - horizonte`. Se logra
    desplazando el bloque de rezagos `horizonte - 1` horas adicionales (la
    Gaceta, un día más por cada día de horizonte), de modo que la información
    más fresca disponible sea exactamente la del origen.

    Devuelve (datos, cols): tabla con `fecha` (la hora objetivo) y las columnas
    `origen_blue`/`origen_brecha` (el valor observado en `t - horizonte`, que es
    el naïve del mismo horizonte), sin NaN, y la lista de predictores.
    """
    hor = grilla_horaria(mercado)

    hor["origen_blue"] = hor["blue_mid"].shift(horizonte)
    hor["origen_brecha"] = hor["brecha_pct"].shift(horizonte)

    hor["brecha_lag1h"] = hor["brecha_pct"].shift(1)
    hor["brecha_lag3h"] = hor["brecha_pct"].shift(3)
    hor["brecha_lag24h"] = hor["brecha_pct"].shift(24)
    hor["oficial_lag1h"] = hor["oficial_mid"].shift(1)
    hor["vol_lag1h"] = hor["vol_24h"].shift(1)
    hor["ret_blue_lag1h"] = hor["blue_mid"].pct_change().shift(1)

    dias = 1 + -((1 - horizonte) // 24)     # ceil((h-1)/24) días de rezago extra
    gac_h = panel.set_index("fecha_d")[CANALES_HORARIO].shift(dias)
    hor["fecha_d"] = hor.index.normalize()
    hor = hor.join(gac_h, on="fecha_d").drop(columns="fecha_d")
    hor[CANALES_HORARIO] = hor[CANALES_HORARIO].fillna(0)

    feats = ["brecha_lag1h", "brecha_lag3h", "brecha_lag24h", "oficial_lag1h",
             "vol_lag1h", "ret_blue_lag1h"] + CANALES_HORARIO
    extra = horizonte - 1                   # la info más fresca es la del origen
    if extra:
        hor[feats] = hor[feats].shift(extra)
    dum_hora = pd.get_dummies(hor.index.hour, prefix="hora", drop_first=True,
                              dtype=float)
    dum_hora.index = hor.index
    dum_dia = pd.get_dummies(hor.index.dayofweek, prefix="dow", drop_first=True,
                             dtype=float)
    dum_dia.index = hor.index

    datos = pd.concat([hor[["blue_mid", "brecha_pct", "origen_blue",
                            "origen_brecha"]], hor[feats], dum_hora, dum_dia],
                      axis=1).dropna()
    datos.insert(0, "fecha", datos.index)
    datos = datos.reset_index(drop=True)
    no_pred = ("fecha", "brecha_pct", "blue_mid", "origen_blue", "origen_brecha")
    cols = [c for c in datos.columns if c not in no_pred]
    return datos, cols


def ajustar_rl_multiple(train, test, cols_x, target):
    """RL múltiple (mínimos cuadrados por SVD) + naïve del mismo horizonte.

    El naïve es el último valor observado en el origen (`origen_<target>`), que
    ya viene alineado fila a fila con el objetivo `target`.
    Devuelve (pred, naive) como Series indexadas por test["fecha"].
    """
    mod = LinearRegression().fit(train[cols_x], train[target])
    pred = pd.Series(mod.predict(test[cols_x]), index=test["fecha"].values)
    # origen_blue (nivel) / origen_brecha (brecha)
    col = "origen_blue" if target == "blue_mid" else "origen_brecha"
    nv = pd.Series(test[col].values, index=test["fecha"].values)
    return pred, nv


def ajustar_ols_hac(train, cols_x, target="brecha_pct", maxlags=7):
    """OLS con errores HAC (Newey-West); descarta columnas constantes.

    Un diseño de rango deficiente produce coeficientes/p-valores espurios.
    Devuelve (fit, cols_usadas).
    """
    cols_ok = [c for c in cols_x if train[c].nunique() > 1]
    X = sm.add_constant(train[cols_ok], has_constant="add")
    fit = sm.OLS(train[target], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return fit, cols_ok
