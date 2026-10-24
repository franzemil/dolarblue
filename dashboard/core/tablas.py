"""Tablas derivadas del informe, listas para mostrar en el dashboard."""
from __future__ import annotations

import pandas as pd

from diccionario_datos import etiqueta, etiqueta_eje, etiqueta_tag

from core.config import HORIZONTES, HORIZONTES_EVENTO


def tabla_rl(R: dict) -> pd.DataFrame:
    filas = []
    for h in HORIZONTES:
        for target in ("brecha_pct", "blue_mid"):
            v = R["por_horizonte"][str(h)]["rl"][target]
            filas.append({
                "Nivel": etiqueta_eje(target), "Horizonte": f"{h} h",
                "n test": v["n_test"], "RMSE": v["rmse"], "MAE": v["mae"],
                "Naïve RMSE": v["rmse_naive"], "Naïve MAE": v["mae_naive"],
                "RMSE ÷ naïve": v["ratio"],
                "MAE ÷ naïve": round(v["mae"] / v["mae_naive"], 2),
            })
    return pd.DataFrame(filas)


def tabla_arima(R: dict) -> pd.DataFrame:
    filas = []
    for h in HORIZONTES:
        for target in ("brecha_pct", "blue_mid"):
            v = R["por_horizonte"][str(h)]["arima"][target]
            filas.append({
                "Nivel": etiqueta_eje(target), "Horizonte": f"{h} h",
                "Modelo": v["modelo"], "AIC": v["aic"],
                "n test": v["n_test"], "RMSE": v["rmse"], "MAE": v["mae"],
                "Naïve RMSE": v["rmse_naive"], "RMSE ÷ naïve": v["ratio"],
            })
    return pd.DataFrame(filas)


def tabla_comparativa(R: dict) -> pd.DataFrame:
    filas = []
    for h in HORIZONTES:
        ph = R["por_horizonte"][str(h)]
        for target in ("brecha_pct", "blue_mid"):
            for clave, nombre in (("arima", ph["arima"][target]["modelo"]),
                                  ("rl", "RL múltiple")):
                v = ph[clave][target]
                filas.append({
                    "Nivel": etiqueta_eje(target), "Horizonte": f"{h} h",
                    "Modelo": nombre, "RMSE": v["rmse"],
                    "RMSE ÷ naïve": v["ratio"], "MAE": v["mae"],
                    "MAE ÷ naïve": round(v["mae"] / v["mae_naive"], 2),
                })
    return pd.DataFrame(filas)


def tabla_ols(R: dict) -> pd.DataFrame:
    coef = pd.DataFrame(R["ols_hac"]["coeficientes"])
    coef["significativo"] = coef["p"] < 0.05
    return coef[["var", "coef", "ee", "p", "lo", "hi", "significativo"]]


def tabla_eventos(R: dict) -> pd.DataFrame:
    filas = []
    for tag, d in R["eventos"].items():
        fila = {"tag": etiqueta_tag(tag), "tag_id": tag}
        for h in HORIZONTES_EVENTO:
            fila[h] = d[h]["media"]
            fila[f"n {h}"] = d[h]["n"]
            fila[f"ic {h}"] = d[h]["ic95"]
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("5 h")


def tabla_spearman(R: dict) -> pd.DataFrame:
    filas = [
        {"tag": etiqueta_tag(tag), "tag_id": tag, "n": v["n"],
         "t": v["t"], "t1": v["t1"]}
        for tag, v in R["spearman"].items()
    ]
    return pd.DataFrame(filas).sort_values("t1", key=abs, ascending=False)


def tabla_parametros(R: dict, target: str) -> pd.DataFrame:
    return pd.DataFrame(R["arima"][target]["parametros"]).rename(
        columns={"parametro": "Parámetro", "coef": "Coef.", "ee": "EE",
                 "z": "z", "p": "p", "ic_inf": "IC 2,5", "ic_sup": "IC 97,5"})


def tabla_criterios(R: dict, target: str) -> pd.DataFrame:
    return pd.DataFrame([R["arima"][target]["criterios"]]).rename(
        columns={"k": "k", "n": "n", "logL": "logL", "AIC": "AIC",
                 "BIC": "BIC", "AICc": "AICc", "HQIC": "HQIC"})


def tabla_direccion(direccion: pd.DataFrame, horizonte: int,
                    target: str = "blue_mid") -> pd.DataFrame:
    d = direccion[(direccion.horizonte == horizonte)
                  & (direccion.target == target)]
    return d.rename(columns={
        "modelo": "Modelo", "n": "n", "aciertos": "Aciertos",
        "accuracy": "Accuracy", "tp": "TP", "fn": "FN", "fp": "FP",
        "tn": "TN"})


def etiqueta_target(target: str) -> str:
    return etiqueta_eje(target)


def etiqueta_corta(target: str) -> str:
    return etiqueta(target)
