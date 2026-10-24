"""Genera las cifras del informe (secciones Modelado y Evaluación) en HORARIO.

Salida: modelado/metricas_reporte.json + resumen por consola.
Todo se calcula desde los CSV vigentes: si el pipeline refresca datos, se
vuelve a correr este script y el informe queda al día.

Frecuencia: horaria (`market_hourly.csv`). Hold-out: las últimas 168 h.
Los canales de la Gaceta entran rezagados un día (no tienen resolución horaria).

Uso:
    cd 01_intro && .venv/bin/python final_project/modelado/reporte_modelado.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.arima import (arima_holdout_detalle, arima_walkforward,
                          seleccionar_por_horizonte)  # noqa: E402
from models.regresion import (ajustar_ols_hac, ajustar_rl_multiple,  # noqa: E402
                              construir_features_horarias, grilla_horaria)
from metrics import naive_ultimo  # noqa: E402

warnings.filterwarnings("ignore")

FP = Path(__file__).resolve().parent.parent      # final_project/
OUT = Path(__file__).resolve().parent            # modelado/

H_TEST = 168                 # hold-out: una semana horaria
VENTANA_H = 90 * 24          # ventana de entrenamiento del ARIMA (acota el grid)
MAXLAGS_HAC = 24             # un ciclo diario

res: dict = {"frecuencia": "horaria", "h_test": H_TEST,
             "ventana_arima_h": VENTANA_H}

# ====================================================== datos horarios ======
mercado = pd.read_csv(FP / "data" / "processed" / "market_hourly.csv",
                      parse_dates=["fecha"])
panel = pd.read_csv(OUT / "panel_diario.csv", parse_dates=["fecha_d"])

datos, cols_x = construir_features_horarias(mercado, panel)
tr, te = datos.iloc[:-H_TEST], datos.iloc[-H_TEST:]
print(f"grilla horaria: {len(datos)} h | train {len(tr)} | test {len(te)} | "
      f"{tr['fecha'].min()} → {te['fecha'].max()}")
print(f"predictores: {len(cols_x)}")

# ==================== regresión lineal múltiple (sklearn) por horizonte ====
# Para cada horizonte h se reconstruyen los predictores con información hasta
# t - h. El naïve es el último valor observado en el origen, y(t - h).
HORIZONTES = (1, 24)
rl, datos_h = {}, {}
for h in HORIZONTES:
    datos, cols_x = construir_features_horarias(mercado, panel, horizonte=h)
    tr, te = datos.iloc[:-H_TEST], datos.iloc[-H_TEST:]
    datos_h[h] = (datos, cols_x, tr, te)
    print(f"h={h:2d} h | filas {len(datos)} | train {len(tr)} | test {len(te)} | "
          f"predictores {len(cols_x)}")
    rl[h] = {}
    for target in ("brecha_pct", "blue_mid"):
        pred, nv = ajustar_rl_multiple(tr, te, cols_x, target)
        y_te = te[target].values
        r, rn = float(np.sqrt(np.mean((y_te - pred.values) ** 2))), \
            float(np.sqrt(np.mean((y_te - nv.values) ** 2)))
        rl[h][target] = {
            "n_test": int(len(te)), "n_train": int(len(tr)),
            "rmse": round(r, 4),
            "mae": round(float(np.mean(np.abs(y_te - pred.values))), 4),
            "rmse_naive": round(rn, 4),
            "mae_naive": round(float(np.mean(np.abs(y_te - nv.values))), 4),
            "ratio": round(r / rn, 3),
        }

# ================================================ OLS + HAC (inferencia) ====
# La inferencia se hace sobre el diseño contemporáneo (h = 1 h).
tr1, cols1 = datos_h[1][2], datos_h[1][1]
fit_hac, cols_ok = ajustar_ols_hac(tr1, cols1, "brecha_pct",
                                   maxlags=MAXLAGS_HAC)
tab = fit_hac.summary2().tables[1]
sig = tab[tab["P>|z|"] < 0.05].sort_values("P>|z|")
res["ols_hac"] = {
    "r2": round(float(fit_hac.rsquared), 4),
    "n": int(fit_hac.nobs),
    "n_train": int(len(tr1)),
    "descartadas": [c for c in cols1 if c not in cols_ok],
    "coeficientes": [
        {"var": k, "coef": round(float(r["Coef."]), 4),
         "ee": round(float(r["Std.Err."]), 4),
         "p": round(float(r["P>|z|"]), 4),
         "lo": round(float(r["[0.025"]), 4),
         "hi": round(float(r["0.975]"]), 4)}
        for k, r in sig.iterrows()
    ],
}

# ========================================= ARIMA horario (univariado) ======
hor = grilla_horaria(mercado)
series = {c: hor[c].dropna() for c in ("blue_mid", "brecha_pct")}

# (a) selección del orden por AIC, con la ventana de entrenamiento de cada
#     horizonte (el origen inicial se desplaza h horas)
arima, sel_h = {}, {}
for nombre, y in series.items():
    ventana = y.iloc[-(VENTANA_H + H_TEST):]
    arima[nombre] = {k: v for k, v in
                     arima_holdout_detalle(ventana, n_test=H_TEST, max_pq=2).items()
                     if k in ("criterios", "parametros", "adf_p_nivel",
                              "adf_p_diff")}
    split = y.index[-(H_TEST + 1)]
    sel_h[nombre] = {h: seleccionar_por_horizonte(y, split, h,
                                                  ventana_h=VENTANA_H, max_pq=2)
                     for h in HORIZONTES}

# (b) evaluación walk-forward en los dos horizontes, con el orden elegido
#     para cada uno
arima_h = {}
for h in HORIZONTES:
    arima_h[h] = {}
    for nombre, y in series.items():
        split = y.index[-(H_TEST + 1)]
        sel = sel_h[nombre][h]
        pred, nv, lo, hi = arima_walkforward(y, split, h, sel["order"],
                                             trend=sel["trend"])
        real = y.loc[pred.index]
        e, en = real - pred, real - nv
        r, rn = float(np.sqrt((e ** 2).mean())), float(np.sqrt((en ** 2).mean()))
        arima_h[h][nombre] = {
            "modelo": sel["modelo"], "aic": sel["aic"],
            "n_train": sel["n_train"], "order": list(sel["order"]),
            "n_test": int(len(pred)), "horizonte": h,
            "rmse": round(r, 4),
            "mae": round(float(e.abs().mean()), 4),
            "rmse_naive": round(rn, 4),
            "mae_naive": round(float(en.abs().mean()), 4),
            "ratio": round(r / rn, 3),
            "cobertura": round(float(((real >= lo) & (real <= hi)).mean()), 3),
        }

res["horizontes"] = list(HORIZONTES)
res["por_horizonte"] = {str(h): {"rl": rl[h], "arima": arima_h[h]}
                        for h in HORIZONTES}
res["arima"] = arima
res["arima_seleccion"] = {k: {str(h): {kk: (str(vv) if kk == "origen0" else vv)
                                       for kk, vv in v.items()}
                              for h, v in d.items()} for k, d in sel_h.items()}

# ======================================= estudio de eventos (5h/1d/3d/5d) ====
mkt = mercado[["fecha", "blue_mid"]].dropna().sort_values("fecha") \
    .reset_index(drop=True)
docs = pd.read_csv(FP / "data" / "processed" / "gaceta_keywords.csv")
tags = pd.read_csv(FP / "data" / "processed" / "gaceta_keywords_tags.csv")

docs_f = docs[["norma_id", "fecha_edicion"]].copy()
docs_f["fecha"] = pd.to_datetime(docs_f["fecha_edicion"], errors="coerce")
eventos = (docs_f.dropna(subset=["fecha"])[["norma_id", "fecha"]]
           .merge(tags[["norma_id", "tag"]], on="norma_id")[["fecha", "tag"]]
           .drop_duplicates())

HOR = {"5 h": 5, "1 d": 24, "3 d": 72, "5 d": 120}


def ancla(f):
    dia = mkt[mkt.fecha.dt.date == f.date()]
    if dia.empty:
        return None
    i0 = (dia.fecha.dt.hour - 12).abs().idxmin()
    return mkt.at[i0, "fecha"], mkt.at[i0, "blue_mid"]


def ret(t0, p0, h):
    i1 = (mkt.fecha - (t0 + pd.Timedelta(hours=h))).abs().idxmin()
    p1 = mkt.at[i1, "blue_mid"]
    return (p1 - p0) / p0 * 100 if p0 and not pd.isna(p1) else np.nan


fechas = pd.DataFrame({"fecha": eventos.fecha.unique()})
for h, hv in HOR.items():
    fechas[h] = [np.nan if (a := ancla(f)) is None else ret(a[0], a[1], hv)
                 for f in fechas.fecha]

ev = eventos.merge(fechas, on="fecha")
largo = ev.melt(id_vars=["fecha", "tag"], value_vars=list(HOR),
                var_name="hor", value_name="r").dropna()
g = largo.groupby(["tag", "hor"])["r"].agg(n="count", media="mean",
                                           desv="std").reset_index()
g["ic95"] = 1.96 * g["desv"].fillna(0) / np.sqrt(g["n"])
piv = g.pivot(index="tag", columns="hor", values="media").reindex(columns=list(HOR))
res["eventos"] = {t: {h: {"media": round(float(piv.loc[t, h]), 3),
                          "n": int(g[(g.tag == t) & (g.hor == h)]["n"].iloc[0]),
                          "ic95": round(float(g[(g.tag == t) & (g.hor == h)]["ic95"].iloc[0]), 3)}
                      for h in HOR}
                  for t in piv.index if pd.notna(piv.loc[t]).any()}
res["eventos_meta"] = {
    "pares_totales": int(len(eventos)),
    "pares_con_datos": int(ev[list(HOR)].notna().any(axis=1).sum()),
    "grilla_desde": str(mkt.fecha.min().date()),
    "grilla_hasta": str(mkt.fecha.max().date()),
    "grilla_dias": int(mkt.fecha.dt.date.nunique()),
}

# ================================================== Spearman (diario) =======
# Los tags se publican por día: esta correlación se mide a escala diaria.
serie_d = pd.read_csv(OUT / "serie_diaria_blue.csv", parse_dates=["fecha"],
                      index_col="fecha")
tag_dia = (docs_f.dropna(subset=["fecha"])
           .merge(tags[["norma_id", "tag"]], on="norma_id")
           .groupby(["fecha", "tag"]).size().unstack(fill_value=0).sort_index())
tag_dia = tag_dia.reindex(pd.date_range(tag_dia.index.min(), tag_dia.index.max(),
                                        freq="D"), fill_value=0)
ret_b = serie_d["ret_blue_pct"].dropna()
base = pd.DataFrame(index=ret_b.index.intersection(tag_dia.index))
base["ret_t"] = ret_b.reindex(base.index)
base["ret_t1"] = ret_b.shift(-1).reindex(base.index)
res["spearman"] = {
    t: {"n": int((tag_dia[t].reindex(base.index) > 0).sum()),
        "t": round(float(tag_dia[t].reindex(base.index).corr(base["ret_t"],
                                                            method="spearman")), 3),
        "t1": round(float(tag_dia[t].reindex(base.index).corr(base["ret_t1"],
                                                             method="spearman")), 3)}
    for t in tag_dia.columns
}

# ========================= matriz de costos (dirección a 1 paso) ===========
# Provenance: la matriz 2x2 sale del notebook de análisis (confusión de
# dirección, ARIMA a un paso sobre blue_mid, modo cambio_esperado).
# Fuente primaria: el notebook de resultados (independiente). Si no existe,
# se usa el CSV del notebook de análisis como respaldo.
_dir_candidatos = [OUT / "outputs" / "informe_direccion.csv",
                   OUT / "outputs" / "nb_confusion_direccion.csv"]
_dir_csv = next((c for c in _dir_candidatos if c.exists()), None)
res["matriz_costos"] = None
if _dir_csv is not None:
    _d = pd.read_csv(_dir_csv)
    if "horizonte" not in _d:                 # compatibilidad con CSV antiguos
        _d["horizonte"] = 1
    base = _d[(_d["target"] == "blue_mid") & (_d["modo"] == "cambio_esperado")]
    res["matriz_costos"] = {}
    for h, g in base.groupby("horizonte"):
        fila = g[g["modelo"].str.startswith("ARIMA")]
        if not len(fila):
            continue
        f = fila.iloc[0]
        tp, fn, fp, tn = (int(f["tp"]), int(f["fn"]),
                          int(f["fp"]), int(f["tn"]))
        res["matriz_costos"][str(int(h))] = {
            "modelo": str(f["modelo"]),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n": int(f["n"]), "accuracy": float(f["accuracy"]),
            "subas": tp + fn, "bajadas": fp + tn,
            "umbral": round(tn / fn, 3) if fn else None,
            "fuente": _dir_csv.name,
        }

# ================================================================= salida ====
(OUT / "metricas_reporte.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False))


def _h(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


print("=== ARIMA: orden elegido por horizonte (AIC) ===")
for k, d in res["arima_seleccion"].items():
    for h, v in d.items():
        print(f"  {k:11s} h={h:>2} h | {v['modelo']:14s} AIC {v['aic']:>9} | "
              f"n_train={v['n_train']} | origen {v['origen0'][:16]}")
print("=== métricas por horizonte (walk-forward, naïve del mismo horizonte) ===")
for h in res["horizontes"]:
    print(f"  --- {h} h ---")
    for modelo in ("arima", "rl"):
        for k, v in res["por_horizonte"][str(h)][modelo].items():
            extra = (f" | cobertura IC95 {v['cobertura']:.3f}"
                     if "cobertura" in v else "")
            print(f"    {modelo:5s} {k:11s} RMSE {v['rmse']:.4f} vs naïve "
                  f"{v['rmse_naive']:.4f} (ratio {v['ratio']:.2f}){extra}")
print(f"=== OLS+HAC (n={res['ols_hac']['n']}, R2={res['ols_hac']['r2']}) ===")
for c in res["ols_hac"]["coeficientes"]:
    print(f"  {c['var']:18s} β={c['coef']:+.4f}  p={c['p']:.4f}")
print("=== matriz de costos (dirección, ARIMA 1 paso, blue) ===")
mc = res.get("matriz_costos")
if mc:
    for h, v in mc.items():
        print(f"  h={h:>2} {v['modelo']:16s} TP {v['tp']:3d} | FN {v['fn']:3d} | "
              f"FP {v['fp']:3d} | TN {v['tn']:3d} | n={v['n']} | "
              f"accuracy {v['accuracy']:.3f} | umbral C_FN/C_FP < {v['umbral']}")
else:
    print("  (sin CSV de dirección: correr el notebook primero)")
print("OK -> metricas_reporte.json")
