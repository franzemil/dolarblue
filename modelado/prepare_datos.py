"""Fase 1 — Preparación de datos (final_project): panel diario mercado + Gaceta.

Los nombres técnicos de las columnas se mantienen como en el pipeline original
(`blue_sell`, `brecha_pct`, `blue_mid`, ...). Para la traducción al español ver
`diccionario_datos.py` (etiquetas, descripciones y unidades).

Entradas:
  - data/history_full.csv    (34k snapshots ~1 año: official/blue buy-sell)
  - data/processed/market_hourly.csv   (grilla horaria 6 meses)
  - data/processed/gaceta_keywords.csv (509 normas con tags por palabras clave)

Salidas:
  - modelado/serie_diaria_blue.csv   (1 año: blue_sell, oficial_sell, brecha_pct)
  - modelado/panel_diario.csv        (179 días: retornos + features de Gaceta)
"""
from pathlib import Path

import numpy as np
import pandas as pd

FP = Path(__file__).resolve().parent.parent   # final_project/
OUT = Path(__file__).resolve().parent         # final_project/modelado/

# =============================================== A) serie diaria (1 año) ====
h = pd.read_csv(FP / "data" / "history_full.csv")
h["fecha"] = pd.to_datetime(h["timestamp"], unit="s")
h = h.set_index("fecha")

# los 0.0 son gaps de captura, no precios
n_ceros = int((h["blue_sell"] == 0).sum())
h = h[(h["blue_sell"] > 0) & (h["official_sell"] > 0)]
print(f"history_full: {len(h)} snapshots válidos ({n_ceros} con blue=0 eliminados)")
print(f"rango: {h.index.min().date()} -> {h.index.max().date()}")

diaria = h.resample("D").agg(
    blue_sell=("blue_sell", "mean"),
    oficial_sell=("official_sell", "mean"),
    n_snapshots=("blue_sell", "count"),
)
diaria = diaria[diaria["n_snapshots"] > 0]
diaria["brecha_pct"] = (diaria["blue_sell"] / diaria["oficial_sell"] - 1) * 100
diaria["ret_blue_pct"] = diaria["blue_sell"].pct_change() * 100
print(f"\nserie diaria: {len(diaria)} días")
print(diaria[["blue_sell", "oficial_sell", "brecha_pct"]].describe().round(3))

# el salto del oficial (6.96 -> ~10.7) es la devaluación: mostrar cuándo
salto = diaria["oficial_sell"].diff() > 1.0
if salto.any():
    d = diaria.index[salto][0].date()
    print(f"\n!! salto del tipo de cambio oficial detectado el {d} "
          f"(devaluación): {diaria['oficial_sell'].shift(1)[salto].iloc[0]:.2f} -> "
          f"{diaria['oficial_sell'][salto].iloc[0]:.2f}")

diaria.to_csv(OUT / "serie_diaria_blue.csv", index_label="fecha")

# ================================== B) panel diario mercado + Gaceta (181d) =
mkt = pd.read_csv(FP / "data" / "processed" / "market_hourly.csv", parse_dates=["fecha"])
mkt["fecha_d"] = mkt["fecha"].dt.tz_localize(None).dt.normalize()
panel = mkt.groupby("fecha_d").agg(
    blue_mid=("blue_mid", "mean"),
    oficial_mid=("oficial_mid", "mean"),
    brecha_pct=("brecha_pct", "mean"),
    vol_media_24h=("vol_24h", "mean"),
).reset_index()
panel["ret_blue_pct"] = panel["blue_mid"].pct_change() * 100
panel["d_oficial_pct"] = panel["oficial_mid"].pct_change() * 100

# ================== canales de la Gaceta: LLM (primario) + keywords ==========
# 1) LLM: extracción estructurada por norma (169 normas, 2026-06-18 -> 10-20)
llm = pd.read_csv(FP / "data" / "processed" / "gaceta_llm.csv")
llm = llm[llm["status"] == "ok"].copy()
llm["fecha"] = pd.to_datetime(llm["fecha_publicacion"], errors="coerce")
llm = llm.dropna(subset=["fecha"])
tags_llm = llm["tags"].fillna("")

llm_d = pd.DataFrame({
    "n_docs_total_llm": 1,
    "n_docs_fx": llm["relevancia_fx"].isin(["media", "alta"]).astype(int),
    "n_docs_fx_alta": (llm["relevancia_fx"] == "alta").astype(int),
    "n_docs_deuda": tags_llm.str.contains("deuda_externa").astype(int),
    "n_docs_comercio": tags_llm.str.contains("comercio_exterior").astype(int),
    "n_docs_mineria": tags_llm.str.contains("mineria_energia").astype(int),
}).groupby(llm["fecha"].dt.normalize()).sum()
print(f"\nGaceta LLM: {len(llm)} normas en {len(llm_d)} días "
      f"({llm.fecha.min().date()} -> {llm.fecha.max().date()})")

# 2) keywords: respaldo para los días sin cobertura LLM
gac = pd.read_csv(FP / "data" / "processed" / "gaceta_keywords.csv")
gac["fecha"] = pd.to_datetime(gac["fecha_edicion"], errors="coerce")
gac = gac.dropna(subset=["fecha"])
gac["fx_alta"] = (gac["relevancia_fx"] == "alta").astype(int)
gac["fx_media_alta"] = gac["relevancia_fx"].isin(["media", "alta"]).astype(int)
kw_d = gac.groupby(gac["fecha"].dt.normalize()).agg(
    n_docs_total_llm=("norma_id", "count"),
    n_docs_fx=("fx_media_alta", "sum"),
    n_docs_fx_alta=("fx_alta", "sum"),
    n_docs_deuda=("primary_tag", lambda s: (s == "deuda_externa").sum()),
    n_docs_comercio=("primary_tag", lambda s: (s == "comercio_exterior").sum()),
    n_docs_mineria=("primary_tag", lambda s: (s == "mineria_energia").sum()),
)
print(f"Gaceta keywords: {len(gac)} normas en {len(kw_d)} días "
      f"({gac.fecha.min().date()} -> {gac.fecha.max().date()})")

# 3) unión: LLM donde existe, keywords en el resto
gac_d = llm_d.combine_first(kw_d).rename_axis("fecha_d")
gac_d["n_docs_total"] = gac_d.pop("n_docs_total_llm")

panel = panel.merge(gac_d, left_on="fecha_d", right_index=True, how="left")
cols_gac = [c for c in gac_d.columns]
panel[cols_gac] = panel[cols_gac].fillna(0)

# rezago de la variable dependiente y del shock de Gaceta (publicación T afecta T+1)
panel["ret_blue_lag1"] = panel["ret_blue_pct"].shift(1)
panel["n_docs_fx_lag1"] = panel["n_docs_fx"].shift(1)
panel["n_docs_deuda_lag1"] = panel["n_docs_deuda"].shift(1)
panel["dia_sem"] = panel["fecha_d"].dt.dayofweek

panel = panel.dropna(subset=["ret_blue_pct", "ret_blue_lag1", "d_oficial_pct"])
panel.to_csv(OUT / "panel_diario.csv", index=False)
print(f"\npanel diario mercado+gaceta: {len(panel)} días "
      f"({panel['fecha_d'].min().date()} -> {panel['fecha_d'].max().date()})")
print(panel[["ret_blue_pct", "d_oficial_pct", "n_docs_fx", "n_docs_fx_alta"]]
      .describe().round(3))
print(f"días con algún doc FX: {(panel['n_docs_fx'] > 0).sum()} de {len(panel)}")
print("\nOK -> serie_diaria_blue.csv, panel_diario.csv")
