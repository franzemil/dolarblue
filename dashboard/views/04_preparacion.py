"""Preparación de datos."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as C
from core import data, formato as F, ui

ui.encabezado("04_preparacion")
R = data.metricas()
hor = data.grilla()

ui.seccion("4.1 · Del JSON crudo a los predictores")
pasos = pd.DataFrame([
    {"Etapa": "1 · Recolección",
     "Qué ocurre": "El colector consulta GET /v1/chart/6m y guarda snapshots "
                   "JSON en data/raw/."},
    {"Etapa": "2 · Normalización",
     "Qué ocurre": "Se elimina el envoltorio de la API y se valida cada punto "
                   "contra un esquema tipado: t, bb, bs, ob, os."},
    {"Etapa": "3 · Características",
     "Qué ocurre": "Puntos medios, spread, brecha, retorno horario y "
                   "volatilidad rodante de 24 h → market_hourly.csv."},
    {"Etapa": "4 · Predictores",
     "Qué ocurre": "Rezagos, canales de la Gaceta y dummies de hora/día, sin "
                   "fuga de información."},
])
ui.tabla(pasos, "El camino desde la API hasta la tabla de modelado.",
         "05_preparacion.tex · 4.1", height=220)
st.image(str(C.OUTPUTS / "informe_snapshot.png"),
         caption="Del snapshot a la serie de modelado: remuestreo horario y "
                 "agregación diaria de la Gaceta.", width="stretch")

ui.seccion("4.2 · Remuestreo a frecuencia fija")
ui.kpis([
    {"label": "Filas de la grilla", "value": F.miles(len(hor)),
     "nota": f'{F.fecha_hora(hor.index.min())} → {F.fecha_hora(hor.index.max())}'},
    {"label": "Paso", "value": "1 h",
     "nota": "se conserva el último precio; huecos ≤ 3 h interpolados"},
    {"label": "Columnas del mercado", "value": "17",
     "nota": "incluye retornos futuros (etiquetas), no predictores"},
    {"label": "Cobertura", "value": "6 meses",
     "nota": "serie horaria alineada blue/oficial"},
])
muestra = hor[["blue_mid", "oficial_mid", "brecha_pct"]].head(6)
ui.tabla(muestra.reset_index().rename(columns={"fecha": "Fecha"}),
         "Primeras filas de la serie horaria lista para modelar.",
         "tab:remuestreo",
         formato={"blue_mid": 2, "oficial_mid": 2, "brecha_pct": 2})

ui.seccion("4.3 · Variable objetivo y conjunto de eventos")
st.markdown(
    """
El problema se formula como **previsión supervisada** a dos horizontes: el
nivel del paralelo (o la brecha) **a una hora y a un día**, con intervalos que
cuantifiquen la incertidumbre. Para cada horizonte $h$ los predictores usan solo
información disponible en $t-h$, y la línea base naïve repite el último valor
observado en ese origen.

Para el **Objetivo 2** se cruza la serie horaria con las fechas de edición de la
Gaceta: por cada publicación se toma el precio ancla (12:00 UTC) y se miden
retornos a 5 h, 1, 3 y 5 días contra la variación media de todos los días de la
serie (línea base incondicional).
"""
)

ui.seccion("4.4 · Partición temporal y línea base")
filas = []
for h in C.HORIZONTES:
    for target in C.TARGETS:
        v = R["por_horizonte"][str(h)]["rl"][target]
        filas.append({
            "Nivel": "Blue (Bs/USD)" if target == "blue_mid" else "Brecha (pp)",
            "Horizonte": f"{h} h",
            "n train": F.miles(v["n_train"]), "n test": F.miles(v["n_test"]),
            "Naïve RMSE": v["rmse_naive"], "Naïve MAE": v["mae_naive"],
        })
ui.tabla(pd.DataFrame(filas),
         "Partición temporal fija: las últimas 168 h son el test; el naïve "
         "repite el último valor observado en el origen.",
         "tab:remuestreo · 4.4",
         formato={"Naïve RMSE": 4, "Naïve MAE": 4})

st.markdown(
    """
- **Sin fuga de información:** para $h=1$ el predictor más fresco es $t-1$;
  para $h=24$ el bloque completo de rezagos se desplaza 23 h adicionales, de
  modo que el predictor más fresco es $t-24$.
- **Un solo corte:** la misma ventana de 168 h para todos los modelos, targets
  y horizontes.
- **El naïve es el rival real:** en tipos de cambio, el paseo aleatorio es
  notoriamente difícil de batir.
"""
)

ui.seccion("Predictores por horizonte")
horizonte = st.segmented_control(
    "Horizonte para inspeccionar features", options=list(C.HORIZONTES),
    format_func=lambda h: f"{h} h", default=1, key="prep_horizonte",
)
datos, cols = data.features(int(horizonte))
train, test = datos.iloc[:-R["h_test"]], datos.iloc[-R["h_test"]:]

rezagos = [c for c in cols if "lag" in c]
gaceta = [c for c in cols if c.startswith("n_docs")]
dummies = [c for c in cols if c.startswith(("hora_", "dow_"))]
ui.kpis([
    {"label": "Predictores", "value": len(cols), "nota": "sin fuga"},
    {"label": "Train", "value": F.miles(len(train)), "nota": "horas"},
    {"label": "Test", "value": len(test),
     "nota": f'{F.fecha_hora(test.fecha.min())} → {F.fecha_hora(test.fecha.max())}'},
    {"label": "Rezagos / Gaceta / Dummies",
     "value": f"{len(rezagos)} / {len(gaceta)} / {len(dummies)}",
     "nota": "composición del diseño"},
])
with st.expander(f"Ver los {len(cols)} predictores"):
    st.markdown(f"**Rezagos y nivel:** {', '.join(f'`{c}`' for c in rezagos)}")
    st.markdown(f"**Gaceta:** {', '.join(f'`{c}`' for c in gaceta)}")
    st.markdown(f"**Dummies:** {len(dummies)} columnas de hora (23) y día de "
                "semana (6).")

ui.tabla(
    train[["fecha", "blue_mid", "brecha_pct", "origen_blue",
           "origen_brecha"] + rezagos[:3]].head(5),
    "Insumos del modelo: el objetivo de cada fila es la hora t; el origen "
    "contiene el naïve del mismo horizonte.",
    "notebook · 4.1",
    formato={"blue_mid": 2, "brecha_pct": 2, "origen_blue": 2,
             "origen_brecha": 2, "brecha_lag1h": 2, "brecha_lag3h": 2,
             "brecha_lag24h": 2},
)

ui.lectura(
    "La partición respeta el orden temporal, nunca una división aleatoria. "
    "Cada decisión de preparación se toma pensando en evitar fuga: el modelo "
    "solo ve lo que existía en el origen del pronóstico.",
)

ui.pie("views/04_preparacion.py")
