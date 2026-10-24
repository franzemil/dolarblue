"""Resumen ejecutivo."""
from __future__ import annotations

import streamlit as st

from core import config as C
from core import data, figuras as G, formato as F, ui

ui.encabezado("00_resumen")
R = data.metricas()
cob = data.cobertura()
hor = data.grilla()

ui.kpis([
    {"label": "Serie horaria", "value": F.miles(cob["mercado_filas"]),
     "nota": f'{F.fecha(cob["mercado_desde"])} → {F.fecha(cob["mercado_hasta"])}'},
    {"label": "Panel Gaceta", "value": f'{cob["panel_filas"]} días',
     "nota": f'{cob["gaceta_filas"]} normas etiquetadas por LLM'},
    {"label": "Ventana de prueba", "value": f"{R['h_test']} h",
     "nota": "última semana, mismo corte para todos"},
    {"label": "Mejor RMSE ÷ naïve",
     "value": F.num(R["por_horizonte"]["1"]["arima"]["blue_mid"]["ratio"], 2),
     "nota": "ARIMA del nivel a 1 h", "tono": "good",
     "delta": "▼ 4,5 % frente al naïve"},
])

st.write("")
ui.figura(
    G.serie_nivel(hor, height=340),
    "Nivel del dólar paralelo (blue) y del oficial del BCB en frecuencia "
    "horaria. La devaluación del oficial del 27/06/2026 marca el único quiebre "
    "estructural de la muestra.",
    "fig:serie_blue_oficial",
)

ui.seccion("Tres hallazgos",
           "Lo que sostiene el informe, en una frase cada uno.")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("**1 · El ARIMA es el único que no pierde**")
    st.caption(
        "Iguala al naïve a un día (0,99–1,00) y mejora 4,5 % el RMSE del nivel "
        "a una hora. La regresión múltiple pierde en los cuatro casos."
    )
with c2:
    st.markdown("**2 · El error de nivel no genera ahorro**")
    st.caption(
        "El MAE del ARIMA es idéntico al del naïve: 0,0050 Bs/USD a 1 h y "
        "0,0277 a 24 h. La precisión extra no se traduce en dinero."
    )
with c3:
    st.markdown("**3 · La dirección solo sirve a 1 hora**")
    st.caption(
        "Accuracy 0,571 a una hora, con umbral TN/FN 1,21. A un día cae "
        "a 0,485: por debajo del azar."
    )

ui.lectura(
    "El valor del modelo no está en acertar más —el tipo de cambio horario se "
    "comporta como un paseo aleatorio— sino en la <b>banda de "
    "incertidumbre</b> y en una señal direccional útil únicamente a una hora.",
)

ui.seccion("Recorrido del informe")
st.markdown(
    "El menú lateral replica las fases de CRISP-DM del informe. "
    "Cada modelo tiene su propia pestaña dentro de **Modelado**."
)
a, b, c = st.columns(3)
with a:
    st.page_link("views/03_datos.py", label="Datos y EDA",
                 icon=":material/database:")
    st.page_link("views/04_preparacion.py", label="Preparación",
                 icon=":material/build:")
with b:
    st.page_link("views/05_modelado.py", label="Modelado",
                 icon=":material/model_training:")
    st.page_link("views/06_evaluacion.py", label="Evaluación de negocio",
                 icon=":material/insights:")
with c:
    st.page_link("views/10_metricas.py", label="Métricas",
                 icon=":material/functions:")
    st.page_link("views/07_conclusiones.py", label="Conclusiones",
                 icon=":material/flag:")
    st.page_link("views/09_reproducibilidad.py", label="Reproducibilidad",
                 icon=":material/settings:")

ui.pie("views/00_resumen.py")
