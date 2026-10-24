"""Comprensión y obtención de datos (incluye EDA)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as C
from core import data, figuras as G, formato as F, ui

ui.encabezado("03_datos")
cob = data.cobertura()
hor = data.grilla()
mercado = data.mercado()
panel = data.panel()

ui.seccion("3.1 · Recolección de datos")
st.markdown(
    """
Un colector consulta el endpoint `GET /v1/chart/6m` de la API y guarda cada
respuesta como snapshot JSON en `data/raw/`. Una etapa de normalización elimina
el envoltorio de la API y valida cada punto contra un esquema tipado; la
indexación deriva los puntos medios, el spread, la brecha porcentual, el retorno
horario y la volatilidad rodante de 24 h.
"""
)
st.image(str(C.OUTPUTS / "informe_flujo_datos.png"),
         caption="Obtención de datos: las dos fuentes convergen en los "
                 "procesadores.", width="stretch")
st.image(str(C.OUTPUTS / "informe_recoleccion.png"),
         caption="Del snapshot crudo a los predictores del modelo.",
         width="stretch")

ui.seccion("3.2 · Fuentes y cobertura")
tabla_cob = pd.DataFrame([
    {"Fuente": "Mercado (crudo)", "Filas": F.miles(cob["mercado_filas"]),
     "Desde": F.fecha(cob["mercado_desde"]),
     "Hasta": F.fecha(cob["mercado_hasta"]),
     "Uso": "serie horaria de precios"},
    {"Fuente": "Grilla horaria (EDA)", "Filas": F.miles(len(hor)),
     "Desde": F.fecha(hor.index.min()), "Hasta": F.fecha(hor.index.max()),
     "Uso": "base de todos los modelos"},
    {"Fuente": "Panel diario (Gaceta)", "Filas": F.miles(cob["panel_filas"]),
     "Desde": F.fecha(cob["panel_desde"]), "Hasta": F.fecha(cob["panel_hasta"]),
     "Uso": "canales de la Gaceta"},
    {"Fuente": "Gaceta etiquetada (LLM)", "Filas": F.miles(cob["gaceta_filas"]),
     "Desde": F.fecha(cob["gaceta_desde"]),
     "Hasta": F.fecha(cob["gaceta_hasta"]),
     "Uso": "normas y tags"},
])
ui.tabla(tabla_cob, "Cobertura de las cuatro fuentes del proyecto.",
         "notebook · 2.2", height=180)

dias_fx = int((panel.n_docs_fx > 0).sum())
ui.kpis([
    {"label": "Paso de la grilla", "value": "1 h",
     "nota": "frecuencia horaria regular"},
    {"label": "Días con norma FX", "value": f"{dias_fx} de {len(panel)}",
     "nota": "panel de la Gaceta"},
    {"label": "Normas etiquetadas", "value": F.miles(cob["gaceta_filas"]),
     "nota": "extracción con LLM"},
    {"label": "Quiebre estructural", "value": "27/06/2026",
     "nota": "devaluación del oficial (+41 %)", "tono": "neutral"},
])

ui.seccion("3.3 · Diccionario de datos")
glosario = data.glosario()
busqueda = st.text_input(
    "Buscar variable, etiqueta o descripción",
    placeholder="p. ej. brecha, volumen, blue…",
    label_visibility="collapsed",
)
vista = glosario
if busqueda:
    m = (glosario["Variable"].str.contains(busqueda, case=False, na=False)
         | glosario["Etiqueta"].str.contains(busqueda, case=False, na=False)
         | glosario["Descripción"].str.contains(busqueda, case=False, na=False))
    vista = glosario[m]
st.caption(f"{len(vista)} de {len(glosario)} campos del diccionario.")
vista_dicc = vista[["Etiqueta", "Variable", "Descripción", "Unidad"]]
ui.tabla(
    vista_dicc,
    "Diccionario único de traducción: el código conserva los nombres técnicos "
    "y la app muestra la etiqueta en español.",
    "modelado/diccionario_datos.py", height=380,
    column_config={
        "Etiqueta": st.column_config.TextColumn(width="medium"),
        "Variable": st.column_config.TextColumn(width="small"),
        "Descripción": st.column_config.TextColumn(width="large"),
        "Unidad": st.column_config.TextColumn(width="small"),
    },
)

ui.seccion("3.4 · Calidad de datos")
q1, q2, q3 = st.columns(3, gap="medium")
with q1:
    nucleo = ["blue_mid", "oficial_mid", "brecha_pct"]
    nulos = int(mercado[nucleo].isna().sum().sum())
    st.metric("Nulos en precios", nulos,
              help="Blue, oficial y brecha. vol_24h tiene 24 nulos de "
                   "calentamiento y las etiquetas fwd_* nulos por diseño.")
with q2:
    dup = int(mercado.fecha.duplicated().sum())
    st.metric("Fechas duplicadas", dup)
with q3:
    huerfanas = int((hor.index.to_series().diff() != pd.Timedelta(hours=1))
                    .iloc[1:].sum())
    st.metric("Saltos > 1 h en la grilla", huerfanas,
              help="Tras el remuestreo, la grilla es regular por construcción.")

ofi = (mercado.set_index("fecha")["oficial_mid"]
       .resample("D").last().dropna())
cambio = ofi.pct_change().abs()
fecha_quiebre = cambio.idxmax()
st.markdown(
    f"El único quiebre estructural es la devaluación del oficial del "
    f"**{F.fecha(fecha_quiebre)}**: el tipo de cambio pasó de "
    f"{F.num(ofi.shift(1).loc[fecha_quiebre], 2)} a "
    f"{F.num(ofi.loc[fecha_quiebre], 2)} Bs/USD "
    f"({F.pct(cambio.loc[fecha_quiebre], 1, decimal=False)}). La ventana de "
    "prueba cae íntegra después del quiebre, así que las métricas miden el "
    "régimen vigente."
)

ui.seccion("3.5 · Exploración (EDA)")
ui.figura(
    G.crudo_vs_eda(data.crudo(), hor),
    "Snapshots crudos (puntos grises) frente a la grilla horaria regular "
    "(línea), en la última semana de la muestra.",
    "fig:crudo_vs_eda",
)
ui.figura(
    G.serie_nivel(hor, height=340),
    "Nivel del blue y del oficial en frecuencia horaria. La línea punteada "
    "marca la devaluación del 27/06/2026.",
    "fig:serie_blue_oficial",
)
ui.figura(
    G.serie_brecha(hor, height=300),
    "Brecha cambiaria (blue / oficial − 1). Es la variable objetivo del "
    "Objetivo 1 junto con el nivel del paralelo.",
    "fig:brecha",
)

desc = hor["brecha_pct"].describe().to_frame("brecha_pct").T
desc.columns = ["n", "Media", "Desv. estándar", "Mín.", "P25", "Mediana",
                "P75", "Máx."]
ui.tabla(desc, "Estadísticos descriptivos de la brecha en la serie horaria.",
         "notebook · 2.2", formato={c: 2 for c in desc.columns[1:]})

ui.lectura(
    "La serie horaria cubre 6 meses y 4.318 horas efectivas. El quiebre del "
    "oficial divide la muestra en dos regímenes; todas las comparaciones se "
    "hacen dentro del régimen vigente.",
)

ui.pie("views/03_datos.py")
