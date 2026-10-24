"""Introducción."""
from __future__ import annotations

import streamlit as st

from core import ui

ui.encabezado("01_introduccion")

ui.seccion("Motivación")
st.markdown(
    """
Desde 2023 Bolivia atraviesa una escasez de divisas que separó de forma
persistente el tipo de cambio oficial, administrado por el Banco Central de
Bolivia (BCB), del precio que el dólar alcanza en el mercado paralelo
(*dólar blue*). Tras el régimen flexible de 2025–2026, el oficial quedó
unificado mientras el paralelo cotiza libremente vía USDT en plataformas P2P
(Binance, Bybit, Bitget) y fintechs (Airtm, Takenos, El Dorado, Wallbit).

Esa **brecha** es el principal termómetro de escasez de divisas y de presión
cambiaria del país: cuando se amplía, el mercado descuenta más restricción;
cuando se cierra, la presión cede.
"""
)

ui.seccion("Alcance", "Un pipeline reproducible, no un sistema en producción.")
st.markdown(
    """
- Recolecta y normaliza cotizaciones de la API pública de
  [dolarbluebolivia.click](https://www.dolarbluebolivia.click) (37 endpoints)
  y de la Gaceta Oficial de Bolivia.
- Mantiene series de tiempo alineadas blue/oficial (compra/venta).
- Produce análisis exploratorio reproducible.
- Ajusta modelos de **previsión** del nivel del paralelo y de la brecha
  (regresión lineal, OLS con errores HAC y ARIMA), evaluados con partición
  temporal y línea base naïve a **1 hora** y **1 día** (Objetivo 1).
- Estudia la **relación** entre las publicaciones de la Gaceta y la brecha en
  ventanas de ±5 días (Objetivo 2).
"""
)

ui.lectura(
    "El despliegue y el monitoreo (fase 6 de CRISP-DM) quedan como trabajo "
    "pendiente. Este dashboard es la vista de resultados del informe, no un "
    "servicio de cotizaciones.",
)

ui.seccion("Estructura del documento")
filas = [
    ("1", "Introducción", "Motivación, alcance y estructura.", "views/01_introduccion.py"),
    ("2", "Comprensión del negocio", "Objetivos 1–2, criterios de éxito, recursos.",
     "views/02_negocio.py"),
    ("3", "Comprensión y obtención de datos", "Fuentes, pipeline, diccionario, calidad y EDA.",
     "views/03_datos.py"),
    ("4", "Preparación de datos", "Remuestreo, features sin fuga, split y naïve.",
     "views/04_preparacion.py"),
    ("5", "Modelado", "RL múltiple, OLS-HAC, ARIMA, comparativa y Gaceta.",
     "views/05_modelado.py"),
    ("6", "Evaluación de negocio", "Ahorro, señal direccional y costos.",
     "views/06_evaluacion.py"),
    ("7", "Conclusiones", "Hallazgos y trabajo futuro.", "views/07_conclusiones.py"),
]
for fase, titulo, desc, ruta in filas:
    a, b, c = st.columns([0.7, 2.2, 5.2])
    with a:
        st.markdown(f"**{fase}**")
    with b:
        st.page_link(ruta, label=titulo)
    with c:
        st.caption(desc)

ui.lectura(
    "La lectura recomendada es lineal: negocio → datos → preparación → modelado "
    "→ evaluación. Cada página enlaza su sección LaTeX de origen.",
)

ui.pie("views/01_introduccion.py")
