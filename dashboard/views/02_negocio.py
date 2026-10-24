"""Comprensión del negocio."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import ui

ui.encabezado("02_negocio")

ui.seccion("Dos sistemas de precios, un problema de previsión")
st.markdown(
    """
En Bolivia conviven **dos sistemas de precios del dólar** que no siempre
coinciden: el **oficial**, administrado por el BCB y válido para operaciones
formales, y el **paralelo (blue)**, que se forma libremente en plataformas P2P
de criptomonedas y fintechs, cotizado en USDT.

La **brecha** entre ambos es el termómetro de escasez de divisas. El problema
es de previsión: importadores, empresas y personas no pueden anticipar el nivel
futuro del paralelo y, por lo tanto, no pueden planificar tesorería ni cobertura.
"""
)

c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**Problema 1 · Falta de anticipación**")
    st.caption(
        "No existen estimaciones del nivel del paralelo ni de la brecha a 1 hora "
        "y 1 día. Predecir el tipo de cambio fuera de muestra es difícil: los "
        "modelos estructurales suelen no superar al paseo aleatorio."
    )
with c2:
    st.markdown("**Problema 2 · Relación no cuantificada**")
    st.caption(
        "Se desconoce si las normas de la Gaceta (combustibles, controles de "
        "cambio, reservas, deuda) anticipan o acompañan los movimientos de la "
        "brecha; sin esa evidencia no pueden usarse como señales de política."
    )

ui.seccion("Objetivos")
objetivos = pd.DataFrame([
    {"#": "Objetivo 1",
     "Objetivo": "Predecir el nivel del paralelo y la brecha oficial–paralelo "
                 "a 1 h y 1 día, con intervalos de predicción."},
    {"#": "Objetivo 2",
     "Objetivo": "Cuantificar la relación entre las publicaciones de la Gaceta "
                 "(canales temáticos) y la brecha, en ventanas de ±5 días."},
])
ui.tabla(objetivos, "Objetivos de predicción del proyecto.", "tab:objetivos",
         height=150)

ui.seccion("Criterios de éxito")
st.markdown(
    """
- Los pronósticos a 1 h y 1 día **superan a la línea base naïve** (paseo
  aleatorio) fuera de muestra, medidos con RMSE y MAE.
- Los intervalos de predicción alcanzan una **cobertura empírica cercana al
  90 %**.
- El estudio de eventos de ±5 días muestra un **efecto medible** (signo y
  magnitud, con errores HAC) de las publicaciones relevantes sobre la brecha,
  o una conclusión documentada de ausencia de relación.
"""
)
st.caption(
    "Nota de lectura: el criterio se evalúa contra el naïve del **mismo** "
    "horizonte; un cociente RMSE ÷ naïve igual a 1 es empate y mayor que 1 es "
    "peor desempeño."
)

ui.seccion("Inventario de recursos")
r1, r2 = st.columns([1.6, 1], gap="large")
with r1:
    st.markdown(
        """
**Datos disponibles (8 conjuntos)**

- Historial completo de snapshots: 16,4 mil registros blue vs oficial.
- ~96 mil snapshots por plataforma.
- Serie horaria alineada de 6 meses: 4,3 mil puntos.
- Historial por plataforma cada 3 h y velas OHLC del P2P.
- Percentiles y estadísticos rodantes.
- Pipeline reproducible sobre 37 endpoints.
- Reporte de calidad de la propia API.
"""
    )
with r2:
    st.markdown(
        """
**Herramientas y equipo**

- Python 3.12: `requests`, `pandas`, `matplotlib`, `seaborn`,
  `scikit-learn`, `statsmodels`.
- EDA previo y auditoría de predictibilidad.
- Un analista con acceso a la API (clave propia con cuota semanal).
"""
    )

ui.lectura(
    "El alcance es deliberadamente acotado: dos objetivos, dos horizontes y "
    "tres modelos. La comparación contra el naïve es el criterio que ordena "
    "todas las decisiones posteriores.",
)

ui.pie("views/02_negocio.py")
