"""Conclusiones (WIP en el informe)."""
from __future__ import annotations

import streamlit as st

from core import ui

ui.encabezado("07_conclusiones")
ui.badge("WIP · no incluida en el PDF")
st.write("")
ui.alerta(
    "Esta sección existe en el repositorio como borrador "
    "(<code>wip_07_conclusiones.tex</code>) y todavía no se compila en el "
    "informe. La versión formal es el PDF. El contenido de abajo combina el "
    "borrador con los hallazgos ya cerrados de las secciones 5 y 6.",
    titulo="Estado",
)

ui.seccion("Hallazgos")
st.markdown(
    """
1. **El naïve es un rival muy fuerte.** El nivel del blue horario se comporta
   como un paseo aleatorio. El ARIMA lo iguala o mejora levemente; la regresión
   lineal múltiple pierde en los cuatro casos.
2. **La precisión no se traduce en ahorro.** El MAE del ARIMA es idéntico al
   del naïve a 1 h y a 24 h. La única ganancia medible es de RMSE (4,5 % a una
   hora), sin contrapartida contable.
3. **La señal direccional es de una hora.** A 1 h la accuracy es 0,571 y el
   umbral C_FN/C_FP es 1,21; a 24 h cae a 0,485 y el umbral a 0,43. No hay
   señal diaria.
4. **La Gaceta no muestra relación explotable.** Ningún canal resulta
   significativo en el OLS-HAC ni en el estudio de eventos; las correlaciones
   diarias son débiles (|ρ| ≤ 0,16).
5. **El pipeline es reproducible.** Datos, figuras, tablas y PDF se regeneran
   desde los CSV con comandos del `Makefile`; este dashboard lee los mismos
   artefactos.
"""
)
ui.lectura(
    "El aporte del proyecto es negativo en el mejor sentido: documenta con "
    "evidencia que el mercado horario es casi impredecible y que la Gaceta no "
    "anticipa la brecha. Eso evita invertir en modelos que no pueden superar "
    "una regla trivial.",
)

ui.seccion("Trabajo futuro")
st.markdown(
    """
- **Validación walk-forward más larga.** Una sola semana de prueba no alcanza
  para concluir sobre estabilidad; repetir con ventanas móviles de varios meses.
- **Intervalos, no puntos.** Profundizar la calibración de la banda del ARIMA:
  es el único componente con valor operativo.
- **Costos asimétricos reales.** Estimar C_FP y C_FN con datos de tesorería de
  una empresa para convertir el umbral en una regla de decisión.
- **Ingesta programada.** Respetar la cuota semanal de la API (~30 llamadas por
  corrida) con ingestas programadas, no exploratorias.
- **Señales alternativas.** El volumen P2P y los spreads por plataforma no
  entraron al modelo; podrían aportar información que la Gaceta no tiene.
"""
)

ui.seccion("Estado del entregable")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.metric("Fases CRISP-DM", "1–5")
    st.caption("despliegue y monitoreo pendientes")
with c2:
    st.metric("Modelos evaluados", "3")
    st.caption("RL · OLS-HAC · ARIMA")
with c3:
    st.metric("Ventana de prueba", "168 h")
    st.caption("una semana horaria")

ui.pie("views/07_conclusiones.py")
