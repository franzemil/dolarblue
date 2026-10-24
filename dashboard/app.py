"""Dólar Blue · Dashboard del informe final (CRISP-DM).

Ejecutar:
    streamlit run dashboard/app.py
    make dashboard
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")          # backend sin ventana: Streamlit lo necesita

import streamlit as st  # noqa: E402

from core import config as C  # noqa: E402
from core import data, formato as F, ui  # noqa: E402

st.set_page_config(
    page_title="Dólar Blue · Informe CRISP-DM",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.css()
st.logo(str(C.ASSETS / "logo.svg"), size="large")

# ── navegación: una página por sección del informe ─────────────────────────
paginas = {
    grupo: [
        st.Page(ruta, title=titulo, icon=icono,
                default=(ruta == "views/00_resumen.py"))
        for ruta, titulo, icono in items
    ]
    for grupo, items in C.GRUPOS.items()
}
nav = st.navigation(paginas, position="sidebar")

# ── barra lateral persistente ──────────────────────────────────────────────
with st.sidebar:
    st.segmented_control(
        "Horizonte de pronóstico",
        options=list(C.HORIZONTES),
        format_func=lambda h: f"{h} h" if h == 1 else "24 h (1 día)",
        default=1,
        key="horizonte",
        help="Filtra las pestañas de Modelado y Evaluación.",
    )

    st.markdown('<div class="db-side-label">Datos</div>', unsafe_allow_html=True)
    corte = "—"
    try:
        cob = data.cobertura()
        st.markdown(
            f'<div class="db-side-data">'
            f'Serie horaria: <b>{F.miles(cob["mercado_filas"])}</b> h<br>'
            f'Panel Gaceta: <b>{cob["panel_filas"]}</b> días<br>'
            f'Corte: <b>{F.fecha(cob["mercado_hasta"])}</b>'
            f"</div>",
            unsafe_allow_html=True,
        )
        corte = F.fecha(cob["mercado_hasta"])
    except data.DataFaltante as exc:
        st.warning(str(exc))

    st.divider()
    try:
        st.download_button(
            "Métricas (JSON)",
            data=(C.MODELADO / "metricas_reporte.json").read_bytes(),
            file_name="metricas_reporte.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
    except FileNotFoundError:
        pass
    pdf = C.REPORT / "build" / "main.pdf"
    if pdf.exists():
        st.download_button(
            "Informe (PDF)",
            data=pdf.read_bytes(),
            file_name="informe_dolar_blue.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            width="stretch",
        )

    with st.expander("Estado de artefactos"):
        for a in data.estado_assets():
            icono = "✅" if a["existe"] else "❌"
            st.caption(f'{icono} `{a["archivo"]}` · {a["uso"]}')
        st.caption("Regenerar: `make dashboard-assets`")

    st.caption(f"v0.1 · Streamlit · datos al {corte}")

# ── página activa ──────────────────────────────────────────────────────────
try:
    nav.run()
except data.DataFaltante as exc:
    st.error("No se puede renderizar esta página.")
    st.code(str(exc))
    st.info("Ejecutá `make dashboard-assets` para regenerar los artefactos.")
