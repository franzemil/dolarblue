"""Componentes de interfaz y estilos del dashboard."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import config as C
from core import formato as F

# ── hoja de estilo ─────────────────────────────────────────────────────────
_CSS = """
<style>
:root {
  --db-blue:#2563EB; --db-orange:#F59E0B; --db-gray:#6B7280;
  --db-dark:#0F172A; --db-slate:#334155; --db-surface:#F8FAFC;
  --db-border:#E2E8F0; --db-green:#10B981; --db-red:#EF4444;
}
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
  "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--db-surface);
  border-right: 1px solid var(--db-border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.db-brand { font-weight: 800; font-size: 1.02rem; letter-spacing: .04em;
  color: var(--db-blue); margin-bottom: .1rem; }
.db-brand-sub { font-size: .74rem; color: var(--db-gray); margin-bottom: 1rem; }
.db-side-label { text-transform: uppercase; letter-spacing: .09em;
  font-size: .64rem; font-weight: 700; color: var(--db-gray);
  margin: .9rem 0 .2rem; }
.db-side-data { font-size: .74rem; color: var(--db-slate); line-height: 1.5; }

/* encabezado */
.db-eyebrow { text-transform: uppercase; letter-spacing: .11em;
  font-size: .7rem; font-weight: 700; color: var(--db-blue); }
.db-title { font-size: 2.05rem; line-height: 1.12; font-weight: 800;
  color: var(--db-dark); margin: .25rem 0 .45rem; letter-spacing: -.02em; }
.db-lead { font-size: 1.02rem; color: var(--db-slate); max-width: 78ch;
  line-height: 1.55; margin-bottom: .35rem; }
.db-fuente { font-size: .68rem; color: var(--db-gray); margin: .1rem 0 1rem; }
.db-fuente code { font-size: .68rem; background: #F1F5F9; padding: .05rem .3rem;
  border-radius: 4px; }

/* secciones */
.db-h2 { font-size: 1.18rem; font-weight: 750; color: var(--db-dark);
  margin: 1.4rem 0 .15rem; letter-spacing: -.01em; }
.db-h2-sub { font-size: .84rem; color: var(--db-gray); margin-bottom: .6rem; }

/* KPI */
.db-kpi { position: relative; border: 1px solid var(--db-border);
  border-radius: 12px; background: #fff; padding: .8rem .9rem .75rem;
  height: 100%; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.db-kpi:hover { z-index: 100; }
.db-kpi-label { font-size: .68rem; text-transform: uppercase;
  letter-spacing: .08em; color: var(--db-gray); font-weight: 700; }
.db-kpi-value { font-size: 1.5rem; font-weight: 800; color: var(--db-dark);
  line-height: 1.25; margin-top: .18rem; }
.db-help { position: relative; display: inline-block; margin-left: .25rem;
  color: var(--db-gray); font-size: .72rem; font-weight: 600;
  cursor: help; outline: none; }
.db-help::after {
  content: attr(data-tip);
  position: absolute; left: 50%; bottom: 1.45rem;
  transform: translateX(-50%);
  width: max-content; max-width: 18rem;
  padding: .5rem .65rem;
  border-radius: 7px;
  background: var(--db-dark); color: #fff;
  font-size: .72rem; font-weight: 400; line-height: 1.4;
  letter-spacing: normal; text-transform: none; text-align: left;
  white-space: normal; z-index: 1000;
  box-shadow: 0 6px 18px rgba(15,23,42,.22);
  opacity: 0; visibility: hidden; pointer-events: none;
  transition: opacity .12s ease;
}
.db-help::before {
  content: ""; position: absolute; left: 50%; bottom: 1.1rem;
  transform: translateX(-50%);
  border: 6px solid transparent; border-top-color: var(--db-dark);
  opacity: 0; visibility: hidden; pointer-events: none;
  transition: opacity .12s ease;
}
.db-kpi:hover .db-help::after,
.db-kpi:hover .db-help::before,
.db-help:focus::after,
.db-help:focus::before { opacity: 1; visibility: visible; }
.db-kpi-nota { font-size: .73rem; color: var(--db-gray); margin-top: .15rem; }
.db-delta { font-size: .74rem; font-weight: 700; margin-top: .15rem; }
.db-good { color: var(--db-green); } .db-bad { color: var(--db-red); }
.db-neutral { color: var(--db-gray); }

/* ficha */
.db-ficha { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
  gap: .55rem .9rem; border: 1px solid var(--db-border); border-radius: 12px;
  background: #fff; padding: .8rem .95rem; margin: .2rem 0 .9rem; }
.db-ficha span { display: block; font-size: .65rem; text-transform: uppercase;
  letter-spacing: .07em; color: var(--db-gray); font-weight: 700; }
.db-ficha b { font-size: .88rem; color: var(--db-dark); font-weight: 650; }

/* lectura */
.db-lectura { border-left: 3px solid var(--db-blue); background: #EFF6FF;
  border-radius: 0 8px 8px 0; padding: .65rem .85rem; margin: .55rem 0 1rem;
  font-size: .86rem; color: #1E3A8A; line-height: 1.5; }
.db-lectura b { color: #1E40AF; }
.db-alerta { border-left: 3px solid var(--db-orange); background: #FFFBEB;
  border-radius: 0 8px 8px 0; padding: .65rem .85rem; margin: .55rem 0 1rem;
  font-size: .86rem; color: #78350F; line-height: 1.5; }

/* captions y badges */
.db-caption { font-size: .72rem; color: var(--db-gray); margin: -.2rem 0 .9rem;
  line-height: 1.4; }
.db-caption code { background: #F1F5F9; padding: .03rem .28rem; border-radius: 4px; }
.db-badge { display: inline-block; font-size: .62rem; font-weight: 800;
  letter-spacing: .07em; text-transform: uppercase; padding: .14rem .45rem;
  border-radius: 999px; background: #FEF3C7; color: #92400E; vertical-align: middle; }
.db-badge-blue { background: #DBEAFE; color: #1E40AF; }
.db-badge-green { background: #D1FAE5; color: #065F46; }
.db-pie { border-top: 1px solid var(--db-border); margin-top: 2rem;
  padding-top: .8rem; }

/* tabla */
[data-testid="stDataFrame"] { border: 1px solid var(--db-border);
  border-radius: 10px; }
</style>
"""


def css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── encabezados y secciones ────────────────────────────────────────────────
def encabezado(key: str) -> None:
    s = C.SECCIONES[key]
    st.markdown(
        f'<div class="db-eyebrow">{html.escape(s["fase"])}</div>'
        f'<div class="db-title">{html.escape(s["titulo"])}</div>'
        f'<div class="db-lead">{s["lead"]}</div>',
        unsafe_allow_html=True,
    )


def seccion(titulo: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="db-h2">{html.escape(titulo)}</div>',
                unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="db-h2-sub">{sub}</div>',
                    unsafe_allow_html=True)


def fuente(texto: str) -> None:
    st.markdown(f'<div class="db-fuente">Fuente: <code>{html.escape(texto)}'
                "</code></div>", unsafe_allow_html=True)


def badge(texto: str, tono: str = "") -> None:
    clase = f"db-badge db-badge-{tono}" if tono else "db-badge"
    st.markdown(f'<span class="{clase}">{html.escape(texto)}</span>',
                unsafe_allow_html=True)


# ── tarjetas ───────────────────────────────────────────────────────────────
def kpis(items: list[dict]) -> None:
    """items: {label, value, nota?, delta?, tono?, ayuda?}.

    ``ayuda`` se muestra como tooltip al pasar el mouse por la tarjeta.
    """
    cols = st.columns(len(items), gap="small")
    for col, it in zip(cols, items):
        tono = it.get("tono", "neutral")
        delta = it.get("delta")
        delta_html = (f'<div class="db-delta db-{tono}">{delta}</div>'
                      if delta else "")
        ayuda = it.get("ayuda")
        icono = ""
        if ayuda:
            tip = html.escape(ayuda, quote=True)
            icono = (f'<span class="db-help" data-tip="{tip}" '
                     f'aria-label="{tip}" tabindex="0">ⓘ</span>')
        with col:
            st.markdown(
                f'<div class="db-kpi">'
                f'<div class="db-kpi-label">{html.escape(it["label"])}'
                f'{icono}</div>'
                f'<div class="db-kpi-value">{html.escape(str(it["value"]))}</div>'
                f'{delta_html}'
                f'<div class="db-kpi-nota">{html.escape(it.get("nota", ""))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def ficha(items: list[tuple[str, object]]) -> None:
    celdas = "".join(
        f"<div><span>{html.escape(str(k))}</span><b>{html.escape(str(v))}</b></div>"
        for k, v in items
    )
    st.markdown(f'<div class="db-ficha">{celdas}</div>', unsafe_allow_html=True)


def lectura(texto: str, titulo: str = "Lectura") -> None:
    st.markdown(
        f'<div class="db-lectura"><b>{html.escape(titulo)}.</b> {texto}</div>',
        unsafe_allow_html=True,
    )


def alerta(texto: str, titulo: str = "Atención") -> None:
    st.markdown(
        f'<div class="db-alerta"><b>{html.escape(titulo)}.</b> {texto}</div>',
        unsafe_allow_html=True,
    )


# ── tablas y figuras ───────────────────────────────────────────────────────
def caption(texto: str, origen: str | None = None) -> None:
    extra = f" · <code>{html.escape(origen)}</code>" if origen else ""
    st.markdown(f'<div class="db-caption">{texto}{extra}</div>',
                unsafe_allow_html=True)


def tabla(df: pd.DataFrame, caption_txt: str | None = None,
          origen: str | None = None, height: int | None = None,
          formato: dict | None = None,
          column_config: dict | None = None) -> None:
    """Tabla con caption editorial. `formato` = {columna: nº decimales}."""
    vista = df.copy()
    if formato:
        for col, nd in formato.items():
            if col in vista.columns:
                vista[col] = vista[col].map(
                    lambda v, n=nd: F.num(v, n) if isinstance(v, (int, float))
                    else v)
    kwargs = {"width": "stretch", "hide_index": True}
    if height:
        kwargs["height"] = height
    if column_config:
        kwargs["column_config"] = column_config
    st.dataframe(vista, **kwargs)
    if caption_txt:
        caption(caption_txt, origen)


def figura(fig: go.Figure, caption_txt: str | None = None,
           origen: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False})
    if caption_txt:
        caption(caption_txt, origen)


def figura_mpl(fig, caption_txt: str | None = None,
               origen: str | None = None) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    if caption_txt:
        caption(caption_txt, origen)


def vacio(mensaje: str, comando: str | None = None) -> None:
    st.warning(mensaje)
    if comando:
        st.code(comando, language="bash")


def pie(actual: str) -> None:
    """Navegación secuencial entre las páginas del informe."""
    paginas = [p for grupo in C.GRUPOS.values() for p in grupo]
    rutas = [p[0].replace("views/", "") for p in paginas]
    i = rutas.index(actual.replace("views/", ""))
    st.markdown('<div class="db-pie"></div>', unsafe_allow_html=True)
    izq, der = st.columns(2)
    with izq:
        if i > 0:
            ruta, titulo, icono = paginas[i - 1]
            st.page_link(ruta, label=f"← {titulo}", icon=icono)
    with der:
        if i < len(paginas) - 1:
            ruta, titulo, icono = paginas[i + 1]
            st.page_link(ruta, label=f"{titulo} →", icon=icono)


def descarga_csv(df: pd.DataFrame, nombre: str, etiqueta: str = "Descargar CSV"):
    st.download_button(
        etiqueta,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre,
        mime="text/csv",
        icon=":material/download:",
    )
