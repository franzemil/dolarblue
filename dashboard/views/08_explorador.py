"""Anexo · Explorador de datos."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import data, formato as F, ui

ui.encabezado("08_explorador")
ui.lectura(
    "Las tablas son las mismas que alimentan el informe. Sirven para verificar "
    "un dato puntual, auditar un rango o descargar el CSV para trabajarlo fuera "
    "del dashboard.",
)

tab_mercado, tab_panel, tab_gaceta = st.tabs(
    ["Mercado horario", "Panel diario", "Normas de la Gaceta"]
)


def _fechas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Copia para mostrar con fechas dd/mm/aaaa (la descarga queda cruda)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%d/%m/%Y %H:%M")
    return out


def _rango(df: pd.DataFrame, col: str) -> pd.DataFrame:
    minimo, maximo = df[col].min().date(), df[col].max().date()
    rango = st.date_input("Rango de fechas", value=(minimo, maximo),
                          min_value=minimo, max_value=maximo)
    if isinstance(rango, (tuple, list)) and len(rango) == 2:
        ini, fin = rango
        return df[(df[col] >= pd.Timestamp(ini)) & (df[col] <= pd.Timestamp(fin)
                                                    + pd.Timedelta(days=1))]
    return df


with tab_mercado:
    df = data.mercado()
    # columnas con NaN que no se usan en la vista (las usadas por el modelo
    # se conservan en data.mercado() para calcular features)
    sin_uso = ["ret_blue_1h", "vol_24h", "fwd_1h", "fwd_6h", "fwd_24h",
               "fwd_72h", "fwd_168h"]
    df = df.drop(columns=[c for c in sin_uso if c in df.columns])
    df = _rango(df, "fecha")
    por_defecto = ["fecha", "blue_mid", "oficial_mid", "brecha_pct"]
    cols = st.multiselect("Columnas", options=list(df.columns),
                          default=[c for c in por_defecto if c in df.columns])
    vista = df[cols] if cols else df
    st.caption(f"{F.miles(len(vista))} filas · {len(vista.columns)} columnas")
    st.dataframe(_fechas(vista, ["fecha"]), width="stretch", hide_index=True,
                 height=430)
    ui.descarga_csv(vista, "mercado_horario.csv")

with tab_panel:
    df = data.panel().rename(columns={"fecha_d": "fecha"})
    df = _rango(df, "fecha")
    por_defecto = ["fecha", "blue_mid", "oficial_mid", "brecha_pct",
                   "n_docs_fx", "n_docs_fx_alta", "n_docs_total"]
    cols = st.multiselect("Columnas", options=list(df.columns),
                          default=[c for c in por_defecto if c in df.columns])
    vista = df[cols] if cols else df
    st.caption(f"{F.miles(len(vista))} filas · {len(vista.columns)} columnas")
    st.dataframe(_fechas(vista, ["fecha"]), width="stretch", hide_index=True,
                 height=430)
    ui.descarga_csv(vista, "panel_diario.csv")

with tab_gaceta:
    df = data.gaceta()
    if df.empty:
        ui.vacio("No hay normas en los assets.", "make dashboard-assets")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            estados = ["Todos"] + sorted(df["status"].dropna().unique().tolist())
            estado = st.selectbox("Estado de extracción", estados)
        with c2:
            texto = st.text_input(
                "Buscar en instrumento, entidad o resumen",
                placeholder="p. ej. combustibles, BCB, reservas…",
                label_visibility="collapsed",
            )
        vista = df if estado == "Todos" else df[df.status == estado]
        if texto:
            m = pd.Series(False, index=vista.index)
            for col in ("instrumento", "entidad_emisora", "resumen_1l", "tags"):
                if col in vista:
                    m |= vista[col].astype(str).str.contains(
                        texto, case=False, na=False)
            vista = vista[m]
        por_defecto = ["fecha_publicacion", "instrumento", "entidad_emisora",
                       "tags", "relevancia_fx", "cost_usd", "status"]
        cols = st.multiselect(
            "Columnas", options=list(vista.columns),
            default=[c for c in por_defecto if c in vista.columns],
            key="cols_gaceta")
        tabla = vista[cols] if cols else vista
        st.caption(f"{F.miles(len(tabla))} normas · "
                   f"{len(tabla.columns)} columnas")
        st.dataframe(_fechas(tabla, ["fecha_publicacion"]),
                     width="stretch", hide_index=True, height=430)
        ui.descarga_csv(tabla, "gaceta_normas.csv")

ui.pie("views/08_explorador.py")
