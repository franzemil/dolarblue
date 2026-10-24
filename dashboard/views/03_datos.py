"""Comprensión y obtención de datos (incluye EDA)."""
from __future__ import annotations

import json

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
ui.imagen(C.ASSETS / "diagrams" / "informe_flujo_datos.png",
          "Obtención de datos: las dos fuentes convergen en los procesadores.")
ui.imagen(C.ASSETS / "diagrams" / "informe_recoleccion.png",
          "Del snapshot crudo a los predictores del modelo.")

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
    "Snapshots crudos de compra y venta frente a la grilla horaria (medio), "
    "en la última semana de la muestra.",
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
         "notebook · 2.3", formato={c: 2 for c in desc.columns[1:]})

ui.seccion("Ejemplo de feature: n_docs_fx_alta")
dias_fx = int((panel["n_docs_fx"] > 0).sum())
dias_alta = int((panel["n_docs_fx_alta"] > 0).sum())
normas_fx = int(panel["n_docs_fx"].sum())
normas_alta = int(panel["n_docs_fx_alta"].sum())
ui.kpis([
    {"label": "Días con norma FX", "value": f"{dias_fx} de {len(panel)}",
     "nota": "relevancia media o alta",
     "ayuda": "n_docs_fx > 0: días con al menos una norma de relevancia cambiaria media o alta."},
    {"label": "Días con FX alta", "value": f"{dias_alta} de {len(panel)}",
     "nota": "solo relevancia alta",
     "ayuda": "n_docs_fx_alta > 0: días con al menos una norma de relevancia cambiaria alta."},
    {"label": "Normas FX", "value": F.miles(normas_fx),
     "nota": "media o alta en el panel",
     "ayuda": f"Suma diaria de n_docs_fx en los {len(panel)} días del panel."},
    {"label": "Normas FX alta", "value": F.miles(normas_alta),
     "nota": "solo alta en el panel",
     "ayuda": f"Suma diaria de n_docs_fx_alta en los {len(panel)} días del panel."},
])
ejemplo = (panel.loc[panel["n_docs_fx_alta"] > 0,
                    ["fecha_d", "n_docs_total", "n_docs_fx", "n_docs_fx_alta"]]
           .sort_values("fecha_d").copy())
for c in ("n_docs_total", "n_docs_fx", "n_docs_fx_alta"):
    ejemplo[c] = ejemplo[c].fillna(0).astype(int)
ejemplo["fecha_d"] = ejemplo["fecha_d"].map(F.fecha)
ui.tabla(
    ejemplo.rename(columns={
        "fecha_d": "Fecha",
        "n_docs_total": "Normas totales",
        "n_docs_fx": "Normas FX (media+alta)",
        "n_docs_fx_alta": "Normas FX alta",
    }),
    "Días del panel con al menos una norma de relevancia FX alta.",
    "panel_diario.csv · n_docs_fx_alta",
    height=260,
)
ui.lectura(
    "<b>n_docs_fx</b> cuenta normas con relevancia media o alta; "
    "<b>n_docs_fx_alta</b> cuenta solo las de relevancia alta. No es un tag "
    "ni un score: es un conteo diario. Un día sin publicación aporta 0, no un "
    f"faltante. Con solo {normas_alta} normas de alta en {len(panel)} días, "
    "el canal es raro: por eso se rezaga un día y no resulta significativo "
    "en el OLS-HAC.",
    titulo="Cómo leerlo",
)

ui.lectura(
    "La serie horaria cubre 6 meses y 4.318 horas efectivas. El quiebre del "
    "oficial divide la muestra en dos regímenes; todas las comparaciones se "
    "hacen dentro del régimen vigente.",
)

ui.seccion("3.6 · Trazabilidad del pipeline")
PIPE_DIR = C.ASSETS / "pipeline"
meta_path = PIPE_DIR / "scenes.json"

if not meta_path.exists():
    ui.vacio(
        "Faltan los frames y el guion de la animación en "
        "`dashboard/assets/pipeline/`.",
        "make pipeline-assets",
    )
else:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    s = meta["stats"]
    hueco = s["gaps"][0][2] if s.get("gaps") else 0
    st.markdown(
        "La animación de 60 segundos recorre las ocho escenas del pipeline. "
        "Cada pestaña muestra el clip animado de una escena (o su frame final "
        "si los clips no están generados) con la misma explicación que la "
        "sección **2.2** del notebook. Los frames se exportan desde "
        "`report/make_animation.py`; no se recalculan al abrir la app."
    )
    st.caption(
        f"Frames generados desde `data/` al {s['updated']} · sin llamadas a la API."
    )
    ui.kpis([
        {"label": "Snapshots crudos", "value": F.miles(s["snapshots"]),
         "nota": "history_full.csv"},
        {"label": "Horas observadas", "value": F.miles(s["hours"]),
         "nota": f"{F.miles(s['hours_expected'])} esperadas · {hueco} h interpoladas"},
        {"label": "PDFs / extracciones",
         "value": f"{s['pdfs']} / {s['extracted']}",
         "nota": "una edición sin detalle"},
        {"label": "Cache LLM",
         "value": f"{s['cache_entries']} / {s['extracted']}",
         "nota": f"{s['provider']} / {s['model']}"},
    ])

    video = C.REPORT / "anim" / "pipeline.mp4"
    if video.exists():
        with st.expander("Ver la animación completa (60 s)"):
            try:
                st.video(str(video))
                st.caption(
                    "MP4 1920×1080 · 30 fps · también disponible en "
                    "`report/anim/pipeline.gif`."
                )
            except Exception:
                st.warning(
                    "No se pudo cargar el MP4 en este entorno; "
                    "usá los frames de las pestañas."
                )
    else:
        st.caption(
            "El MP4 no está disponible en este entorno; los ocho frames sí. "
            "Generalo con `make anim`."
        )

    clip_dir = C.REPORT / "anim" / "scenes"
    tabs = st.tabs([f"{x['n']}. {x['short']}" for x in meta["scenes"]])
    for tab, spec in zip(tabs, meta["scenes"]):
        with tab:
            st.markdown(f"#### {spec['title']}")
            st.markdown(spec["lead"])
            clip = clip_dir / spec.get("clip", "")
            if spec.get("clip") and clip.exists():
                ui.imagen(clip)
                st.caption(
                    f"Escena {spec['n']}/8 · {spec['start']:.0f}–{spec['end']:.0f} s "
                    f"· animación en loop ({spec['duration']:.0f} s) · "
                    f"{spec['subtitle']}"
                )
            else:
                ui.imagen(PIPE_DIR / spec["image"])
                st.caption(
                    f"Escena {spec['n']}/8 · {spec['start']:.0f}–{spec['end']:.0f} s "
                    f"· frame final · generá `make anim` para ver la animación"
                )
            st.markdown("**Qué mirar**")
            for bullet in spec["bullets"]:
                st.markdown(f"- {bullet}")
            ui.lectura(spec["rule"], titulo="Regla del pipeline")
            ui.fuente(spec["source"])

ui.pie("views/03_datos.py")
