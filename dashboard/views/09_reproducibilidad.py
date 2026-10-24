"""Anexo · Reproducibilidad."""
from __future__ import annotations

import platform

import pandas as pd
import streamlit as st

from core import config as C
from core import data, formato as F, ui

ui.encabezado("09_reproducibilidad")
ui.lectura(
    "La app no re-entrena: lee artefactos generados por el pipeline. "
    "Regenerarlos es un comando, y cada cifra del informe se puede rastrear "
    "hasta su archivo de origen.",
)

ui.seccion("Pipeline")
ui.imagen(C.ASSETS / "diagrams" / "informe_flujo_datos.png",
          "Fuentes → recolectores → procesadores → CSV de análisis.")
st.code(
    """data/raw/*.json ──► processors ──► data/processed/*.csv
                                     ├── market_hourly.csv
                                     ├── gaceta_daily.csv
                                     └── gaceta_llm.csv
modelado/reporte_modelado.py ──► modelado/metricas_reporte.json
dashboard/scripts/export_assets.py ──► dashboard/assets/*.parquet
streamlit run dashboard/app.py ──► esta app""",
    language="text",
)

ui.seccion("Comandos")
comandos = pd.DataFrame([
    {"Comando": "make dashboard",
     "Qué hace": "Levanta el dashboard en local."},
    {"Comando": "make dashboard-assets",
     "Qué hace": "Regenera los parquet de dashboard/assets (predicciones, "
                 "OLS-HAC, ROC)."},
    {"Comando": "make data",
     "Qué hace": "Refresca los datos de la API (cuota semanal)."},
    {"Comando": "make process",
     "Qué hace": "Reprocesa snapshots y etiqueta la Gaceta con LLM."},
    {"Comando": "make render",
     "Qué hace": "Diagramas + figuras + PDF del informe."},
])
ui.tabla(comandos, "Comandos del proyecto.", "Makefile", height=250)

ui.seccion("Versiones")
versiones = {
    "Python": platform.python_version(),
    "pandas": pd.__version__,
    "Streamlit": st.__version__,
}
for modulo, nombre in (("numpy", "numpy"), ("plotly", "plotly"),
                       ("statsmodels", "statsmodels"),
                       ("sklearn", "scikit-learn"),
                       ("matplotlib", "matplotlib"),
                       ("pyarrow", "pyarrow")):
    try:
        m = __import__(modulo)
        versiones[nombre] = getattr(m, "__version__", "—")
    except Exception:
        versiones[nombre] = "no instalado"
ui.tabla(pd.DataFrame([versiones]).T.reset_index().rename(
    columns={"index": "Paquete", 0: "Versión"}),
    "Entorno con el que se generaron los artefactos.",
    "pyproject.toml", height=320)

ui.seccion("Artefactos")
estado = pd.DataFrame(data.estado_assets())
estado["tamaño"] = estado["bytes"].map(
    lambda b: f"{F.num(b / 1024, 1)} KB" if b else "—")
estado["existe"] = estado["existe"].map(lambda x: "sí" if x else "no")
ui.tabla(estado[["archivo", "uso", "existe", "tamaño", "ruta"]].rename(
    columns={"archivo": "Archivo", "uso": "Uso", "existe": "Presente",
             "tamaño": "Tamaño", "ruta": "Ruta"}),
    "Linaje de los artefactos que consume el dashboard.",
    "dashboard/assets · modelado/outputs", height=380)

m = data.manifest()
if m:
    st.markdown("**Manifiesto de assets**")
    filas = [
        ("Generado", m.get("generado", "—")),
        ("Datos desde", F.fecha_hora(m.get("datos_desde"))),
        ("Datos hasta", F.fecha_hora(m.get("datos_hasta"))),
        ("Horas de la grilla", F.miles(m.get("horas", 0))),
        ("Ventana de prueba", f'{m.get("h_test", "—")} h'),
    ]
    ui.ficha(filas)
    with st.expander("Ver hashes SHA-256 (12 caracteres)"):
        for nombre, sha in m.get("sha256", {}).items():
            st.code(f"{sha}  {nombre}", language="text")
else:
    ui.vacio("No hay manifest.json. Ejecutá `make dashboard-assets`.")

ui.seccion("Estructura del dashboard")
st.code(
    """dashboard/
├── app.py                 # navegación y sidebar
├── core/                  # config, datos, formato, figuras, UI, tablas
├── views/                 # una página por sección del informe
├── assets/                # parquet curados (versionados)
└── scripts/export_assets.py""",
    language="text",
)

ui.pie("views/09_reproducibilidad.py")
