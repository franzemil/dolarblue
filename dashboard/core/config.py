"""Constantes, rutas y metadatos de navegación del dashboard."""
from __future__ import annotations

from pathlib import Path

# ── rutas ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # raíz del proyecto
DASH = ROOT / "dashboard"
ASSETS = DASH / "assets"
MODELADO = ROOT / "modelado"
DATA = ROOT / "data" / "processed"
REPORT = ROOT / "report"
OUTPUTS = MODELADO / "outputs"

# el dashboard reutiliza los módulos del pipeline
import sys  # noqa: E402
for _p in (ROOT, MODELADO, DASH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── constantes del proyecto ────────────────────────────────────────────────
H = 168                    # ventana de prueba (una semana horaria)
HORIZONTES = (1, 24)
TARGETS = ("blue_mid", "brecha_pct")
NOMBRE_HORIZONTE = {1: "1 hora", 24: "1 día"}
HORIZONTES_EVENTO = ("5 h", "1 d", "3 d", "5 d")

# ── paleta (charts/estilo.py + main.tex) ───────────────────────────────────
BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
GRAY = "#6B7280"
DARK = "#0F172A"
SLATE = "#334155"
SURFACE = "#F8FAFC"
BORDER = "#E2E8F0"

COLOR_MODELO = {"RL múltiple": ORANGE, "ARIMA": BLUE}
COLOR_TARGET = {"blue_mid": BLUE, "brecha_pct": RED}

# ── navegación: una entrada por sección del informe ────────────────────────
GRUPOS = {
    "Portada": [
        ("views/00_resumen.py", "Resumen", ":material/home:"),
    ],
    "Fases CRISP-DM": [
        ("views/01_introduccion.py", "Introducción", ":material/info:"),
        ("views/02_negocio.py", "Comprensión del negocio",
         ":material/business_center:"),
        ("views/03_datos.py", "Datos y EDA", ":material/database:"),
        ("views/04_preparacion.py", "Preparación", ":material/build:"),
        ("views/05_modelado.py", "Modelado", ":material/model_training:"),
        ("views/06_evaluacion.py", "Evaluación de negocio",
         ":material/insights:"),
    ],
    "Cierre": [
        ("views/07_conclusiones.py", "Conclusiones", ":material/flag:"),
    ],
    "Anexos": [
        ("views/10_metricas.py", "Métricas", ":material/functions:"),
        ("views/08_explorador.py", "Explorador de datos",
         ":material/table_view:"),
        ("views/09_reproducibilidad.py", "Reproducibilidad",
         ":material/settings:"),
    ],
}

# metadatos editoriales de cada página
SECCIONES = {
    "00_resumen": {
        "fase": "Informe final · CRISP-DM",
        "titulo": "Dólar blue, oficial y P2P en Bolivia",
        "lead": ("Previsión del nivel del paralelo y de la brecha cambiaria a "
                 "una hora y a un día: datos, modelos y evaluación de negocio "
                 "en una sola vista."),
        "fuente": "report/sections/00_resumen.tex",
    },
    "01_introduccion": {
        "fase": "1 · Introducción",
        "titulo": "Motivación y alcance",
        "lead": ("Dos sistemas de precios del dólar conviven en Bolivia. El "
                 "proyecto documenta su relación y evalúa si es posible "
                 "anticipar el paralelo."),
        "fuente": "report/sections/01_introduccion.tex",
    },
    "02_negocio": {
        "fase": "2 · Comprensión del negocio",
        "titulo": "Objetivos, KPIs y criterios de éxito",
        "lead": ("Dos objetivos medibles: predecir el nivel y la brecha "
                 "(Objetivo 1) y medir el vínculo entre la Gaceta y la brecha "
                 "(Objetivo 2)."),
        "fuente": "report/sections/02_negocio.tex",
    },
    "03_datos": {
        "fase": "3 · Comprensión y obtención de datos",
        "titulo": "Fuentes, calidad y exploración",
        "lead": ("Del snapshot de la API y el PDF de la Gaceta a una serie "
                 "horaria regular: cobertura, huecos, diccionario y EDA."),
        "fuente": "report/sections/03_datos.tex",
    },
    "04_preparacion": {
        "fase": "4 · Preparación de datos",
        "titulo": "De la fuente cruda a los predictores",
        "lead": ("Remuestreo horario, features sin fuga de información, "
                 "partición temporal y línea base naïve."),
        "fuente": "report/sections/05_preparacion.tex",
    },
    "05_modelado": {
        "fase": "5 · Modelado",
        "titulo": "Tres modelos, dos horizontes, una comparación",
        "lead": ("Regresión lineal múltiple, OLS con errores HAC y ARIMA, "
                 "evaluados contra el naïve en las últimas 168 h."),
        "fuente": "report/sections/06_modelado.tex",
    },
    "06_evaluacion": {
        "fase": "6 · Evaluación de negocio",
        "titulo": "¿Sirve el modelo para decidir?",
        "lead": ("El error de nivel no genera ahorro; la señal direccional "
                 "solo aporta a una hora. Los números detrás de esa "
                 "conclusión."),
        "fuente": "report/sections/07_evaluacion_negocio.tex",
    },
    "07_conclusiones": {
        "fase": "7 · Conclusiones",
        "titulo": "Hallazgos y trabajo futuro",
        "lead": ("Lo que el proyecto demuestra, lo que no, y qué haría falta "
                 "para llevar la señal a producción."),
        "fuente": "report/sections/wip_07_conclusiones.tex",
    },
    "08_explorador": {
        "fase": "Anexo",
        "titulo": "Explorador de datos",
        "lead": ("Consulta, filtra y descarga las tablas que alimentan el "
                 "informe."),
        "fuente": "data/processed · modelado/panel_diario.csv",
    },
    "10_metricas": {
        "fase": "Anexo",
        "titulo": "Métricas: de la MAE al R²",
        "lead": ("Qué mide cada número del informe, cuál es su mejor valor, "
                 "dónde aparece y cuál es la trampa habitual al leerlo."),
        "fuente": "metrics/ · report/sections/06_modelado.tex",
    },
    "09_reproducibilidad": {
        "fase": "Anexo",
        "titulo": "Reproducibilidad",
        "lead": ("Pipeline, comandos, versiones y linaje de cada artefacto "
                 "que muestra el dashboard."),
        "fuente": "Makefile · pyproject.toml",
    },
}
