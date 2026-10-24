"""Taxonomía FX del proyecto: vocabulario controlado de tags + relevancia.

Única fuente de verdad para el prompt del LLM (enum) y para las columnas de
gaceta_daily.csv (n_docs_{tag}_llm). Cambiar aquí cambia todo el pipeline.
"""

from __future__ import annotations

TAGS_FX: list[str] = [
    # -- heredadas del tagger legacy por palabras clave
    "bcb_tipo_de_cambio",
    "divisas_control_cambios",
    "deuda_externa",
    "comercio_exterior",
    "credito_financiero",
    "mineria_energia",
    "fiscal_presupuesto",
    # -- canales de la crisis 2023-2026
    "combustibles_subsidios",
    "restricciones_importaciones",
    "banca_dolarizacion",
    "reservas_internacionales",
    "lavado_activos",
    "remesas",
    "servicios_basicos_precios",
    "politica_monetaria",
]

RELEVANCIA_FX: dict[str, int] = {"alta": 3, "media": 2, "baja": 1, "nula": 0}

TAGS_BOUNDARY_RULES: str = """
Reglas de frontera entre tags afines:
- combustibles_subsidios: importaciones, precios y subsidios de COMBUSTIBLES
  (gasolina, diésel, GGL, jet fuel); contrataciones de YPFB/DHMA para ese fin.
- servicios_basicos_precios: tarifas de servicios públicos (electricidad,
  agua, alcantarillado, telecomunicaciones, transporte).
- mineria_energia: producción, exportaciones y regalías del sector minero,
  hidrocarburos y litio (oferta de divisas).
- restricciones_importaciones: autorizaciones, cupos y prohibiciones de
  importación (SEMINEX y similares); comercio_exterior: política comercial
  general, aranceles, tratados y balanza comercial.
- politica_monetaria: tasas de política, operaciones de mercado abierto y
  encajes generales del BCB (canal UIP); banca_dolarizacion: normas
  específicas de moneda extranjera en bancos; credito_financiero: crédito y
  servicios del sistema financiero.
""".strip()
