"""Colectores del proyecto: base común, modelos tipados y fuentes concretas.

- models.py:      TypedResult[T] + payloads Pydantic documentados con Field(description)
- base.py:        DataCollector (ABC) -> ApiCollector[T], FileCollector
- endpoints.py:   registro de URLs de la API dolarblue (un EndpointSpec por URL)
- normalizers.py: envelopes -> payloads tipados -> filas con procedencia
- dolarblue.py:   DolarBlueCollector  (ApiCollector por URL)
- gaceta.py:      GacetaCollector     (FileCollector)
- collect_all.py: orquestador run_all (lo usa main.py)
- utils.py:       helpers de compatibilidad (chart_6m, manifiesto Gaceta)
- rotate_key.py:  CLI de rotación de la DBB_API_KEY
"""

from collectors.base import ApiCollector, DataCollector, FileCollector
from collectors.models import FileResult, TypedResult

__all__ = ["ApiCollector", "DataCollector", "FileCollector", "FileResult", "TypedResult"]
