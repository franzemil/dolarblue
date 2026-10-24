"""Processors: pipeline de procesamiento replayable (sin re-descargar).

Etapas:
- SnapshotReader: manifiesto + archivos crudos ya en disco -> inputs
- JsonProcessor: replay de data/sources/*.json -> data/normalized/*.csv
- LLMFileProcessor: PDFs de Gaceta -> registros tipados (GLM-5.3-Flash)
- MarketHourlyIndexer / GacetaDailyIndexer: un archivo final por rama,
  cada uno a la resolución temporal nativa de su fuente.
"""

from processors.results import GacetaNorma, ProcessResult
from processors.stage import Pipeline, Stage

__all__ = ["Pipeline", "Stage", "GacetaNorma", "ProcessResult"]
