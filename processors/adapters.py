"""Adapter DataCollector -> Stage.

Los collectors existentes no cambian sus internals: este adapter les da el
contrato ``Stage`` para que compongan en pipelines igual que los processors
(pipeline de colecta semanal, separado del pipeline de procesamiento).
"""

from __future__ import annotations

from typing import Any

from collectors.base import DataCollector
from processors.stage import Stage


class CollectorAdapter(Stage):
    """Envuelve un DataCollector: transform() = collect()."""

    def __init__(self, collector: DataCollector):
        self.collector = collector

    def transform(self, inputs: Any) -> Any:
        return self.collector.collect()
