"""Stage ABC + Pipeline: composición estilo sklearn (collectors == processors).

Un ``Stage`` es la unidad mínima: ``fit`` prepara (no-op por defecto) y
``transform`` hace el trabajo. Un ``Pipeline`` ES un ``Stage`` (composite),
así las ramas se anidan sin casos especiales.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """Contrato compartido por collectors (vía adapter) y processors."""

    def fit(self, inputs: Any) -> "Stage":
        """Preparación opcional; por defecto no-op."""
        return self

    @abstractmethod
    def transform(self, inputs: Any) -> Any:
        """Ejecuta la etapa sobre ``inputs`` y devuelve su salida."""

    def fit_transform(self, inputs: Any) -> Any:
        return self.fit(inputs).transform(inputs)


class Pipeline(Stage):
    """Secuencia de Stages con nombre; la salida de cada uno entra al siguiente."""

    def __init__(self, steps: list[tuple[str, Stage]]):
        if not steps:
            raise ValueError("Pipeline necesita al menos un step (nombre, Stage)")
        self.steps = steps
        self.named_steps: dict[str, Stage] = dict(steps)

    def transform(self, inputs: Any) -> Any:
        for _name, stage in self.steps:
            inputs = stage.fit_transform(inputs)
        return inputs
