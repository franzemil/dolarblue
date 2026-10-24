"""Modelos del pipeline de procesamiento.

- ``ProcessResult``: espejo de ``TypedResult`` para etapas de procesamiento
  (procedencia de archivos + estado + costo LLM).
- ``GacetaNorma``: registro tipado que el LLM debe producir por cada norma de
  la Gaceta; se valida con Pydantic contra la taxonomía de processors/taxonomy.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from processors.taxonomy import RELEVANCIA_FX, TAGS_FX


class ProcessResult(BaseModel):
    """Resultado de una etapa de procesamiento: entrada/salida + estado + costo."""

    input_path: Path | None = Field(None, description="Archivo consumido (PDF, JSON, CSV)")
    output_path: Path | None = Field(None, description="Archivo producido (CSV, cache, raw)")
    status: str = Field(
        "ok",
        description="ok | cache | skipped | error | schema_invalid | llm_failed | llm_cost_cap",
    )
    extracted_date: datetime = Field(description="Momento (UTC) del procesamiento")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Detalles: endpoint, familia, norma_id, error..."
    )
    cost_usd: float = Field(0.0, description="Costo LLM de este ítem (0 si no hubo llamada)")


class GacetaNorma(BaseModel):
    """Registro extraído de un PDF de norma de la Gaceta por el LLM."""

    model_config = {"extra": "ignore"}

    norma_id: str = Field(description="ID de la norma (se sobreescribe con el del path)")
    edicion: str = Field(description="Edición de la Gaceta (se sobreescribe con la del path)")
    instrumento: str | None = Field(
        None, description="Tipo de instrumento: Decreto Supremo, Resolución Ministerial, Ley, ..."
    )
    entidad_emisora: str | None = Field(None, description="Entidad que emite la norma")
    fecha_publicacion: str | None = Field(None, description="Fecha de publicación (YYYY-MM-DD)")
    fecha_vigencia: str | None = Field(None, description="Fecha de vigencia (YYYY-MM-DD)")
    monto_usd: float | None = Field(
        None,
        description="Monto explícito en dólares americanos (número plano); null si no hay o es otra moneda",
    )
    moneda: str | None = Field(None, description="USD | BOB | null — moneda del monto mencionado")
    resumen_1l: str | None = Field(None, description="Resumen en una línea de qué trata la norma")
    tags: list[str] = Field(default_factory=list, description="Tags FX aplicables de TAGS_FX")
    relevancia_fx: str = Field(
        "nula", description="alta | media | baja | nula — impacto esperado en el tipo de cambio"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Certeza global de la extracción (0-1)")

    @field_validator("tags")
    @classmethod
    def _tags_conocidos(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in TAGS_FX]
        if unknown:
            raise ValueError(f"tags fuera del vocabulario: {unknown}")
        return v

    @field_validator("relevancia_fx")
    @classmethod
    def _relevancia_conocida(cls, v: str) -> str:
        if v not in RELEVANCIA_FX:
            raise ValueError(f"relevancia_fx inválida: {v!r} (usar alta|media|baja|nula)")
        return v

    @model_validator(mode="after")
    def _monto_coherente(self) -> "GacetaNorma":
        if self.monto_usd is None and self.moneda == "USD":
            raise ValueError("moneda=USD pero monto_usd es null")
        if self.monto_usd is not None and self.monto_usd < 0:
            raise ValueError("monto_usd negativo")
        return self
