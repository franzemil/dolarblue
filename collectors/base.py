"""Clases base de los colectores.

``DataCollector`` es la raíz abstracta: el ``__init__`` configura las
dependencias (rutas, sesión HTTP, credenciales) y ``collect()`` ejecuta la
colecta devolviendo ``TypedResult`` tipados. Las dos ramas abstractas cubren
los dos tipos de fuente:

- ``ApiCollector[T]``: APIs JSON; cada instancia concreta recibe su modelo
  Pydantic (``model``) para validar/normalizar la respuesta antes de guardar.
- ``FileCollector``: descargas de archivos (PDFs, CSVs) con reintentos,
  validación de contenido y cache reanudable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from collectors.models import TypedResult


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DataCollector(ABC):
    """Base abstracta de todo colector del proyecto."""

    def __init__(
        self,
        out_dir: Path,
        session: requests.Session | None = None,
        retries: int = 3,
        backoff: float = 1.5,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.retries = retries
        self.backoff = backoff

    @abstractmethod
    def collect(self) -> list[TypedResult]:
        """Ejecuta la colecta y devuelve resultados con procedencia tipada."""

    def _get(self, url: str, **kw) -> requests.Response | None:
        """GET con reintentos y backoff lineal; None si la fuente no responde."""
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=kw.pop("timeout", 60), **kw)
                if r.status_code == 200:
                    return r
            except requests.RequestException:
                pass
            time.sleep(self.backoff * (attempt + 1))
        return None


class ApiCollector(DataCollector, ABC):
    """Colector de un endpoint JSON. Cada URL usa su propia instancia con
    ``model`` = anotación Pydantic de su familia (``Quote``,
    ``list[Vela]``, ...) para validar/normalizar la respuesta y tipar el
    TypedResult como ``TypedResult[model]``."""

    def __init__(self, model: Any, out_dir: Path, session: requests.Session | None = None):
        super().__init__(out_dir, session)
        self.model = model


class FileCollector(DataCollector, ABC):
    """Colector de archivos: descarga binaria validada y cache reanudable."""

    def download_file(self, url: str, dest: Path, magic: bytes | None = None) -> tuple[Path | None, str]:
        """Descarga ``url`` a ``dest``. Devuelve (path, estado).

        estado: cache (ya existía) | ok | descarga_fallida.
        Si ``magic`` se indica, valida los primeros bytes (p. ej. b"%PDF").
        """
        if dest.exists() and dest.stat().st_size > 1000:
            return dest, "cache"
        r = self._get(url)
        if r is None or (magic and not r.content.startswith(magic)):
            return None, "descarga_fallida"
        dest.write_bytes(r.content)
        time.sleep(0.8)  # cortesía con el servidor fuente
        return dest, "ok"
