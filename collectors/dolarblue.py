"""Collector de la API Dolar Blue Bolivia.

Un ``DolarBlueCollector`` itera los ``EndpointSpec`` de collectors/endpoints.py
y por cada URL construye un ``_EndpointCollector`` (ApiCollector) con el modelo
Pydantic de su familia: fetch -> unwrap envelope -> normalización tipada ->
snapshot crudo + tabla normalizada en data/normalized/.

Salidas (compatibles con la versión anterior):
- data/raw/snapshot_<UTC>.json            (respuestas crudas por endpoint)
- data/sources/<endpoint>.json            (payload interno de cada endpoint ok)
- data/normalized/<familia>.csv           (filas tipadas + procedencia)
- export streaming -> data/history_full.csv
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pydantic import ValidationError

from collectors.base import ApiCollector, DataCollector, utcnow
from collectors.endpoints import (
    BASE_URL,
    EXPORT_ENDPOINTS,
    PRIVATE_ENDPOINTS,
    PUBLIC_ENDPOINTS,
    WINDOW_DAYS,
    EndpointSpec,
)
from collectors.models import FAMILY_MODELS, RawDict, TypedResult
from collectors.normalizers import NormalizeError, family_records, normalize, unwrap_envelope
from collectors.writers import write_tables


def _window_params() -> dict:
    now = int(time.time())
    return {"from_ts": now - WINDOW_DAYS * 86400, "to_ts": now}


class _EndpointCollector(ApiCollector):
    """ApiCollector concreto para UNA URL de la API dolarblue.

    __init__ recibe sus dependencias (spec, modelo de familia, sesión) y
    collect() devuelve el TypedResult tipado; el envelope crudo viaja en
    meta["raw_envelope"] para reconstruir el snapshot.
    """

    def __init__(self, spec: EndpointSpec, out_dir: Path, session: requests.Session):
        super().__init__(FAMILY_MODELS.get(spec.family, RawDict), out_dir, session)
        self.spec = spec

    def _path(self) -> str:
        if not (self.spec.window and "from_ts" not in self.spec.path):
            return self.spec.path
        sep = "&" if "?" in self.spec.path else "?"
        qs = "&".join(f"{k}={v}" for k, v in _window_params().items())
        return f"{self.spec.path}{sep}{qs}"

    def collect(self) -> TypedResult:  # type: ignore[override]
        url = f"{BASE_URL}{self._path()}"
        extracted = utcnow()
        meta: dict = {"endpoint": self.spec.name, "familia": self.spec.family}
        try:
            r = self.session.get(url, timeout=60)
        except requests.RequestException as e:
            meta["_error"] = str(e)
            return TypedResult(url=url, extracted_date=extracted, status="error", meta=meta)
        ok = r.ok and "json" in r.headers.get("content-type", "")
        body = r.json() if ok else {"_http_status": r.status_code, "_body": r.text[:300]}
        envelope = {"url": url, "status": r.status_code, "ok": ok, "data": body}
        inner, published = unwrap_envelope(body if ok else {})

        payload = None
        status = "ok" if ok else "error"
        if ok:
            try:
                payload = normalize(self.spec.family, inner, self.spec.name)
            except (NormalizeError, ValueError, ValidationError) as e:
                status = "norm_error"
                meta["norm_error"] = f"{type(e).__name__}: {e}"

        meta.update(
            http_status=r.status_code,
            cached=body.get("cached") if isinstance(body, dict) else None,
            raw_envelope=envelope,
        )
        return TypedResult[self.model](
            url=url, extracted_date=extracted, published_date=published,
            status=status, meta=meta, payload=payload,
        )


class DolarBlueCollector(DataCollector):
    """Orquesta un collector por URL: snapshots crudos + tablas normalizadas."""

    def __init__(
        self,
        out_dir: Path,
        api_key: str | None = None,
        session: requests.Session | None = None,
        delay: float = 0.3,
    ):
        super().__init__(out_dir, session)
        self.api_key = api_key or os.environ.get("DBB_API_KEY")
        self.delay = delay
        self.session.headers.update({"Accept": "application/json", "User-Agent": "ds-project/0.1"})
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key
        self.norm_dir = self.out_dir / "normalized"
        self.norm_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> list[TypedResult]:
        results: list[TypedResult] = []
        for spec in PUBLIC_ENDPOINTS:
            results.append(_EndpointCollector(spec, self.out_dir, self.session).collect())
            time.sleep(self.delay)
        if self.api_key:
            for spec in PRIVATE_ENDPOINTS:
                results.append(_EndpointCollector(spec, self.out_dir, self.session).collect())
                time.sleep(self.delay)
        self._write_snapshot(results)
        self._write_sources(results)
        self._write_normalized(results)
        self._run_exports()
        return results

    # ------------------------------ salidas ---------------------------------

    def _write_snapshot(self, results: list[TypedResult]) -> Path:
        """data/raw/snapshot_<UTC>.json — mismo formato de siempre."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_dir = self.out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"snapshot_{stamp}.json"
        snapshot = {
            res.meta["endpoint"]: res.meta["raw_envelope"]
            for res in results
            if "raw_envelope" in res.meta
        }
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _write_sources(self, results: list[TypedResult]) -> None:
        """data/sources/<endpoint>.json — payload interno de cada endpoint ok."""
        src_dir = self.out_dir / "sources"
        src_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            envelope = res.meta.get("raw_envelope", {})
            if envelope.get("ok"):
                (src_dir / f"{res.meta['endpoint']}.json").write_text(
                    json.dumps(envelope["data"], ensure_ascii=False, indent=2), encoding="utf-8"
                )

    def _write_normalized(self, results: list[TypedResult]) -> list[Path]:
        """data/normalized/<familia>.csv — filas tipadas con procedencia."""
        tables: dict[str, list[dict]] = {}
        for res in results:
            if res.status != "ok" or res.payload is None:
                continue
            table, rows = family_records(res.meta["familia"], res.payload, res.url, res.extracted_date)
            if table:
                tables.setdefault(table, []).extend(rows)
        return write_tables(tables, self.norm_dir)

    def _run_exports(self) -> list[Path]:
        files = []
        for spec in EXPORT_ENDPOINTS:
            path = self._export_streaming(spec, self.out_dir / "history_full")
            if path:
                files.append(path)
        return files

    def _export_streaming(self, spec: EndpointSpec, dest: Path) -> Path | None:
        sep = "&" if "?" in spec.path else "?"
        qs = "&".join(f"{k}={v}" for k, v in _window_params().items())
        try:
            with self.session.get(f"{BASE_URL}{spec.path}{sep}{qs}", stream=True, timeout=180) as r:
                if not r.ok:
                    print(f"{spec.path} -> {r.status_code}: {r.text[:200]}")
                    return None
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            final = dest.with_suffix(".csv" if b"," in tmp.open("rb").readline() else ".json")
            tmp.rename(final)
            return final
        except requests.RequestException as e:
            print(f"{spec.path} error: {e}")
            return None
