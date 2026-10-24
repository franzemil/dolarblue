"""Collector de la Gaceta Oficial de Bolivia (FileCollector).

Descarga los PDFs de las normas de las ediciones publicadas en los últimos N
días para un tipo de edición:
    10 = Especial · 11 = Propiedad Intelectual · 13 = Normal (diaria)

Cada norma descargada produce un ``FileResult`` (hijo de TypedResult) con:
url (página de descarga), extracted_date, published_date (fecha de edición),
path (PDF local) y meta {edicion, tipo, norma_id}.

Uso:
    from collectors.gaceta import GacetaCollector
    results = GacetaCollector(tipo=13, days=30).collect()
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from collectors.base import FileCollector, utcnow
from collectors.models import FileResult

BASE = "http://www.gacetaoficialdebolivia.gob.bo"
UA = {"User-Agent": "Mozilla/5.0 (research; ds-project)"}
CARD_RE = re.compile(
    rb'N.{0,4} de Edici.{0,4}n : <strong>([^<]+)</strong>.*?'
    rb'Tipo de Edici.{0,4}n : <strong>([^<]+)</strong>.*?'
    rb'Fecha de Publicaci.{0,4}n :\s*<strong>(\d{2}-\d{2}-\d{4})</strong>',
    re.S,
)
NORMA_RE = re.compile(rb"/normas/descargarNrms/(\d+)")
H6_RE = re.compile(rb"<h6><b>\s*(.*?)\s*</b></h6>", re.S)
SUMA_RE = re.compile(
    rb"<strong>\s*(\d{1,2}\s+DE\s+[A-Z]+\s+DE\s+\d{4}\s*\.-)\s*</strong>(.*?)</p>",
    re.S,
)


def _clean(fragment: bytes) -> str:
    import html as ihtml

    txt = re.sub(rb"<[^>]+>", b" ", fragment).decode("utf-8", "replace")
    return re.sub(r"\s+", " ", ihtml.unescape(txt)).strip()


class GacetaCollector(FileCollector):
    """Colecta ediciones y normas (PDFs) de la Gaceta. Reanudable por cache."""

    def __init__(
        self,
        tipo: int = 13,
        days: int = 122,
        max_pages: int = 25,
        out_dir: Path | None = None,
        session=None,
    ):
        super().__init__(out_dir or Path(__file__).parent.parent / "data" / "gaceta", session)
        self.session.headers.update(UA)
        self.tipo = tipo
        self.days = days
        self.max_pages = max_pages

    def collect(self) -> list[FileResult]:
        cutoff = datetime.now() - timedelta(days=self.days)
        print(f"buscando ediciones tipo {self.tipo} desde {cutoff.date()}")
        editions = self._listing_editions(cutoff)
        print(f"{len(editions)} ediciones en ventana: {[e[0] for e in editions]}")

        results: list[FileResult] = []
        for eid, fecha, tipo_txt in editions:
            results += self._scrape_edition(eid, fecha, tipo_txt)
            time.sleep(0.8)
        self._write_docs_csv(results)
        return results

    def _listing_editions(self, cutoff: datetime) -> list[tuple[str, str, str]]:
        """Itera páginas del listado; devuelve (id, fecha, tipo) en la ventana."""
        found, seen = [], set()
        for page in range(1, self.max_pages + 1):
            suffix = "" if page == 1 else f"/page:{page}"
            r = self._get(f"{BASE}/edicions/buscarEspeciales/{self.tipo}{suffix}")
            if r is None:
                print(f"listado {page}: sin respuesta, fin")
                break
            cards = [(m[0].decode(), m[2].decode(), m[1].decode()) for m in CARD_RE.findall(r.content)]
            fresh = [(e, f, t) for e, f, t in cards if e not in seen]
            if not fresh:
                break
            for eid, fecha, tipo_txt in fresh:
                seen.add(eid)
                if datetime.strptime(fecha, "%d-%m-%Y") >= cutoff:
                    found.append((eid, fecha, tipo_txt.strip()))
            youngest = min(datetime.strptime(f, "%d-%m-%Y") for _, f, _ in fresh)
            print(f"página {page}: {len(fresh)} ediciones (más antigua {youngest.date()})")
            if youngest < cutoff:
                break
            time.sleep(0.8)
        return found

    def _write_docs_csv(self, results: list[FileResult]) -> Path:
        """docs_gaceta.csv: metadatos + suma por norma (entrada del Tagger)."""
        import csv

        path = self.out_dir / "docs_gaceta.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["norma_id", "edicion", "fecha_edicion", "fecha_norma",
                        "titulo", "suma", "archivo", "estado"])
            for r in results:
                w.writerow([
                    r.meta.get("norma_id", ""),
                    r.meta.get("edicion", ""),
                    r.published_date.strftime("%Y-%m-%d") if r.published_date else "",
                    r.meta.get("fecha_norma", ""),
                    r.meta.get("titulo", ""),
                    r.meta.get("suma", ""),
                    r.path.name if r.path else "",
                    r.status,
                ])
        return path

    @staticmethod
    def _parse_cards(page: bytes) -> dict[int, dict]:
        """HTML de la edición -> {norma_id: {titulo, fecha_norma, suma}}."""
        cards: dict[int, dict] = {}
        titles = [(m.start(), _clean(m.group(1))) for m in H6_RE.finditer(page)]
        for idx, (pos, titulo) in enumerate(titles):
            block = page[pos:titles[idx + 1][0] if idx + 1 < len(titles) else len(page)]
            m_id = NORMA_RE.search(block)
            if not m_id:
                continue
            fecha_norma, suma = "", ""
            m_suma = SUMA_RE.search(block)
            if m_suma:
                fecha_norma, suma = _clean(m_suma.group(1)), _clean(m_suma.group(2))
            cards[int(m_id.group(1))] = {"titulo": titulo, "fecha_norma": fecha_norma, "suma": suma}
        return cards

    def _scrape_edition(self, eid: str, fecha: str, tipo_txt: str) -> list[FileResult]:
        """Una edición -> lista de FileResults, uno por norma."""
        published = datetime.strptime(fecha, "%d-%m-%Y") if fecha else None
        r = self._get(f"{BASE}/edicions/view/{eid}")
        if r is None or len(r.content) < 1000:
            return [FileResult(
                url=f"{BASE}/edicions/view/{eid}", extracted_date=utcnow(),
                published_date=published, path=None, status="detalle_vacio",
                meta={"edicion": eid, "tipo": tipo_txt},
            )]
        cards = self._parse_cards(r.content)
        results = []
        norma_ids = list(dict.fromkeys(NORMA_RE.findall(r.content)))
        ed_dir = self.out_dir / eid
        ed_dir.mkdir(parents=True, exist_ok=True)
        for nid in norma_ids:
            nid = nid.decode()
            card = cards.get(int(nid), {})
            dest, estado = self.download_file(
                f"{BASE}/normas/descargarNrms/{nid}", ed_dir / f"norma_{nid}.pdf", magic=b"%PDF"
            )
            results.append(FileResult(
                url=f"{BASE}/normas/descargarNrms/{nid}", extracted_date=utcnow(),
                published_date=published, path=dest, status=estado,
                meta={"edicion": eid, "tipo": tipo_txt, "norma_id": nid,
                      "titulo": card.get("titulo", ""), "fecha_norma": card.get("fecha_norma", ""),
                      "suma": card.get("suma", "")},
            ))
        return results
