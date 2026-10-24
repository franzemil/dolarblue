"""Indexers finales: un archivo por rama a su resolución temporal nativa.

- MarketHourlyIndexer: chart_6m (horario nativo) -> data/processed/market_hourly.csv
- GacetaDailyIndexer: gaceta_llm.csv -> data/processed/gaceta_daily.csv

El cruce mercado x eventos se hace aguas abajo (merge_asof / resample): el
pipeline no fuerza una grilla común y no pierde la resolución de ninguna fuente.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from processors.results import ProcessResult
from processors.stage import Stage
from processors.taxonomy import RELEVANCIA_FX, TAGS_FX

FWD_HORIZONS: tuple[int, ...] = (1, 6, 24, 72, 168)


class MarketHourlyIndexer(Stage):
    """Serie horaria del paralelo desde chart_6m (sin remuestreo)."""

    def __init__(self, data_dir: Path, dry_run: bool = False):
        self.data_dir = Path(data_dir)
        self.out = self.data_dir / "processed" / "market_hourly.csv"
        self.dry_run = dry_run

    def transform(self, inputs=None) -> ProcessResult:
        src = self.data_dir / "normalized" / "chart_6m.csv"
        if not src.exists():
            raise FileNotFoundError(f"falta {src}: corré JsonProcessor (rama dolarblue) primero")
        df = pd.read_csv(src).sort_values("t").drop_duplicates("t")
        df["fecha"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["blue_mid"] = (df["bb"] + df["bs"]) / 2
        df["oficial_mid"] = (df["ob"] + df["os"]) / 2
        df["spread_blue"] = df["bs"] - df["bb"]
        df["brecha_pct"] = (df["blue_mid"] - df["oficial_mid"]) / df["oficial_mid"] * 100
        df["ret_blue_1h"] = df["blue_mid"].pct_change(fill_method=None) * 100
        df["vol_24h"] = df["ret_blue_1h"].rolling(24).std()
        for h in FWD_HORIZONS:
            df[f"fwd_{h}h"] = df["blue_mid"].shift(-h) / df["blue_mid"] - 1

        cols = [
            "fecha",
            "t",
            "bb",
            "bs",
            "ob",
            "os",
            "blue_mid",
            "oficial_mid",
            "spread_blue",
            "brecha_pct",
            "ret_blue_1h",
            "vol_24h",
            *(f"fwd_{h}h" for h in FWD_HORIZONS),
        ]
        df = df[cols]
        numeric = df.select_dtypes(include="number").columns
        df[numeric] = df[numeric].round(4)
        if not self.dry_run:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.out, index=False)
        return ProcessResult(
            input_path=src,
            output_path=self.out,
            status="ok",
            extracted_date=datetime.now(timezone.utc),
            meta={
                "rows": len(df),
                "desde": str(df["fecha"].min()),
                "hasta": str(df["fecha"].max()),
            },
        )


class EventWindowIndexer(Stage):
    """Ventanas -5..+5 días alrededor de cada fecha de publicación de la Gaceta.

    Una fila por (fecha de publicación, offset) con el agregado diario del
    mercado y las features del evento: insumo del event study con OLS.
    """

    OFFSETS: tuple[int, ...] = tuple(range(-5, 6))

    def __init__(self, data_dir: Path, dry_run: bool = False):
        self.data_dir = Path(data_dir)
        self.out = self.data_dir / "processed" / "gaceta_event_windows.csv"
        self.dry_run = dry_run

    def transform(self, inputs=None) -> ProcessResult:
        market = self._daily_market()
        events = self._event_days()
        rows: list[dict] = []
        for fecha, event in events.iterrows():
            for offset in self.OFFSETS:
                day = fecha + pd.Timedelta(days=int(offset))
                row: dict = {
                    "fecha_publicacion": fecha.date().isoformat(),
                    "offset": offset,
                    "fecha": day.date().isoformat(),
                }
                if day in market.index:
                    row.update(market.loc[day].to_dict())
                row.update({col: event[col] for col in events.columns})
                rows.append(row)
        df = pd.DataFrame(rows)
        if not self.dry_run and not df.empty:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.out, index=False)
        return ProcessResult(
            output_path=self.out,
            status="ok",
            extracted_date=datetime.now(timezone.utc),
            meta={
                "publicaciones": int(len(events)),
                "filas": int(len(df)),
                "offsets": len(self.OFFSETS),
            },
        )

    def _daily_market(self) -> pd.DataFrame:
        src = self.data_dir / "processed" / "market_hourly.csv"
        if not src.exists():
            raise FileNotFoundError(f"falta {src}: corré la rama dolarblue primero")
        df = pd.read_csv(src, usecols=["fecha", "blue_mid", "brecha_pct", "vol_24h"])
        df["fecha"] = pd.to_datetime(df["fecha"], utc=True).dt.tz_localize(None).dt.normalize()
        first = df.groupby("fecha")["blue_mid"].first()
        last = df.groupby("fecha")["blue_mid"].last()
        daily = df.groupby("fecha").agg(
            blue_mid=("blue_mid", "mean"),
            brecha_pct=("brecha_pct", "mean"),
            vol_24h=("vol_24h", "last"),
            horas=("blue_mid", "count"),
        )
        daily["ret_blue_dia"] = last / first - 1
        daily.index.name = "fecha"
        return daily

    def _event_days(self) -> pd.DataFrame:
        src = self.data_dir / "processed" / "gaceta_daily.csv"
        if not src.exists():
            raise FileNotFoundError(f"falta {src}: corré la rama Gaceta primero")
        df = pd.read_csv(src)
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
        df = df[df["n_docs_total"] > 0].set_index("fecha")
        return df


class GacetaDailyIndexer(Stage):
    """Eventos de Gaceta agregados por día calendario (grilla completa)."""

    def __init__(self, data_dir: Path, backend: str = "llm", dry_run: bool = False):
        if backend not in ("llm", "keywords"):
            raise ValueError(f"backend inválido: {backend!r}")
        self.data_dir = Path(data_dir)
        self.backend = backend
        self.out = self.data_dir / "processed" / "gaceta_daily.csv"
        self.dry_run = dry_run

    def transform(self, inputs=None) -> ProcessResult:
        idx = self._calendar()
        if self.backend == "llm":
            daily = self._from_llm(idx)
        else:
            daily = self._from_keywords(idx)
        if not self.dry_run:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            numeric = daily.select_dtypes(include="number").columns
            daily[numeric] = daily[numeric].round(4)
            daily.to_csv(self.out)
        return ProcessResult(
            output_path=self.out,
            status="ok",
            extracted_date=datetime.now(timezone.utc),
            meta={
                "backend": self.backend,
                "rows": len(daily),
                "docs": int(daily["n_docs_total"].sum()) if len(daily) else 0,
            },
        )

    # ------------------------------ helpers ----------------------------------
    def _calendar(self) -> pd.DatetimeIndex:
        fechas = []
        chart = self.data_dir / "normalized" / "chart_6m.csv"
        if chart.exists():
            t = pd.read_csv(chart, usecols=["t"])["t"]
            fechas.append(pd.to_datetime(t, unit="s", utc=True).dt.tz_localize(None))
        g = self.data_dir / "processed" / "gaceta_llm.csv"
        if g.exists():
            f = pd.read_csv(g, usecols=["fecha_publicacion"])["fecha_publicacion"].dropna()
            fechas.append(pd.to_datetime(f, errors="coerce").dropna())
        series = pd.concat(fechas) if fechas else pd.Series(dtype="datetime64[ns]")
        if series.empty:
            return pd.DatetimeIndex([])
        return pd.date_range(
            series.min().normalize(), series.max().normalize(), freq="D", name="fecha"
        )

    def _from_llm(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        src = self.data_dir / "processed" / "gaceta_llm.csv"
        tag_cols = [f"n_docs_{tag}_llm" for tag in TAGS_FX]
        cols = ["n_docs_total", *tag_cols, "monto_usd_sum", "confidence_mean", "relevancia_max"]
        if not src.exists() or idx.empty:
            return pd.DataFrame(0, index=idx, columns=cols)

        df = pd.read_csv(src, dtype={"norma_id": str})
        df = df[df["status"].isin(("ok", "cache"))].copy()
        df["fecha"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce").dt.normalize()
        df = df[df["fecha"].notna()]

        daily = pd.DataFrame(index=idx)
        daily["n_docs_total"] = df.groupby("fecha")["norma_id"].count()
        tags = df.assign(tag=df["tags"].fillna("").str.split("|")).explode("tag")
        tags = tags[tags["tag"].isin(TAGS_FX)]
        for tag in TAGS_FX:
            daily[f"n_docs_{tag}_llm"] = tags[tags["tag"] == tag].groupby("fecha")["norma_id"].count()
        daily["monto_usd_sum"] = df.groupby("fecha")["monto_usd"].sum(min_count=0)
        daily["confidence_mean"] = df.groupby("fecha")["confidence"].mean()
        relevancia = (
            df.assign(r=df["relevancia_fx"].map(RELEVANCIA_FX).fillna(0))
            .groupby("fecha")["r"]
            .max()
        )
        daily["relevancia_max"] = relevancia

        daily = daily[cols]
        daily["n_docs_total"] = daily["n_docs_total"].fillna(0).astype(int)
        for col in tag_cols:
            daily[col] = daily[col].fillna(0).astype(int)
        daily["monto_usd_sum"] = daily["monto_usd_sum"].fillna(0.0)
        daily["relevancia_max"] = daily["relevancia_max"].fillna(0).astype(int)
        return daily

    def _from_keywords(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        gaceta = self.data_dir / "gaceta"
        docs_path = gaceta / "docs_etiquetados.csv"
        tags_path = gaceta / "doc_tags.csv"
        tag_cols = [f"n_docs_{tag}" for tag in TAGS_FX]
        cols = ["n_docs_total", *tag_cols, "monto_usd_sum", "confidence_mean", "relevancia_max"]
        if not docs_path.exists() or not tags_path.exists() or idx.empty:
            return pd.DataFrame(0, index=idx, columns=cols)

        docs = pd.read_csv(docs_path, dtype={"norma_id": str})
        tags = pd.read_csv(tags_path, dtype={"norma_id": str})
        docs["fecha"] = pd.to_datetime(docs["fecha_edicion"], errors="coerce").dt.normalize()
        tags = tags.merge(
            docs[["norma_id", "fecha"]].drop_duplicates("norma_id"), on="norma_id", how="left"
        )

        daily = pd.DataFrame(index=idx)
        daily["n_docs_total"] = docs.groupby("fecha")["norma_id"].count()
        for tag in TAGS_FX:
            sub = tags[tags["tag"] == tag]
            daily[f"n_docs_{tag}"] = sub.groupby("fecha")["norma_id"].count()
        daily["monto_usd_sum"] = 0.0
        daily["confidence_mean"] = pd.NA
        relevancia = (
            docs.assign(r=docs["relevancia_fx"].map(RELEVANCIA_FX).fillna(0))
            .groupby("fecha")["r"]
            .max()
        )
        daily["relevancia_max"] = relevancia

        daily = daily[cols]
        daily["n_docs_total"] = daily["n_docs_total"].fillna(0).astype(int)
        for col in tag_cols:
            daily[col] = daily[col].fillna(0).astype(int)
        daily["relevancia_max"] = daily["relevancia_max"].fillna(0).astype(int)
        return daily
