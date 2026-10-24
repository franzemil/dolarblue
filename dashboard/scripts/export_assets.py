"""Exporta los artefactos que consume el dashboard (offline, una sola vez).

Genera `dashboard/assets/*.parquet` a partir del pipeline y de los mismos
módulos que usa el notebook (`models/`, `metrics/`). No se ejecuta al arrancar
la app.

Uso:
    .venv/bin/python dashboard/scripts/export_assets.py
    make dashboard-assets
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── rutas: dashboard/scripts/ -> raíz del proyecto ─────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
ASSETS = DASH / "assets"
MODELADO = ROOT / "modelado"

for _p in (ROOT, MODELADO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.arima import (seleccionar_por_horizonte,  # noqa: E402
                          arima_walkforward)
from models.regresion import (ajustar_ols_hac, ajustar_rl_multiple,  # noqa: E402
                              construir_features_horarias, grilla_horaria)

warnings.filterwarnings("ignore")

H = 168
HORIZONTES = (1, 24)
TARGETS = ("blue_mid", "brecha_pct")
OUT = ASSETS


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def cargar():
    mercado = pd.read_csv(ROOT / "data/processed/market_hourly.csv",
                          parse_dates=["fecha"])
    panel = pd.read_csv(MODELADO / "panel_diario.csv", parse_dates=["fecha_d"])
    gaceta = pd.read_csv(ROOT / "data/processed/gaceta_llm.csv",
                         parse_dates=["fecha_publicacion"])
    crudo = pd.read_csv(ROOT / "data/history_full.csv")
    crudo["fecha"] = pd.to_datetime(crudo["timestamp"], unit="s")
    crudo = crudo[(crudo.blue_sell > 0) & (crudo.official_sell > 0)].copy()
    crudo["brecha_snapshot"] = (crudo.blue_sell / crudo.official_sell - 1) * 100
    crudo = crudo[["fecha", "blue_sell", "brecha_snapshot"]]
    return mercado, panel, gaceta, crudo


def exportar_predicciones(mercado, panel):
    """Predicciones walk-forward de RL y ARIMA por horizonte y target."""
    hor = grilla_horaria(mercado)
    series = {c: hor[c].dropna() for c in TARGETS}
    filas, resumen = [], []

    for h in HORIZONTES:
        datos, cols = construir_features_horarias(mercado, panel, horizonte=h)
        train, test = datos.iloc[:-H], datos.iloc[-H:]

        for target in TARGETS:
            pred, naive = ajustar_rl_multiple(train, test, cols, target)
            real = series[target].reindex(pred.index)
            filas.append(pd.DataFrame({
                "fecha": pred.index, "horizonte": h, "target": target,
                "modelo": "RL múltiple", "real": real.values,
                "pred": pred.values,
                "naive": naive.reindex(pred.index).values,
                "lo": np.nan, "hi": np.nan,
            }))

            y = series[target]
            split = y.index[-(H + 1)]
            sel = seleccionar_por_horizonte(y, split, h, ventana_h=90 * 24,
                                            max_pq=2)
            p_ar, nv_ar, lo, hi = arima_walkforward(
                y, split, h, sel["order"], trend=sel["trend"])
            real_ar = y.reindex(p_ar.index)
            filas.append(pd.DataFrame({
                "fecha": p_ar.index, "horizonte": h, "target": target,
                "modelo": f"ARIMA ({h} pasos)", "real": real_ar.values,
                "pred": p_ar.values,
                "naive": nv_ar.reindex(p_ar.index).values,
                "lo": lo.reindex(p_ar.index).values,
                "hi": hi.reindex(p_ar.index).values,
            }))

            resumen.append({
                "horizonte": h, "target": target,
                "rmse_rl": round(rmse(real, pred), 4),
                "mae_rl": round(mae(real, pred), 4),
                "rmse_arima": round(rmse(real_ar, p_ar), 4),
                "mae_arima": round(mae(real_ar, p_ar), 4),
                "orden_arima": sel["modelo"],
            })
            print(f"  h={h:2d} · {target:11s} · RL RMSE {rmse(real, pred):.4f} "
                  f"· ARIMA {sel['modelo']} RMSE {rmse(real_ar, p_ar):.4f}")

    preds = pd.concat(filas, ignore_index=True)
    for h in HORIZONTES:
        preds[preds.horizonte == h].to_parquet(
            OUT / f"predicciones_h{h}.parquet", index=False)
    return preds, pd.DataFrame(resumen)


def exportar_ols(mercado, panel):
    """Coeficientes completos del OLS con errores HAC (h=1, brecha)."""
    datos, cols = construir_features_horarias(mercado, panel, horizonte=1)
    train = datos.iloc[:-H]
    fit, cols_ok = ajustar_ols_hac(train, cols, "brecha_pct", maxlags=24)
    ic = fit.conf_int()
    filas = [{
        "var": nombre, "coef": float(fit.params[nombre]),
        "ee": float(fit.bse[nombre]), "z": float(fit.tvalues[nombre]),
        "p": float(fit.pvalues[nombre]),
        "lo": float(ic.loc[nombre].iloc[0]),
        "hi": float(ic.loc[nombre].iloc[1]),
    } for nombre in fit.params.index]
    out = pd.DataFrame(filas)
    out.to_parquet(OUT / "ols_hac.parquet", index=False)
    print(f"  OLS-HAC · {len(cols_ok)} predictores · R² {fit.rsquared:.4f} "
          f"· n {int(fit.nobs)}")
    return out


def exportar_roc(preds):
    """Curvas ROC de la señal direccional (misma lógica que la celda 5.1)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    def signo(x, eps=0.0):
        return pd.Series(np.select([x > eps, x < -eps], [1, -1], 0),
                         index=x.index)

    filas = []
    for (h, target), grupo in preds.groupby(["horizonte", "target"]):
        base = grupo[grupo.modelo == "RL múltiple"].set_index("fecha")
        for modelo, sub in grupo.groupby("modelo"):
            sub = sub.set_index("fecha").reindex(base.index)
            real_dir = signo(sub.real - sub.naive)
            score = sub.pred - sub.naive
            ok = (real_dir != 0) & score.notna() & (score != 0)
            y_up = (real_dir[ok] == 1).astype(int)
            fpr, tpr, _ = roc_curve(y_up, score[ok])
            auc = float(roc_auc_score(y_up, score[ok]))
            for f, t in zip(fpr, tpr):
                filas.append({"horizonte": int(h), "target": target,
                              "modelo": modelo, "fpr": float(f),
                              "tpr": float(t), "auc": round(auc, 3)})
    roc = pd.DataFrame(filas)
    roc.to_parquet(OUT / "roc_curvas.parquet", index=False)
    return roc


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"→ exportando assets en {OUT.relative_to(ROOT)}")
    mercado, panel, gaceta, crudo = cargar()

    mercado.to_parquet(OUT / "mercado.parquet", index=False)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    gaceta.to_parquet(OUT / "gaceta.parquet", index=False)
    crudo.to_parquet(OUT / "crudo.parquet", index=False)

    preds, resumen = exportar_predicciones(mercado, panel)
    roc = exportar_roc(preds)
    ols = exportar_ols(mercado, panel)

    R = json.loads((MODELADO / "metricas_reporte.json").read_text())
    for _, r in resumen.iterrows():
        ph = R["por_horizonte"][str(r.horizonte)]
        assert abs(r.rmse_rl - ph["rl"][r.target]["rmse"]) < 1e-2, r.to_dict()
        assert abs(r.rmse_arima - ph["arima"][r.target]["rmse"]) < 1e-2, \
            r.to_dict()
        assert r.orden_arima == ph["arima"][r.target]["modelo"], r.to_dict()
    b1 = float(ols.loc[ols["var"] == "brecha_lag1h", "coef"].iloc[0])
    assert abs(b1 - R["ols_hac"]["coeficientes"][0]["coef"]) < 5e-3, b1
    print("✔ predicciones, órdenes y coef HAC coinciden con el JSON")

    hor = grilla_horaria(mercado)
    manifest = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_desde": str(hor.index.min()), "datos_hasta": str(hor.index.max()),
        "horas": int(len(hor)), "panel_dias": int(len(panel)),
        "h_test": H, "horizontes": list(HORIZONTES),
        "filas": {
            "mercado": int(len(mercado)), "panel": int(len(panel)),
            "gaceta": int(len(gaceta)), "crudo": int(len(crudo)),
            "predicciones": int(len(preds)), "roc": int(len(roc)),
            "ols_hac": int(len(ols)),
        },
        "sha256": {p.name: _sha(p) for p in sorted(OUT.glob("*.parquet"))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  ensure_ascii=False))
    print(f"✔ assets escritos en {time.time() - t0:.1f}s")
    print(json.dumps(manifest["filas"], indent=2))


if __name__ == "__main__":
    main()
