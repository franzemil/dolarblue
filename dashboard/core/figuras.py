"""Figuras interactivas (Plotly) del dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import config as C

QUIEBRE = pd.Timestamp("2026-06-27")


def _layout(fig: go.Figure, height: int = 380, ytitle: str | None = None,
            xtitle: str | None = None, leyenda: bool = True,
            hover: str = "x unified") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=12, t=26, b=10),
        hovermode=hover,
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
                       "Arial, sans-serif", size=12, color=C.DARK),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        showlegend=leyenda,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, title_text=xtitle,
                     linecolor=C.BORDER, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, title_text=ytitle,
                     tickfont=dict(size=11))
    return fig


def _quiebre(fig: go.Figure) -> None:
    fig.add_vline(x=QUIEBRE, line_width=1, line_dash="dot",
                  line_color=C.GRAY)
    fig.add_annotation(x=QUIEBRE, y=1, yref="paper", yanchor="bottom",
                       text="Devaluación del oficial", showarrow=False,
                       font=dict(size=10, color=C.GRAY), xanchor="left",
                       xshift=4)


# ── series del mercado ─────────────────────────────────────────────────────
def serie_nivel(grilla: pd.DataFrame, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grilla.index, y=grilla.blue_mid, name="Blue (medio)",
        line=dict(color=C.BLUE, width=1.4)))
    fig.add_trace(go.Scatter(
        x=grilla.index, y=grilla.oficial_mid, name="Oficial (BCB)",
        line=dict(color=C.ORANGE, width=1.2)))
    _layout(fig, height, ytitle="Bs por USD")
    _quiebre(fig)
    return fig


def serie_brecha(grilla: pd.DataFrame, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=grilla.index, y=grilla.brecha_pct, name="Brecha",
        line=dict(color=C.RED, width=1.2),
        fill="tozeroy", fillcolor="rgba(239,68,68,.08)"))
    _layout(fig, height, ytitle="Brecha (%)")
    _quiebre(fig)
    return fig


def crudo_vs_eda(crudo: pd.DataFrame, grilla: pd.DataFrame,
                 height: int = 380) -> go.Figure:
    """Dos snapshots crudos (compra/venta) vs la serie EDA (medio)."""
    fin = grilla.index.max() + pd.Timedelta(hours=1)
    inicio = fin - pd.Timedelta(hours=168)
    sub = grilla.loc[inicio:fin - pd.Timedelta(hours=1)]
    raw = crudo[(crudo.fecha >= inicio) & (crudo.fecha < fin)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=("Nivel del blue (Bs/USD)",
                                        "Brecha cambiaria (%)"))
    # Panel de nivel: compra y venta crudas + medio EDA
    fig.add_trace(go.Scatter(x=raw.fecha, y=raw.blue_buy, mode="markers",
                             name="Crudo compra",
                             marker=dict(color="#94A3B8", size=3, opacity=.7)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=raw.fecha, y=raw.blue_sell, mode="markers",
                             name="Crudo venta",
                             marker=dict(color="#475569", size=3, opacity=.7)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub.blue_mid, name="EDA (medio)",
                             line=dict(color=C.BLUE, width=1.6)), row=1, col=1)
    # Panel de brecha: compra y venta crudas + brecha EDA
    fig.add_trace(go.Scatter(x=raw.fecha, y=raw.brecha_buy, mode="markers",
                             name="Brecha compra",
                             marker=dict(color="#94A3B8", size=3, opacity=.7),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=raw.fecha, y=raw.brecha_sell, mode="markers",
                             name="Brecha venta",
                             marker=dict(color="#475569", size=3, opacity=.7),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub.brecha_pct, name="Brecha EDA",
                             showlegend=False,
                             line=dict(color=C.RED, width=1.6)), row=2, col=1)
    _layout(fig, height, hover="x unified")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
    return fig

# ── predicciones ───────────────────────────────────────────────────────────
def pred_vs_real(preds: pd.DataFrame, target: str, modelo: str,
                 height: int = 380) -> go.Figure:
    sub = preds[(preds.target == target) & (preds.modelo == modelo)]
    sub = sub.sort_values("fecha")
    color = C.COLOR_TARGET.get(target, C.BLUE)
    fig = go.Figure()
    if sub.lo.notna().any():
        fig.add_trace(go.Scatter(x=sub.fecha, y=sub.hi, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=sub.fecha, y=sub.lo, line=dict(width=0), fill="tonexty",
            fillcolor="rgba(37,99,235,.12)", name="IC 95 %",
            hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sub.fecha, y=sub.real, name="Observado",
                             line=dict(color=C.DARK, width=1.6)))
    fig.add_trace(go.Scatter(x=sub.fecha, y=sub.pred, name="Predicción",
                             line=dict(color=color, width=1.8)))
    fig.add_trace(go.Scatter(x=sub.fecha, y=sub.naive, name="Naïve",
                             line=dict(color=C.GRAY, width=1.2, dash="dash")))
    _layout(fig, height, ytitle=("Bs/USD" if target == "blue_mid"
                                 else "Brecha (%)"))
    return fig


def error_modelo(preds: pd.DataFrame, target: str, modelo: str,
                 height: int = 240) -> go.Figure:
    sub = preds[(preds.target == target) & (preds.modelo == modelo)].copy()
    sub["error"] = sub.pred - sub.real
    sesgo = float(sub.error.mean())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub.fecha, y=sub.error, name="Error",
                         marker_color=np.where(sub.error >= 0, C.BLUE, C.RED),
                         opacity=.75))
    fig.add_hline(y=sesgo, line_dash="dash", line_color=C.ORANGE,
                  annotation_text=f"Sesgo medio {sesgo:+.3f}",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=C.ORANGE))
    fig.add_hline(y=0, line_color=C.BORDER)
    _layout(fig, height, ytitle="Predicción − observado")
    return fig


def ratio_barras(comp: pd.DataFrame, height: int = 380) -> go.Figure:
    """RMSE ÷ naïve por target, horizonte y modelo."""
    comp = comp.copy()
    comp["etiqueta"] = (comp["Nivel"] + " · " + comp["Horizonte"])
    fig = go.Figure()
    for modelo, color in ((comp["Modelo"].iloc[0], C.BLUE),
                          (comp["Modelo"].iloc[1], C.ORANGE)):
        sub = comp[comp.Modelo == modelo]
        fig.add_trace(go.Bar(
            x=sub.etiqueta, y=sub["RMSE ÷ naïve"], name=modelo,
            marker_color=color,
            text=[f"{v:.2f}".replace(".", ",") for v in sub["RMSE ÷ naïve"]],
            textposition="outside", cliponaxis=False))
    fig.add_hline(y=1, line_dash="dash", line_color=C.GRAY)
    _layout(fig, height, ytitle="RMSE ÷ naïve", leyenda=True)
    fig.update_layout(barmode="group")
    return fig


def aic_barras(grid: pd.DataFrame, target: str, height: int = 420) -> go.Figure:
    col = "blue_sell" if target == "blue_mid" else "brecha_pct"
    d = grid.copy()
    d["modelo"] = d.apply(lambda r: f"({int(r.p)},{int(r.d)},{int(r.q)}) "
                                    f"{'c' if r.trend == 'c' else 'n'}", axis=1)
    d = d.sort_values(col)
    minimo = d[col].min()
    colores = [C.GREEN if v == minimo else "#CBD5E1" for v in d[col]]
    fig = go.Figure(go.Bar(
        x=d[col], y=d.modelo, orientation="h", marker_color=colores,
        text=[f"{v:,.1f}".replace(",", ".") for v in d[col]],
        textposition="outside", cliponaxis=False))
    _layout(fig, height, xtitle="AIC (menor es mejor)", hover="closest")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return fig


# ── señal direccional ──────────────────────────────────────────────────────
def confusion_heatmap(cm: dict, height: int = 330) -> go.Figure:
    z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
    total = max(sum(z[0]) + sum(z[1]), 1)
    texto = [[f"{cm['tp']}<br>{cm['tp'] / total:.0%}",
              f"{cm['fn']}<br>{cm['fn'] / total:.0%}"],
             [f"{cm['fp']}<br>{cm['fp'] / total:.0%}",
              f"{cm['tn']}<br>{cm['tn'] / total:.0%}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predice sube", "Predice baja"],
        y=["Real sube", "Real baja"], text=texto, texttemplate="%{text}",
        textfont=dict(size=15, color=C.DARK),
        colorscale=[[0, "#F1F5F9"], [1, "#BFDBFE"]],
        showscale=False, hoverongaps=False))
    _layout(fig, height, hover="closest", leyenda=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return fig


def roc_horizonte(roc: pd.DataFrame, horizonte: int,
                  height: int = 340) -> go.Figure:
    d = roc[roc.horizonte == horizonte]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Nivel del blue", "Brecha cambiaria"))
    for j, target in enumerate(C.TARGETS, start=1):
        sub = d[d.target == target]
        aucs = {}
        for modelo, color in (("ARIMA", C.BLUE), ("RL múltiple", C.ORANGE)):
            s = sub[sub.modelo.str.contains(modelo, case=False)]
            if s.empty:
                continue
            aucs[modelo] = float(s.auc.iloc[0])
            fig.add_trace(go.Scatter(
                x=s.fpr, y=s.tpr, mode="lines", name=modelo,
                line=dict(color=color, width=2), legendgroup=modelo,
                showlegend=(j == 1)), row=1, col=j)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=C.GRAY, dash="dot", width=1),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=j)
        partes = [f"{m} {aucs[m]:.2f}".replace(".", ",")
                  for m in ("ARIMA", "RL múltiple") if m in aucs]
        fig.layout.annotations[j - 1].text = (
            ("Nivel del blue" if target == "blue_mid" else "Brecha cambiaria")
            + "<br><sup>AUC " + " · ".join(partes) + "</sup>")
        fig.update_xaxes(title_text="Tasa de falsos positivos", row=1, col=j)
    fig.update_yaxes(title_text="Tasa de verdaderos positivos", row=1, col=1)
    _layout(fig, height, hover="closest")
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    return fig


# ── Gaceta ─────────────────────────────────────────────────────────────────
def eventos_barh(ev: pd.DataFrame, horizonte: str = "5 h",
                 height: int = 400) -> go.Figure:
    d = ev.sort_values(horizonte)
    colores = [C.RED if v < 0 else C.BLUE for v in d[horizonte]]
    fig = go.Figure(go.Bar(
        x=d[horizonte], y=d["tag"], orientation="h", marker_color=colores,
        error_x=dict(type="data", array=d["ic95"], color=C.GRAY, thickness=1),
        text=[f"{v:+.2f}" for v in d[horizonte]], textposition="outside",
        cliponaxis=False))
    fig.add_vline(x=0, line_color=C.DARK, line_width=1)
    _layout(fig, height, xtitle="Exceso de retorno (pp)")
    fig.update_layout(showlegend=False)
    return fig


def spearman_heatmap(sp: pd.DataFrame, height: int = 400) -> go.Figure:
    z = sp[["t", "t1"]].values * 100
    texto = [[f"{v:.3f}".replace(".", ",") for v in fila] for fila in sp[["t", "t1"]].values]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Mismo día (t)", "Día siguiente (t+1)"], y=sp["tag"],
        text=texto, texttemplate="%{text}", colorscale="RdBu",
        zmid=0, zmin=-20, zmax=20,
        colorbar=dict(title="ρ ×100", thickness=12, len=.7)))
    _layout(fig, height, hover="closest", leyenda=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return fig


def z_hac(ols: pd.DataFrame, umbral: float = 1.96,
          height: int = 520) -> go.Figure:
    """|z| de cada coeficiente HAC en escala lineal.

    `brecha_lag1h` tiene un |z| ~155 y aplasta al resto; se deja fuera del eje
    con una anotación para que los demás coeficientes sean legibles. También se
    excluyen `const` y `sigma2`, que no son predictores.
    """
    d = ols[~ols["var"].isin(["const", "sigma2"])].copy()
    d["absz"] = d["z"].abs()
    fuera = d[d.absz > 10]
    d = d[d.absz <= 10].sort_values("absz").tail(20)
    colores = [C.BLUE if v >= umbral else "#CBD5E1" for v in d.absz]
    texto = [f"{v:.2f}".replace(".", ",") if v >= umbral else ""
             for v in d.absz]
    fig = go.Figure(go.Bar(
        x=d.absz, y=d["var"], orientation="h", marker_color=colores,
        text=texto, textposition="outside", cliponaxis=False))
    fig.add_vline(x=umbral, line_dash="dash", line_color=C.RED,
                  annotation_text="|z| = 1,96 (5 %)",
                  annotation_position="top",
                  annotation_font=dict(size=10, color=C.RED))
    if not fuera.empty:
        etiqueta = ", ".join(
            f"{r['var']} |z| = {r.absz:.1f}".replace(".", ",")
            for _, r in fuera.iterrows())
        fig.add_annotation(x=0.99, y=0.02, xref="paper", yref="paper",
                           xanchor="right", yanchor="bottom",
                           text=f"Fuera de escala: {etiqueta}",
                           showarrow=False,
                           font=dict(size=10, color=C.GRAY))
    _layout(fig, height, xtitle="|z|", hover="closest")
    fig.update_xaxes(range=[0, max(3.2, float(d.absz.max()) * 1.15)])
    fig.update_layout(showlegend=False)
    return fig
