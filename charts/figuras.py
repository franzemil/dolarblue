"""Figuras del notebook, como funciones puras.

Cada función recibe los datos ya preparados y devuelve la figura sin guardar:
el llamador hace `fig.savefig(...)` con el nombre y dpi que quiera. Las
operaciones de render son idénticas a las celdas originales del notebook
`analisis_gaceta_fx.ipynb` (extracción fiel, sin cambios de estilo).

Nota: algunas figuras usan `rmse` (de metrics/) para armar las etiquetas,
igual que hacían las celdas.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from metrics import rmse

COLORES_ROC = {"Regresión lineal": "tab:orange", "ARIMA (1 pasos)": "tab:red",
               "ARIMA (24 pasos)": "tab:purple"}
COLORES_HORIZONTES = {"5 h": "#1d4ed8", "1 d": "#0891b2", "3 d": "#f59e0b", "5 d": "#b91c1c"}


def raw_vs_eda(crudo_semana, eda_hora_semana, *, etiqueta, etiqueta_eje):
    """Crudo (snapshots) -> EDA horario (grilla regular), ambos cada hora."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    PANELES = [(axes[0], "blue_sell", "blue_mid"),
               (axes[1], "brecha_snapshot", "brecha_pct")]
    for ax, col_crudo, col_eda_h in PANELES:
        ax.plot(crudo_semana.fecha, crudo_semana[col_crudo],
                color="#94a3b8", lw=0.5, alpha=0.8, label="crudo (snapshots)")
        ax.plot(eda_hora_semana.index, eda_hora_semana[col_eda_h],
                color="#0891b2", lw=1.4, marker="o", ms=2.5,
                label="EDA horario (grilla regular)")
        ax.set_title(etiqueta("blue_sell") if col_crudo == "blue_sell"
                     else etiqueta("brecha_pct"))
        ax.set_ylabel(etiqueta_eje("blue_mid" if col_crudo == "blue_sell"
                                   else "brecha_pct"))
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle("La ventana de prueba: crudo → EDA horario", fontsize=12)
    fig.tight_layout()
    return fig


def rl_test(te_h, pred_rl_h, naive_h, *, etiqueta, etiqueta_eje,
            horizonte=1, naive1_h=None):
    """Observado vs RL vs naïve, con un panel de error debajo de cada target.

    El panel de error (predicción − observado) es el que permite juzgar la
    cercanía: una línea que "se ve cerca" puede tener sesgo sistemático.
    La línea base dibujada es el naïve del mismo horizonte: el último valor
    observado en el origen del pronóstico.
    """
    fig, axes = plt.subplots(4, 1, figsize=(6.9, 8.6),
                             gridspec_kw={"height_ratios": [3, 1, 3, 1]},
                             sharex=True)
    for k, target in enumerate(["brecha_pct", "blue_mid"]):
        ax, axe = axes[2 * k], axes[2 * k + 1]
        real, pred = te_h[target], pred_rl_h[target]
        naive = naive_h[target]
        ax.plot(real.index, real, color="tab:green", lw=1, label="observado (prueba)")
        ax.plot(pred.index, pred, color="tab:orange", ls="--", lw=1.4,
                label=f"Regresión lineal h={horizonte} h "
                      f"(RMSE {rmse(real, pred):.3f})")
        ax.plot(naive.index, naive, color="tab:gray", ls=":", lw=1.1,
                label=f"naïve h={horizonte} h (RMSE {rmse(real, naive):.3f})")
        ax.set_title(etiqueta(target), fontsize=9.5)
        ax.set_ylabel(etiqueta_eje(target), fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.6)

        e_rl, e_nv = pred - real, naive - real
        axe.axhline(0, color="black", lw=0.8)
        axe.plot(e_rl.index, e_rl, color="tab:orange", ls="--", lw=1.1,
                 label=f"error RL (sesgo {e_rl.mean():+.3f})")
        axe.plot(e_nv.index, e_nv, color="tab:gray", ls=":", lw=1.1,
                 label=f"error naïve (sesgo {e_nv.mean():+.3f})")
        axe.set_ylabel("error", fontsize=8)
        axe.tick_params(labelsize=7.5)
        axe.legend(fontsize=6.5, loc="lower left", ncol=2)
        axe.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"Regresión lineal (sklearn) — pronóstico a {horizonte} h",
                 fontsize=10.5)
    fig.tight_layout()
    return fig


def arima_horizontes(datos, *, etiqueta, etiqueta_eje, horizontes=(1, 24)):
    """ARIMA frente al observado y al naïve, una fila por horizonte.

    `datos` es {horizonte: {target: (pred, real, naive, ci)}}. La columna
    izquierda es el nivel del blue y la derecha la brecha. A 24 h las líneas del
    modelo y del naïve casi coinciden: el naïve va discontinuo y por encima, de
    modo que se ven las dos.
    """
    objetivos = ["blue_mid", "brecha_pct"]
    fig, axes = plt.subplots(len(horizontes), len(objetivos),
                             figsize=(6.9, 2.55 * len(horizontes) + 0.9),
                             squeeze=False)
    for i, h in enumerate(horizontes):
        for j, target in enumerate(objetivos):
            ax = axes[i, j]
            pred, real, naive, ci = datos[h][target]
            ax.fill_between(ci.index, ci["lo"], ci["hi"], color="0.5",
                            alpha=0.13, zorder=1, label="IC95")
            ax.plot(real.index, real, color="tab:green", lw=0.9, zorder=2,
                    label="observado")
            ax.plot(pred.index, pred, color="tab:red", ls="-", lw=1.4, zorder=3,
                    label=f"ARIMA (RMSE {rmse(real, pred):.3f})")
            ax.plot(naive.index, naive, color="tab:blue", ls=(0, (4, 2)),
                    lw=1.5, zorder=4,
                    label=f"naïve (RMSE {rmse(real, naive):.3f})")
            ax.set_title(f"{etiqueta(target)} · horizonte {h} h", fontsize=9)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.set_ylabel(etiqueta_eje(target), fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)
            ax.grid(alpha=0.25, lw=0.55)
    fig.suptitle("ARIMA: pronóstico a una hora y a un día", fontsize=10.5)
    fig.tight_layout()
    return fig


def confusion_direccion(tabla, *, etiqueta, horizonte=1, modelos=None):
    """Una matriz 2x2 por target y modelo, para un horizonte dado.

    `tabla` es el DataFrame con columnas target, horizonte, modelo, tp, fn,
    fp, tn, n, aciertos y accuracy.
    """
    d = tabla[tabla["horizonte"] == horizonte]
    if modelos is None:
        modelos = sorted(d["modelo"].unique())
    fig, axes = plt.subplots(2, len(modelos), figsize=(4.6 * len(modelos), 8.2),
                             squeeze=False)
    for fila, target in enumerate(["blue_mid", "brecha_pct"]):
        for col, modelo in enumerate(modelos):
            ax = axes[fila, col]
            f = d[(d["target"] == target) & (d["modelo"] == modelo)].iloc[0]
            cm = np.array([[f["tp"], f["fn"]], [f["fp"], f["tn"]]])
            pct = cm / cm.sum(axis=1, keepdims=True) * 100

            ax.imshow(cm, cmap="Blues", vmin=0)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]}\n({pct[a, b]:.0f}%)",
                            ha="center", va="center", fontsize=11,
                            color="white" if cm[a, b] > cm.max() * 0.6 else "#1e293b")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["pred sube", "pred baja"], fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["real sube", "real baja"], fontsize=9)
            ax.set_title(f"{etiqueta(target)} — {modelo}\n"
                         f"aciertos {int(f['aciertos'])}/{int(f['n'])}  "
                         f"(acc {f['accuracy']:.2f})", fontsize=9.5)
    fig.suptitle(f"Matriz de confusión de dirección — horizonte {horizonte} h",
                 fontsize=12)
    fig.tight_layout()
    return fig


def roc_direccion(roc_datos, *, etiqueta, horizontes=(1, 24), colores=None):
    """Curvas ROC por horizonte (filas) y target (columnas).

    `roc_datos` va indexado por (horizonte, target, modelo) y devuelve
    (fpr, tpr, auc). Responde: ¿se distingue una suba de una bajada?
    """
    if colores is None:
        colores = COLORES_ROC
    objetivos = ["blue_mid", "brecha_pct"]
    fig, axes = plt.subplots(len(horizontes), len(objetivos),
                             figsize=(6.9, 2.75 * len(horizontes) + 0.8),
                             squeeze=False)
    for i, h in enumerate(horizontes):
        for j, target in enumerate(objetivos):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#94a3b8",
                    label="azar (0,50)")
            for nombre, color in colores.items():
                clave = (h, target, nombre)
                if clave not in roc_datos:
                    continue
                fpr, tpr, auc = roc_datos[clave]
                ax.plot(fpr, tpr, lw=1.6, color=color,
                        label=f"{nombre} ({auc:.2f})")
            ax.set_title(f"{etiqueta(target)} · horizonte {h} h", fontsize=9)
            ax.set_xlabel("falsos positivos", fontsize=7.5)
            ax.set_ylabel("verdaderos positivos", fontsize=7.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.55)
            ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
    fig.suptitle("Curvas ROC de la señal direccional", fontsize=10.5)
    fig.tight_layout()
    return fig


def event_study_tag(graf, err, n_txt, *, etiqueta_tag, hora_ref):
    """Barras del exceso de variación por tag y horizonte, con IC95."""
    colores = COLORES_HORIZONTES
    alto = 0.20
    y = np.arange(len(graf))

    fig, ax = plt.subplots(figsize=(11, 0.8 * len(graf) + 2.3))
    for k, (h, color) in enumerate(colores.items()):
        off = (k - (len(colores) - 1) / 2) * alto
        ax.barh(y + off, graf[h].values, height=alto, color=color, alpha=0.9,
                xerr=err[h].values, label=h,
                error_kw={"ecolor": "#555", "lw": 0.9, "capsize": 2})
        # valor pegado al extremo de la barra (no al final del intervalo)
        for yi, v in zip(y + off, graf[h].values):
            ax.text(v + (0.035 if v >= 0 else -0.035), yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color="#222")

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{etiqueta_tag(t)}  (n={n_txt[t]})" for t in graf.index],
                       fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Exceso de variación del blue sobre la línea base (%)")
    ax.set_title("Impacto de la Gaceta en el dólar blue por tag y horizonte (5 h · 1 d · 3 d · 5 d)\n"
                 f"Ancla {hora_ref}:00 UTC · barras = exceso vs. benchmark incondicional · "
                 "líneas = IC95", fontsize=11)
    ax.legend(title="Horizonte", fontsize=9, title_fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.margins(x=0.14)
    fig.tight_layout()
    return fig


def corr_tag_heatmap(corr, *, etiqueta_tag):
    """Heatmap de correlaciones Spearman (t y t+1) por tag."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    mat = corr[["spearman_t", "spearman_t1"]]
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(["Spearman t", "Spearman t+1"])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([etiqueta_tag(t) for t in mat.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.18 else "black")
    fig.colorbar(im, ax=ax, label="ρ de Spearman")
    ax.set_title("Correlación de rangos (Spearman)\ntag ↔ retorno diario del blue",
                 fontsize=10)
    fig.tight_layout()
    return fig


def z_hac(tabla, *, umbral=1.96, etiqueta=None, top=12, titulo=None):
    """Barras del |z| de los predictores más significativos (escala log).

    El |z| de la persistencia es ~2 órdenes de magnitud mayor que el resto;
    por eso se usa escala logarítmica y se muestran solo los `top` mayores.
    """
    d = tabla.copy()
    d["abs_z"] = d["z"].abs()
    d = d.sort_values("abs_z", ascending=False).head(top)
    d = d.iloc[::-1]                       # el mayor arriba
    fig, ax = plt.subplots(figsize=(6.9, 0.30 * len(d) + 1.7))
    colores = ["#b91c1c" if v > umbral else "#cbd5e1" for v in d["abs_z"]]
    ax.barh(range(len(d)), d["abs_z"], color=colores, height=0.7)
    ax.set_xscale("log")
    ax.axvline(umbral, color="black", ls="--", lw=1,
               label=f"umbral 5 % (|z| = {umbral:.2f})")
    for yi, v in enumerate(d["abs_z"]):
        ax.text(v * 1.12, yi, f"{v:.1f}", va="center", fontsize=7.5, color="#333")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([etiqueta(n) if etiqueta else n for n in d.index],
                       fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("|z| del coeficiente (errores HAC, escala log)")
    ax.set_title(titulo or f"Significancia de los predictores · {len(d)} de "
                           f"{len(tabla)} con mayor |z|", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, lw=0.6, which="both")
    fig.tight_layout()
    return fig
