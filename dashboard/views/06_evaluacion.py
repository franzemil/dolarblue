"""Evaluación de negocio."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as C
from core import data, figuras as G, formato as F, tablas as T, ui

ui.encabezado("06_evaluacion")
R = data.metricas()
h = int(st.session_state.get("horizonte", 1))

tabs = st.tabs(["Error de nivel", "Señal direccional", "Limitaciones"])

# ══════════════════════════════════════════════════════════════════════════
# 6.1 · El error de nivel no genera ahorro
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    a1 = R["por_horizonte"]["1"]["arima"]["blue_mid"]
    a24 = R["por_horizonte"]["24"]["arima"]["blue_mid"]
    d1 = a1["mae_naive"] - a1["mae"]
    d24 = a24["mae_naive"] - a24["mae"]
    ui.kpis([
        {"label": "MAE ARIMA · 1 h", "value": F.num(a1["mae"], 4),
         "nota": f'naïve {F.num(a1["mae_naive"], 4)} Bs/USD'},
        {"label": "MAE ARIMA · 24 h", "value": F.num(a24["mae"], 4),
         "nota": f'naïve {F.num(a24["mae_naive"], 4)} Bs/USD'},
        {"label": "ΔMAE · 1 h", "value": F.num(d1, 4),
         "nota": "ahorro por precisión (Bs/USD)", "tono": "bad"},
        {"label": "ΔMAE · 24 h", "value": F.num(d24, 4),
         "nota": "ahorro por precisión (Bs/USD)", "tono": "bad"},
    ])
    ui.lectura(
        "El error que importa en una operación de compra/venta es el MAE en Bs "
        "por USD multiplicado por el monto operado. A una hora el ARIMA y el "
        "naïve empatan (<b>0,0050</b> Bs/USD) y a un día también "
        "(<b>0,0277</b>): <b>el ahorro por precisión es cero a cualquier "
        "volumen</b>.",
    )

    st.markdown("**Calculadora de ahorro**")
    st.caption(
        "Supone que el precio se fija con el pronóstico. A precio de mercado, "
        "el ahorro es cero por definición."
    )
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        miles_usd = st.slider(
            "Volumen semanal operado (miles de USD)", min_value=10,
            max_value=1_000, value=100, step=10,
        )
        volumen = miles_usd * 1_000
        st.caption(f"Volumen: {F.miles(volumen)} USD por semana")
        horizonte_roi = st.radio(
            "Horizonte", options=[1, 24], horizontal=True,
            format_func=lambda x: f"{x} h", key="roi_h",
        )
    delta = d1 if horizonte_roi == 1 else d24
    ahorro_sem = delta * volumen
    with c2:
        st.metric("ΔMAE aplicado", f"{F.num(delta, 4)} Bs/USD")
        st.metric("Ahorro semanal", f"{F.num(ahorro_sem, 0)} Bs")
        st.metric("Ahorro anual", f"{F.num(ahorro_sem * 52, 0)} Bs")

    st.markdown(
        "La tabla del informe muestra el mismo cálculo para tres volúmenes. "
        "Con ΔMAE = 0, todas las filas son cero."
    )
    kpi_vol = pd.DataFrame([
        {"Volumen semanal (USD)": F.miles(v), "Ahorro semanal (Bs)": F.num(0, 0),
         "Ahorro anual (Bs)": F.num(0, 0)}
        for v in (10_000, 100_000, 1_000_000)
    ])
    ui.tabla(kpi_vol, "Ahorro por precisión a tres volúmenes de operación.",
             "tab:kpi_volumen")

    st.markdown("**Costo de adquisición de datos (LLM)**")
    gaceta = data.gaceta()
    if not gaceta.empty and "cost_usd" in gaceta:
        ok = gaceta[gaceta["status"] == "ok"] if "status" in gaceta else gaceta
        costo = float(ok["cost_usd"].sum())
        n = len(ok)
        proy = costo / n * 510 if n else 0
        ui.kpis([
            {"label": "Normas etiquetadas", "value": F.miles(n),
             "nota": "status ok"},
            {"label": "Costo total", "value": f"$ {F.num(costo, 4)}",
             "nota": "extracción con LLM"},
            {"label": "Costo por norma", "value": f"$ {F.num(costo / n, 6)}",
             "nota": "promedio"},
            {"label": "Proyección anual", "value": f"$ {F.num(proy, 2)}",
             "nota": "510 normas estimadas"},
        ])
        costo_tab = pd.DataFrame([
            {"Concepto": "Costo total de la extracción",
             "Valor": f"$ {F.num(costo, 4)}"},
            {"Concepto": "Costo por norma", "Valor": f"$ {F.num(costo / n, 6)}"},
            {"Concepto": "Proyección anual (510 normas)",
             "Valor": f"$ {F.num(proy, 2)}"},
        ])
        ui.tabla(costo_tab, "Costo de adquisición y etiquetado de la Gaceta.",
                 "tab:costo_llm")
    else:
        ui.vacio("No hay datos de costo LLM en los assets.",
                 "make dashboard-assets")

    ui.lectura(
        "La precisión extra del ARIMA a una hora no se traduce en ahorro: el "
        "MAE es idéntico al naïve. Lo único medible es una mejora de RMSE del "
        "4,5 %, que no tiene contrapartida contable.",
    )

# ══════════════════════════════════════════════════════════════════════════
# 6.2 · La señal direccional solo sirve a una hora
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    target = st.radio(
        "Nivel", options=list(C.TARGETS), horizontal=True,
        format_func=lambda t: "Blue (Bs/USD)" if t == "blue_mid" else "Brecha (pp)",
        key="dir_target",
    )
    modelo = st.radio(
        "Modelo", options=["ARIMA", "Regresión lineal"], horizontal=True,
        key="dir_modelo",
    )
    d = data.direccion()
    if d.empty:
        ui.vacio("Falta modelado/outputs/informe_direccion.csv.",
                 "make dashboard-assets")
        st.stop()
    patron = "ARIMA" if modelo == "ARIMA" else "Regresión"
    sub = d[(d.horizonte == h) & (d.target == target)
            & d.modelo.str.contains(patron, case=False)]
    if sub.empty:
        ui.vacio(f"No hay matriz para {modelo} a {h} h.", None)
        st.stop()
    fila = sub.iloc[0]
    mc = {k: int(fila[k]) for k in ("tp", "fn", "fp", "tn")}
    umbral = mc["tn"] / mc["fn"] if mc["fn"] else float("nan")
    bajadas = mc["fp"] + mc["tn"]
    subas = mc["tp"] + mc["fn"]

    ui.kpis([
        {"label": "Accuracy", "value": F.num(fila.accuracy, 3),
         "nota": f'{modelo} · {C.NOMBRE_HORIZONTE.get(h, f"{h} h")}',
         "tono": "good" if fila.accuracy > 0.5 else "bad"},
        {"label": "Aciertos", "value": f"{int(fila.aciertos)} / {int(fila.n)}",
         "nota": f'{subas} subas · {bajadas} bajadas'},
        {"label": "Umbral C_FN / C_FP",
         "value": F.num(umbral, 2),
         "nota": "aporta si el costo relativo es menor",
         "tono": "good" if umbral > 1 else "bad"},
        {"label": "Ahorro con costos simétricos",
         "value": f"{F.num(mc['tn'] - mc['fn'], 0)} / {bajadas}",
         "nota": "unidades frente a no operar",
         "tono": "good" if mc["tn"] > mc["fn"] else "bad"},
    ])

    izq, der = st.columns([1, 1.15], gap="large")
    with izq:
        ui.figura(
            G.confusion_heatmap(mc, height=330),
            f"Matriz de confusión de la dirección del nivel, {modelo} a {h} h. "
            "La celda muestra el conteo y su participación en el total.",
            "tab:matriz_costos",
        )
    with der:
        st.markdown(
            f"""
**Cómo se lee**

- **TP** {mc['tp']} · el modelo anticipó una suba.
- **FN** {mc['fn']} · subió y el modelo no la vio.
- **FP** {mc['fp']} · anticipó una suba que no ocurrió.
- **TN** {mc['tn']} · anticipó una baja correctamente.

El modelo conviene frente a no operar mientras
`TN · C_FP > FN · C_FN`, es decir mientras
`C_FN / C_FP < {F.num(umbral, 2)}`.
"""
        )
        if umbral < 1:
            st.caption(
                "Con un umbral menor que 1, la señal solo sirve si las subas "
                "perdidas cuestan menos que las bajadas evitadas: en la "
                "práctica, no sirve."
            )

    roc = data.roc()
    if not roc.empty:
        ui.figura(
            G.roc_horizonte(roc, h, height=360),
            "Curvas ROC de la señal direccional por nivel y modelo. La diagonal "
            "punteada es el azar (AUC = 0,50).",
            "fig:roc",
        )
        aucs = (roc[roc.horizonte == h][["target", "modelo", "auc"]]
                .drop_duplicates().sort_values("auc", ascending=False))
        aucs["target"] = aucs["target"].map(
            lambda t: "Blue" if t == "blue_mid" else "Brecha")
        ui.tabla(aucs.rename(columns={"target": "Nivel", "modelo": "Modelo",
                                      "auc": "AUC"}),
                 "AUC por horizonte, nivel y modelo.",
                 "fig:roc", formato={"AUC": 3})

    tabla_dir = T.tabla_direccion(d, h, target)[
        ["Modelo", "n", "Aciertos", "Accuracy", "TP", "FN", "FP", "TN"]]
    ui.tabla(
        tabla_dir,
        "Matriz de confusión y accuracy por modelo, calculadas con las mismas "
        "predicciones walk-forward de la sección 5.",
        "tab:matriz_costos",
        formato={"accuracy": 3},
    )

    mc1 = R["matriz_costos"]["1"]
    mc24 = R["matriz_costos"]["24"]
    ui.lectura(
        f"A una hora el modelo evita {mc1['tn']} de las "
        f"{mc1['fp'] + mc1['tn']} bajadas y pierde {mc1['fn']} subas: conviene "
        f"mientras {mc1['tn']}·C_FP > {mc1['fn']}·C_FN, es decir siempre que "
        f"C_FN/C_FP < <b>{F.num(mc1['umbral'], 2)}</b>. A un día la relación se "
        f"invierte: el umbral cae a {F.num(mc24['umbral'], 2)} y la accuracy "
        f"queda por debajo del azar ({F.num(mc24['accuracy'], 3)}). "
        "<b>La señal direccional es un fenómeno de una hora, no de un día.</b>",
    )

# ══════════════════════════════════════════════════════════════════════════
# 6.3 · Limitaciones de la evaluación
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        """
- **El ahorro supone fijar el precio con el pronóstico.** A precio de mercado
  es cero: la comparación es contra el naïve, no contra una mesa de cambios
  real.
- **C_FP y C_FN no se estimaron con datos de la empresa.** El resultado es un
  umbral de decisión, no una cifra contable.
- **Una sola semana de evaluación.** Los umbrales deben revisarse con más
  ventanas antes de usarse operativamente.
- **La accuracy no basta.** Un modelo que siempre predice "sube" acierta la
  tasa base; por eso el informe reporta TP/FN/FP/TN y el umbral de costos.
"""
    )
    ui.alerta(
        "La evaluación de negocio no convierte una mejora estadística en una "
        "promesa financiera: acota qué decisión cambia y bajo qué costos.",
        titulo="Alcance",
    )

ui.pie("views/06_evaluacion.py")
