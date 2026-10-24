"""Anexo · Métricas: qué mide cada número del informe."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as C
from core import data, formato as F, ui
from diccionario_datos import etiqueta_eje, etiqueta_tag

ui.encabezado("10_metricas")
R = data.metricas()
panel = data.panel()


def _tabla(filas: list[tuple], cabecera: tuple[str, ...]) -> None:
    """Tabla markdown: se ajusta al ancho y permite fórmulas largas."""
    out = ["| " + " | ".join(cabecera) + " |",
           "|" + "---|" * len(cabecera)]
    out += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    st.markdown("\n".join(out))


ui.lectura(
    "Todas las cifras del dashboard salen de estas definiciones. La página "
    "explica qué mide cada métrica, cuál es su mejor valor, dónde aparece y "
    "cuál es la trampa habitual al leerla. Los ejemplos usan los números "
    "reales de la ventana de prueba de 168 h.",
)

ui.ficha([
    ("Ventana de prueba", "168 h"),
    ("Horizontes", "1 h · 24 h"),
    ("Targets", "nivel · brecha"),
    ("Modelos", "RL · OLS-HAC · ARIMA"),
    ("Naïve", "último valor del origen"),
])

ui.seccion("Qué métrica responde a cada pregunta")
_tabla([
    ("¿Cuál es el error típico?", "MAE", "Error medio en Bs/USD o pp; es el que entra en el costo de operación.", "tab:kpi_mae"),
    ("¿Hay errores extremos?", "RMSE", "Penaliza más los errores grandes que el MAE.", "tab:comparativa"),
    ("¿Supera al paseo aleatorio?", "RMSE ÷ naïve", "Menor que 1 es mejor; 1 es empate.", "tab:comparativa"),
    ("¿Se traduce en dinero?", "ΔMAE × volumen", "Ahorro por precisión; si ΔMAE = 0, el ahorro es cero.", "tab:kpi_volumen"),
    ("¿Acierta la dirección?", "Accuracy, TP/FN/FP/TN", "Útil solo si las clases están balanceadas.", "tab:matriz_direccion"),
    ("¿Conviene comprar o esperar?", "TN / FN", "Umbral de decisión entre evitar una baja y perderse una suba.", "tab:matriz_direccion"),
    ("¿Qué variable importa?", "β, p-valor, IC 95 %", "Efecto estimado y su incertidumbre con errores HAC.", "tab:ols_hac"),
    ("¿Qué tan bien ajusta el OLS?", "R²", "Proporción de varianza explicada dentro de la muestra.", "tab:ols_hac"),
    ("¿Qué orden ARIMA elegir?", "AIC, BIC, AICc, HQIC", "Comparan verosimilitud y penalizan parámetros.", "tab:arima_grid"),
    ("¿La serie es estacionaria?", "ADF", "p < 0,05 rechaza raíz unitaria.", "tab:arima"),
    ("¿La Gaceta se relaciona?", "Exceso, IC95 %, Spearman ρ", "Efecto de un evento y asociación de rangos.", "tab:eventos · tab:spearman"),
], ("Pregunta", "Métrica", "Qué responde", "Dónde aparece"))

tabs = st.tabs(["Errores de pronóstico", "Señal direccional",
                "Inferencia (OLS-HAC)", "Series y selección", "Gaceta"])

# ══════════════════════════════════════════════════════════════════════════
# Errores de pronóstico
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.latex(r"e_t = \hat y_t - y_t")
    st.markdown(
        "El error se define como **predicción menos observado**. Su signo "
        "importa para leer el sesgo; su valor absoluto, para medir magnitud."
    )
    _tabla([
        ("Error (e)", "ŷ − y", "Desvío con signo.", "0", "fig:rl_test"),
        ("Sesgo", "(1/n) Σ e", "Error medio con signo; negativo indica subestimación.", "0", "fig:rl_test"),
        ("MAE", "(1/n) Σ abs(e)", "Error típico en unidades del target.", "0", "tab:rl_multiple · tab:arima · tab:kpi_mae"),
        ("RMSE", "sqrt((1/n) Σ e²)", "Error cuadrático medio; castiga los errores grandes.", "0", "tab:comparativa · tab:rl_multiple"),
        ("Naïve MAE / RMSE", "ŷ_t = y_(t−h)", "Línea base del paseo aleatorio: repetir el último valor del origen.", "—", "tab:rl_multiple · tab:arima"),
        ("RMSE ÷ naïve", "RMSE_modelo / RMSE_naïve", "Mejora relativa frente al naïve.", "< 1", "tab:comparativa"),
        ("MAE ÷ naïve", "MAE_modelo / MAE_naïve", "Mejora relativa en error típico.", "< 1", "tab:comparativa"),
        ("ΔMAE", "MAE_naïve − MAE_modelo", "Ahorro por precisión, en unidades del target.", "> 0", "tab:kpi_mae"),
        ("Cobertura del IC", "(# y dentro de [lo, hi]) / n", "Calibración empírica de la banda de pronóstico.", "≈ 95 %", "tab:arima"),
    ], ("Métrica", "Fórmula", "Qué mide", "Mejor valor", "Dónde aparece"))

    st.markdown("**Ejemplo real: nivel del blue a 1 h**")
    a = R["por_horizonte"]["1"]["arima"]["blue_mid"]
    r = R["por_horizonte"]["1"]["rl"]["blue_mid"]
    filas = pd.DataFrame([
        {"Modelo": "ARIMA(2,1,1)", "RMSE": a["rmse"], "MAE": a["mae"],
         "Naïve RMSE": a["rmse_naive"], "Naïve MAE": a["mae_naive"],
         "RMSE ÷ naïve": a["ratio"],
         "ΔMAE": round(a["mae_naive"] - a["mae"], 4)},
        {"Modelo": "RL múltiple", "RMSE": r["rmse"], "MAE": r["mae"],
         "Naïve RMSE": r["rmse_naive"], "Naïve MAE": r["mae_naive"],
         "RMSE ÷ naïve": r["ratio"],
         "ΔMAE": round(r["mae_naive"] - r["mae"], 4)},
    ])
    ui.tabla(
        filas,
        f"El ARIMA mejora el RMSE un {F.num((1 - a['ratio']) * 100, 1)} % "
        f"({F.num(a['ratio'], 2)}), pero su MAE es idéntico al del naïve y el "
        f"ΔMAE es 0. La regresión, en cambio, pierde {F.num(r['ratio'], 1)} "
        "veces en RMSE y su ΔMAE es negativo.",
        "tab:comparativa · tab:kpi_mae",
        formato={"RMSE": 4, "MAE": 4, "Naïve RMSE": 4, "Naïve MAE": 4,
                 "RMSE ÷ naïve": 2, "ΔMAE": 4},
    )
    ui.lectura(
        "El RMSE y el MAE no cuentan la misma historia. El RMSE premia acercarse "
        "en las horas atípicas; el MAE describe el error típico, que es el que "
        "se multiplica por el volumen operado. Cuando el MAE no mejora, no hay "
        "ahorro aunque el RMSE baje.",
    )

    with st.expander("Comprobar las fórmulas con las predicciones reales"):
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Horizonte", options=list(C.HORIZONTES),
                         format_func=lambda x: f"{x} h", key="met_h")
        target = c2.selectbox(
            "Nivel", options=list(C.TARGETS),
            format_func=lambda t: "Blue (Bs/USD)" if t == "blue_mid"
            else "Brecha (pp)", key="met_target")
        modelos = ["RL múltiple", f"ARIMA ({h} pasos)"]
        modelo = c3.selectbox("Modelo", options=modelos, key="met_modelo")
        preds = data.predicciones(h)
        sub = preds[(preds.target == target) & (preds.modelo == modelo)]
        e = sub.pred - sub.real
        rmse = float((e ** 2).mean() ** 0.5)
        mae = float(e.abs().mean())
        e_naive = sub.naive - sub.real
        rmse_n = float((e_naive ** 2).mean() ** 0.5)
        mae_n = float(e_naive.abs().mean())
        ui.kpis([
            {"label": "RMSE", "value": F.num(rmse, 4),
             "nota": f'naïve {F.num(rmse_n, 4)}'},
            {"label": "MAE", "value": F.num(mae, 4),
             "nota": f'naïve {F.num(mae_n, 4)}'},
            {"label": "RMSE ÷ naïve", "value": F.num(rmse / rmse_n, 2),
             "tono": "good" if rmse / rmse_n < 1 else "bad"},
            {"label": "MAE ÷ naïve", "value": F.num(mae / mae_n, 2),
             "tono": "good" if mae / mae_n < 1 else "bad"},
        ])
        st.caption(
            f"Calculado sobre {len(sub)} horas de la ventana de prueba. "
            "El cociente se lee igual que en el informe: 1 es empate."
        )

# ══════════════════════════════════════════════════════════════════════════
# Señal direccional
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.latex(r"\text{direccion}_t = \operatorname{signo}(y_t - y_{t-h})")
    st.markdown(
        "Se predice si el precio **sube o baja** frente al último valor del "
        "origen. Las horas sin cambio se excluyen. La matriz compara la "
        "dirección realizada con la predicha."
    )
    _tabla([
        ("TP", "real sube y predice sube", "Acierto de una suba.", "alto", "tab:matriz_direccion"),
        ("FN", "real sube y predice baja", "Suba perdida: no se compró.", "bajo", "tab:matriz_direccion"),
        ("FP", "real baja y predice sube", "Compra equivocada: el precio bajó.", "bajo", "tab:matriz_direccion"),
        ("TN", "real baja y predice baja", "Acierto de una baja.", "alto", "tab:matriz_direccion"),
        ("Accuracy", "(TP + TN) / n", "Proporción de aciertos.", "1", "tab:matriz_direccion"),
        ("Recall", "TP / (TP + FN)", "Cuántas subas se detectan.", "1", "fig:roc"),
        ("FPR", "FP / (FP + TN)", "Cuántas bajadas se confunden con subas.", "0", "fig:roc"),
        ("AUC", "Área bajo la curva ROC", "Capacidad de ordenar subas antes que bajas.", "1", "fig:roc"),
        ("Umbral TN / FN", "TN / FN", "Bajadas evitadas por cada suba perdida.", "> 1", "tab:matriz_direccion"),
        ("Bajadas evitadas", "TN / (FP + TN)", "Proporción de bajadas anticipadas correctamente.", "1", "tab:matriz_direccion"),
    ], ("Métrica", "Definición", "Qué mide", "Mejor valor", "Dónde aparece"))

    st.markdown("**Ejemplo real: nivel del blue, ARIMA con origen móvil**")
    # compatibilidad con una caché de Streamlit anterior al renombre
    md = R.get("matriz_direccion") or R.get("matriz_costos") or {}
    mc1 = md["1"]
    mc24 = md["24"]
    ui.tabla(
        pd.DataFrame([
            {"Horizonte": "1 h", "TP": mc1["tp"], "FN": mc1["fn"],
             "FP": mc1["fp"], "TN": mc1["tn"], "n": mc1["n"],
             "Accuracy": mc1["accuracy"], "TN / FN": mc1["umbral"]},
            {"Horizonte": "24 h", "TP": mc24["tp"], "FN": mc24["fn"],
             "FP": mc24["fp"], "TN": mc24["tn"], "n": mc24["n"],
             "Accuracy": mc24["accuracy"], "TN / FN": mc24["umbral"]},
        ]),
        "A una hora la señal supera el azar; a un día la accuracy cae por "
        "debajo de 0,5 y el umbral se vuelve más exigente.",
        "tab:matriz_direccion",
        formato={"Accuracy": 3, "TN / FN": 2},
    )
    st.markdown("**Matriz de confusión a 1 hora**")
    matriz = pd.DataFrame(
        [[f"TP {mc1['tp']}", f"FN {mc1['fn']}"],
         [f"FP {mc1['fp']}", f"TN {mc1['tn']}"]],
        index=["Real sube", "Real baja"],
        columns=["Predice sube", "Predice baja"],
    )
    st.dataframe(matriz, width="stretch")
    st.caption(
        f"Accuracy = (TP + TN) / n = ({mc1['tp']} + {mc1['tn']}) / "
        f"{mc1['n']} = {F.num(mc1['accuracy'], 3)}. "
        f"Umbral TN/FN = {mc1['tn']} / {mc1['fn']} = "
        f"{F.num(mc1['umbral'], 2)}."
    )

    roc = data.roc()
    if not roc.empty:
        st.markdown("**AUC por horizonte y nivel**")
        auc = (roc[["horizonte", "target", "modelo", "auc"]]
               .drop_duplicates()
               .pivot_table(index=["horizonte", "target"],
                            columns="modelo", values="auc")
               .reset_index())
        auc["horizonte"] = auc["horizonte"].map(lambda h: f"{h} h")
        auc["target"] = auc["target"].map(
            lambda t: "Blue" if t == "blue_mid" else "Brecha")
        auc = auc.rename(columns={"horizonte": "Horizonte",
                                  "target": "Nivel",
                                  "ARIMA (1 pasos)": "ARIMA 1 h",
                                  "ARIMA (24 pasos)": "ARIMA 24 h"})
        cols = ["Horizonte", "Nivel"] + [c for c in ("ARIMA 1 h", "ARIMA 24 h",
                                                     "RL múltiple") if c in auc]
        max1 = float(roc[roc.horizonte == 1].auc.max())
        max24_brecha = float(roc[(roc.horizonte == 24)
                                 & (roc.target == "brecha_pct")].auc.max())
        ui.tabla(
            auc[cols],
            f"AUC = 0,5 es azar; 1 es separación perfecta. A 1 h ninguna curva "
            f"supera {F.num(max1, 2)}; a 24 h la brecha llega a "
            f"{F.num(max24_brecha, 2)} mientras el nivel del blue cae por "
            "debajo de 0,5.",
            "fig:roc",
            formato={c: 3 for c in cols if c not in ("Horizonte", "Nivel")},
        )
    ui.alerta(
        "La accuracy sola engaña: si el 70 % de las horas sube, predecir "
        "siempre \"sube\" acierta 0,70 sin información. Por eso el informe "
        "reporta TP, FN, FP y TN, y resume la señal en un umbral TN/FN.",
        titulo="Trampa frecuente",
    )

# ══════════════════════════════════════════════════════════════════════════
# Inferencia (OLS-HAC)
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.latex(r"y_t = \beta_0 + \beta_1 x_{1t} + \cdots + \beta_k x_{kt} + \varepsilon_t")
    st.markdown(
        "El OLS con errores HAC estima el efecto de cada predictor sobre la "
        "brecha. Los errores de Newey–West (24 rezagos) son robustos a "
        "heterocedasticidad y autocorrelación: **no cambian los β, cambian los "
        "errores estándar**."
    )
    _tabla([
        ("β", "coeficiente estimado", "Cambio del target por unidad del predictor.", "—", "tab:ols_hac"),
        ("EE HAC", "desvío estándar robusto", "Incertidumbre del β sin suponer errores i.i.d.", "0", "tab:ols_hac"),
        ("z", "β / EE HAC", "Distancia del cero en errores estándar.", "abs(z) > 1,96", "fig:hac_z"),
        ("p-valor", "2 · P(Z > abs(z))", "Probabilidad de observar ese β si el efecto real fuera 0.", "0", "tab:ols_hac"),
        ("IC 95 %", "β ± 1,96 · EE HAC", "Rango plausible del efecto.", "no cruza 0", "tab:ols_hac"),
        ("R²", "1 − SS_res / SS_tot", "Proporción de varianza explicada dentro de la muestra.", "1", "tab:ols_hac"),
    ], ("Métrica", "Fórmula", "Qué mide", "Mejor valor", "Dónde aparece"))
    st.latex(r"R^2 = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \bar y)^2}")

    coef = pd.DataFrame(R["ols_hac"]["coeficientes"])
    vista = pd.DataFrame({
        "Coeficiente": coef["var"],
        "β": coef["coef"].map(lambda v: F.num(v, 4)),
        "EE HAC": coef["ee"].map(lambda v: F.num(v, 4)),
        "p-valor": coef["p"].map(F.pvalor),
        "IC 2,5": coef["lo"].map(lambda v: F.num(v, 4)),
        "IC 97,5": coef["hi"].map(lambda v: F.num(v, 4)),
        "z = β / EE": (coef["coef"] / coef["ee"]).map(lambda v: F.num(v, 2)),
    })
    ui.tabla(
        vista,
        f"Únicos coeficientes significativos al 5 %. R² dentro de muestra = "
        f"{F.num(R['ols_hac']['r2'], 4)}, n = {F.miles(R['ols_hac']['n'])}. "
        "El R² alto no es un logro predictivo: brecha_lag1h casi copia el "
        "target.",
        "tab:ols_hac",
        height=150,
    )
    ui.lectura(
        "Un p-valor pequeño indica que el efecto es difícil de explicar por "
        "azar, no que sea grande ni útil. La persistencia de la brecha "
        "(β ≈ 0,99) domina el ajuste; el resto de los predictores, incluida la "
        "Gaceta, no aporta evidencia suficiente.",
    )
    ui.alerta(
        "R² se calcula dentro de la muestra y no es comparable entre targets. "
        f"Un R² de {F.num(R['ols_hac']['r2'], 4)} en la brecha no significa "
        "que el modelo prediga mejor el nivel del blue: son escalas y "
        "dinámicas distintas.",
        titulo="Trampa frecuente",
    )

# ══════════════════════════════════════════════════════════════════════════
# Series y selección
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.latex(r"\mathrm{AIC} = 2k - 2\ln L")
    st.markdown(
        "Los criterios de información combinan el ajuste (verosimilitud) con "
        "una penalización por cantidad de parámetros. **Menor es mejor**, pero "
        "solo se comparan modelos sobre la misma serie y la misma muestra."
    )
    _tabla([
        ("ADF (nivel)", "p-valor de raíz unitaria", "p < 0,05: se rechaza raíz unitaria en el nivel.", "< 0,05", "tab:arima"),
        ("ADF (diff)", "p-valor sobre Δy", "p < 0,05: la serie es estacionaria en diferencias.", "< 0,05", "tab:arima"),
        ("logL", "log-verosimilitud", "Qué tan probable es la muestra bajo el modelo.", "mayor", "tab:criterios_info"),
        ("k", "número de parámetros", "Complejidad del modelo.", "—", "tab:criterios_info"),
        ("AIC", "2k − 2 ln L", "Ajuste penalizado por parámetros.", "menor", "tab:arima_grid"),
        ("BIC", "k ln n − 2 ln L", "Penaliza más la complejidad que el AIC.", "menor", "tab:criterios_info"),
        ("AICc", "AIC + 2k(k+1)/(n−k−1)", "Corrección del AIC para muestras chicas.", "menor", "tab:criterios_info"),
        ("HQIC", "−2 ln L + 2k ln(ln n)", "Penalización intermedia entre AIC y BIC.", "menor", "tab:criterios_info"),
    ], ("Métrica", "Fórmula", "Qué mide", "Mejor valor", "Dónde aparece"))

    filas = []
    for target in C.TARGETS:
        c = R["arima"][target]["criterios"]
        filas.append({
            "Nivel": etiqueta_eje(target), "k": c["k"], "n": c["n"],
            "logL": c["logL"], "AIC": c["AIC"], "BIC": c["BIC"],
            "AICc": c["AICc"], "HQIC": c["HQIC"],
            "ADF nivel": R["arima"][target]["adf_p_nivel"],
            "ADF diff": R["arima"][target]["adf_p_diff"],
        })
    ui.tabla(
        pd.DataFrame(filas),
        "Criterios del ARIMA elegido por AIC y p-valores ADF. El orden se "
        "reselecciona por horizonte con la ventana de entrenamiento que "
        "corresponde.",
        "tab:criterios_info · tab:arima",
        formato={"logL": 1, "AIC": 1, "BIC": 1, "AICc": 1, "HQIC": 1,
                 "ADF nivel": 4, "ADF diff": 4},
    )
    ui.alerta(
        "El AIC no dice si el modelo es bueno en términos absolutos: solo "
        "ordena candidatos sobre los mismos datos. Un AIC de −9.305 no es "
        "\"mejor\" que uno de 2.926 si cambia el target.",
        titulo="Trampa frecuente",
    )

# ══════════════════════════════════════════════════════════════════════════
# Gaceta
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.latex(r"\text{exceso}_{i,h} = r_{i,h} - \bar r_h")
    st.markdown(
        "Para cada publicación de la Gaceta se mide el retorno del blue a 5 h, "
        "1, 3 y 5 días. El **benchmark** es la variación media de todos los "
        "días de la serie al mismo horizonte; el **exceso** es lo que se "
        "atribuye al evento."
    )
    _tabla([
        ("Evento", "par (fecha, tag)", "Una norma de un canal temático en una fecha.", "—", "tab:eventos"),
        ("Retorno", "(P_t+h / P_t − 1) · 100", "Variación porcentual del blue.", "—", "tab:eventos"),
        ("Benchmark", "media de todos los días a h", "Deriva propia de la muestra.", "—", "tab:eventos"),
        ("Exceso", "retorno_evento − benchmark", "Efecto atribuible a la publicación.", "—", "tab:eventos"),
        ("Media del exceso", "promedio sobre eventos", "Efecto medio por tag y horizonte.", "—", "tab:eventos"),
        ("IC 95 %", "media ± 1,96 · EE", "Precisión del efecto medio; es una semi-amplitud.", "angosto", "tab:eventos"),
        ("n", "eventos con datos", "Cuántos pares fecha–tag entran.", "grande", "tab:eventos"),
        ("Spearman ρ", "correlación de rangos", "Asociación monótona entre conteo diario y retorno.", "abs(ρ) → 1", "tab:spearman"),
    ], ("Métrica", "Fórmula", "Qué mide", "Mejor valor", "Dónde aparece"))

    meta = R["eventos_meta"]
    ui.ficha([
        ("Pares fecha–tag", meta["pares_totales"]),
        ("Con datos de mercado", meta["pares_con_datos"]),
        ("Días de la grilla", meta["grilla_dias"]),
        ("Ancla", "12:00 UTC"),
    ])

    filas = []
    for tag in ("divisas_control_cambios", "deuda_externa",
                "fiscal_presupuesto"):
        d = R["eventos"][tag]
        filas.append({
            "Tag": etiqueta_tag(tag),
            "n 5 h": d["5 h"]["n"],
            "Exceso 5 h": d["5 h"]["media"],
            "IC95 5 h": d["5 h"]["ic95"],
            "Exceso 1 d": d["1 d"]["media"],
            "Exceso 3 d": d["3 d"]["media"],
        })
    ui.tabla(
        pd.DataFrame(filas),
        "Los tags cambiarios tienen exceso negativo a 5 h (la dirección de la "
        "hipótesis), pero con n ≤ 6 el IC95 es demasiado ancho para concluir. "
        f"El caso fiscal es ruido: n = "
        f"{R['eventos']['fiscal_presupuesto']['3 d']['n']} e IC95 de ±"
        f"{F.num(R['eventos']['fiscal_presupuesto']['3 d']['ic95'], 1)} pp.",
        "tab:eventos",
        formato={"Exceso 5 h": 2, "IC95 5 h": 2, "Exceso 1 d": 2,
                 "Exceso 3 d": 2},
    )

    sp = pd.DataFrame([
        {"Tag": etiqueta_tag(tag), "n días": v["n"],
         "ρ mismo día": v["t"], "ρ día siguiente": v["t1"]}
        for tag, v in R["spearman"].items()
    ]).sort_values("ρ día siguiente", key=abs, ascending=False).head(5)
    max_rho = max(max(abs(v["t"]), abs(v["t1"]))
                  for v in R["spearman"].values())
    ui.tabla(
        sp,
        "Spearman usa rangos, no valores: es robusto a conteos discretos y "
        f"outliers. Todas las correlaciones son débiles (|ρ| ≤ "
        f"{F.num(max_rho, 2)}).",
        "tab:spearman",
        formato={"ρ mismo día": 3, "ρ día siguiente": 3},
    )
    dias_fx = int((panel["n_docs_fx"] > 0).sum())
    ui.alerta(
        "Correlación no es causalidad y un evento no es independiente de otro: "
        "las ventanas de ±5 días se solapan. Con "
        f"{dias_fx} de {len(panel)} días con norma FX, la ausencia de "
        "significancia es informativa, no un defecto del test.",
        titulo="Trampa frecuente",
    )

# ══════════════════════════════════════════════════════════════════════════
# Mapa y glosario
# ══════════════════════════════════════════════════════════════════════════
ui.seccion("Mapa de métricas por tabla y figura")
mapa = pd.DataFrame([
    {"Elemento": "tab:rl_multiple", "Métricas": "RMSE, MAE, naïve, ratios"},
    {"Elemento": "tab:arima", "Métricas": "RMSE, MAE, naïve, ratio, cobertura"},
    {"Elemento": "tab:comparativa", "Métricas": "RMSE y MAE contra el naïve"},
    {"Elemento": "tab:ols_hac", "Métricas": "β, EE HAC, p, IC 95 %, R²"},
    {"Elemento": "tab:arima_grid", "Métricas": "AIC de 16 candidatos"},
    {"Elemento": "tab:criterios_info", "Métricas": "logL, AIC, BIC, AICc, HQIC"},
    {"Elemento": "tab:matriz_direccion", "Métricas": "TP, FN, FP, TN, accuracy, TN/FN"},
    {"Elemento": "fig:roc", "Métricas": "TPR, FPR, AUC"},
    {"Elemento": "tab:eventos", "Métricas": "exceso medio, IC95 %, n"},
    {"Elemento": "tab:spearman", "Métricas": "ρ mismo día y día siguiente"},
    {"Elemento": "tab:kpi_mae · tab:kpi_volumen", "Métricas": "ΔMAE, ahorro"},
])
ui.tabla(mapa, None, None, height=380,
         column_config={"Elemento": st.column_config.TextColumn(width="medium"),
                        "Métricas": st.column_config.TextColumn(width="large")})

ui.seccion("Glosario de símbolos")
_tabla([
    ("n", "número de observaciones o de eventos"),
    ("h", "horizonte del pronóstico: 1 h o 24 h"),
    ("y_t", "valor observado en t"),
    ("ŷ_t", "valor predicho para t"),
    ("e_t", "error con signo: ŷ − y"),
    ("k", "número de parámetros del modelo"),
    ("L", "verosimilitud maximizada"),
    ("ρ", "correlación de rangos de Spearman"),
    ("TN / FN", "bajadas evitadas por cada suba perdida"),
    ("pp", "puntos porcentuales"),
], ("Símbolo", "Significado"))

ui.lectura(
    "La regla de oro: el RMSE y el R² describen el ajuste; el MAE y el ΔMAE "
    "deciden si hay ahorro; la accuracy y el umbral TN/FN deciden si la "
    "señal sirve para operar. Ninguna métrica reemplaza a las otras.",
)

ui.pie("views/10_metricas.py")
