"""Modelado — una pestaña por modelo."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as C
from core import data, figuras as G, formato as F, tablas as T, ui

ui.encabezado("05_modelado")
R = data.metricas()
h = int(st.session_state.get("horizonte", 1))

ui.ficha([
    ("Ventana de prueba", f"{R['h_test']} h"),
    ("Horizontes", "1 h · 24 h"),
    ("Métrica de corte", "RMSE ÷ naïve"),
    ("Quiebre estructural", "27/06/2026"),
    ("Modelos", "RL · OLS-HAC · ARIMA"),
])
ui.lectura(
    "Las últimas <b>168 h</b> son el test, con el mismo corte para todos los "
    "modelos. Cada modelo se compara con el <b>naïve de su horizonte</b>: "
    "repetir el último valor observado en el origen. ÷ naïve = 1 es empate; "
    "> 1 es peor."
)

tabs = st.tabs(["RL múltiple", "OLS + HAC", "ARIMA", "Comparativa",
                "Objetivo 2 · Gaceta", "Limitaciones"])

# ══════════════════════════════════════════════════════════════════════════
# 5.2 · Regresión lineal múltiple
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    target = st.radio(
        "Nivel a predecir", options=list(C.TARGETS), horizontal=True,
        format_func=lambda t: "Blue (Bs/USD)" if t == "blue_mid" else "Brecha (pp)",
        key="rl_target",
    )
    v = R["por_horizonte"][str(h)]["rl"][target]
    ratio_mae = v["mae"] / v["mae_naive"]
    ui.ficha([
        ("Modelo", "Mínimos cuadrados (SVD)"),
        ("Predictores", "37"),
        ("Horizonte", f"{h} h"),
        ("n train", F.miles(v["n_train"])),
        ("n test", v["n_test"]),
    ])
    ui.kpis([
        {"label": "RMSE", "value": F.num(v["rmse"], 4),
         "nota": f'naïve {F.num(v["rmse_naive"], 4)}'},
        {"label": "MAE", "value": F.num(v["mae"], 4),
         "nota": f'naïve {F.num(v["mae_naive"], 4)}'},
        {"label": "RMSE ÷ naïve", "value": F.num(v["ratio"], 2),
         "delta": (f"▼ {F.num((1 - v['ratio']) * 100, 1)} % vs naïve"
                   if v["ratio"] < 1 else
                   f"▲ {F.num((v['ratio'] - 1) * 100, 1)} % vs naïve"),
         "tono": "good" if v["ratio"] < 1 else "bad"},
        {"label": "MAE ÷ naïve", "value": F.num(ratio_mae, 2),
         "delta": ("▼ mejor" if ratio_mae < 1 else "▲ peor"),
         "tono": "good" if ratio_mae < 1 else "bad"},
    ])
    preds = data.predicciones(h)
    ui.figura(
        G.pred_vs_real(preds, target, "RL múltiple", height=380),
        "Observado, predicción de la regresión y naïve en la ventana de prueba.",
        "fig:rl_test",
    )
    ui.figura(
        G.error_modelo(preds, target, "RL múltiple", height=230),
        "Error (predicción − observado) por hora. La línea punteada es el sesgo "
        "medio: la regresión predice sistemáticamente por debajo del nivel.",
        "fig:rl_test · panel inferior",
    )
    tabla = T.tabla_rl(R)
    ui.tabla(
        tabla,
        "Regresión lineal múltiple en la ventana de prueba, a dos horizontes. "
        "En cada caso el naïve es el último valor observado en el origen.",
        "tab:rl_multiple",
        formato={"RMSE": 4, "MAE": 4, "Naïve RMSE": 4, "Naïve MAE": 4,
                 "RMSE ÷ naïve": 2, "MAE ÷ naïve": 2},
    )
    ui.lectura(
        "La regresión <b>pierde contra el naïve en los cuatro casos</b>. A una "
        "hora el nivel es casi un paseo aleatorio: repetir la hora anterior "
        "deja un error de 0,0077 Bs y el modelo, con su sesgo de nivel, llega a "
        "0,0819. A un día ambos se degradan, pero la brecha empeora mucho más "
        "(0,886 contra 0,235).",
    )

# ══════════════════════════════════════════════════════════════════════════
# 5.3 · OLS con errores HAC
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    ols = data.ols_hac()
    oh = R["ols_hac"]
    ui.ficha([
        ("Diseño", "Contemporáneo (h = 1)"),
        ("Errores", "HAC Newey–West, 24 rezagos"),
        ("R² dentro de muestra", F.num(oh["r2"], 4)),
        ("n", F.miles(oh["n"])),
        ("Predictores", "37"),
    ])
    ui.kpis([
        {"label": "Coeficientes significativos", "value": "2",
         "nota": "de 37, al 5 %"},
        {"label": "brecha_lag1h", "value": "0,9899",
         "nota": "persistencia casi perfecta", "tono": "good"},
        {"label": "oficial_lag1h", "value": "−0,0263",
         "nota": "efecto denominador", "tono": "bad"},
        {"label": "R²", "value": F.num(oh["r2"], 4),
         "nota": "dentro de muestra"},
    ])
    if not ols.empty:
        ui.figura(
            G.z_hac(ols, umbral=1.96, height=540),
            "Significancia de los coeficientes HAC en escala lineal: solo "
            "brecha_lag1h y oficial_lag1h superan el umbral del 5 %. "
            "brecha_lag1h (|z| ≈ 155) queda fuera del eje para poder leer el "
            "resto; se excluyen const y sigma2.",
            "fig:hac_z",
        )
    else:
        st.image(str(C.OUTPUTS / "informe_hac_z.png"),
                 caption="Significancia de los coeficientes HAC.",
                 width="stretch")

    coef = T.tabla_ols(R)
    coef_vista = pd.DataFrame({
        "Coeficiente": coef["var"],
        "β": coef["coef"],
        "ee HAC": coef["ee"],
        "p-valor": coef["p"].map(F.pvalor),
        "IC 2,5": coef["lo"],
        "IC 97,5": coef["hi"],
        "Sig. 5 %": coef["significativo"].map(lambda b: "sí" if b else "no"),
    })
    ui.tabla(
        coef_vista,
        "Coeficientes del OLS con errores HAC. Solo dos son significativos al "
        "5 %.",
        "tab:ols_hac",
        formato={"β": 4, "ee HAC": 4, "IC 2,5": 4, "IC 97,5": 4},
    )
    ui.lectura(
        "Solo dos coeficientes son significativos: <b>brecha_lag1h</b> "
        "(≈ 0,99, persistencia casi perfecta) y <b>oficial_lag1h</b> "
        "(−0,026, efecto denominador). Ningún canal de la Gaceta ni el volumen "
        "resultan significativos.",
    )

# ══════════════════════════════════════════════════════════════════════════
# 5.4 · ARIMA
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    target = st.radio(
        "Nivel a predecir", options=list(C.TARGETS), horizontal=True,
        format_func=lambda t: "Blue (Bs/USD)" if t == "blue_mid" else "Brecha (pp)",
        key="arima_target",
    )
    v = R["por_horizonte"][str(h)]["arima"][target]
    sel = R["arima_seleccion"][target][str(h)]
    ui.ficha([
        ("Modelo", v["modelo"]),
        ("Orden (p,d,q)", str(tuple(sel["order"]))),
        ("Tendencia", "sin deriva" if sel["trend"] == "n" else "constante"),
        ("AIC", F.num(v["aic"], 1)),
        ("Ventana de train", f'{F.miles(sel["n_train"])} h'),
        ("Origen inicial", F.fecha_hora(sel["origen0"])),
    ])
    ui.kpis([
        {"label": "RMSE", "value": F.num(v["rmse"], 4),
         "nota": f'naïve {F.num(v["rmse_naive"], 4)}'},
        {"label": "MAE", "value": F.num(v["mae"], 4),
         "nota": f'naïve {F.num(v["mae_naive"], 4)}'},
        {"label": "RMSE ÷ naïve", "value": F.num(v["ratio"], 2),
         "delta": (f"▼ {F.num((1 - v['ratio']) * 100, 1)} % vs naïve"
                   if v["ratio"] < 1 else
                   f"▲ {F.num((v['ratio'] - 1) * 100, 1)} % vs naïve"),
         "tono": "good" if v["ratio"] <= 1 else "bad"},
        {"label": "Cobertura IC 95 %", "value": F.cobertura(v["cobertura"]),
         "nota": "empírica en el test"},
    ])
    preds = data.predicciones(h)
    ui.figura(
        G.pred_vs_real(preds, target, f"ARIMA ({h} pasos)", height=380),
        "ARIMA con origen móvil: pronóstico puntual, banda de IC 95 % y naïve, "
        "en la ventana de prueba. El valor del modelo está en la banda, no en "
        "acertar más que el naïve.",
        "fig:arima_test",
    )
    ui.figura(
        G.error_modelo(preds, target, f"ARIMA ({h} pasos)", height=230),
        "Error por hora del ARIMA. A una hora es casi nulo; a un día se amplía "
        "con la incertidumbre del paseo aleatorio.",
        "fig:arima_test",
    )

    st.markdown("**Selección del orden por AIC**")
    grid = data.grid_arima()
    col = "blue_sell" if target == "blue_mid" else "brecha_pct"
    gv = grid.copy()
    gv["Modelo"] = gv.apply(
        lambda r: f"({int(r.p)},{int(r.d)},{int(r.q)}) "
                  f"{'c' if r.trend == 'c' else 'n'}", axis=1)
    gv = gv[["Modelo", col]].rename(columns={col: "AIC"}).sort_values("AIC")
    minimo = gv["AIC"].min()
    gv["AIC"] = gv["AIC"].map(lambda x: F.num(x, 1))
    st.dataframe(gv, width="stretch", hide_index=True, height=250)
    st.caption(f"16 candidatos con p, q ≤ 2 y d ∈ {{0,1}}. Mínimo: "
               f"{F.num(minimo, 1)} (en negrita en el informe).")
    ui.figura(
        G.aic_barras(grid, target, height=430),
        "Grilla completa de AIC: en verde el mínimo de cada serie. El orden se "
        "recalcula por horizonte con la ventana de entrenamiento que "
        "corresponde.",
        "tab:arima_grid",
    )

    ci, pa = st.columns(2, gap="large")
    with ci:
        st.markdown("**Criterios de información**")
        crit = T.tabla_criterios(R, target)
        st.dataframe(crit, width="stretch", hide_index=True)
    with pa:
        st.markdown("**Parámetros estimados**")
        par = T.tabla_parametros(R, target)
        par_vista = par.copy()
        par_vista["p"] = par_vista["p"].map(F.pvalor)
        for c in ("Coef.", "EE", "IC 2,5", "IC 97,5"):
            par_vista[c] = par_vista[c].map(lambda v: F.num(v, 4))
        st.dataframe(par_vista, width="stretch", hide_index=True,
                     height=210)

    ui.tabla(
        T.tabla_arima(R),
        "ARIMA en la ventana de prueba de 168 h, a dos horizontes, con "
        "evaluación de origen móvil.",
        "tab:arima",
        formato={"RMSE": 4, "MAE": 4, "Naïve RMSE": 4, "RMSE ÷ naïve": 2,
                 "AIC": 1},
    )
    ui.lectura(
        "El ARIMA <b>iguala o mejora levemente al naïve en los cuatro casos</b>: "
        "a una hora reduce el RMSE del nivel un 4,5 % y a un día las diferencias "
        "son de una décima de punto. El valor del modelo no está en acertar más "
        "—el tipo de cambio horario se comporta como un paseo aleatorio— sino "
        "en la banda de incertidumbre.",
    )

# ══════════════════════════════════════════════════════════════════════════
# 5.5 · Comparativa
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    comp = T.tabla_comparativa(R)
    ui.figura(
        G.ratio_barras(comp, height=380),
        "RMSE ÷ naïve por nivel, horizonte y modelo. La línea punteada marca el "
        "empate con el naïve.",
        "tab:comparativa",
    )
    ui.tabla(
        comp,
        "Resumen del Objetivo 1: cada modelo contra el naïve de su mismo "
        "horizonte, en RMSE y MAE.",
        "tab:comparativa",
        formato={"RMSE": 4, "RMSE ÷ naïve": 2, "MAE": 4, "MAE ÷ naïve": 2},
    )
    ui.lectura(
        "El <b>ARIMA domina a la regresión en los cuatro pares</b> y es el único "
        "que no pierde contra su línea base: iguala al naïve a un día "
        "(0,99–1,00) y lo mejora levemente a una hora en el nivel (0,96). La "
        "regresión múltiple pierde en todos los casos, con el caso extremo del "
        "nivel a 1 h (10,6× en RMSE y 16,2× en MAE).",
    )
    ui.alerta(
        "El MAE es la magnitud que entra en el costo de operación. Aunque el "
        "ARIMA mejore el RMSE a 1 h, su MAE es idéntico al naïve: la ganancia "
        "está en las horas atípicas, no en el error típico. Ver la sección 6.",
        titulo="Matiz importante",
    )

# ══════════════════════════════════════════════════════════════════════════
# 5.6 · Objetivo 2 — Gaceta
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    meta = R["eventos_meta"]
    ui.ficha([
        ("Pares fecha–tag", f'{meta["pares_totales"]}'),
        ("Con datos de mercado", f'{meta["pares_con_datos"]}'),
        ("Grilla", f'{meta["grilla_desde"]} → {meta["grilla_hasta"]}'),
        ("Días", f'{meta["grilla_dias"]}'),
        ("Ancla", "12:00 UTC"),
    ])
    ev = T.tabla_eventos(R)
    horizonte_ev = st.segmented_control(
        "Horizonte del evento", options=list(C.HORIZONTES_EVENTO),
        default="5 h", key="ev_horizonte",
    )
    ev_chart = ev[["tag", horizonte_ev, f"ic {horizonte_ev}",
                   f"n {horizonte_ev}"]].rename(
        columns={horizonte_ev: "media", f"ic {horizonte_ev}": "ic95",
                 f"n {horizonte_ev}": "n"})
    ui.figura(
        G.eventos_barh(ev_chart, "media", height=420),
        f"Exceso de variación del blue sobre la línea base incondicional por "
        f"tag, a {horizonte_ev}. Barras rojas: exceso negativo (el blue subió "
        "menos que la media); azules: positivo. Bigotes = IC 95 %.",
        "tab:eventos · fig:eventos_gaceta",
    )

    media = ev[["tag"] + list(C.HORIZONTES_EVENTO)].rename(
        columns={h_: f"media {h_}" for h_ in C.HORIZONTES_EVENTO})
    ui.tabla(
        media,
        "Exceso de variación (pp) por tag y horizonte, ordenado por el efecto "
        "a 5 h. Los tags cambiarios son los únicos con exceso negativo a 5 h.",
        "tab:eventos",
        formato={c: 2 for c in media.columns if c != "tag"},
    )
    with st.expander("Ver n por tag y horizonte"):
        ns = ev[["tag"] + [f"n {h_}" for h_ in C.HORIZONTES_EVENTO]]
        st.dataframe(ns, width="stretch", hide_index=True)

    sp = T.tabla_spearman(R)
    ui.figura(
        G.spearman_heatmap(sp, height=420),
        "Correlación de Spearman entre el conteo diario de normas de cada tag y "
        "el retorno diario del blue, en el mismo día (t) y con un día de rezago "
        "(t+1).",
        "tab:spearman",
    )
    sp_tabla = sp[["tag", "n", "t", "t1"]].rename(
        columns={"tag": "Tag", "n": "n días", "t": "ρ (t)", "t1": "ρ (t+1)"})
    ui.tabla(
        sp_tabla,
        "Correlaciones diarias: todas débiles (|ρ| ≤ 0,16). No hay relación "
        "monótona apreciable entre la intensidad de publicación y el retorno.",
        "tab:spearman",
        formato={"ρ (t)": 3, "ρ (t+1)": 3},
    )
    ui.lectura(
        "Los tags cambiarios (divisas_control_cambios, deuda_externa) son los "
        "únicos con exceso negativo a 5 h —la dirección que predice la "
        "hipótesis—, pero <b>ninguno es significativo</b> con n ≥ 5 en los "
        "horizontes cambiarios. El caso fiscal_presupuesto (+3,8 pp a 3 d) es "
        "ruido: n = 4 y un IC95 de ±7,2 pp.",
    )

# ══════════════════════════════════════════════════════════════════════════
# 5.7 · Limitaciones
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        """
- **Una sola semana de prueba.** El RMSE depende de pocas horas atípicas y
  las diferencias entre modelos no son concluyentes con esta muestra.
- **Gaceta dispersa.** Solo **23 de 179 días** tienen una norma con relevancia
  cambiaria; por eso ningún canal resulta significativo.
- **El nivel del blue es casi un paseo aleatorio.** Cualquier modelo compite
  contra una regla trivial muy fuerte; la ganancia está en la incertidumbre.
- **Un único quiebre estructural.** La devaluación del 27/06/2026 deja la
  ventana de prueba íntegra en el régimen vigente, pero impide evaluar
  estabilidad entre regímenes.
"""
    )
    ui.alerta(
        "Estas limitaciones no invalidan la comparación: la ordenan. El informe "
        "las usa para acotar las conclusiones, no para relativizarlas.",
        titulo="Cómo leerlas",
    )

ui.pie("views/05_modelado.py")
