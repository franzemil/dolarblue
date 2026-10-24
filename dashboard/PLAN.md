# Plan — Dashboard Streamlit: Dólar Blue (CRISP-DM)

> **Estado: implementado.**
> Once páginas navegables, seis pestañas de modelado, artefactos curados en
> `dashboard/assets/` y verificación manual en navegador.
>
> **Fuente de verdad:** `modelado/metricas_reporte.json` + CSV del pipeline.
> La app **lee**; no re-entrena modelos al abrir una página.

---

## 1. Objetivo y alcance

### Objetivo

Reproducir el informe (`report/main.tex`) y el notebook de resultados
(`modelado/proyecto_notebook.ipynb`) como una aplicación navegable:

- **Cada sección del informe es una entrada del menú lateral.**
- **Cada modelo es una pestaña** dentro de *Modelado*.
- Las cifras y las lecturas son las mismas del PDF, con interacción de consulta
  (horizonte, zoom, filtros, descargas).

### Dentro

- Navegación por fases CRISP-DM: Resumen, 1 Introducción, 2 Negocio,
  3 Datos y EDA, 4 Preparación, 5 Modelado, 6 Evaluación, 7 Conclusiones,
  más tres anexos (Métricas, Explorador y Reproducibilidad).
- Pestañas: RL múltiple, OLS + HAC, ARIMA, Comparativa, Objetivo 2 (Gaceta),
  Limitaciones.
- Series interactivas con Plotly; figuras de diagnóstico con matplotlib.
- Descarga de métricas, PDF, CSV y manifiesto de artefactos.

### Fuera

- Recolectar datos en vivo (eso es `collectors/` + `main.py`).
- Re-entrenar modelos en cada request.
- Autenticación, base de datos o escritura de artefactos.
- Sustituir el PDF como entregable formal.

---

## 2. Principios

| Principio | Regla concreta |
|---|---|
| Espeja el informe | Mismo orden, títulos y numeración que `main.tex` |
| Una fuente de verdad | Todo número sale de `metricas_reporte.json` o de un CSV; si falta, se calcula con la misma función del pipeline |
| Cero re-entrenamiento | La UI lee artefactos; el recálculo es offline (`export_assets.py`) |
| Trazabilidad | Cada tabla/figura declara su origen (`tab:arima`, `fig:roc`) en un caption discreto |
| Lectura, no metodología | Cada bloque lleva su callout *Lectura* con la conclusión del informe |
| Español y coma decimal | `0,0074` · `Bs/USD` · `pp` · fechas `dd/mm/aaaa` |
| Rápido y offline | Sin red; caché de datos; primer render < 2 s con artefactos |

---

## 3. Navegación

| # | Menú | Fuente editorial | Contenido |
|---|---|---|---|
| 0 | Resumen | `00_resumen.tex` | KPIs, serie blue/oficial, tres hallazgos, accesos |
| 1 | Introducción | `01_introduccion.tex` | Motivación, alcance, estructura del documento |
| 2 | Comprensión del negocio | `02_negocio.tex` | Objetivos 1–2, criterios de éxito, recursos |
| 3 | Datos y EDA | `03_datos.tex` + `04_analisis.tex` | Recolección, fuentes, diccionario, calidad, EDA |
| 4 | Preparación | `05_preparacion.tex` | Pipeline, remuestreo, target, split, features |
| 5 | Modelado | `06_modelado.tex` | Diseño de validación + seis pestañas |
| 6 | Evaluación de negocio | `07_evaluacion_negocio.tex` | Ahorro, señal direccional, limitaciones |
| 7 | Conclusiones | `wip_07_conclusiones.tex` | Hallazgos, trabajo futuro, badge WIP |
| 8 | Métricas | `metrics/` + `06_modelado.tex` | Qué mide cada número, fórmula, mejor valor y trampas |
| 9 | Explorador de datos | — | Tablas filtrables y descarga CSV |
| 10 | Reproducibilidad | — | Pipeline, comandos, versiones, manifiesto |

**Sidebar persistente:** logo, selector global de horizonte (1 h / 24 h), corte
de datos, descargas (métricas JSON e informe PDF), estado de artefactos y
versión.

El horizonte elegido se guarda en `st.session_state["horizonte"]` y lo consumen
Resumen, Modelado y Evaluación.

---

## 4. Arquitectura de archivos

```
dashboard/
├── app.py                     # página, tema, st.navigation, sidebar
├── PLAN.md                    # este documento
├── README.md                  # arranque rápido
├── requirements.txt           # dependencias mínimas de despliegue
├── .streamlit/config.toml     # tema dbblue / dborange
├── assets/                    # artefactos curados y versionados
│   ├── logo.svg
│   ├── mercado.parquet
│   ├── panel.parquet
│   ├── gaceta.parquet
│   ├── crudo.parquet
│   ├── predicciones_h1.parquet
│   ├── predicciones_h24.parquet
│   ├── roc_curvas.parquet
│   ├── ols_hac.parquet
│   └── manifest.json
├── core/
│   ├── config.py              # rutas, constantes, paleta, navegación
│   ├── data.py                # loaders cacheados y modo degradado
│   ├── formato.py             # números y fechas en español
│   ├── figuras.py             # figuras Plotly y reutilización de matplotlib
│   ├── tablas.py              # tablas derivadas de metricas_reporte.json
│   └── ui.py                  # estilos, encabezados, KPIs, callouts, tablas
├── views/
│   ├── 00_resumen.py
│   ├── 01_introduccion.py
│   ├── 02_negocio.py
│   ├── 03_datos.py
│   ├── 04_preparacion.py
│   ├── 05_modelado.py         # seis pestañas
│   ├── 06_evaluacion.py       # tres pestañas
│   ├── 07_conclusiones.py
│   ├── 08_explorador.py
│   ├── 09_reproducibilidad.py
│   └── 10_metricas.py
└── scripts/
    └── export_assets.py       # offline: predicciones, OLS-HAC, ROC, parquet
```

**Decisiones**

- `st.navigation` + `st.Page` con grupos, iconos y orden explícito.
- Carpeta `views/` (no `pages/`) para evitar la auto-descubierta de Streamlit.
- `dashboard/assets/*.parquet` versionados: `data/` está en `.gitignore`, así
  que la app arranca sin pipeline y se puede desplegar.
- Matplotlib con backend `Agg`; Plotly para todo lo interactivo.

---

## 5. Capa de datos

| Loader | Archivo | Uso |
|---|---|---|
| `metricas()` | `modelado/metricas_reporte.json` | KPIs, tablas, comparativa, eventos, costos |
| `mercado()` | `assets/mercado.parquet` | serie horaria |
| `panel()` | `assets/panel.parquet` | Gaceta diaria |
| `gaceta()` | `assets/gaceta.parquet` | normas etiquetadas y costo LLM |
| `crudo()` | `assets/crudo.parquet` | snapshots para el EDA |
| `predicciones(h)` | `assets/predicciones_h*.parquet` | observado, predicho, naïve, banda |
| `roc()` | `assets/roc_curvas.parquet` | curvas ROC y AUC |
| `ols_hac()` | `assets/ols_hac.parquet` | coeficientes HAC completos |
| `direccion()` | `modelado/outputs/informe_direccion.csv` | matrices de confusión |
| `grid_arima()` | tabla del informe | 16 candidatos de `tab:arima_grid` |

- Todos los loaders usan `st.cache_data`; los artefactos son estáticos.
- Si falta un archivo, la app muestra la ruta y el comando que lo genera.
- `metricas_reporte.json` es obligatorio: sin él se muestra un diagnóstico, no
  un traceback.

### Regenerar artefactos

`dashboard/scripts/export_assets.py` reutiliza `models/regresion.py` y
`models/arima.py` para reconstruir las predicciones walk-forward, el OLS-HAC y
las curvas ROC, y escribe los parquet + `manifest.json` con hashes.
Se ejecuta **fuera** de la app:

```bash
make dashboard-assets
```

---

## 6. Modelado: una pestaña por modelo

Cabecera común con el diseño de validación: test de 168 h, dos horizontes,
RMSE y MAE contra el naïve, quiebre del 27/06/2026.

| Pestaña | Bloques |
|---|---|
| **RL múltiple** | Ficha (37 predictores, ventana), KPIs por horizonte, observado vs predicho vs naïve, error y sesgo, `tab:rl_multiple`, lectura |
| **OLS + HAC** | Ficha (R², n, 24 rezagos), coeficientes significativos, \|z\| en escala lineal, `tab:ols_hac`, lectura |
| **ARIMA** | Ficha (orden, AIC, ventana), KPIs, walk-forward con banda 95 %, grilla AIC (16 candidatos), criterios y parámetros, `tab:arima`, lectura |
| **Comparativa** | Barras RMSE ÷ naïve, `tab:comparativa`, lectura y matiz MAE |
| **Objetivo 2** | Estudio de eventos por tag (5 h, 1 d, 3 d, 5 d, IC95, n), Spearman (t, t+1), `tab:eventos` y `tab:spearman`, lectura |
| **Limitaciones** | Una semana de prueba, Gaceta dispersa (23 de 179 días), paseo aleatorio, quiebre único |

**Escala del gráfico HAC:** lineal. `brecha_lag1h` (|z| ≈ 155) se deja fuera del
eje con una anotación para que el resto de los coeficientes sea legible; se
excluyen `const` y `sigma2`.

**Estado compartido:** un solo selector de horizonte en el sidebar. Las
pestañas de ARIMA y RL tienen su propio selector de target.

---

## 7. Sistema de diseño

| Token | Valor | Uso |
|---|---|---|
| `dbblue` | `#2563EB` | primario, serie blue |
| `dborange` | `#F59E0B` | acento, oficial y RL |
| `dbgray` | `#6B7280` | texto secundario, ejes |
| `green` | `#10B981` | mejor que el naïve |
| `red` | `#EF4444` | peor que el naïve, umbral HAC |
| `surface` | `#F8FAFC` | sidebar y tarjetas |

- Tema claro en `.streamlit/config.toml`; ancho `wide`; sidebar expandido.
- Componentes: encabezado con eyebrow, `kpi_card`, `lectura`, `alerta`,
  `ficha`, `tabla`, `figura`, `vacio`, `pie` de navegación.
- Captions de tabla/figura con su etiqueta LaTeX de origen.
- Botón *Deploy* oculto; logo propio en la cabecera del sidebar.

---

## 8. Contenido por página

- **Resumen:** 4 KPIs, serie blue vs oficial, tres hallazgos, accesos directos.
- **Introducción:** motivación, alcance, tabla de estructura con enlaces.
- **Negocio:** dos problemas, objetivos, criterios de éxito, recursos.
- **Datos:** diagramas, cobertura de fuentes, diccionario buscable, calidad y
  quiebre, EDA (crudo vs grilla, nivel, brecha), descriptivos.
- **Preparación:** pipeline en cuatro etapas, remuestreo, target y eventos,
  split y naïve, predictores por horizonte con inspección de features.
- **Modelado:** diseño de validación y las seis pestañas.
- **Evaluación:** error de nivel con calculadora de ahorro y costo LLM; señal
  direccional con matriz, umbral C_FN/C_FP y ROC; limitaciones.
- **Conclusiones:** badge WIP, hallazgos, trabajo futuro, estado del entregable.
- **Métricas:** de la MAE al R², con fórmulas, ejemplos reales, mapa por tabla
  y glosario de símbolos; incluye una calculadora de RMSE/MAE sobre las
  predicciones exportadas.
- **Explorador:** mercado, panel y Gaceta con filtros, búsqueda y descarga.
- **Reproducibilidad:** pipeline, comandos, versiones, artefactos y manifiesto.

---

## 9. Estado de implementación

- [x] Navegación de once páginas y sidebar persistente.
- [x] Capa de datos con caché y modo degradado.
- [x] Exportación offline de predicciones, OLS-HAC y ROC.
- [x] Seis pestañas de modelado y tres de evaluación.
- [x] KPIs, tablas y figuras con paridad numérica con el informe.
- [x] Formato español (coma decimal, miles, fechas).
- [x] Tema, logo, componentes y navegación secuencial.
- [x] Sin símbolo de sección ni línea de fuente editorial en las páginas.
- [x] Gráfico HAC en escala lineal.
- [x] Página de métricas en Anexos (MAE, RMSE, ratios, accuracy, ROC/AUC, β, R², AIC, ADF, Spearman).
- [x] Verificación manual en navegador: las once páginas y las pestañas
      renderizan sin tracebacks; los valores del informe aparecen en pantalla.

---

## 10. Cómo ejecutar

```bash
# 1. Instalar (una vez)
uv sync

# 2. Levantar
make dashboard
# o: uv run streamlit run dashboard/app.py

# 3. Regenerar artefactos si cambiaron los datos
make dashboard-assets
```

La app queda en `http://localhost:8501`.

### Flujo de actualización

1. `make data` y `make process` refrescan los CSV del pipeline.
2. `python modelado/reporte_modelado.py` actualiza `metricas_reporte.json`.
3. `make dashboard-assets` regenera los parquet del dashboard.
4. Recargar la app.

---

## 11. Decisiones y riesgos

| Decisión | Motivo |
|---|---|
| Artefactos parquet versionados | `data/` no se versiona; la app debe arrancar sola |
| Predicciones exportadas offline | Evita walk-forward en cada request |
| Híbrido Plotly + matplotlib | Interacción donde aporta; figuras del informe donde importa la paridad |
| Grilla AIC del informe, no el CSV | El informe documenta 16 candidatos; el CSV del pipeline puede ampliarse |
| Escala lineal en HAC | Legibilidad; el outlier se anota y se excluyen parámetros de nuisance |

| Riesgo | Mitigación |
|---|---|
| Datos nuevos sin regenerar assets | `manifest.json` y estado de artefactos en el sidebar |
| Fechas del dataset en 2026 | Sello *Datos al* siempre visible |
| Formato español en tablas | Preformateo de strings; gráficos conservan números |
| Figuras del informe que cambian | Se reutilizan funciones de `charts/figuras.py` cuando aplica |
| Despliegue sin pipeline | `requirements.txt` + parquet curados |

---

## 12. Trabajo futuro

- Descarga del informe PDF por página (no solo global).
- Enlaces profundos a pestañas (`st.query_params`).
- Comparador de horizontes lado a lado.
- Modo oscuro.
- Recalculo opcional de un modelo desde la UI, con caché por hash de datos.
