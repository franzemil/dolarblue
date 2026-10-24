# Dólar blue Bolivia (BOB/USD) — pronóstico a 1 hora y 1 día

Trabajo final de MCD-701. Pipeline CRISP-DM completo: recolección de cotizaciones
P2P y de la Gaceta Oficial, preparación, modelado (regresión lineal y ARIMA),
evaluación de negocio y un informe LaTeX.

El informe (`report/build/main.pdf`) contiene **solo los resultados y las
decisiones**. Este README guarda el detalle técnico que no entra en sus páginas:
parámetros estimados del ARIMA, criterios de información y notas de
interpretación.

---

## Cómo se ejecuta

```bash
cd 01_intro

# 1. datos (mercado horario + panel de la Gaceta)
.venv/bin/python final_project/modelado/prepare_datos.py

# 2. métricas -> modelado/metricas_reporte.json
.venv/bin/python final_project/modelado/reporte_modelado.py

# 3. notebook de resultados (proyecto_notebook.ipynb)
.venv/bin/python final_project/modelado/run_notebook.py

#    el notebook de análisis (registro histórico) solo si se pide
.venv/bin/python final_project/modelado/run_notebook.py --nb analisis_gaceta_fx.ipynb

# 4. informe
cd final_project/report
tectonic --keep-intermediates --keep-logs -o build main.tex
```

La animación del pipeline (60 s) no se incrusta en `main.pdf`: se genera
aparte y el informe incluye solo su fotograma final. Recorre fuentes →
ingesta → LLM/caché → replay → indexer horario → indexer diario → EDA/NA →
resumen.

```bash
make anim              # MP4 + GIF + póster + frames + clips por escena
make poster            # solo el póster vectorial (lo usa `make render`)
make pipeline-assets   # solo los 8 frames + guion para el dashboard
make pipeline-clips    # solo los 8 GIF animados por escena
```

El notebook (sección 2.2) reproduce las ocho escenas como clips animados; el
dashboard las muestra en la sección 3.6, también con el video completo.

Orden importante: `reporte_modelado.py` consume
`modelado/outputs/informe_direccion.csv`, que produce el notebook. Si se
regeneran datos, correr el notebook **antes** del reporte.

## Estructura

| Ruta | Contenido |
|---|---|
| `data/processed/market_hourly.csv` | serie horaria del mercado (4 318 h) |
| `modelado/panel_diario.csv` | panel mercado × Gaceta (179 días × 17 columnas) |
| `models/` | funciones puras: regresión, ARIMA, event study |
| `charts/figuras.py` | figuras del informe como funciones |
| `modelado/reporte_modelado.py` | métricas y `metricas_reporte.json` |
| `modelado/proyecto_notebook.ipynb` | notebook de resultados (el que alimenta el informe) |
| `report/make_animation.py` | animación del pipeline (MP4, GIF, póster y guion) |
| `report/anim/` | `pipeline.mp4` y `pipeline.gif` |
| `dashboard/assets/pipeline/` | 8 frames + `scenes.json` para la sección 3.6 |
| `report/anim/scenes/` | 8 GIF animados, uno por escena |
| `report/` | informe LaTeX y sus secciones |

---

## Detalle técnico del ARIMA

### Estimación

El ARIMA se ajusta por **máxima verosimilitud gaussiana**, no por mínimos
cuadrados. La diferencia práctica:

- **Mínimos cuadrados (OLS)** minimiza la suma de residuos al cuadrado; la
  varianza se estima como $RSS/(n-k)$.
- **Máxima verosimilitud (MLE)** maximiza la densidad conjunta de los datos bajo
  el supuesto gaussiano; los errores son los de predicción a un paso que produce
  la recursión ARMA y $\sigma^2$ se estima como $RSS/n$.

Con las mismas variables ambos coinciden asintóticamente, pero en ARIMA la MLE es
la vía estándar porque el modelo tiene una estructura dinámica (AR y MA) que la
verosimilitud trata de forma conjunta.

### Por qué $d=1$ cambia la interpretación

Con $d=1$ el modelo opera sobre las **primeras diferencias**. Para el nivel del
blue:

| | varianza |
|---|---:|
| del nivel $y_t$ | 12,35 |
| de la diferencia $\Delta y_t$ | 0,229 |
| $\sigma^2$ estimada | 0,226 |

Los coeficientes describen la **memoria de los cambios horarios**, no el efecto
de variables externas ni el nivel del precio. Un AR(1) de $-0{,}35$ significa que
un cambio de esta hora se corrige parcialmente en la siguiente, no que el nivel
baje 0,35 Bs.

### Parámetros estimados

Ventana de entrenamiento de 90 días ($n = 2\,160$), serie en primeras
diferencias. $ee$ es el error estándar, $z = \beta/ee$ y $p$ el p-valor;
$|z| > 1{,}96$ equivale a significancia al 5 %.

**ARIMA(2,1,1) — `blue_mid`**

| Parámetro | $\beta$ | $ee$ | $z$ | $p$ |
|---|---:|---:|---:|---:|
| `ar.L1` | −0,3504 | 0,1236 | −2,83 | 0,005 |
| `ar.L2` | 0,0431 | 0,0292 | 1,48 | 0,139 |
| `ma.L1` | 0,5579 | 0,1226 | 4,55 | <0,001 |
| `sigma2` | 0,0008 | 0,0000 | 125,39 | <0,001 |

**ARIMA(2,1,2) — `brecha_pct`**

| Parámetro | $\beta$ | $ee$ | $z$ | $p$ |
|---|---:|---:|---:|---:|
| `ar.L1` | 0,0677 | 0,0965 | 0,70 | 0,483 |
| `ar.L2` | 0,6443 | 0,0728 | 8,85 | <0,001 |
| `ma.L1` | −0,0446 | 0,0876 | −0,51 | 0,611 |
| `ma.L2` | −0,7310 | 0,0641 | −11,40 | <0,001 |
| `sigma2` | 0,2260 | 0,0012 | 189,60 | <0,001 |

**Lectura.** En el nivel del blue el AR(1) es negativo (−0,35) y el MA(1)
positivo (0,56): corrección de corto plazo, sin memoria más allá de una hora. En
la brecha mandan los términos de *lag* 2 (+0,64 y −0,73), que describen un ciclo
de dos horas; los de *lag* 1 no son significativos.

**Dos advertencias.** `sigma2` es la **varianza residual** y no un efecto: su
p-valor es significativo por construcción y no aporta evidencia. Y estos errores
estándar son los del Hessiano (máxima verosimilitud), **no** están corregidos por
HAC; statsmodels permite la variante robusta con `cov_type="robust"`.

### Selección del orden

Grilla de 16 candidatos con $p,q \le 2$, $d \in \{0,1\}$, evaluada **solo con
entrenamiento** y repetida para cada horizonte (la ventana se desplaza $h$ horas).
El detalle está en la tabla 6 del informe; el archivo con los valores crudos es
`modelado/outputs/arima_grid_aic_horario.csv`.

- AIC elige **ARIMA(2,1,1)** para el nivel y **ARIMA(2,1,2)** para la brecha.
- BIC, más severo, prefiere órdenes menores (ARIMA(0,1,1) y ARIMA(1,1,1)) con
  RMSE casi idéntico fuera de muestra: 0,0776 frente a 0,0775, y 0,2121 frente a
  0,2172. Se mantiene el AIC por ser el criterio estándar de pronóstico.
- La prueba de Dickey–Fuller rechaza raíz unitaria en el nivel ($p = 0{,}001$ y
  $0{,}031$), pero se deja que el AIC decida entre $d=0$ y $d=1$: en ambas series
  elige $d=1$.

### Evaluación

Origen móvil: para cada hora objetivo $t$ se pronostica desde $t-h$ y el modelo
se actualiza con los valores reales (`refit=False`). El naïve del mismo horizonte
es $y(t-h)$.

| Target | $h$ | ARIMA | naïve | ÷ |
|---|---:|---:|---:|---:|
| `blue_mid` | 1 h | 0,0074 | 0,0077 | 0,96 |
| `blue_mid` | 24 h | 0,0390 | 0,0390 | 1,00 |
| `brecha_pct` | 1 h | 0,0907 | 0,0910 | 1,00 |
| `brecha_pct` | 24 h | 0,2338 | 0,2350 | 0,99 |

### Lectura de los resultados (versión larga)

El ARIMA iguala o mejora levemente al naïve en los cuatro casos, nunca lo
empeora. A una hora el nivel baja el RMSE un 4,5 % (0,0074 frente a 0,0077) y la
brecha queda en tablas; a un día las diferencias son de una décima de punto
porcentual.

El resultado coincide con la literatura: en frecuencia horaria el tipo de cambio
se comporta como un paseo aleatorio (Meese & Rogoff, 1983), de modo que el margen
de mejora es pequeño y el valor del modelo no está en acertar más sino en
entregar una banda de incertidumbre y una señal de dirección.

El estadístico de **Diebold–Mariano** se omite: con $n = 168$ y errores
autocorrelacionados, la corrección de Harvey–Leybourne–Newbold no es fiable en
esta muestra. Con $n_{\text{test}} = h$ la corrección colapsa a cero y el
estimador HAC con $h-1$ rezagos no es identificable.

---

## Reproducibilidad

- Semilla fija en todo el pipeline; sin divisiones aleatorias: la partición es
  temporal (últimas 168 h).
- `metricas_reporte.json` es la única fuente de las cifras del informe; el
  notebook incluye `assert` contra él.
- Las figuras del informe las genera el notebook (`figures/fig_*.pdf`).
