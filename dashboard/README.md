# Dashboard — Dólar Blue (CRISP-DM)

Vista Streamlit del informe final: navega las fases de CRISP-DM y compara los
tres modelos (regresión lineal múltiple, OLS con errores HAC y ARIMA) en
pestañas separadas.

## Arranque

```bash
uv sync                 # instala dependencias
make dashboard          # http://localhost:8501
```

Alternativa directa:

```bash
uv run streamlit run dashboard/app.py
```

## Qué incluye

- **Once páginas:** Resumen, Introducción, Negocio, Datos y EDA, Preparación,
  Modelado, Evaluación de negocio, Conclusiones, Métricas, Explorador y
  Reproducibilidad.
- **Seis pestañas de modelado:** RL múltiple, OLS + HAC, ARIMA, Comparativa,
  Objetivo 2 (Gaceta) y Limitaciones.
- **Tres pestañas de evaluación:** error de nivel, señal direccional y
  limitaciones.
- Selector global de horizonte (1 h / 24 h), descarga de métricas, PDF y CSV.
- Formato español: coma decimal, miles y fechas `dd/mm/aaaa`.

## Datos

La app **no re-entrena**: lee artefactos ya generados.

| Archivo | Contenido |
|---|---|
| `modelado/metricas_reporte.json` | métricas, tablas y matrices del informe |
| `dashboard/assets/*.parquet` | series, predicciones, OLS-HAC y ROC curados |
| `modelado/outputs/informe_direccion.csv` | matriz de confusión |

`data/` está en `.gitignore`, por eso los parquet de `dashboard/assets/` se
versionan: permiten arrancar y desplegar sin correr el pipeline.

### Regenerar artefactos

```bash
make dashboard-assets
# equivale a: uv run python dashboard/scripts/export_assets.py
```

El script reutiliza `models/regresion.py` y `models/arima.py`, reconstruye las
predicciones walk-forward, el OLS-HAC y las curvas ROC, y actualiza
`assets/manifest.json` con hashes.

### Actualización completa

1. `make data` y `make process` refrescan los CSV.
2. `python modelado/reporte_modelado.py` actualiza el JSON de métricas.
3. `make dashboard-assets` regenera los parquet.
4. Recargar la app.

## Estructura

```
dashboard/
├── app.py                 # navegación, tema y sidebar
├── assets/                # parquet curados + logo
├── core/                  # config, datos, formato, figuras, tablas, UI
├── views/                 # una página por sección del informe
├── scripts/export_assets.py
├── .streamlit/config.toml
└── requirements.txt       # dependencias mínimas de despliegue
```

## Notas

- El símbolo de sección y la línea de fuente editorial se eliminaron de las
  páginas para dejar una lectura más limpia; la trazabilidad queda en los
  captions de tablas y figuras (`tab:arima`, `fig:roc`).
- El gráfico de significancia HAC usa escala lineal. El coeficiente
  `brecha_lag1h` (|z| ≈ 155) se anota fuera del eje para poder leer el resto.
