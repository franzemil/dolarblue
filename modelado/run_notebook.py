"""Ejecuta y verifica notebooks del proyecto (paso de verificación del plan).

Uso:
    cd 01_intro && .venv/bin/python final_project/modelado/run_notebook.py
    cd 01_intro && .venv/bin/python final_project/modelado/run_notebook.py --nb proyecto_notebook.ipynb
    cd 01_intro && .venv/bin/python final_project/modelado/run_notebook.py --nb analisis_gaceta_fx.ipynb

Sin `--nb` ejecuta solo el notebook de resultados (`proyecto_notebook.ipynb`),
que es el que alimenta el informe. El notebook de análisis sigue disponible con
`--nb analisis_gaceta_fx.ipynb`. Con `--nb` acepta uno o varios nombres/rutas
(relativas a modelado/).

Falla (exit 1) si:
  - alguna celda lanza una excepción,
  - alguna métrica de evaluación queda en NaN,
  - faltan figuras generadas por el notebook de análisis.
"""
import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

MODELADO = Path(__file__).resolve().parent

# Figuras que produce el notebook de análisis (se verifican solo para él)
FIGURAS = ["nb_rl_horaria_test.png", "nb_arima_horaria_test.png",
           "nb_raw_vs_eda.png", "nb_confusion_direccion.png",
           "nb_roc_direccion.png", "nb_event_study_tag.png",
           "nb_corr_tag_heatmap.png"]

NB_ANALISIS = "analisis_gaceta_fx.ipynb"
NB_RESULTADOS = "proyecto_notebook.ipynb"
POR_DEFECTO = [NB_RESULTADOS]

try:
    from dotenv import load_dotenv
    load_dotenv(MODELADO / ".env", override=False)
except ImportError:
    pass
KERNEL = "python3"   # el del propio .venv (evita depender de un kernelspec global)


def ejecutar(ruta: Path) -> list[str]:
    """Ejecuta un notebook y devuelve la lista de problemas encontrados."""
    problemas: list[str] = []
    print("=" * 72)
    print(f"NOTEBOOK: {ruta.name}")
    print("=" * 72)

    nb = nbformat.read(ruta, as_version=4)
    client = NotebookClient(
        nb, kernel_name=KERNEL, timeout=900,
        resources={"metadata": {"path": str(MODELADO)}},
    )
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001
        problemas.append(f"{ruta.name}: ejecución abortada: {exc}")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problemas.append(f"{ruta.name} celda {i}: "
                                 f"{out.get('ename')}: {out.get('evalue')}")
            if out.get("output_type") == "stream" and "NaN" in out.get("text", ""):
                problemas.append(f"{ruta.name} celda {i}: salida contiene NaN")

    nbformat.write(nb, ruta)

    if ruta.name == NB_ANALISIS:
        faltan = [f for f in FIGURAS if not (MODELADO / "outputs" / f).exists()]
        if faltan:
            problemas.append(f"{ruta.name}: figuras ausentes: {faltan}")

    return problemas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nb", nargs="+", default=None,
                    help="notebooks a ejecutar (default: proyecto_notebook.ipynb)")
    args = ap.parse_args()

    nombres = args.nb or POR_DEFECTO
    rutas = [(MODELADO / n).resolve() for n in nombres]

    inexistentes = [r.name for r in rutas if not r.exists()]
    if inexistentes:
        print(f"❌ no encontrados: {inexistentes}")
        return 1

    print(f"kernel: {KERNEL}")
    problemas: list[str] = []
    for ruta in rutas:
        problemas += ejecutar(ruta)

    print("=" * 72)
    if problemas:
        print(f"❌ {len(problemas)} problema(s):")
        for p in problemas:
            print("  -", p)
        return 1

    print(f"✅ {len(rutas)} notebook(s) ejecutado(s) sin errores ni NaN "
          f"({', '.join(r.name for r in rutas)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
