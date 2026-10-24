"""Estilo compartido del proyecto: paleta, rcParams y utilidades de guardado.

Fuente única de verdad para todos los gráficos (informe y notebook). La
paleta y los rcParams provienen de `report/make_figures.py` (patrón original
de class_one/eda_report.py).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Paleta del proyecto
BLUE, ORANGE, GREEN, RED, GRAY = "#2563eb", "#f59e0b", "#10b981", "#ef4444", "#6b7280"

# Colores auxiliares del notebook
GRIS_CRUDO = "#94a3b8"   # series crudas (snapshots)
AZUL_EDA = "#0891b2"     # EDA horario (grilla regular)


def aplicar_estilo() -> None:
    """rcParams del informe. Opt-in: solo quien quiera el estilo del informe
    lo aplica (el notebook no lo usa, para no alterar sus figuras)."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})


def fmt_es(v: float, nd: int = 2) -> str:
    """Número con coma decimal (español) para tablas LaTeX."""
    return f"{v:.{nd}f}".replace(".", ",")


def tex(s: str) -> str:
    """Escapa _ y & para LaTeX."""
    return s.replace("_", r"\_").replace("&", r"\&")


def finish(fig, path, label: str | None = None) -> None:
    """Guarda la figura en `path` (bbox tight), la cierra e informa."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(label if label else f"guardado: {path}")
