"""Diccionario de datos en español para el modelado del dólar blue.

El código y los CSV conservan sus nombres técnicos originales
(`blue_sell`, `brecha_pct`, `blue_mid`, ...). Este módulo es la **única**
fuente de traducción: se usa para etiquetas de tablas y gráficos, sin tocar
los datos ni renombrar columnas.

Uso:
    from diccionario_datos import etiqueta, descripcion, tabla_glosario, renombrar

    etiqueta("blue_sell")            -> "Blue (venta)"
    descripcion("brecha_pct")        -> "Brecha cambiaria: ..."
    renombrar(df)                    -> copia con columnas en español (solo display)
    tabla_glosario()                 -> DataFrame listo para `display()`
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Campo:
    etiqueta: str          # nombre corto en español (tablas/gráficos)
    descripcion: str       # qué mide, en una frase
    unidad: str = ""       # unidad de medida
    ejemplo: str = ""      # valor ilustrativo


# --------------------------------------------------------------------------
# Núcleo del mercado cambiario
# --------------------------------------------------------------------------
DICCIONARIO: dict[str, Campo] = {
    "blue_sell": Campo(
        "Blue (venta)",
        "Precio de venta del dólar paralelo: lo que pagas por 1 USD en el "
        "mercado informal.",
        "Bs/USD", "11,85"),
    "blue_buy": Campo(
        "Blue (compra)",
        "Precio de compra del dólar paralelo: a cuánto te compran 1 USD.",
        "Bs/USD", "11,83"),
    "blue_mid": Campo(
        "Blue (medio)",
        "Precio medio del dólar blue. Es horario: en cada hora de la rejilla se "
        "conserva el último valor observado (interpolado si el hueco es corto).",
        "Bs/USD", "11,84"),
    "oficial_mid": Campo(
        "Oficial (medio)",
        "Tipo de cambio oficial del BCB. Se publica una vez por día y se repite "
        "en todas las horas de ese día.",
        "Bs/USD", "11,69"),
    "oficial_sell": Campo(
        "Oficial (venta)",
        "Precio de venta del dólar oficial del BCB (promedio del día).",
        "Bs/USD", "6,96"),
    "official_sell": Campo(
        "Oficial (venta, dato crudo)",
        "Precio de venta oficial tal como llega en history_full.csv.",
        "Bs/USD", "6,96"),
    "official_buy": Campo(
        "Oficial (compra, dato crudo)",
        "Precio de compra oficial tal como llega en history_full.csv.",
        "Bs/USD", "6,86"),
    "brecha_pct": Campo(
        "Brecha cambiaria",
        "Brecha blue vs oficial: (blue / oficial − 1) × 100. Mide cuánto más "
        "caro está el paralelo, no su nivel.",
        "%", "1,3 %"),
    "ret_blue_pct": Campo(
        "Retorno diario del blue",
        "Variación porcentual del blue respecto del día anterior.",
        "%", "+0,42 %"),
    "d_oficial_pct": Campo(
        "Variación diaria del oficial",
        "Variación porcentual diaria del tipo de cambio oficial.",
        "%", "0,00 %"),
    "vol_media_24h": Campo(
        "Volumen P2P (24 h)",
        "Volumen negociado en las últimas 24 h en el mercado P2P.",
        "USD", "105 309"),
}

# --------------------------------------------------------------------------
# Variables derivadas en el panel diario
# --------------------------------------------------------------------------
DICCIONARIO.update({
    "brecha_lag1": Campo("Brecha (rezago 1 d)", "Brecha del día anterior.", "%"),
    "brecha_lag2": Campo("Brecha (rezago 2 d)", "Brecha de hace 2 días.", "%"),
    "brecha_lag7": Campo("Brecha (rezago 7 d)", "Brecha de hace 7 días.", "%"),
    "brecha_fwd1": Campo("Brecha a 1 día", "Brecha del día siguiente (target h=1).", "%"),
    "brecha_fwd7": Campo("Brecha a 7 días", "Brecha 7 días después (target h=7).", "%"),
    "oficial_lag1": Campo("Oficial (rezago 1 d)", "Nivel oficial del día anterior.", "Bs/USD"),
    "d_oficial_lag1": Campo("Δ oficial (rezago 1 d)", "Variación del oficial del día anterior.", "%"),
    "ret_blue_lag1": Campo("Retorno blue (rezago 1 d)", "Retorno del blue del día anterior (momentum).", "%"),
    "vol_lag1": Campo("Volumen (rezago 1 d)", "Volumen P2P del día anterior.", "USD"),
    "dia_sem": Campo("Día de la semana", "0 = lunes … 6 = domingo.", "—"),
    "n_snapshots": Campo("Snapshots del día", "Nº de capturas usadas para promediar el día.", "—"),
})

# --------------------------------------------------------------------------
# Canales de la Gaceta (todos rezagados 1 día en el modelado)
# --------------------------------------------------------------------------
DICCIONARIO.update({
    "n_docs_total": Campo("Normas publicadas", "Nº total de normas publicadas ese día.", "—"),
    "n_docs_fx": Campo("Normas con relevancia FX",
                       "Normas con relevancia cambiaria media o alta.", "—"),
    "n_docs_fx_alta": Campo("Normas FX alta",
                            "Normas con relevancia cambiaria alta.", "—"),
    "n_docs_deuda": Campo("Normas de deuda externa",
                          "Normas cuyo tag principal es deuda externa.", "—"),
    "n_docs_comercio": Campo("Normas de comercio exterior",
                             "Normas cuyo tag principal es comercio exterior.", "—"),
    "n_docs_mineria": Campo("Normas de minería y energía",
                            "Normas cuyo tag principal es minería y energía.", "—"),
    "n_docs_fx_lag1": Campo("Normas FX (rezago 1 d)",
                            "Normas con relevancia FX del día anterior.", "—"),
    "n_docs_deuda_lag1": Campo("Deuda externa (rezago 1 d)",
                               "Normas de deuda externa del día anterior.", "—"),
    # variantes _lag1 usadas como predictores en el notebook
    "n_docs_fx_alta_lag1": Campo("Normas FX alta (rezago 1 d)",
                                 "Normas con relevancia FX alta del día anterior.", "—"),
    "n_docs_comercio_lag1": Campo("Comercio exterior (rezago 1 d)",
                                  "Normas de comercio exterior del día anterior.", "—"),
    "n_docs_mineria_lag1": Campo("Minería y energía (rezago 1 d)",
                                 "Normas de minería y energía del día anterior.", "—"),
})

# dummies de día de semana generadas con pd.get_dummies (0 = lunes,
# drop_first descarta dow_0 para evitar colinealidad con la constante)
DICCIONARIO.update({
    f"dow_{k}": Campo(
        ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"][k],
        f"Dummy de día de la semana: 1 si el día es {['lunes','martes','miércoles','jueves','viernes','sábado','domingo'][k]}.",
        "0/1")
    for k in range(7)
})

# --------------------------------------------------------------------------
# Tags temáticos de la Gaceta (usados en los gráficos de eventos)
# --------------------------------------------------------------------------
DICCIONARIO.update({
    "divisas_control_cambios": Campo("Divisas / control de cambios",
                                     "Tag temático: normas cambiarias y de divisas.", "—"),
    "deuda_externa": Campo("Deuda externa",
                           "Tag temático: endeudamiento externo.", "—"),
    "comercio_exterior": Campo("Comercio exterior",
                               "Tag temático: exportaciones e importaciones.", "—"),
    "general_administrativo": Campo("General / administrativo",
                                    "Tag temático: normas sin impacto económico directo.", "—"),
    "mineria_energia": Campo("Minería y energía",
                             "Tag temático: hidrocarburos, minería y electricidad.", "—"),
    "impuestos": Campo("Impuestos", "Tag temático: tributos.", "—"),
    "credito_financiero": Campo("Crédito financiero",
                                "Tag temático: crédito y sistema financiero.", "—"),
    "fiscal_presupuesto": Campo("Fiscal / presupuesto",
                                "Tag temático: gasto público y presupuesto.", "—"),
    "contrataciones_estado": Campo("Contrataciones del Estado",
                                   "Tag temático: compras y contrataciones públicas.", "—"),
})

# --------------------------------------------------------------------------
# Columnas de fecha
# --------------------------------------------------------------------------
DICCIONARIO.update({
    "fecha": Campo("Fecha", "Día calendario (YYYY-MM-DD).", "—"),
    "fecha_d": Campo("Fecha", "Día calendario (YYYY-MM-DD).", "—"),
    "timestamp": Campo("Marca de tiempo", "Instante del snapshot (epoch).", "s"),
})


def campo(nombre: str) -> Campo | None:
    """Devuelve el Campo asociado a un nombre técnico (o None)."""
    return DICCIONARIO.get(nombre)


def etiqueta(nombre: str, con_unidad: bool = False) -> str:
    """Nombre corto en español; si se pide, agrega la unidad entre paréntesis."""
    c = DICCIONARIO.get(nombre)
    if c is None:
        return nombre
    if con_unidad and c.unidad and c.unidad != "—":
        return f"{c.etiqueta} ({c.unidad})"
    return c.etiqueta


def descripcion(nombre: str) -> str:
    """Descripción larga en español (o cadena vacía si no está en el diccionario)."""
    c = DICCIONARIO.get(nombre)
    return c.descripcion if c else ""


def etiqueta_eje(nombre: str) -> str:
    """Etiqueta para el eje de un gráfico: 'Etiqueta [unidad]'.

    Se usan corchetes para no encadenar paréntesis cuando la etiqueta ya
    trae uno (p. ej. 'Blue (venta) [Bs/USD]').
    """
    c = DICCIONARIO.get(nombre)
    if c is None:
        return nombre
    if c.unidad and c.unidad != "—":
        return f"{c.etiqueta} [{c.unidad}]"
    return c.etiqueta


def etiquetas(nombres) -> list[str]:
    """Traduce una lista de columnas a etiquetas en español."""
    return [etiqueta(n) for n in nombres]


def etiqueta_tag(tag: str) -> str:
    """Etiqueta legible de un tag de la Gaceta (con respaldo genérico)."""
    return etiqueta(tag) if tag in DICCIONARIO else tag.replace("_", " ").capitalize()


def renombrar(df, con_unidad: bool = False):
    """Copia del DataFrame con las columnas en español. Solo para mostrar:
    no debe usarse para volver a indexar los datos."""
    return df.rename(columns={c: etiqueta(c, con_unidad) for c in df.columns})


# --------------------------------------------------------------------------
# Agrupación del glosario en 2 grupos
#   1. Modelo     → predictores de la regresión + targets
#   2. Referencia → datos crudos, canales base de la Gaceta y fechas
# --------------------------------------------------------------------------
GRUPO_MODELO = "1. Modelo (predictores y targets)"
GRUPO_REFERENCIA = "2. Referencia (datos crudos y Gaceta)"

_VARS_MODELO = {
    # targets de los modelos
    "brecha_pct", "blue_mid", "blue_sell", "brecha_fwd1", "brecha_fwd7",
    # predictores: lags de mercado
    "brecha_lag1", "brecha_lag2", "brecha_lag7", "oficial_lag1",
    "d_oficial_lag1", "ret_blue_lag1", "vol_lag1",
    # predictores: canales de la Gaceta rezagados
    "n_docs_fx_lag1", "n_docs_fx_alta_lag1", "n_docs_deuda_lag1",
    "n_docs_comercio_lag1", "n_docs_mineria_lag1",
    # predictores: dummies de calendario
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
}


def grupo(nombre: str) -> str:
    """Grupo del glosario para una variable (Referencia si no está mapeada)."""
    return GRUPO_MODELO if nombre in _VARS_MODELO else GRUPO_REFERENCIA


def tabla_glosario(nombres=None, grupo_filtro: str | None = None):
    """DataFrame (Grupo, Etiqueta, Variable, Descripción, Unidad, Ejemplo).

    Si `nombres` es None se listan todas las entradas. `grupo_filtro` acepta
    "modelo" o "referencia" (búsqueda por substring, insensible a mayúsculas).
    """
    import pandas as pd

    claves = list(DICCIONARIO) if nombres is None else list(nombres)
    filas = []
    for k in claves:
        c = DICCIONARIO.get(k)
        if c is None:
            continue
        filas.append({
            "Grupo": grupo(k),
            "Etiqueta": c.etiqueta,
            "Variable": k,
            "Descripción": c.descripcion,
            "Unidad": c.unidad,
            "Ejemplo": c.ejemplo,
        })
    df = pd.DataFrame(filas, columns=["Grupo", "Etiqueta", "Variable",
                                      "Descripción", "Unidad", "Ejemplo"])
    if grupo_filtro:
        df = df[df["Grupo"].str.lower().str.contains(
            grupo_filtro.lower(), regex=False)]
    return df


if __name__ == "__main__":
    print(tabla_glosario().to_string(index=False))
