"""
ine_loader.py
─────────────
Carga y preprocesa los ficheros de datos del INE para alimentar el generador
de registros demográficos sintéticos.

Fuentes de datos esperadas en data/raw/:
  - ine_nombres_hombres.csv   (nombre, frecuencia)
  - ine_nombres_mujeres.csv   (nombre, frecuencia)
  - ine_apellidos.csv         (apellido, frecuencia)
  - ine_padron.csv            (provincia_cod, grupo_edad, sexo, total)
  - ine_municipios.csv        (provincia_cod, municipio, cod_postal, poblacion)

Si los ficheros no existen se usan datos de demostración embebidos para que
el módulo sea ejecutable sin descargas previas.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# ──────────────────────────────────────────────────────────────────────────────
# DATOS DE DEMOSTRACIÓN (usados cuando faltan los CSVs del INE)
# ──────────────────────────────────────────────────────────────────────────────

_DEMO_NOMBRES_H = [
    ("Antonio", 680_000), ("Manuel", 600_000), ("José", 580_000),
    ("Francisco", 470_000), ("David", 430_000), ("Juan", 400_000),
    ("Javier", 380_000), ("José Antonio", 360_000), ("Miguel", 340_000),
    ("Alejandro", 320_000),
]

_DEMO_NOMBRES_M = [
    ("María", 660_000), ("Carmen", 420_000), ("Josefa", 380_000),
    ("Isabel", 360_000), ("Ana María", 300_000), ("Laura", 290_000),
    ("María Dolores", 280_000), ("Cristina", 270_000), ("Marta", 250_000),
    ("Elena", 240_000),
]

_DEMO_APELLIDOS = [
    ("García", 1_500_000), ("González", 920_000), ("Rodríguez", 870_000),
    ("Fernández", 780_000), ("López", 750_000), ("Martínez", 720_000),
    ("Sánchez", 660_000), ("Pérez", 620_000), ("Gómez", 490_000),
    ("Martín", 450_000), ("Jiménez", 440_000), ("Ruiz", 420_000),
    ("Hernández", 390_000), ("Díaz", 370_000), ("Moreno", 350_000),
    ("Álvarez", 340_000), ("Muñoz", 330_000), ("Romero", 320_000),
    ("Alonso", 300_000), ("Gutiérrez", 290_000),
]

# Provincias (código INE 2 dígitos) con población aproximada
_DEMO_PROVINCIAS = [
    ("28", 6_700_000), ("08", 5_700_000), ("41", 1_960_000),
    ("46", 2_600_000), ("28", 6_700_000), ("30", 1_500_000),
    ("50", 970_000), ("47", 510_000), ("18", 930_000),
    ("10", 390_000),
]

_DEMO_MUNICIPIOS: Dict[str, list] = {
    "28": [("Madrid", "28001", 3_300_000), ("Móstoles", "28931", 210_000),
           ("Alcalá de Henares", "28801", 200_000)],
    "08": [("Barcelona", "08001", 1_600_000), ("Hospitalet de Llobregat", "08901", 260_000),
           ("Badalona", "08911", 220_000)],
    "41": [("Sevilla", "41001", 690_000), ("Dos Hermanas", "41701", 140_000)],
    "46": [("Valencia", "46001", 800_000), ("Torrent", "46900", 110_000)],
    "30": [("Murcia", "30001", 450_000), ("Cartagena", "30201", 220_000)],
}

_DEMO_GRUPOS_EDAD = [
    ("0-4", "M", 1_100_000), ("5-9", "M", 1_200_000), ("10-14", "M", 1_250_000),
    ("15-19", "M", 1_190_000), ("20-24", "M", 1_240_000), ("25-29", "M", 1_430_000),
    ("30-34", "M", 1_620_000), ("35-39", "M", 1_900_000), ("40-44", "M", 2_000_000),
    ("45-49", "M", 2_100_000), ("50-54", "M", 1_900_000), ("55-59", "M", 1_750_000),
    ("60-64", "M", 1_600_000), ("65-69", "M", 1_400_000), ("70+", "M", 3_500_000),
    ("0-4", "F", 1_060_000), ("5-9", "F", 1_160_000), ("10-14", "F", 1_200_000),
    ("15-19", "F", 1_140_000), ("20-24", "F", 1_190_000), ("25-29", "F", 1_390_000),
    ("30-34", "F", 1_580_000), ("35-39", "F", 1_860_000), ("40-44", "F", 1_960_000),
    ("45-49", "F", 2_060_000), ("50-54", "F", 1_880_000), ("55-59", "F", 1_740_000),
    ("60-64", "F", 1_610_000), ("65-69", "F", 1_440_000), ("70+", "F", 4_600_000),
]

# Prefijos telefónicos por provincia (CNMC)
PREFIJOS_PROVINCIALES: Dict[str, str] = {
    "28": "91", "08": "93", "41": "95", "46": "96", "30": "968",
    "50": "976", "47": "983", "18": "958", "10": "927",
}


def _normalize_probs(df: pd.DataFrame, freq_col: str = "frecuencia") -> pd.DataFrame:
    total = df[freq_col].sum()
    df = df.copy()
    df["prob"] = df[freq_col] / total
    return df


def load_nombres(sexo: str) -> pd.DataFrame:
    """
    Devuelve DataFrame [nombre, prob] para el sexo indicado ('M' o 'F').
    Lee ine_nombres_hombres.csv / ine_nombres_mujeres.csv si existen,
    o usa datos de demostración.
    """
    fname = "ine_nombres_hombres.csv" if sexo == "M" else "ine_nombres_mujeres.csv"
    path = _DATA_RAW / fname
    if path.exists():
        df = pd.read_csv(path, names=["nombre", "frecuencia"], header=0)
    else:
        warnings.warn(f"{fname} no encontrado — usando datos de demostración.", stacklevel=2)
        demo = _DEMO_NOMBRES_H if sexo == "M" else _DEMO_NOMBRES_M
        df = pd.DataFrame(demo, columns=["nombre", "frecuencia"])
    return _normalize_probs(df)[["nombre", "prob"]]


def load_apellidos() -> pd.DataFrame:
    """
    Devuelve DataFrame [apellido, prob] con el top-1000 de apellidos del INE.
    """
    path = _DATA_RAW / "ine_apellidos.csv"
    if path.exists():
        df = pd.read_csv(path, names=["apellido", "frecuencia"], header=0)
    else:
        warnings.warn("ine_apellidos.csv no encontrado — usando datos de demostración.", stacklevel=2)
        df = pd.DataFrame(_DEMO_APELLIDOS, columns=["apellido", "frecuencia"])
    return _normalize_probs(df)[["apellido", "prob"]]


def load_padron() -> pd.DataFrame:
    """
    Devuelve DataFrame [provincia_cod, grupo_edad, sexo, prob] con la distribución
    conjunta del Padrón Municipal.
    """
    # Preferir el fichero preprocesado (provincia + edad + sexo); fallback al raw
    path = _DATA_RAW / "ine_padron_procesado.csv"
    if not path.exists():
        path = _DATA_RAW / "ine_padron.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"provincia_cod": str})
        df.columns = df.columns.str.lower()
    else:
        warnings.warn("ine_padron.csv no encontrado — usando datos de demostración.", stacklevel=2)
        df = pd.DataFrame(_DEMO_GRUPOS_EDAD, columns=["grupo_edad", "sexo", "total"])
        # Añadir provincia aleatoria para demo
        provincias_demo = ["28", "08", "41", "46", "30"]
        rng = np.random.default_rng(0)
        df["provincia_cod"] = rng.choice(provincias_demo, size=len(df))
        df.rename(columns={"total": "frecuencia"}, inplace=True)
    total = df["frecuencia"].sum()
    df["prob"] = df["frecuencia"] / total
    return df[["provincia_cod", "grupo_edad", "sexo", "prob"]]


def load_municipios() -> Dict[str, pd.DataFrame]:
    """
    Devuelve dict {provincia_cod: DataFrame([municipio, cod_postal, prob_municipio])}.
    """
    path = _DATA_RAW / "ine_municipios.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"provincia_cod": str, "cod_postal": str})
    else:
        warnings.warn("ine_municipios.csv no encontrado — usando datos de demostración.", stacklevel=2)
        rows = []
        for prov, entries in _DEMO_MUNICIPIOS.items():
            for mun, cp, pob in entries:
                rows.append({"provincia_cod": prov, "municipio": mun,
                             "cod_postal": cp, "poblacion": pob})
        df = pd.DataFrame(rows)

    result: Dict[str, pd.DataFrame] = {}
    for prov, grp in df.groupby("provincia_cod"):
        grp = grp.copy()
        grp["prob_municipio"] = grp["poblacion"] / grp["poblacion"].sum()
        result[str(prov)] = grp[["municipio", "cod_postal", "prob_municipio"]].reset_index(drop=True)
    return result


def load_all() -> dict:
    """
    Carga todos los recursos del INE y los devuelve en un diccionario listo
    para pasarse a generate_record().
    """
    return {
        "nombres_m": load_nombres("M"),
        "nombres_f": load_nombres("F"),
        "apellidos": load_apellidos(),
        "padron": load_padron(),
        "municipios": load_municipios(),
    }
