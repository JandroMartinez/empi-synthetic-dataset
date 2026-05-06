"""
record_generator.py
───────────────────
Genera un único registro demográfico sintético limpio (sin duplicados ni errores)
a partir de los datos del INE cargados por ine_loader.

Atributos del registro (perfil IHE PDQ):
  record_id, nombre, primer_apellido, segundo_apellido,
  fecha_nacimiento, sexo, provincia, municipio, direccion,
  cod_postal, telefono, correo_electronico,
  is_duplicate (False), original_id (None)
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ine_loader import PREFIJOS_PROVINCIALES

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

_DOMINIOS_EMAIL = ["gmail.com", "hotmail.es", "yahoo.es", "outlook.es", "telefonica.net"]
_TIPOS_VIA = ["Calle", "Avenida", "Plaza", "Paseo", "Ronda", "Camino", "Travesía"]
_NOMBRES_VIA = [
    "Mayor", "Real", "España", "Cervantes", "Constitución", "Goya",
    "Velázquez", "Picasso", "Rosaleda", "Libertad", "Paz", "Europa",
    "América", "Gran Vía", "Príncipe", "Infanta", "Alameda",
]

# Mapeo grupo_edad → rango de años de nacimiento (referencia: 2024)
_EDAD_RANGES = {
    "0-4": (2020, 2024), "5-9": (2015, 2019), "10-14": (2010, 2014),
    "15-19": (2005, 2009), "20-24": (2000, 2004), "25-29": (1995, 1999),
    "30-34": (1990, 1994), "35-39": (1985, 1989), "40-44": (1980, 1984),
    "45-49": (1975, 1979), "50-54": (1970, 1974), "55-59": (1965, 1969),
    "60-64": (1960, 1964), "65-69": (1955, 1959), "70+": (1920, 1954),
}


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def _sample(df: pd.DataFrame, prob_col: str, value_col: str, rng: np.random.Generator) -> str:
    """Muestrea un valor de df[value_col] con probabilidades df[prob_col]."""
    idx = rng.choice(len(df), p=df[prob_col].values)
    return df[value_col].iloc[idx]


def _normalize_for_email(text: str) -> str:
    """Convierte texto a ASCII sin tildes y en minúsculas para usar en emails."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", ascii_str.lower())


def _gen_fecha(grupo_edad: str, rng: np.random.Generator) -> date:
    """Genera una fecha de nacimiento aleatoria dentro del grupo quinquenal."""
    anio_min, anio_max = _EDAD_RANGES.get(grupo_edad, (1960, 2000))
    start = date(anio_min, 1, 1)
    end = date(anio_max, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta_days + 1)))


def _gen_telefono(provincia_cod: str, rng: np.random.Generator) -> str:
    """Genera un número de teléfono con prefijo provincial (CNMC)."""
    prefijo = PREFIJOS_PROVINCIALES.get(provincia_cod, "9")
    digitos_restantes = 9 - len(prefijo)
    resto = "".join(str(d) for d in rng.integers(0, 10, size=digitos_restantes))
    return prefijo + resto


def _gen_email(nombre: str, apellido: str, rng: np.random.Generator) -> str:
    """Construye un correo electrónico a partir del nombre y primer apellido."""
    base = _normalize_for_email(nombre.split()[0]) + "." + _normalize_for_email(apellido)
    sufijo = int(rng.integers(1, 999))
    dominio = _DOMINIOS_EMAIL[int(rng.integers(0, len(_DOMINIOS_EMAIL)))]
    return f"{base}{sufijo}@{dominio}"


def _gen_direccion(rng: np.random.Generator) -> str:
    """Genera una dirección ficticia pero con formato válido."""
    tipo = _TIPOS_VIA[int(rng.integers(0, len(_TIPOS_VIA)))]
    nombre_via = _NOMBRES_VIA[int(rng.integers(0, len(_NOMBRES_VIA)))]
    numero = int(rng.integers(1, 200))
    return f"{tipo} {nombre_via}, {numero}"


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def generate_record(
    nombres_m: pd.DataFrame,
    nombres_f: pd.DataFrame,
    apellidos: pd.DataFrame,
    padron: pd.DataFrame,
    municipios: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Genera un único registro demográfico sintético limpio.

    Parámetros
    ----------
    nombres_m, nombres_f : DataFrame [nombre, prob]
    apellidos            : DataFrame [apellido, prob]
    padron               : DataFrame [provincia_cod, grupo_edad, sexo, prob]
    municipios           : dict {provincia_cod: DataFrame [municipio, cod_postal, prob_municipio]}
    rng                  : numpy Generator para reproducibilidad

    Devuelve
    --------
    dict con 14 claves: los 10 atributos demográficos + cod_postal +
    is_duplicate=False + original_id=None + record_id
    """
    # 1. Muestrear fila del Padrón → determina provincia, grupo_edad, sexo
    padron_row = padron.sample(n=1, weights="prob", random_state=int(rng.integers(0, 2**31))).iloc[0]
    provincia_cod = str(padron_row["provincia_cod"])
    grupo_edad = str(padron_row["grupo_edad"])
    sexo = str(padron_row["sexo"])  # 'M' o 'F'

    # 2. Fecha de nacimiento dentro del grupo quinquenal
    fecha_nacimiento = _gen_fecha(grupo_edad, rng)

    # 3. Nombre coherente con sexo
    nombres_df = nombres_m if sexo == "M" else nombres_f
    nombre = _sample(nombres_df, "prob", "nombre", rng)

    # 4–5. Dos apellidos independientes
    primer_apellido = _sample(apellidos, "prob", "apellido", rng)
    segundo_apellido = _sample(apellidos, "prob", "apellido", rng)

    # 6. Municipio dentro de la provincia
    muns = municipios.get(provincia_cod)
    if muns is not None and not muns.empty:
        mun_row = muns.sample(n=1, weights="prob_municipio",
                              random_state=int(rng.integers(0, 2**31))).iloc[0]
        municipio = str(mun_row["municipio"])
        cod_postal = str(mun_row["cod_postal"])
    else:
        municipio = "Desconocido"
        cod_postal = "00000"

    # 7. Dirección ficticia
    direccion = _gen_direccion(rng)

    # 8. Teléfono con prefijo provincial
    telefono = _gen_telefono(provincia_cod, rng)

    # 9. Correo electrónico
    correo_electronico = _gen_email(nombre, primer_apellido, rng)

    return {
        "record_id": str(uuid.uuid4()),
        "nombre": nombre,
        "primer_apellido": primer_apellido,
        "segundo_apellido": segundo_apellido,
        "fecha_nacimiento": fecha_nacimiento.isoformat(),
        "sexo": sexo,
        "provincia": provincia_cod,
        "municipio": municipio,
        "direccion": direccion,
        "cod_postal": cod_postal,
        "telefono": telefono,
        "correo_electronico": correo_electronico,
        "is_duplicate": False,
        "original_id": None,
    }
