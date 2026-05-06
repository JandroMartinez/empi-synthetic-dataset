"""
dataset_builder.py
──────────────────
Construye el dataset sintético completo (registros limpios + duplicados con errores)
y lo exporta en CSV, Excel y JSON Lines.

Uso desde línea de comandos:
  python dataset_builder.py --records 10000 --dup-rate 0.20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import ine_loader
from record_generator import generate_record

_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO DE SIMULACIÓN DE ERRORES (inline — Fase 4 lo expandirá)
# ──────────────────────────────────────────────────────────────────────────────

_TILDES = str.maketrans("áéíóúÁÉÍÓÚüÜñÑ", "aeiouAEIOUuUnN")

_CONFUSIONES = [("b", "v"), ("ll", "y"), ("c", "z"), ("s", "z")]


def _drop_tilde(text: str, rng: np.random.Generator, p: float = 0.30) -> str:
    if rng.random() < p:
        return text.translate(_TILDES)
    return text


def _typo_swap(text: str, rng: np.random.Generator, p: float = 0.05) -> str:
    """Transpone dos caracteres adyacentes aleatorios."""
    if len(text) < 3 or rng.random() >= p:
        return text
    i = int(rng.integers(0, len(text) - 1))
    lst = list(text)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)


def _confusion(text: str, rng: np.random.Generator, p: float = 0.10) -> str:
    """Aplica confusiones ortográficas (b/v, ll/y, c/z)."""
    if rng.random() >= p:
        return text
    original, replacement = _CONFUSIONES[int(rng.integers(0, len(_CONFUSIONES)))]
    return text.replace(original, replacement, 1)


def _corrupt_name(text: str, rng: np.random.Generator) -> str:
    text = _drop_tilde(text, rng, p=0.30)
    text = _typo_swap(text, rng, p=0.05)
    text = _confusion(text, rng, p=0.10)
    return text


def _corrupt_date(fecha: str, rng: np.random.Generator) -> str:
    """Cambia formato o añade error ±1 año."""
    if rng.random() < 0.20:  # cambio de formato
        parts = fecha.split("-")
        if len(parts) == 3:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
    if rng.random() < 0.05:  # error ±1 año
        year = int(fecha[:4])
        year += int(rng.choice([-1, 1]))
        return str(year) + fecha[4:]
    return fecha


def _corrupt_phone(telefono: str, rng: np.random.Generator) -> str:
    """Transposición de dígitos o borrado de un carácter."""
    if rng.random() < 0.10 and len(telefono) > 3:
        i = int(rng.integers(1, len(telefono) - 1))
        lst = list(telefono)
        lst[i], lst[i + 1] = lst[i + 1], lst[i]
        return "".join(lst)
    if rng.random() < 0.10 and len(telefono) > 5:
        i = int(rng.integers(1, len(telefono)))
        return telefono[:i] + telefono[i + 1:]
    return telefono


def _make_duplicate(record: dict, rng: np.random.Generator) -> dict:
    """Crea una copia modificada del registro con errores controlados."""
    dup = record.copy()
    dup["record_id"] = str(uuid.uuid4())
    dup["original_id"] = record["record_id"]
    dup["is_duplicate"] = True

    dup["nombre"] = _corrupt_name(dup["nombre"], rng)
    dup["primer_apellido"] = _corrupt_name(dup["primer_apellido"], rng)

    # Omisión del segundo apellido (p=0.05)
    if rng.random() < 0.05:
        dup["segundo_apellido"] = ""
    else:
        dup["segundo_apellido"] = _corrupt_name(dup["segundo_apellido"], rng)

    # Inversión de apellidos (p=0.03)
    if rng.random() < 0.03:
        dup["primer_apellido"], dup["segundo_apellido"] = (
            dup["segundo_apellido"], dup["primer_apellido"])

    dup["fecha_nacimiento"] = _corrupt_date(dup["fecha_nacimiento"], rng)
    dup["telefono"] = _corrupt_phone(dup["telefono"], rng)

    # Garantía: el duplicado debe diferir en al menos un campo del original.
    # Si todas las corrupciones probabilísticas fallaron (frecuente con vocabulario
    # reducido de demo), forzamos una transposición en el nombre.
    campos = ["nombre", "primer_apellido", "segundo_apellido", "fecha_nacimiento", "telefono"]
    if all(str(dup[c]) == str(record[c]) for c in campos):
        nombre = dup["nombre"]
        if len(nombre) >= 3:
            i = int(rng.integers(0, len(nombre) - 1))
            lst = list(nombre)
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            dup["nombre"] = "".join(lst)
        else:
            dup["fecha_nacimiento"] = _corrupt_date(
                record["fecha_nacimiento"], rng) or (record["fecha_nacimiento"] + "X")

    return dup


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ──────────────────────────────────────────────────────────────────────────────

def build_clean_dataset(n_records: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Genera n_records registros demográficos sintéticos limpios (sin duplicados).

    Parámetros
    ----------
    n_records : número de registros base a generar
    seed      : semilla para reproducibilidad

    Devuelve
    --------
    pd.DataFrame con 14 columnas
    """
    rng = np.random.default_rng(seed)
    data = ine_loader.load_all()
    records = [generate_record(**data, rng=rng) for _ in range(n_records)]
    return pd.DataFrame(records)


def build_dataset_with_duplicates(
    n_records: int = 10_000,
    dup_rate: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera el dataset completo: registros limpios + duplicados con errores.

    Parámetros
    ----------
    n_records : número de registros base (limpios)
    dup_rate  : fracción de registros base que tendrán un duplicado (0.10 – 0.30)
    seed      : semilla para reproducibilidad

    Devuelve
    --------
    pd.DataFrame con registros limpios + duplicados mezclados aleatoriamente.
    Las columnas is_duplicate y original_id forman el ground truth.
    """
    if not 0.10 <= dup_rate <= 0.30:
        raise ValueError(f"dup_rate debe estar en [0.10, 0.30], se recibió {dup_rate}")

    rng = np.random.default_rng(seed)
    data = ine_loader.load_all()

    clean = [generate_record(**data, rng=rng) for _ in range(n_records)]
    n_dups = int(n_records * dup_rate)
    dup_indices = rng.choice(n_records, size=n_dups, replace=False)

    duplicates = [_make_duplicate(clean[i], rng) for i in dup_indices]

    all_records = clean + duplicates
    rng.shuffle(all_records)  # mezcla para que duplicados no estén al final
    return pd.DataFrame(all_records)


def export(df: pd.DataFrame, output_dir: Path = _OUTPUT_DIR) -> dict:
    """
    Exporta el DataFrame en CSV UTF-8, Excel y JSON Lines.

    Devuelve dict con las rutas generadas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    csv_path = output_dir / "dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    paths["csv"] = str(csv_path)

    xlsx_path = output_dir / "dataset.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    paths["xlsx"] = str(xlsx_path)

    jsonl_path = output_dir / "dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")
    paths["jsonl"] = str(jsonl_path)

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset sintético EMPI")
    parser.add_argument("--records", type=int, default=10_000,
                        help="Número de registros base (default: 10000)")
    parser.add_argument("--dup-rate", type=float, default=0.20,
                        help="Fracción de duplicados (0.10-0.30, default: 0.20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--clean-only", action="store_true",
                        help="Genera solo registros limpios (sin duplicados)")
    args = parser.parse_args()

    print(f"Generando {args.records} registros base (seed={args.seed})…")
    if args.clean_only:
        df = build_clean_dataset(n_records=args.records, seed=args.seed)
    else:
        df = build_dataset_with_duplicates(
            n_records=args.records, dup_rate=args.dup_rate, seed=args.seed)
        n_dups = df["is_duplicate"].sum()
        print(f"  → {len(df)} filas totales ({n_dups} duplicados, {len(df)-n_dups} originales)")

    paths = export(df)
    for fmt, path in paths.items():
        print(f"  Exportado: {path}")


if __name__ == "__main__":
    _cli()
