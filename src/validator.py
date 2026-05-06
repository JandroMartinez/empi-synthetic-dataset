"""
validator.py
────────────
Validación del dataset sintético mediante Python Record Linkage Toolkit.

Pipeline:
  1. Bloqueo por primer_apellido (inicial) para reducir pares candidatos.
  2. Comparación de pares usando Jaro-Winkler (nombre, apellidos) y
     coincidencia exacta (fecha_nacimiento).
  3. Clasificación por umbral de puntuación total.
  4. Evaluación contra el ground truth (columnas is_duplicate / original_id).
  5. Informe con precisión, recall, F1 y análisis por tipo de error.

Uso:
    python src/validator.py [--input output/dataset.csv] [--threshold 2.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import recordlinkage

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 3.5   # suma de similitudes (rango 0-5)
WEIGHTS = {               # peso de cada campo en la puntuación total
    "nombre":           1.0,
    "primer_apellido":  1.0,
    "segundo_apellido": 1.0,
    "fecha_nacimiento": 1.0,
    "telefono":         1.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPROCESADO
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["is_duplicate"] = df["is_duplicate"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    df = df.reset_index(drop=True)
    return df


def build_ground_truth(df: pd.DataFrame) -> pd.MultiIndex:
    """
    Construye el MultiIndex de pares verdaderos (original, duplicado)
    a partir de las columnas is_duplicate y original_id.
    """
    dups = df[df["is_duplicate"] == True].copy()

    id_to_pos = {rid: pos for pos, rid in enumerate(df["record_id"].values)}

    pairs = []
    for _, row in dups.iterrows():
        orig_id = row["original_id"]
        dup_pos = id_to_pos.get(row["record_id"])
        orig_pos = id_to_pos.get(str(orig_id))
        if orig_pos is not None and dup_pos is not None:
            pairs.append((orig_pos, dup_pos))

    return pd.MultiIndex.from_tuples(pairs, names=["idx_orig", "idx_dup"])


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE DE RECORD LINKAGE
# ──────────────────────────────────────────────────────────────────────────────

def run_linkage(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Ejecuta el pipeline completo y devuelve un dict con métricas y detalles.
    Estrategia de bloqueo: primer_apellido + año de nacimiento (2 campos).
    Reduce ~98% los pares candidatos frente al bloqueo simple por apellido.
    """
    # Columnas auxiliares para bloqueo
    df = df.copy()
    df["anyo_nac"]     = df["fecha_nacimiento"].str[:4].fillna("0000")
    df["inicial_nombre"] = df["nombre"].str[0].str.upper().fillna("X")

    # 1. Bloqueo multi-pasada: unión de dos estrategias ──────────────────────
    #    Pasada A: mismo primer_apellido + mismo año de nacimiento
    #    Pasada B: mismo primer_apellido + misma inicial del nombre
    #    → mayor recall que una sola pasada
    idx_a = recordlinkage.Index()
    idx_a.block(left_on=["primer_apellido", "anyo_nac"])
    cand_a = idx_a.index(df)

    idx_b = recordlinkage.Index()
    idx_b.block(left_on=["primer_apellido", "inicial_nombre"])
    cand_b = idx_b.index(df)

    candidates = cand_a.union(cand_b)
    n_candidatos = len(candidates)

    # 2. Comparación de campos ───────────────────────────────────────────────
    comp = recordlinkage.Compare()
    comp.string("nombre",           "nombre",           method="jarowinkler", label="sim_nombre")
    comp.string("primer_apellido",  "primer_apellido",  method="jarowinkler", label="sim_p_apellido")
    comp.string("segundo_apellido", "segundo_apellido", method="jarowinkler", label="sim_s_apellido")
    comp.exact( "fecha_nacimiento", "fecha_nacimiento",                       label="sim_fecha")
    comp.string("telefono",         "telefono",         method="jarowinkler", label="sim_telefono")

    features = comp.compute(candidates, df)

    # 3. Puntuación y clasificación ──────────────────────────────────────────
    features["score"] = (
        features["sim_nombre"]     * WEIGHTS["nombre"] +
        features["sim_p_apellido"] * WEIGHTS["primer_apellido"] +
        features["sim_s_apellido"] * WEIGHTS["segundo_apellido"] +
        features["sim_fecha"]      * WEIGHTS["fecha_nacimiento"] +
        features["sim_telefono"]   * WEIGHTS["telefono"]
    )

    predicted_links = features[features["score"] >= threshold].index

    # 4. Ground truth ────────────────────────────────────────────────────────
    true_links = build_ground_truth(df)

    # 5. Métricas ────────────────────────────────────────────────────────────
    tp = len(predicted_links.intersection(true_links))
    fp = len(predicted_links) - tp
    fn = len(true_links) - tp
    tn = n_candidatos - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # 6. Curva precisión-recall a distintos umbrales ─────────────────────────
    curve = []
    for thr in np.arange(0.5, 5.1, 0.25):
        pred = features[features["score"] >= thr].index
        _tp = len(pred.intersection(true_links))
        _fp = len(pred) - _tp
        _fn = len(true_links) - _tp
        _p  = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _r  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        _f1 = 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
        curve.append({"umbral": round(thr, 2), "precision": round(_p, 4),
                      "recall": round(_r, 4), "f1": round(_f1, 4)})

    # 7. Análisis de falsos negativos por tipo de error ──────────────────────
    missed_pairs = true_links.difference(predicted_links)
    error_analysis = _analyze_missed(df, features, missed_pairs)

    return {
        "n_registros":    len(df),
        "n_duplicados":   int(df["is_duplicate"].sum()),
        "n_candidatos":   n_candidatos,
        "umbral":         threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "curva_pr":       curve,
        "error_analysis": error_analysis,
    }


def _analyze_missed(df: pd.DataFrame, features: pd.DataFrame,
                    missed: pd.MultiIndex) -> dict:
    """Cuenta qué campos difieren en los pares no detectados."""
    if len(missed) == 0:
        return {}

    missed_feats = features.loc[features.index.intersection(missed)]
    if missed_feats.empty:
        return {}

    umbral_campo = 0.85
    return {
        "total_fn":              len(missed),
        "falla_nombre":          int((missed_feats["sim_nombre"]     < umbral_campo).sum()),
        "falla_primer_apellido": int((missed_feats["sim_p_apellido"] < umbral_campo).sum()),
        "falla_segundo_apellido":int((missed_feats["sim_s_apellido"] < umbral_campo).sum()),
        "falla_fecha":           int((missed_feats["sim_fecha"]      < 1.0).sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# INFORME
# ──────────────────────────────────────────────────────────────────────────────

def print_report(r: dict) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print("  INFORME DE VALIDACIÓN — EMPI Synthetic Dataset")
    print(sep)
    print(f"  Registros totales   : {r['n_registros']:,}")
    print(f"  Duplicados reales   : {r['n_duplicados']:,}")
    print(f"  Pares candidatos    : {r['n_candidatos']:,}")
    print(f"  Umbral de decisión  : {r['umbral']}")
    print(sep)
    print(f"  Verdaderos positivos: {r['tp']:,}")
    print(f"  Falsos positivos    : {r['fp']:,}")
    print(f"  Falsos negativos    : {r['fn']:,}")
    print(sep)
    print(f"  Precisión           : {r['precision']:.4f}  ({r['precision']*100:.1f}%)")
    print(f"  Recall              : {r['recall']:.4f}  ({r['recall']*100:.1f}%)")
    print(f"  F1-Score            : {r['f1']:.4f}  ({r['f1']*100:.1f}%)")
    print(sep)
    if r.get("error_analysis"):
        ea = r["error_analysis"]
        print("  Análisis de falsos negativos:")
        print(f"    Fallo en nombre          : {ea.get('falla_nombre',0):,}")
        print(f"    Fallo en primer apellido : {ea.get('falla_primer_apellido',0):,}")
        print(f"    Fallo en segundo apellido: {ea.get('falla_segundo_apellido',0):,}")
        print(f"    Fallo en fecha           : {ea.get('falla_fecha',0):,}")
        print(sep)
    print("  Curva Precisión-Recall (selección):")
    print(f"  {'Umbral':>7}  {'Precisión':>9}  {'Recall':>7}  {'F1':>7}")
    for row in r["curva_pr"]:
        if row["umbral"] in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
            print(f"  {row['umbral']:>7.2f}  {row['precision']:>9.4f}  "
                  f"{row['recall']:>7.4f}  {row['f1']:>7.4f}")
    print(sep)


def save_report(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"  Informe JSON → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el dataset EMPI sintético.")
    parser.add_argument("--input",     default="output/dataset.csv",
                        help="CSV del dataset generado")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Umbral de puntuación para declarar par duplicado")
    parser.add_argument("--output",    default="output/validation_report.json",
                        help="Fichero JSON con el informe completo")
    args = parser.parse_args()

    print(f"Cargando dataset: {args.input}")
    df = load_dataset(Path(args.input))

    print(f"Ejecutando record linkage (umbral={args.threshold})...")
    report = run_linkage(df, threshold=args.threshold)

    print_report(report)
    save_report(report, Path(args.output))


if __name__ == "__main__":
    main()
