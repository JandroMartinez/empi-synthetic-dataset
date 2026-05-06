"""
tests/test_dataset.py
─────────────────────
Batería de tests automáticos (pytest) para validar el dataset generado.

Ejecutar:
  cd empi-synthetic-dataset
  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_builder import build_clean_dataset, build_dataset_with_duplicates
from record_generator import generate_record
import ine_loader

# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_df():
    """Dataset limpio de 1000 registros (suficiente para tests estadísticos)."""
    return build_clean_dataset(n_records=1_000, seed=42)


@pytest.fixture(scope="module")
def full_df():
    """Dataset completo con duplicados al 20%."""
    return build_dataset_with_duplicates(n_records=1_000, dup_rate=0.20, seed=42)


# ──────────────────────────────────────────────────────────────────────────────
# TESTS DE ESTRUCTURA
# ──────────────────────────────────────────────────────────────────────────────

class TestEstructura:

    def test_columnas_presentes(self, clean_df):
        expected = {
            "record_id", "nombre", "primer_apellido", "segundo_apellido",
            "fecha_nacimiento", "sexo", "provincia", "municipio", "direccion",
            "cod_postal", "telefono", "correo_electronico",
            "is_duplicate", "original_id",
        }
        assert expected.issubset(set(clean_df.columns))

    def test_sin_nulos_en_campos_obligatorios(self, clean_df):
        obligatorios = [
            "record_id", "nombre", "primer_apellido", "fecha_nacimiento",
            "sexo", "provincia", "telefono", "correo_electronico",
        ]
        for col in obligatorios:
            assert clean_df[col].notna().all(), f"Nulos en columna: {col}"

    def test_record_id_unico(self, clean_df):
        assert clean_df["record_id"].nunique() == len(clean_df), \
            "record_id no es único"

    def test_is_duplicate_false_en_limpios(self, clean_df):
        assert not clean_df["is_duplicate"].any(), \
            "Dataset limpio contiene registros marcados como duplicados"

    def test_original_id_none_en_limpios(self, clean_df):
        assert clean_df["original_id"].isna().all(), \
            "Dataset limpio contiene original_id no nulos"


# ──────────────────────────────────────────────────────────────────────────────
# TESTS DE FORMATO
# ──────────────────────────────────────────────────────────────────────────────

class TestFormato:

    def test_sexo_enum(self, clean_df):
        valores_sexo = set(clean_df["sexo"].unique())
        assert valores_sexo.issubset({"M", "F"}), \
            f"Valores inesperados en sexo: {valores_sexo}"

    def test_fecha_nacimiento_iso(self, clean_df):
        import re
        patron = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        # Solo registros limpios (no duplicados que pueden tener formato alterado)
        limpios = clean_df[~clean_df["is_duplicate"]]
        invalidas = limpios["fecha_nacimiento"].apply(
            lambda x: not bool(patron.match(str(x))))
        assert not invalidas.any(), \
            f"Fechas con formato incorrecto: {limpios.loc[invalidas, 'fecha_nacimiento'].head()}"

    def test_email_formato(self, clean_df):
        import re
        patron = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
        invalidos = clean_df["correo_electronico"].apply(
            lambda x: not bool(patron.match(str(x))))
        assert not invalidos.any(), \
            f"Emails con formato incorrecto: {clean_df.loc[invalidos, 'correo_electronico'].head()}"

    def test_telefono_solo_digitos(self, clean_df):
        invalidos = clean_df["telefono"].apply(
            lambda x: not str(x).isdigit())
        assert not invalidos.any(), \
            f"Teléfonos con caracteres no numéricos: {clean_df.loc[invalidos, 'telefono'].head()}"


# ──────────────────────────────────────────────────────────────────────────────
# TESTS ESTADÍSTICOS
# ──────────────────────────────────────────────────────────────────────────────

class TestEstadisticos:

    def test_distribucion_sexo(self, clean_df):
        """Distribución 49/51 ±3pp (Padrón Municipal)."""
        ratio_m = (clean_df["sexo"] == "M").mean()
        assert 0.46 <= ratio_m <= 0.54, \
            f"Proporción de hombres fuera de rango: {ratio_m:.2%}"

    def test_top5_provincias_acumulan_suficiente(self, clean_df):
        """Las 5 provincias más pobladas deben acumular ≥30% del total."""
        top5 = clean_df["provincia"].value_counts(normalize=True).head(5).sum()
        assert top5 >= 0.30, \
            f"Top-5 provincias acumulan solo {top5:.2%}"

    def test_record_id_uuid_formato(self, clean_df):
        """record_id debe ser UUID v4 válido."""
        import uuid as _uuid
        invalidos = 0
        for rid in clean_df["record_id"]:
            try:
                _uuid.UUID(str(rid), version=4)
            except ValueError:
                invalidos += 1
        assert invalidos == 0, f"{invalidos} record_id no son UUID válidos"


# ──────────────────────────────────────────────────────────────────────────────
# TESTS DE DUPLICADOS
# ──────────────────────────────────────────────────────────────────────────────

class TestDuplicados:

    def test_tasa_duplicados_objetivo(self, full_df):
        """La tasa real de duplicados debe estar cerca del 20% ±2pp."""
        tasa = full_df["is_duplicate"].mean()
        assert 0.15 <= tasa <= 0.25, \
            f"Tasa de duplicados fuera de rango: {tasa:.2%}"

    def test_original_id_referencia_valida(self, full_df):
        """original_id de cada duplicado debe existir como record_id."""
        ids_originales = set(full_df["record_id"].values)
        duplicados = full_df[full_df["is_duplicate"]]
        invalidos = duplicados["original_id"].apply(
            lambda x: x not in ids_originales)
        assert not invalidos.any(), \
            f"{invalidos.sum()} duplicados con original_id inválido"

    def test_duplicados_difieren_del_original(self, full_df):
        """Un duplicado debe diferir en al menos un campo del registro original."""
        originales = full_df[~full_df["is_duplicate"]].set_index("record_id")
        duplicados = full_df[full_df["is_duplicate"]]
        campos = ["nombre", "primer_apellido", "segundo_apellido",
                  "fecha_nacimiento", "telefono"]
        n_identicos = 0
        for _, dup in duplicados.iterrows():
            orig = originales.loc[dup["original_id"]]
            if all(str(dup[c]) == str(orig[c]) for c in campos):
                n_identicos += 1
        assert n_identicos == 0, \
            f"{n_identicos} duplicados idénticos al original en todos los campos"

    def test_record_id_unico_en_dataset_completo(self, full_df):
        assert full_df["record_id"].nunique() == len(full_df), \
            "record_id no es único en el dataset con duplicados"


# ──────────────────────────────────────────────────────────────────────────────
# TESTS DE CARGA INE
# ──────────────────────────────────────────────────────────────────────────────

class TestIneLoader:

    def test_load_nombres_hombres(self):
        df = ine_loader.load_nombres("M")
        assert "nombre" in df.columns
        assert "prob" in df.columns
        assert abs(df["prob"].sum() - 1.0) < 1e-6

    def test_load_nombres_mujeres(self):
        df = ine_loader.load_nombres("F")
        assert abs(df["prob"].sum() - 1.0) < 1e-6

    def test_load_apellidos(self):
        df = ine_loader.load_apellidos()
        assert len(df) >= 10
        assert abs(df["prob"].sum() - 1.0) < 1e-6

    def test_load_padron(self):
        df = ine_loader.load_padron()
        assert set(["provincia_cod", "grupo_edad", "sexo", "prob"]).issubset(df.columns)
        assert abs(df["prob"].sum() - 1.0) < 1e-6

    def test_load_municipios(self):
        muns = ine_loader.load_municipios()
        assert isinstance(muns, dict)
        assert len(muns) > 0
        for prov, df in muns.items():
            assert abs(df["prob_municipio"].sum() - 1.0) < 1e-6
