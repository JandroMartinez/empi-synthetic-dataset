# empi-synthetic-dataset

Generador de un dataset sintético de registros demográficos españoles con duplicados y errores para la evaluación de sistemas EMPI.

TFM — Máster en Big Data, Universidad Internacional Isabel I de Castilla  
Autor: Alejandro Martínez Gracia · Tutor: Juan Manuel Pascual

---

## Estructura del repositorio

```
empi-synthetic-dataset/
├── data/
│   ├── raw/           # Ficheros INE originales (no versionados)
│   └── processed/     # CSVs normalizados (generados por ine_loader)
├── src/
│   ├── ine_loader.py       # Carga y preprocesa datos del INE
│   ├── record_generator.py # Genera un registro individual
│   └── dataset_builder.py  # Escala a N registros y exporta
├── tests/
│   └── test_dataset.py     # Batería pytest
├── output/                 # Dataset generado (no versionado)
├── requirements.txt
└── Makefile
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Generar dataset completo (10 000 registros + 20% duplicados)
make generate

# Solo registros limpios
make generate-clean

# Ejecutar tests
make test
```

O directamente:

```bash
cd src
python dataset_builder.py --records 10000 --dup-rate 0.20 --seed 42
```

## Fuentes de datos (INE)

Colocar en `data/raw/` los ficheros descargados del INE:

| Fichero | Fuente |
|---|---|
| `ine_nombres_hombres.csv` | INE — Nombres más frecuentes (hombres) |
| `ine_nombres_mujeres.csv` | INE — Nombres más frecuentes (mujeres) |
| `ine_apellidos.csv` | INE — Apellidos más frecuentes |
| `ine_padron.csv` | INE — Padrón Municipal (provincia/edad/sexo) |
| `ine_municipios.csv` | CNIG — Catálogo de municipios con código postal |

Si los ficheros no están presentes, el generador usa datos de demostración embebidos.

## Formato del dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `record_id` | UUID | Clave primaria única |
| `nombre` | str | Nombre según distribución INE |
| `primer_apellido` | str | Primer apellido |
| `segundo_apellido` | str | Segundo apellido (puede omitirse en duplicados) |
| `fecha_nacimiento` | str | ISO 8601 (YYYY-MM-DD) |
| `sexo` | enum | M / F |
| `provincia` | str | Código INE 2 dígitos |
| `municipio` | str | Nombre del municipio |
| `direccion` | str | Vía y número (ficticia) |
| `cod_postal` | str | Código postal de 5 dígitos |
| `telefono` | str | 9 dígitos con prefijo provincial |
| `correo_electronico` | str | Generado a partir de nombre+apellido |
| `is_duplicate` | bool | True si el registro es un duplicado |
| `original_id` | UUID / None | record_id del registro original |

## Taxonomía de errores simulados

- **Tipográficos**: omisión de tildes (p=0.30), confusión b/v, ll/y, c/z (p=0.10), transposición de caracteres (p=0.05)
- **Variabilidad**: omisión del segundo apellido (p=0.05), inversión de apellidos (p=0.03)
- **Fecha**: cambio de formato YYYY-MM-DD → DD/MM/YYYY (p=0.20), error ±1 año (p=0.05)
- **Teléfono**: transposición de dígitos (p=0.10), borrado de carácter (p=0.10)

## Licencia

MIT — véase [LICENSE](LICENSE)
