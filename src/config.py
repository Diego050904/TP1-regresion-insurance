"""Configuracion central: rutas, semilla y esquema de particion."""

from pathlib import Path

# Se calcula desde este archivo, no desde el directorio de ejecucion.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Unica fuente de datos del proyecto.
RAW_DATA_PATH = RAW_DIR / "insurance.csv"

TARGET = "charges"

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
