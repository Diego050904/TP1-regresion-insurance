"""Configuracion central del proyecto: rutas, semilla y esquema de particion.

Todo el codigo del TP importa las rutas desde aca en lugar de escribirlas a mano.
Asi hay un unico lugar donde se define de donde salen los datos y con que semilla
se parte el dataset, lo cual es la base de que el trabajo sea reproducible.
"""

from pathlib import Path

# --- Rutas ------------------------------------------------------------------
# PROJECT_ROOT se calcula relativo a este archivo (src/config.py -> raiz del repo),
# no al directorio desde el que se ejecuta. De esa forma los notebooks funcionan
# igual sin importar desde donde se los abra.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Unica fuente de datos del proyecto. No se descarga nada en tiempo de ejecucion:
# el analisis no depende de internet ni de que cambie la version publicada en Kaggle.
RAW_DATA_PATH = RAW_DIR / "insurance.csv"

# --- Problema ---------------------------------------------------------------
TARGET = "charges"

# --- Reproducibilidad -------------------------------------------------------
# Semilla unica para todo el TP: split train/test, k-fold y cualquier modelo con
# componente aleatoria. Fijarla es lo que permite que el resultado que se muestra
# en la defensa sea el mismo que se obtiene al re-ejecutar el codigo.
RANDOM_SEED = 42

# --- Esquema de particion ---------------------------------------------------
# El test se separa una sola vez, apenas cargados los datos, y no se toca hasta
# la evaluacion final (Fase 6). La validacion se hace con k-fold sobre el train.
TEST_SIZE = 0.20
N_SPLITS = 5
