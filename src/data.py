"""Carga del dataset crudo y separacion train/test.

Cubre los pasos 2 y 3 del pipeline clasico de un proyecto de ML (Clase 3, slide 16):

    1. Definicion del problema
    2. Recoleccion de datos        <- este modulo
    3. Data Splitting              <- este modulo
    4. Limpieza de datos
    5. EDA
    ...

REGLA CENTRAL: aca NO se limpia, NO se imputa y NO se calcula ningun estadistico
para transformar los datos. Lo unico que se hace es leer el archivo, verificar su
integridad estructural y partirlo. Cualquier transformacion que dependa de un
estadistico (media, mediana, IQR, escalado...) se ajusta mas adelante usando
exclusivamente el train.

Uso como script (regenera los splits en data/processed/):
    python -m src.data
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
)

# Esquema esperado del archivo crudo. Se valida en la carga para que, si el CSV
# fuera reemplazado por otra version, el error aparezca aca y no cinco fases
# despues disfrazado de un resultado raro.
COLUMNAS_ESPERADAS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
FILAS_ESPERADAS = 1338

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def cargar_datos_crudos(verificar: bool = True) -> pd.DataFrame:
    """Lee el dataset desde la ruta local fija y valida su estructura.

    No se descarga nada: la unica fuente es data/raw/insurance.csv. Ver la
    seccion "Origen del dataset" del README.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {RAW_DATA_PATH}. "
            "Colocar insurance.csv en data/raw/ (ver README)."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    if verificar:
        faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas esperadas en el CSV: {sorted(faltantes)}")
        if len(df) != FILAS_ESPERADAS:
            raise ValueError(
                f"El CSV tiene {len(df)} filas y se esperaban {FILAS_ESPERADAS}. "
                "Puede tratarse de otra version del dataset."
            )
        # Reordena a las columnas esperadas para que el orden sea estable.
        df = df[COLUMNAS_ESPERADAS]

    return df


def separar_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el dataset en train y test mediante muestreo aleatorio simple.

    Criterio (Clase 3, slide 30): "crear el conjunto de test y separar esos datos
    para no volverlos a usar hasta el final del proyecto. Tipicamente seleccionar
    aleatoriamente 20% de los datos".

    Se hace inmediatamente despues de la carga, antes de cualquier limpieza, para
    evitar data leakage: si se imputaran valores o se quitaran outliers usando el
    dataset completo, informacion que luego va al test estaria influyendo en el
    train (Clase 2, slides 93-96).
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Se conserva el indice original del CSV: permite auditar despues que ninguna
    # fila haya pasado de un conjunto al otro.
    return train, test


def verificar_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Controles de integridad del split. Devuelve un resumen y falla si algo no cierra.

    El control importante es el tercero: dos filas identicas repartidas entre train
    y test serian una forma silenciosa de leakage (el modelo habria visto en
    entrenamiento un dato exactamente igual al que se le evalua).
    """
    n_total = len(train) + len(test)

    # 1) No se perdio ni se duplico ninguna fila.
    if n_total != FILAS_ESPERADAS:
        raise ValueError(f"train + test = {n_total}, se esperaban {FILAS_ESPERADAS}.")

    # 2) Los conjuntos son disjuntos por indice.
    solapamiento = set(train.index) & set(test.index)
    if solapamiento:
        raise ValueError(f"Hay {len(solapamiento)} filas en train y test a la vez.")

    # 3) Ninguna fila duplicada quedo repartida entre ambos conjuntos.
    claves_train = set(map(tuple, train[COLUMNAS_ESPERADAS].to_numpy().tolist()))
    claves_test = set(map(tuple, test[COLUMNAS_ESPERADAS].to_numpy().tolist()))
    filas_compartidas = claves_train & claves_test

    return {
        "n_train": len(train),
        "n_test": len(test),
        "prop_test": len(test) / n_total,
        "solapamiento_indices": len(solapamiento),
        "filas_identicas_compartidas": len(filas_compartidas),
        "media_target_train": train[TARGET].mean(),
        "media_target_test": test[TARGET].mean(),
    }


def guardar_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Persiste train y test en data/processed/.

    Se guarda el indice original (index=True) para poder rastrear cada fila
    hasta el CSV crudo.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=True, index_label="idx_original")
    test.to_csv(TEST_PATH, index=True, index_label="idx_original")


def cargar_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los splits ya generados. Es la puerta de entrada de las fases 2 en adelante.

    A partir de aca ningun notebook vuelve a abrir el CSV crudo: asi es imposible
    recalcular por error un estadistico sobre el dataset completo.
    """
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "No existen los splits en data/processed/. Generarlos con: python -m src.data"
        )
    train = pd.read_csv(TRAIN_PATH, index_col="idx_original")
    test = pd.read_csv(TEST_PATH, index_col="idx_original")
    return train, test


def main() -> None:
    df = cargar_datos_crudos()
    train, test = separar_train_test(df)
    resumen = verificar_split(train, test)
    guardar_splits(train, test)

    print(f"Dataset crudo      : {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Semilla            : {RANDOM_SEED}")
    print(f"Metodo             : muestreo aleatorio simple")
    print("-" * 58)
    print(f"Train              : {resumen['n_train']:5d} filas "
          f"({1 - resumen['prop_test']:.1%})")
    print(f"Test               : {resumen['n_test']:5d} filas "
          f"({resumen['prop_test']:.1%})")
    print("-" * 58)
    print(f"Media charges train: {resumen['media_target_train']:10,.2f}")
    print(f"Media charges test : {resumen['media_target_test']:10,.2f}")
    print("-" * 58)
    print(f"Indices solapados        : {resumen['solapamiento_indices']}")
    print(f"Filas identicas en ambos : {resumen['filas_identicas_compartidas']}")
    print("-" * 58)
    print(f"Guardado en {TRAIN_PATH}")
    print(f"Guardado en {TEST_PATH}")


if __name__ == "__main__":
    main()
