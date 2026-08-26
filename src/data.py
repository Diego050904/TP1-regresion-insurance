"""Carga del dataset y separacion train/test (pasos 2 y 3 del pipeline).

Aca no se limpia, no se imputa y no se calcula ningun estadistico.

Uso como script:  python -m src.data
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TEST_SIZE,
)

COLUMNAS_ESPERADAS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
FILAS_ESPERADAS = 1338

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def cargar_datos_crudos(verificar: bool = True) -> pd.DataFrame:
    """Lee el CSV desde data/raw/ y valida que tenga las columnas y filas esperadas."""
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
        df = df[COLUMNAS_ESPERADAS]

    return df


def eliminar_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Quita filas exactamente repetidas. Devuelve (dataset, cantidad eliminada)."""
    n_antes = len(df)
    df_sin_duplicados = df.drop_duplicates(keep="first")
    return df_sin_duplicados, n_antes - len(df_sin_duplicados)


def separar_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa en train y test por muestreo aleatorio simple, con semilla fija."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train, test


def verificar_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_esperado: int | None = None,
) -> dict:
    """Audita la particion: filas totales, indices disjuntos y filas identicas compartidas.

    Los tres controles son estructurales: no calculan estadisticos del target.
    """
    n_total = len(train) + len(test)

    if n_esperado is not None and n_total != n_esperado:
        raise ValueError(f"train + test = {n_total}, se esperaban {n_esperado}.")

    solapamiento = set(train.index) & set(test.index)
    if solapamiento:
        raise ValueError(f"Hay {len(solapamiento)} filas en train y test a la vez.")

    # Cada fila se convierte en tupla para poder compararlas como conjuntos.
    claves_train = set(map(tuple, train[COLUMNAS_ESPERADAS].to_numpy().tolist()))
    claves_test = set(map(tuple, test[COLUMNAS_ESPERADAS].to_numpy().tolist()))

    return {
        "n_train": len(train),
        "n_test": len(test),
        "prop_test": len(test) / n_total,
        "solapamiento_indices": len(solapamiento),
        "filas_identicas_compartidas": len(claves_train & claves_test),
    }


def guardar_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Escribe train.csv y test.csv en data/processed/, conservando el indice original."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=True, index_label="idx_original")
    test.to_csv(TEST_PATH, index=True, index_label="idx_original")


def cargar_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los splits ya generados. Es la entrada de todas las fases posteriores a la 1."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "No existen los splits en data/processed/. Generarlos con: python -m src.data"
        )
    train = pd.read_csv(TRAIN_PATH, index_col="idx_original")
    test = pd.read_csv(TEST_PATH, index_col="idx_original")
    return train, test


def main() -> None:
    """Carga, deduplica, separa, audita y guarda. Imprime el resumen por consola."""
    df = cargar_datos_crudos()
    n_crudo = len(df)

    df, n_duplicados = eliminar_duplicados(df)

    train, test = separar_train_test(df)
    resumen = verificar_split(train, test, n_esperado=len(df))
    guardar_splits(train, test)

    print(f"Dataset crudo      : {n_crudo} filas x {df.shape[1]} columnas")
    print(f"Duplicados exactos : {n_duplicados} eliminados -> {len(df)} filas")
    print(f"Semilla            : {RANDOM_SEED}")
    print(f"Metodo             : muestreo aleatorio simple")
    print("-" * 58)
    print(f"Train              : {resumen['n_train']:5d} filas "
          f"({1 - resumen['prop_test']:.1%})")
    print(f"Test               : {resumen['n_test']:5d} filas "
          f"({resumen['prop_test']:.1%})")
    print("-" * 58)
    print(f"Indices solapados        : {resumen['solapamiento_indices']}")
    print(f"Filas identicas en ambos : {resumen['filas_identicas_compartidas']}")
    print("-" * 58)
    print(f"Guardado en {TRAIN_PATH}")
    print(f"Guardado en {TEST_PATH}")


if __name__ == "__main__":
    main()
