"""Carga del dataset y separacion train/test.

Cubre los pasos 2 (Recoleccion de datos) y 3 (Data Splitting) del pipeline
clasico de un proyecto de ML (Clase 3, slide 16).

Regla que respeta todo el modulo: aca NO se limpia, NO se imputa y NO se calcula
ningun estadistico. Solo se lee el archivo, se verifica su estructura y se parte.

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
    TEST_SIZE,
)

# Columnas y cantidad de filas que esperamos encontrar en el CSV.
# Sirven para detectar si el archivo fue reemplazado por otra version.
COLUMNAS_ESPERADAS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
FILAS_ESPERADAS = 1338

# Rutas donde se guardan los conjuntos ya separados.
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def cargar_datos_crudos(verificar: bool = True) -> pd.DataFrame:
    """Lee el dataset original desde disco y valida que sea el esperado.

    Que hace
    --------
    1. Comprueba que exista el archivo data/raw/insurance.csv.
    2. Lo lee con pandas.
    3. Si verificar=True, controla que esten las 7 columnas esperadas y que
       tenga 1338 filas, y reordena las columnas para que el orden sea estable.

    Por que valida
    --------------
    Si alguien reemplazara el CSV por otra version del dataset, queremos que el
    error aparezca aca y no varias fases despues disfrazado de un resultado raro.

    Parametros
    ----------
    verificar : bool
        Si es False, solo lee el archivo sin controlar su estructura.

    Devuelve
    --------
    pd.DataFrame
        El dataset completo, 1338 filas x 7 columnas.

    Lanza
    -----
    FileNotFoundError
        Si no existe data/raw/insurance.csv.
    ValueError
        Si faltan columnas o la cantidad de filas no es la esperada.
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
        df = df[COLUMNAS_ESPERADAS]

    return df


def eliminar_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina las filas exactamente duplicadas, conservando la primera aparicion.

    Que hace
    --------
    Busca filas identicas en las 7 columnas y deja una sola copia de cada una.

    Por que se hace ANTES del split, y por que eso no contradice la regla de
    "separar antes de limpiar"
    -----------------------------------------------------------------------
    Una fila exactamente repetida es un defecto de la recoleccion de datos (el
    mismo registro cargado dos veces), no un valor que haya que estimar. Al
    eliminarla no se calcula ningun estadistico, asi que no viaja informacion del
    test hacia el train: pertenece al paso 2 del pipeline (Recoleccion), no al
    paso 4 (Limpieza).

    Si en cambio se dejara para despues del split, las dos copias pueden quedar
    repartidas una en train y otra en test. En ese caso el test contendria un
    registro que el modelo ya vio al entrenar y dejaria de ser un conjunto de
    datos verdaderamente nuevos, que es justamente lo que la separacion busca
    garantizar.

    Parametros
    ----------
    df : pd.DataFrame
        Dataset tal como sale de cargar_datos_crudos().

    Devuelve
    --------
    tuple[pd.DataFrame, int]
        (dataset sin duplicados, cantidad de filas eliminadas).
    """
    n_antes = len(df)
    df_sin_duplicados = df.drop_duplicates(keep="first")
    return df_sin_duplicados, n_antes - len(df_sin_duplicados)


def separar_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el dataset en train y test por muestreo aleatorio simple.

    Que hace
    --------
    Mezcla las filas y aparta el 20% para test, usando una semilla fija para que
    el sorteo sea siempre el mismo. Conserva el indice original del CSV, lo que
    permite auditar despues si alguna fila cambio de conjunto.

    Por que se hace aca y no despues de limpiar
    -------------------------------------------
    Es el paso 3 del pipeline. Si limpiaramos antes, los estadisticos usados
    (medianas, IQR) incluirian informacion de las filas que van al test, y el
    error de test dejaria de ser una estimacion honesta (Clase 2, slides 93-96).

    Parametros
    ----------
    df : pd.DataFrame
        Dataset completo, tal como sale de cargar_datos_crudos().
    test_size : float
        Fraccion destinada a test. Por defecto 0.20 (Clase 3, slide 30).
    random_state : int
        Semilla del sorteo. Fija para que el resultado sea reproducible.

    Devuelve
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test), de 1070 y 268 filas respectivamente.
    """
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
    """Audita que la separacion train/test sea correcta.

    Que controla
    ------------
    1. Que la suma de filas de ambos conjuntos sea la esperada (no se perdio ni
       se duplico nada).
    2. Que ninguna fila este en los dos conjuntos a la vez (indices disjuntos).
    3. Cuantas filas del test son identicas a alguna fila del train. Este es el
       control importante: dos filas iguales repartidas entre train y test
       significan que el modelo se evalua sobre un dato que ya vio al entrenar.
       Tras eliminar los duplicados antes del split, este valor debe ser 0.

    Por que la auditoria NO reporta estadisticos del target
    ------------------------------------------------------
    Los tres controles son puramente estructurales: cuentan filas y detectan
    solapamientos. Ninguno calcula la media, el desvio ni la distribucion de
    charges en el test.

    Es deliberado. La Clase 2 (slide 89) establece que el test "solamente se
    utiliza para estimar el error de prediccion en datos nuevos, nunca para tomar
    decisiones". Mirar el target del test antes de la evaluacion final no aporta
    ninguna accion posible, y en cambio abre la puerta a repetir el sorteo hasta
    obtener una particion favorable, que si seria adulterar el resultado.

    La representatividad de la particion se verifica sobre las variables de
    entrada, que es donde la pregunta tiene sentido (ver notebook 01).

    Parametros
    ----------
    train, test : pd.DataFrame
        Los conjuntos devueltos por separar_train_test().
    n_esperado : int | None
        Cantidad total de filas que deberian sumar ambos conjuntos. Si es None,
        se omite ese control.

    Devuelve
    --------
    dict
        Resumen con: n_train, n_test, prop_test, solapamiento_indices y
        filas_identicas_compartidas.

    Lanza
    -----
    ValueError
        Si se perdieron filas o si hay indices repetidos entre conjuntos.
    """
    n_total = len(train) + len(test)

    # Control 1: no se perdio ni se duplico ninguna fila.
    if n_esperado is not None and n_total != n_esperado:
        raise ValueError(f"train + test = {n_total}, se esperaban {n_esperado}.")

    # Control 2: los conjuntos son disjuntos por indice.
    solapamiento = set(train.index) & set(test.index)
    if solapamiento:
        raise ValueError(f"Hay {len(solapamiento)} filas en train y test a la vez.")

    # Control 3: se convierte cada fila en una tupla para poder compararlas como
    # conjuntos y contar cuantas aparecen en ambos lados.
    claves_train = set(map(tuple, train[COLUMNAS_ESPERADAS].to_numpy().tolist()))
    claves_test = set(map(tuple, test[COLUMNAS_ESPERADAS].to_numpy().tolist()))
    filas_compartidas = claves_train & claves_test

    return {
        "n_train": len(train),
        "n_test": len(test),
        "prop_test": len(test) / n_total,
        "solapamiento_indices": len(solapamiento),
        "filas_identicas_compartidas": len(filas_compartidas),
    }


def guardar_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Guarda train y test como CSV en data/processed/.

    Conserva el indice original del dataset en una columna llamada
    'idx_original', para poder rastrear cada fila hasta el archivo crudo.

    Parametros
    ----------
    train, test : pd.DataFrame
        Los conjuntos a persistir.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=True, index_label="idx_original")
    test.to_csv(TEST_PATH, index=True, index_label="idx_original")


def cargar_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los splits ya generados desde data/processed/.

    Es la puerta de entrada de todas las fases posteriores a la 1: a partir de
    ahi ningun notebook vuelve a abrir el CSV crudo, para que sea imposible
    calcular por error un estadistico sobre el dataset completo.

    Devuelve
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test), con el indice original restaurado.

    Lanza
    -----
    FileNotFoundError
        Si los splits todavia no fueron generados.
    """
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "No existen los splits en data/processed/. Generarlos con: python -m src.data"
        )
    train = pd.read_csv(TRAIN_PATH, index_col="idx_original")
    test = pd.read_csv(TEST_PATH, index_col="idx_original")
    return train, test


def main() -> None:
    """Ejecuta el proceso completo e imprime un resumen por consola.

    Carga el CSV, elimina duplicados exactos, separa train/test, audita el
    resultado y guarda ambos conjuntos en data/processed/.
    """
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
