"""Preprocesamiento: encoding de categoricas y escalado de numericas.

Cubre el paso 6 del pipeline clasico (Feature engineering / selection) y las
consignas 1.1 (variables categoricas) y 1.4 (features y escalado).

Idea central: todo el preprocesamiento vive dentro de un unico objeto
`ColumnTransformer`. Ese objeto se ajusta (`fit`) SOLO con datos de entrenamiento
y despues se aplica (`transform`) al test o al fold de validacion, con los mismos
parametros aprendidos.

Por que importa que sea un solo objeto
--------------------------------------
Si el escalado se hiciera "a mano" habria que acordarse de guardar la media y el
desvio del train y reaplicarlos al test. Encapsulandolo, esa disciplina deja de
depender de la memoria: el mismo objeto que aprendio del train es el que
transforma el test, y es imposible recalcular los estadisticos por error.

Ademas, al meterlo dentro de un `Pipeline` de scikit-learn, la validacion cruzada
reajusta el preprocesamiento en cada fold automaticamente, que es lo correcto: si
se ajustara una sola vez sobre todo el train, los folds de validacion habrian
participado en el calculo de la media y el desvio.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET

# Variables de entrada, agrupadas por el tratamiento que reciben.
NUMERICAS = ["age", "bmi", "children"]
CATEGORICAS = ["sex", "smoker", "region"]


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa un conjunto en variables de entrada (X) y target (y).

    Parametros
    ----------
    df : pd.DataFrame
        Train o test, tal como los devuelve cargar_splits().

    Devuelve
    --------
    tuple[pd.DataFrame, pd.Series]
        (X con las 6 variables de entrada, y con la columna charges).
    """
    X = df[NUMERICAS + CATEGORICAS]
    y = df[TARGET]
    return X, y


def construir_preprocesador(escalar: bool = True) -> ColumnTransformer:
    """Arma el ColumnTransformer que transforma las 6 variables de entrada.

    Que hace con cada grupo
    -----------------------
    - Numericas (age, bmi, children): estandarizacion z-score con StandardScaler,
      es decir (x - media) / desvio. La media y el desvio se aprenden del train.
    - Categoricas (sex, smoker, region): one-hot encoding con drop="first".

    Por que z-score y no min-max
    ----------------------------
    La Clase 3 (slide 43) lo indica como opcion por defecto y senala que es "mucho
    menos sensible a outliers" que min-max, que se define por el minimo y el maximo
    y por lo tanto queda determinado por los valores extremos.

    Por que hace falta escalar
    --------------------------
    Para una regresion lineal por minimos cuadrados el escalado no cambia las
    predicciones: los coeficientes absorben las unidades de cada variable. Se
    vuelve imprescindible con regularizacion L1/L2 (Fase 5): la penalizacion
    castiga coeficientes grandes, y un coeficiente es grande o chico segun las
    unidades de su variable. Sin escalar, `age` (desvio 14) y `children`
    (desvio 1.2) reciben penalizaciones incomparables.

    Por que las columnas one-hot NO se escalan
    ------------------------------------------
    Ya son 0/1, un rango acotado y comparable, con desvios de 0.40 a 0.50 —del
    mismo orden que una variable estandarizada. Escalarlas ademas dificultaria
    leer los coeficientes y amplificaria las categorias poco frecuentes.

    Por que drop="first"
    --------------------
    Con las k columnas de una categorica y un modelo con intercepto, la suma de
    las dummies es siempre 1 y la matriz de diseno queda mal condicionada (la
    "trampa de las variables dummy"): los coeficientes dejan de ser unicos.
    Descartando una categoria, esa queda como referencia y las demas se
    interpretan respecto de ella.

    Parametros
    ----------
    escalar : bool
        Si es False, las numericas pasan sin transformar. Sirve para comparar el
        efecto del escalado; el pipeline del TP usa siempre True.

    Devuelve
    --------
    ColumnTransformer
        Sin ajustar. Se ajusta con .fit(X_train) o al usarlo dentro de un Pipeline.
    """
    transformador_numericas = StandardScaler() if escalar else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", transformador_numericas, NUMERICAS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="error"),
                CATEGORICAS,
            ),
        ],
        remainder="drop",   # cualquier columna no listada se descarta
        verbose_feature_names_out=False,
    )


def nombres_de_features(preprocesador: ColumnTransformer) -> list[str]:
    """Devuelve los nombres de las columnas que produce el preprocesador.

    Hace falta porque el ColumnTransformer devuelve un array de numpy sin nombres,
    y para interpretar los coeficientes del modelo (Fase 4) necesitamos saber a que
    variable corresponde cada uno.

    Parametros
    ----------
    preprocesador : ColumnTransformer
        Ya ajustado con .fit().

    Devuelve
    --------
    list[str]
        Nombres en el mismo orden que las columnas de salida.
    """
    return list(preprocesador.get_feature_names_out())
