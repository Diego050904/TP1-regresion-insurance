"""Encoding de categoricas y escalado de numericas (paso 6 del pipeline).

Todo el preprocesamiento vive en un unico ColumnTransformer, que se ajusta con
train y se aplica al resto con los mismos parametros aprendidos.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET

NUMERICAS = ["age", "bmi", "children"]
CATEGORICAS = ["sex", "smoker", "region"]


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Divide un conjunto en las 6 variables de entrada (X) y el target (y)."""
    return df[NUMERICAS + CATEGORICAS], df[TARGET]


def construir_preprocesador(escalar: bool = True) -> ColumnTransformer:
    """Arma el ColumnTransformer: z-score en las numericas, one-hot en las categoricas.

    El escalado es necesario para que la penalizacion L1 no castigue a una
    variable por sus unidades. drop="first" evita la trampa de las dummies.
    """
    transformador_numericas = StandardScaler() if escalar else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", transformador_numericas, NUMERICAS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="error"), CATEGORICAS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def nombres_de_features(preprocesador: ColumnTransformer) -> list[str]:
    """Nombres de las columnas de salida, para poder interpretar los coeficientes."""
    return list(preprocesador.get_feature_names_out())
