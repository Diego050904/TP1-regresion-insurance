"""Entrenamiento y evaluacion de modelos con validacion cruzada.

Cubre los pasos 7 (Modelado) y 9 (Evaluacion en Dev) del pipeline clasico
(Clase 3, slide 16), y las consignas 2.2, 2.3 y 4 del enunciado.

Regla que respeta todo el modulo: la validacion cruzada se hace UNICAMENTE sobre
el train. Ninguna funcion de aca recibe el conjunto de test.

Por que los modelos se arman como Pipeline
------------------------------------------
Un `Pipeline` encadena el preprocesamiento y el modelo en un solo objeto. Cuando
`cross_validate` lo evalua, reajusta la cadena completa dentro de cada fold: el
escalado aprende la media y el desvio del sub-conjunto de entrenamiento de ese
fold, no de todo el train.

Si en cambio se transformaran los datos una sola vez antes del k-fold, cada fold
de validacion habria participado en el calculo de esos estadisticos y el error de
validacion saldria optimista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import N_SPLITS, RANDOM_SEED
from src.preprocesamiento import construir_preprocesador


def crear_kfold(n_splits: int = N_SPLITS, semilla: int = RANDOM_SEED) -> KFold:
    """Crea el esquema de validacion cruzada que usan todas las fases.

    Que hace
    --------
    Divide el train en `n_splits` partes iguales. En cada iteracion una parte se
    usa para validar y el resto para entrenar, de modo que cada observacion se
    valida exactamente una vez.

    Por que k = 5
    -------------
    La Clase 2 (slide 85) indica que "k = 5 o 10 son las mas usadas y son
    suficientes". Con 1069 filas, k=5 deja unas 214 observaciones por fold de
    validacion, cantidad razonable para estimar el error.

    Parametros
    ----------
    n_splits : int
        Cantidad de folds.
    semilla : int
        Fija el reparto de filas entre folds, para que el resultado sea reproducible.

    Devuelve
    --------
    KFold
        Con shuffle activado: las filas se mezclan antes de repartirse.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=semilla)


def crear_modelo_lineal() -> Pipeline:
    """Arma el modelo de regresion lineal completo: preprocesamiento + regresion.

    Que hace
    --------
    Devuelve un Pipeline de dos pasos:
      1. `preprocesamiento`: one-hot para las categoricas y z-score para las
         numericas (ver src/preprocesamiento.py).
      2. `regresion`: LinearRegression, que ajusta los coeficientes minimizando
         la suma de los errores al cuadrado.

    Devuelve
    --------
    Pipeline
        Sin entrenar. Se entrena con .fit() o dentro de cross_validate().
    """
    return Pipeline([
        ("preprocesamiento", construir_preprocesador()),
        ("regresion", LinearRegression()),
    ])


def evaluar_cv(
    modelo: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    nombre: str,
    cv: KFold | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Evalua un modelo con validacion cruzada y devuelve el detalle por fold.

    Que hace
    --------
    Para cada fold: entrena el modelo con los datos de entrenamiento del fold y
    calcula el RMSE y el R2 tanto sobre esos datos (train) como sobre los datos
    reservados (validacion).

    Por que se reporta el RMSE de train ademas del de validacion
    -----------------------------------------------------------
    La consigna 2.3 pide los dos. La diferencia entre ambos es el
    "generalization gap" (Clase 2, slides 76-77): si el error de train es mucho
    menor que el de validacion, el modelo esta memorizando ruido del
    entrenamiento en lugar de aprender la relacion real.

    Parametros
    ----------
    modelo : Pipeline
        Modelo sin entrenar.
    X, y : pd.DataFrame, pd.Series
        Variables de entrada y target del TRAIN. Nunca del test.
    nombre : str
        Etiqueta del modelo, para identificarlo en la tabla comparativa.
    cv : KFold | None
        Esquema de validacion. Si es None se usa crear_kfold().

    Devuelve
    --------
    tuple[pd.DataFrame, dict]
        - DataFrame con una fila por fold: rmse_train, rmse_val, r2_train, r2_val.
        - dict con el resumen promediado sobre los folds, listo para apilar en
          una tabla comparativa de modelos.
    """
    if cv is None:
        cv = crear_kfold()

    # scoring negativo: scikit-learn asume "mas alto es mejor", asi que devuelve
    # el RMSE con signo cambiado. Lo revertimos abajo.
    resultados = cross_validate(
        modelo,
        X,
        y,
        cv=cv,
        scoring=["neg_root_mean_squared_error", "r2"],
        return_train_score=True,
    )

    por_fold = pd.DataFrame({
        "fold": range(1, cv.get_n_splits() + 1),
        "rmse_train": -resultados["train_neg_root_mean_squared_error"],
        "rmse_val": -resultados["test_neg_root_mean_squared_error"],
        "r2_train": resultados["train_r2"],
        "r2_val": resultados["test_r2"],
    }).set_index("fold")

    resumen = {
        "modelo": nombre,
        "rmse_train": por_fold["rmse_train"].mean(),
        "rmse_val": por_fold["rmse_val"].mean(),
        "rmse_val_desvio": por_fold["rmse_val"].std(),
        "gap": por_fold["rmse_val"].mean() - por_fold["rmse_train"].mean(),
        "r2_val": por_fold["r2_val"].mean(),
    }

    return por_fold, resumen


def tabla_coeficientes(modelo: Pipeline) -> pd.DataFrame:
    """Extrae los coeficientes de un modelo lineal ya entrenado.

    Que hace
    --------
    Asocia cada coeficiente con el nombre de la feature que le corresponde y los
    ordena por magnitud.

    Como se interpretan
    -------------------
    Las numericas estan estandarizadas, asi que su coeficiente indica cuanto
    cambia el costo predicho al aumentar esa variable en un desvio estandar. Las
    columnas one-hot valen 0 o 1, asi que su coeficiente es la diferencia
    respecto de la categoria de referencia.

    Parametros
    ----------
    modelo : Pipeline
        Ya entrenado con .fit().

    Devuelve
    --------
    pd.DataFrame
        Columnas: coeficiente y coef_abs, ordenado por magnitud descendente.
    """
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    coefs = modelo.named_steps["regresion"].coef_

    tabla = pd.DataFrame({"coeficiente": coefs}, index=nombres)
    tabla["coef_abs"] = tabla["coeficiente"].abs()
    return tabla.sort_values("coef_abs", ascending=False)
