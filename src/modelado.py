"""Modelos y evaluacion con validacion cruzada (pasos 7 y 9 del pipeline).

Ninguna funcion de este modulo recibe el conjunto de test.

Los modelos se arman como Pipeline para que cross_validate reajuste el
preprocesamiento dentro de cada fold, y no una sola vez sobre todo el train.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import N_SPLITS, RANDOM_SEED
from src.preprocesamiento import construir_preprocesador


def crear_kfold(n_splits: int = N_SPLITS, semilla: int = RANDOM_SEED) -> KFold:
    """KFold con shuffle y semilla fija. k=5 segun la Clase 2."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=semilla)


def crear_repeated_kfold(
    n_splits: int = N_SPLITS,
    n_repeats: int = 10,
    semilla: int = RANDOM_SEED,
) -> RepeatedKFold:
    """Repite el k-fold con repartos distintos, para que la comparacion entre
    modelos no dependa de una unica semilla de validacion."""
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)


def crear_modelo_lineal() -> Pipeline:
    """Pipeline de dos pasos: preprocesamiento + LinearRegression."""
    return Pipeline([
        ("preprocesamiento", construir_preprocesador()),
        ("regresion", LinearRegression()),
    ])


def crear_modelo_polinomico(
    grado: int,
    alpha: float | None = None,
    max_iter: int = 1_000_000,
) -> Pipeline:
    """Pipeline de cuatro pasos: preprocesamiento, expansion polinomica, reescalado y regresion.

    Con alpha=None usa minimos cuadrados; con un valor, Lasso (el alpha de
    scikit-learn es el lambda de la clase). Se reescala despues de la expansion
    porque los cuadrados y productos tienen escalas nuevas.
    """
    pasos = [
        ("preprocesamiento", construir_preprocesador()),
        ("polinomica", PolynomialFeatures(degree=grado, include_bias=False)),
        ("reescalado", StandardScaler()),
    ]

    if alpha is None:
        pasos.append(("regresion", LinearRegression()))
    else:
        pasos.append(("regresion", Lasso(alpha=alpha, max_iter=max_iter,
                                         random_state=RANDOM_SEED)))

    return Pipeline(pasos)


def contar_features(modelo: Pipeline, X: pd.DataFrame) -> tuple[int, int]:
    """Devuelve (columnas totales, columnas con coeficiente distinto de cero).

    Con Lasso la segunda cifra es menor: la penalizacion L1 selecciona variables.
    """
    coefs = modelo.named_steps["regresion"].coef_
    return len(coefs), int((coefs != 0).sum())


def evaluar_cv(
    modelo: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    nombre: str,
    cv: KFold | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Corre la validacion cruzada y devuelve (detalle por fold, resumen promediado).

    Reporta RMSE de train y de validacion. La diferencia entre ambos es el
    generalization gap.
    """
    if cv is None:
        cv = crear_kfold()

    # scikit-learn asume "mas alto es mejor", asi que devuelve el RMSE negado.
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
    """Coeficientes del modelo entrenado, con el nombre de cada feature y ordenados por magnitud."""
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    coefs = modelo.named_steps["regresion"].coef_

    tabla = pd.DataFrame({"coeficiente": coefs}, index=nombres)
    tabla["coef_abs"] = tabla["coeficiente"].abs()
    return tabla.sort_values("coef_abs", ascending=False)
