"""Utilidades para la limpieza de datos y el analisis exploratorio.

Cubre los pasos 4 y 5 del pipeline clasico (Clase 3, slide 16):

    4. Limpieza de datos  - missing values, outliers, inconsistencias
    5. EDA                - distribuciones, correlaciones, insights

ADVERTENCIA DE USO: todas las funciones de este modulo calculan estadisticos
(medias, desvios, cuartiles, IQR). Por lo tanto deben aplicarse UNICAMENTE sobre
el conjunto de train. Calcular estos valores sobre el dataset completo, o sobre el
test, seria data leakage (Clase 2, slides 93-96).

Los limites que devuelven estas funciones son parametros aprendidos del train: si
mas adelante se decidiera recortar o transformar en base a ellos, habria que
aplicar esos mismos limites al test, sin recalcularlos.
"""

from __future__ import annotations

import pandas as pd


def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad y porcentaje de valores faltantes por columna.

    El porcentaje es el criterio que la Clase 2 (slide 31) usa para decidir la
    estrategia: por encima de ~20% de faltantes en una columna, se evalua quitar
    la variable entera en lugar de imputarla.
    """
    return pd.DataFrame({
        "n_faltantes": df.isna().sum(),
        "pct_faltantes": (100 * df.isna().mean()).round(2),
    })


def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Limites inferior y superior del criterio IQR (Clase 2, slide 41).

        IQR = Q3 - Q1
        limite inferior = Q1 - k * IQR
        limite superior = Q3 + k * IQR

    k = 1.5 es el valor convencional, y es el que usa el boxplot para dibujar los
    bigotes: por eso los puntos que el boxplot marca como outliers son exactamente
    los que este criterio detecta.
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outliers_iqr(serie: pd.Series, k: float = 1.5) -> pd.Series:
    """Mascara booleana: True donde el valor cae fuera de los limites IQR."""
    li, ls = limites_iqr(serie, k)
    return (serie < li) | (serie > ls)


def outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Mascara booleana segun z-score (Clase 2, slide 41): es outlier si |z| > 3.

        z = (x - media) / desvio

    Si la variable fuera gaussiana, |z| > 3 dejaria afuera aproximadamente el 0.3%
    de los datos. En variables muy asimetricas el criterio es mas conservador que
    el IQR, porque la media y el desvio ya estan "inflados" por los propios valores
    extremos que se busca detectar.
    """
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > umbral


def resumen_outliers(
    df: pd.DataFrame,
    columnas: list[str],
    k: float = 1.5,
    umbral_z: float = 3.0,
) -> pd.DataFrame:
    """Compara ambos criterios de deteccion para cada variable numerica."""
    filas = []
    for col in columnas:
        li, ls = limites_iqr(df[col], k)
        m_iqr = outliers_iqr(df[col], k)
        m_z = outliers_zscore(df[col], umbral_z)
        filas.append({
            "variable": col,
            "lim_inf_IQR": round(li, 2),
            "lim_sup_IQR": round(ls, 2),
            "n_IQR": int(m_iqr.sum()),
            "pct_IQR": round(100 * m_iqr.mean(), 2),
            "n_zscore": int(m_z.sum()),
            "pct_zscore": round(100 * m_z.mean(), 2),
        })
    return pd.DataFrame(filas).set_index("variable")
