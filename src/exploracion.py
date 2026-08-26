"""Limpieza de datos y analisis exploratorio (pasos 4 y 5 del pipeline).

Todas las funciones calculan estadisticos: aplicarlas UNICAMENTE sobre el train.
"""

from __future__ import annotations

import pandas as pd


def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad y porcentaje de valores nulos por columna."""
    return pd.DataFrame({
        "n_faltantes": df.isna().sum(),
        "pct_faltantes": (100 * df.isna().mean()).round(2),
    })


def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Limites del criterio IQR: (Q1 - k*IQR, Q3 + k*IQR). Con k=1.5 son los bigotes del boxplot."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outliers_iqr(serie: pd.Series, k: float = 1.5) -> pd.Series:
    """Mascara booleana: True donde el valor cae fuera de los limites IQR."""
    li, ls = limites_iqr(serie, k)
    return (serie < li) | (serie > ls)


def outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Mascara booleana segun |z| > umbral. Usa media y desvio, que no son robustos."""
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > umbral


def resumen_outliers(
    df: pd.DataFrame,
    columnas: list[str],
    k: float = 1.5,
    umbral_z: float = 3.0,
) -> pd.DataFrame:
    """Tabla comparativa de ambos criterios de deteccion para varias variables."""
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
