"""Utilidades para la limpieza de datos y el analisis exploratorio.

Cubre los pasos 4 (Limpieza de datos) y 5 (EDA) del pipeline clasico
(Clase 3, slide 16).

ADVERTENCIA DE USO
------------------
Todas las funciones de este modulo calculan estadisticos: medias, desvios,
cuartiles, IQR. Por lo tanto deben aplicarse UNICAMENTE sobre el train.
Calcularlos sobre el dataset completo o sobre el test seria data leakage
(Clase 2, slides 93-96).

Los limites que devuelven son parametros aprendidos del train: si mas adelante
se decidiera recortar o transformar en base a ellos, habria que aplicar esos
mismos limites al test sin recalcularlos.
"""

from __future__ import annotations

import pandas as pd


def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta los valores faltantes de cada columna.

    Que hace
    --------
    Devuelve, por columna, cuantos valores nulos (NaN) hay y que porcentaje del
    total representan.

    Para que sirve el porcentaje
    ----------------------------
    Es el criterio que usa la Clase 2 (slide 31) para decidir la estrategia: por
    encima de ~20% de faltantes en una columna se evalua descartar la variable
    entera en vez de imputarla.

    Parametros
    ----------
    df : pd.DataFrame
        Conjunto a analizar (debe ser el train).

    Devuelve
    --------
    pd.DataFrame
        Indice = nombre de columna. Columnas: n_faltantes y pct_faltantes.
    """
    return pd.DataFrame({
        "n_faltantes": df.isna().sum(),
        "pct_faltantes": (100 * df.isna().mean()).round(2),
    })


def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Calcula los limites del criterio IQR para detectar outliers.

    Que hace
    --------
    Aplica la formula de la Clase 2 (slide 41):

        IQR = Q3 - Q1
        limite inferior = Q1 - k * IQR
        limite superior = Q3 + k * IQR

    Un valor es outlier si cae fuera de ese intervalo.

    Por que k = 1.5
    ---------------
    Es el valor convencional y el que usa el boxplot para dibujar los bigotes.
    Por eso los puntos que el boxplot marca como outliers son exactamente los
    que detecta esta funcion.

    Parametros
    ----------
    serie : pd.Series
        Variable numerica a analizar.
    k : float
        Multiplicador del IQR. Mas alto = criterio mas permisivo.

    Devuelve
    --------
    tuple[float, float]
        (limite_inferior, limite_superior).
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outliers_iqr(serie: pd.Series, k: float = 1.5) -> pd.Series:
    """Marca que valores son outliers segun el criterio IQR.

    Parametros
    ----------
    serie : pd.Series
        Variable numerica a analizar.
    k : float
        Multiplicador del IQR (ver limites_iqr).

    Devuelve
    --------
    pd.Series
        Serie de booleanos del mismo largo que la entrada: True donde el valor
        cae fuera de los limites. Sirve para filtrar (df[mascara]) o para contar
        (mascara.sum()).
    """
    li, ls = limites_iqr(serie, k)
    return (serie < li) | (serie > ls)


def outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Marca que valores son outliers segun el criterio z-score.

    Que hace
    --------
    Calcula z = (x - media) / desvio para cada valor y marca los que superan el
    umbral en valor absoluto. La Clase 2 (slide 41) usa |z| > 3.

    Diferencia con el criterio IQR
    ------------------------------
    El z-score usa media y desvio, que son estadisticos NO robustos: los propios
    valores extremos los inflan y el umbral se corre hacia afuera. El IQR usa
    cuartiles, que si son robustos. Por eso en variables muy asimetricas el
    z-score detecta bastantes menos outliers que el IQR.

    Parametros
    ----------
    serie : pd.Series
        Variable numerica a analizar.
    umbral : float
        Valor de |z| a partir del cual se considera outlier.

    Devuelve
    --------
    pd.Series
        Serie de booleanos: True donde |z| supera el umbral.
    """
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > umbral


def resumen_outliers(
    df: pd.DataFrame,
    columnas: list[str],
    k: float = 1.5,
    umbral_z: float = 3.0,
) -> pd.DataFrame:
    """Compara los dos criterios de deteccion de outliers para varias variables.

    Que hace
    --------
    Para cada variable numerica indicada, calcula los limites IQR y cuenta
    cuantos outliers detecta cada criterio (IQR y z-score), en cantidad y en
    porcentaje. Permite ver de un vistazo donde ambos coinciden y donde no.

    Parametros
    ----------
    df : pd.DataFrame
        Conjunto a analizar (debe ser el train).
    columnas : list[str]
        Nombres de las variables numericas a evaluar.
    k : float
        Multiplicador del IQR.
    umbral_z : float
        Umbral de |z| para el criterio z-score.

    Devuelve
    --------
    pd.DataFrame
        Una fila por variable, con: lim_inf_IQR, lim_sup_IQR, n_IQR, pct_IQR,
        n_zscore y pct_zscore.
    """
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
