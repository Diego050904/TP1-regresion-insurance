# TP1 — Regresión e Introducción a la Evaluación de Modelos

**Materia:** 72.75 — Aprendizaje Automático (Machine Learning) — ITBA
**Cuatrimestre:** 2026 Q2
**Fecha de defensa:** 26/08/2026 (entrega de código + presentación: 25/08/2026)

---

## Objetivo

Implementar modelos de regresión para predecir una variable numérica y aprender a
**entrenar y evaluar modelos usando validación cruzada**, controlando el overfitting
mediante una separación correcta de los datos.

---

## Dataset elegido: Insurance Charges

**Target:** `charges` — costo médico anual facturado por el seguro (variable continua).

| Variable   | Tipo                  | Descripción |
|------------|-----------------------|-------------|
| `age`      | Numérica discreta     | Edad del titular |
| `sex`      | Categórica nominal    | `female` / `male` |
| `bmi`      | Numérica continua     | Índice de masa corporal |
| `children` | Numérica discreta     | Cantidad de hijos a cargo |
| `smoker`   | Categórica nominal    | `yes` / `no` |
| `region`   | Categórica nominal    | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges`  | **Numérica continua** | **Target:** costo médico anual |

**Dimensiones:** 1338 filas × 7 columnas.


## Estructura del repositorio

```
TP1-regresion-insurance/
├── data/
│   ├── raw/
│   │   └── insurance.csv        # Fuente de verdad. Versionado en git.
│   └── processed/               # Splits generados (regenerables, no versionados)
├── notebooks/
│   ├── 01_carga_y_split.ipynb   # Fase 1: carga y separación train/test
│   ├── 02_limpieza_y_eda.ipynb  # Fase 2: limpieza y análisis exploratorio
│   ├── 03_preprocesamiento_y_features.ipynb   # Fase 3: encoding, escalado, features
│   ├── 04_regresion_lineal.ipynb              # Fase 4: k-fold y regresión lineal
│   ├── 05_polinomica_y_regularizacion.ipynb   # Fase 5: polinómica y L1
│   └── 06_evaluacion_final.ipynb              # Fase 6: evaluación en test
├── src/
│   ├── config.py                # Rutas, semilla y esquema de partición
│   ├── data.py                  # Carga, deduplicación, split y persistencia
│   ├── exploracion.py           # Faltantes y detección de outliers (IQR, z-score)
│   ├── preprocesamiento.py      # ColumnTransformer: one-hot + escalado z-score
│   └── modelado.py              # k-fold, modelos (lineal y polinómico) y evaluación
├── reports/
│   └── figures/                 # Gráficos para la presentación
├── requirements.txt
├── .gitignore
└── README.md
```

La lógica reutilizable vive en `src/` y los notebooks la importan. Así, el mismo
objeto de preprocesamiento que se ajusta con el train es literalmente el que se
aplica al test: no hay forma de que se filtre un estadístico calculado sobre el
dataset completo.

---

## Checklist de fases

- [x] **Fase 0 — Setup inicial**
  - [x] Estructura de carpetas
  - [x] Entorno virtual y `requirements.txt` con versiones fijadas
  - [x] `.gitignore` y README inicial
  - [x] Dataset en `data/raw/insurance.csv`
- [x] **Fase 1 — Carga y separación de datos** *(consigna 2.1)* → [`notebooks/01_carga_y_split.ipynb`](notebooks/01_carga_y_split.ipynb)
  - [x] Carga desde ruta local y verificación de integridad
  - [x] Split train/test con semilla fija, antes de cualquier limpieza
  - [x] Justificación de la proporción (80/20) y de la estratificación (por `smoker`)
  - [x] Auditoría del split: sin solapamiento de índices ni filas idénticas compartidas
  - [x] Splits persistidos en `data/processed/`
- [x] **Fase 2 — Limpieza de datos y EDA (sólo sobre train)** *(consignas 1.2, 1.3)*
  - [x] Valores faltantes: análisis y estrategia
  - [x] Duplicados dentro del train
  - [x] Inconsistencias y rangos válidos
  - [x] Outliers: detección visual (histograma, boxplot, scatter) y estadística (IQR, z-score)
  - [x] Decisión justificada sobre qué hacer con los outliers
  - [x] EDA: distribuciones, relación con el target y correlación entre features
- [x] **Fase 3 — Preprocesamiento y features** *(consignas 1.1, 1.4)*
  - [x] Encoding de categóricas, con justificación por variable
  - [x] Imputación: no aplica (no hay faltantes)
  - [x] Escalado por z-score (`StandardScaler`), ajustado sólo con train
  - [x] Filtros de selección: correlación de Pearson e información mutua
  - [x] Todo encapsulado en un `ColumnTransformer` ajustado sólo con train
- [x] **Fase 4 — Regresión lineal** *(consignas 2.2, 2.3)*
  - [x] k-fold cross-validation sobre el train (k = 5)
  - [x] Entrenamiento dentro del CV
  - [x] RMSE de train y de validación
- [x] **Fase 5 — Regresión polinómica y regularización** *(consignas 3.1, 3.2, 3.3, 4)*
  - [x] Transformación polinómica: grados 1, 2 y 3
  - [x] Entrenamiento sobre las variables transformadas
  - [x] Regularización L1 (Lasso) con 7 valores de lambda por grado (selección embedded)
  - [x] Tabla de RMSE de train y validación por grado y lambda
- [x] **Fase 6 — Evaluación final y comparación** *(consigna 5)*
  - [x] Evaluación en test (una única vez)
  - [x] ¿Qué modelo obtuvo menor error?
  - [x] ¿Cuál implementarían en una aplicación real? Justificar
  - [x] ¿Qué RMSE esperar en datos nuevos?


---

## Fuentes

- Enunciado: *TP1. Regresión e Introducción a la evaluación de modelos* — 72.75, 2026 Q2.
- *Clase 2: Datos, variables, overfitting y métricas* — Carlos Bibián Nogueras, PhD.
- *Clase 3: EDA, Feature selection, Regularización y Métricas* — Carlos Bibián Nogueras, PhD.
