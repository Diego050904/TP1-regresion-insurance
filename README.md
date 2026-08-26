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

### ¿Por qué este dataset?

1. **Es un problema de regresión genuino.** El target es continuo y de rango amplio,
   a diferencia de *Wine Quality*, donde la "calidad" es un entero de 3 a 8 y el
   problema queda a mitad de camino entre regresión y clasificación ordinal.
2. **Tiene variables categóricas reales que obligan a decidir y justificar el encoding**
   (punto 1.1 del enunciado): dos binarias (`sex`, `smoker`) y una nominal de cuatro
   niveles (`region`). Eso permite contrastar las estrategias vistas en clase
   (one-hot, frequency, target y regularized target encoding) sobre un caso concreto.
3. **La relación con el target no es puramente lineal.** El efecto de `smoker` es
   fuerte y además interactúa con `bmi`, lo que da sentido real —y no meramente
   mecánico— a la regresión polinómica de los puntos 3 y 4.
4. **Tamaño adecuado para validación cruzada.** Con 1338 observaciones, un holdout
   fijo de dev dejaría pocos datos para elegir modelo; k-fold sobre el train usa los
   datos de manera más eficiente, que es exactamente el escenario que la Clase 2
   recomienda para datasets pequeños/medianos.
5. **Es interpretable.** Las conclusiones se pueden explicar en lenguaje de negocio
   durante la defensa, sin necesidad de conocimiento de dominio especializado.

### Origen del dataset

El dataset proviene de Kaggle: <https://www.kaggle.com/datasets/mirichoi0218/insurance>

A modo de referencia de procedencia, también puede obtenerse programáticamente así:

```python
# NOTA: esto NO forma parte del pipeline de este proyecto.
# Se documenta únicamente para dejar registro del origen del dataset.
import kagglehub
path = kagglehub.dataset_download("mirichoi0218/insurance")
```

> **El código de este repositorio nunca descarga los datos.** Siempre los lee desde
> la ruta local `data/raw/insurance.csv`. La razón es la reproducibilidad: si el
> pipeline dependiera de `kagglehub`, el análisis quedaría atado a que haya internet
> el día de la defensa y a que la *"latest version"* publicada en Kaggle no cambie
> entre que se arma el TP y que se presenta. El CSV se versiona en el repositorio
> para congelar exactamente los datos sobre los que se obtuvieron los resultados.

---

## Reglas metodológicas del proyecto

Estas cuatro reglas se respetan en todas las fases y son la defensa contra el
**data leakage**. La Clase 2 (slides 93–96) plantea explícitamente dónde separar el
test y advierte: *"si quitamos outliers e imputamos valores usando TODO el dataset,
información que luego irá al test estará siendo usada para train"*.

1. **El split train/test se hace inmediatamente después de la carga**, antes de
   cualquier limpieza, imputación o cálculo de estadísticos.
2. **Todo estadístico de limpieza/imputación/escalado se calcula sólo con el train**
   (medias, medianas, IQR, etc.) y esa misma transformación —con esos mismos
   parámetros— se aplica después al test y al fold de validación.
3. **La validación cruzada se hace únicamente sobre el train.** Nunca toca el test.
4. **El test se usa una sola vez, al final, y jamás para elegir modelo.**

> Nota sobre el orden del enunciado: el enunciado numera "1. Limpieza de datos" antes
> de "2.1 Separación de datos". Ese es el orden de *exposición* de las consignas, no
> el de *ejecución*. Este repositorio ejecuta el split primero, siguiendo el criterio
> explícito de la Clase 2.

---

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

## Setup

Requiere Python 3.14 (verificado sobre 3.14.4, Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir `.venv`.

### Regenerar los splits

Los archivos de `data/processed/` no se versionan porque se reconstruyen de forma
determinista a partir del CSV crudo y la semilla de `src/config.py`:

```bash
python -m src.data
```

---

## Entrega

Formato confirmado con la cátedra:

- **El código se entrega como este repositorio de GitHub** (alternativa aceptada: los
  notebooks sueltos). La cátedra no va a ejecutarlo.
- Lo que debe quedar visible es **el código con el que se entrenaron los modelos y se
  eligieron los hiperparámetros**, no sólo el que produce los gráficos finales. Por eso las
  Fases 4 y 5 conservan la búsqueda completa —todos los grados de polinomio y todos los
  valores de lambda evaluados— en una tabla de resultados, y no únicamente el modelo ganador.
- **Semilla fija en el split train/test**: confirmado como criterio correcto. Para la
  validación, la cátedra señala que sería válido usar semillas aleatorias y repetir el
  k-fold con folds distintos en cada iteración (*repeated k-fold*), como variación del
  esquema visto en clase.

Entrega: 25/08/2026 (24 h antes de la defensa).

---

## Decisiones tomadas

| Fase | Decisión | Valor | Justificación breve |
|---|---|---|---|
| 1 | Momento del split | Inmediatamente tras la carga | Paso 3 del pipeline (Clase 3, slide 16); evitar data leakage (Clase 2, slides 93–96) |
| 1 | Método de separación | Muestreo aleatorio simple | Clase 3, slide 30 |
| 1 | Proporción train/test | 80 / 20 → 1069 / 268 filas | Clase 3 slide 30; el dev fijo del 60-20-20 se reemplaza por k-fold sobre train (Clase 2 slide 85) |
| 1 | Semilla | 42, en `src/config.py` | Reproducibilidad del resultado en la defensa |
| 1 | Qué se observa del test | Sólo las variables de entrada | La auditoría del split es estructural; el target del test no se mira hasta la Fase 6 (Clase 2, slide 89) |
| 1 | Duplicados exactos | **Eliminados antes del split** (1 fila) | Si no, una copia caía en train y otra en test, y el test dejaba de ser datos nuevos. Es integridad de la recolección, no limpieza estadística |
| 2 | Valores faltantes | Ninguno (0 en las 7 columnas) | No hay nada que imputar ni eliminar; se documenta el análisis (consigna 1.2) |
| 2 | Duplicados en train | 0 | Ya eliminados en la Fase 1; el control lo confirma |
| 2 | Outliers de `charges` | **Se mantienen** (106 por IQR, 9.9%) | El 97.2% son fumadores: no son errores sino la señal principal. Eliminarlos borraría el 48% de los fumadores del train |
| 2 | Outliers de `bmi` y `children` | Se mantienen | Valores fisiológicamente plausibles, sin efecto sobre el costo medio |
| 2 | Rango de validez del modelo | Documentado: `age` 18–64, `bmi` 15.96–53.13 | Predecir fuera de ese rango es extrapolación; se retoma en la Fase 6 |
| 2 | Robustez frente a extremos | Vía regularización L1 (Fase 5) | Estrategia 3 de la Clase 2 slide 42: modelos más robustos, en lugar de eliminar datos |
| 3 | Encoding de las 3 categóricas | **One-hot con `drop="first"`** | Son nominales; sin leakage; 6 variables → 8 columnas |
| 3 | Ordinal / frequency / target encoding | Descartados | Sin jerarquía; las 4 regiones tienen frecuencias casi iguales (0.2367–0.2685); target encoding arriesga leakage sin resolver ningún problema de cardinalidad |
| 3 | Escalado | **z-score (`StandardScaler`)** sobre las numéricas | Clase 3 slide 43; imprescindible para que L1 penalice de forma pareja |
| 3 | Features incluidas | **Las 6** (todas) | Ninguna irrelevante ni redundante; sin problema de dimensionalidad |
| 4 | Esquema de validación | **k-fold con k = 5**, sólo sobre el train | Clase 2 slide 85; usa los datos de forma más eficiente que un dev fijo |
| 4 | Preprocesamiento dentro del `Pipeline` | Sí | `cross_validate` lo reajusta en cada fold; evita que los folds de validación contaminen la media y el desvío |
| 4 | Resultado regresión lineal | RMSE val **6.123,65** · train 6.075,92 · R² 0.72 | Gap de 47,73 (0.8%): sin overfitting |
| 5 | Grados de polinomio evaluados | 1, 2 y 3 | Grado 4 descartado: 494 features para 1069 filas (2.2 por parámetro) |
| 5 | Reescalado tras la expansión polinómica | Sí | Los cuadrados y productos tienen escalas nuevas; sin reescalar L1 penalizaría por magnitud |
| 5 | Valores de lambda (L1) | 0.001, 0.01, 0.1, 1, 10, 100, 1000 | Los 4 de la Clase 3 más 3 mayores: con `charges` en miles, λ < 1 no tiene efecto medible |
| 5 | **Mejor modelo** | **Grado 2, λ = 100** · RMSE val **4.905,3** · R² 0.822 | 19.9% mejor que el lineal; 20 de 44 features sobreviven a L1 |
| 5 | Estabilidad de la selección | Repeated k-fold, 50 particiones | Grado 2 y grado 3 con λ=100 quedan empatados (6 de diferencia, desvío 382); se desempata por simplicidad |
| 6 | Modelo entregado | Polinómica g2 + Lasso λ=100, entrenado con las 1069 filas | Clase 2 slide 88: el modelo elegido se reentrena con todo el train |
| 6 | **RMSE en test** | **4.630,26** · MAE 2.879,09 · R² 0.880 | IC 95% bootstrap: [3.814 , 5.403]. 22.3% mejor que el lineal (5.956,34) |
| 6 | RMSE que se promete | ~4.900 | Se reporta la estimación de validación cruzada, más conservadora que el test |

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
- [ ] **Fase 7 — Entrega**
  - [ ] Presentación de 10 minutos (incluye intro teórica: train/validación/test)
  - [ ] README final
  - [ ] Envío de código + presentación (25/08/2026)

---

## Fuentes

- Enunciado: *TP1. Regresión e Introducción a la evaluación de modelos* — 72.75, 2026 Q2.
- *Clase 2: Datos, variables, overfitting y métricas* — Carlos Bibián Nogueras, PhD.
- *Clase 3: EDA, Feature selection, Regularización y Métricas* — Carlos Bibián Nogueras, PhD.
