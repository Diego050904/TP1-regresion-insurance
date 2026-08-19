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
├── notebooks/                   # Un notebook por fase, narrativo
├── src/
│   └── config.py                # Rutas, semilla y esquema de partición
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

---

## Checklist de fases

- [x] **Fase 0 — Setup inicial**
  - [x] Estructura de carpetas
  - [x] Entorno virtual y `requirements.txt` con versiones fijadas
  - [x] `.gitignore` y README inicial
  - [x] Dataset en `data/raw/insurance.csv`
- [ ] **Fase 1 — Carga y separación de datos** *(consigna 2.1)*
  - [ ] Carga desde ruta local y verificación de integridad
  - [ ] Split train/test con semilla fija, antes de cualquier limpieza
  - [ ] Justificación de la proporción y del método de separación
- [ ] **Fase 2 — Análisis exploratorio (sólo sobre train)** *(consignas 1.2, 1.3)*
  - [ ] Tipos de variables
  - [ ] Valores faltantes: análisis y estrategia
  - [ ] Duplicados
  - [ ] Outliers: detección visual (boxplot, histograma, scatter) y estadística (IQR, z-score)
  - [ ] Decisión justificada sobre qué hacer con los outliers
- [ ] **Fase 3 — Preprocesamiento** *(consignas 1.1, 1.4)*
  - [ ] Encoding de categóricas, con justificación por variable
  - [ ] Imputación (si corresponde), con estadísticos del train
  - [ ] Selección de features y escalado, justificados
  - [ ] Todo encapsulado en un `Pipeline` ajustado sólo con train
- [ ] **Fase 4 — Regresión lineal** *(consignas 2.2, 2.3)*
  - [ ] k-fold cross-validation sobre el train
  - [ ] Entrenamiento dentro del CV
  - [ ] RMSE de train y de validación
- [ ] **Fase 5 — Regresión polinómica y regularización** *(consignas 3.1, 3.2, 3.3, 4)*
  - [ ] Transformación polinómica para varios grados
  - [ ] Entrenamiento sobre las variables transformadas
  - [ ] Regularización L1 (Lasso) con un par de valores de lambda
  - [ ] Tabla de RMSE de validación por grado y lambda
- [ ] **Fase 6 — Evaluación final y comparación** *(consigna 5)*
  - [ ] Evaluación en test (una única vez)
  - [ ] ¿Qué modelo obtuvo menor error?
  - [ ] ¿Cuál implementarían en una aplicación real? Justificar
  - [ ] ¿Qué RMSE esperar en datos nuevos?
- [ ] **Fase 7 — Entrega**
  - [ ] Presentación de 10 minutos (incluye intro teórica: train/validación/test)
  - [ ] README final
  - [ ] Envío de código + presentación (25/08/2026)

---

## Fuentes

- Enunciado: *TP1. Regresión e Introducción a la evaluación de modelos* — 72.75, 2026 Q2.
- *Clase 2: Datos, variables, overfitting y métricas* — Carlos Bibián Nogueras, PhD.
