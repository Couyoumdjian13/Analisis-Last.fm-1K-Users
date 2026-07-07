# Repeat-Aware Recommendation — IIC3633 Grupo 32

Repositorio del proyecto final del curso **IIC3633 Sistemas Recomendadores** (semestre 2026-1, Pontificia Universidad Católica de Chile). Aborda el problema de **recomendación de ítems repetidos** (*Repeat-Aware Recommendation*): aprender, a partir del historial temporal de interacciones, cuándo un usuario tendrá mayor probabilidad de repetir el consumo de un ítem ya visitado en lugar de explorar uno nuevo.

**Equipo:** Pedro Munita · José Racioppi · Tomás Couyoumdjian.

---

## Tabla de Contenidos

1. [Resumen del proyecto](#1-resumen-del-proyecto)
2. [Estado actual del repositorio](#2-estado-actual-del-repositorio)
3. [Hallazgos principales](#3-hallazgos-principales)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Datasets](#5-datasets)
6. [Modelos](#6-modelos)
7. [Protocolo de evaluación](#7-protocolo-de-evaluación)
8. [Setup y reproducción de resultados](#8-setup-y-reproducción-de-resultados)
9. [Documentos del curso](#9-documentos-del-curso)
10. [Cronograma y entregables](#10-cronograma-y-entregables)
11. [Bibliografía clave](#11-bibliografía-clave)

---

## 1. Resumen del proyecto

Los sistemas recomendadores convencionales tratan el historial de interacciones como evidencia de preferencias positivas, asumiendo que el interés en un ítem se agota tras su primer consumo. Esta hipótesis es incorrecta en escenarios reales como streaming musical (los usuarios re-escuchan sus canciones favoritas en ráfagas) o e-commerce de productos de consumo frecuente (recompra periódica). Ignorar este fenómeno **penaliza por construcción** ítems que constituyen las preferencias más consolidadas del usuario.

El proyecto modela explícitamente dos modos de consumo coexistentes:

* **Repeat consumption** — gobernado por intervalos temporales y frecuencia histórica de repetición.
* **Explore consumption** — el filtrado colaborativo clásico orientado al descubrimiento.

El **objetivo general** es desarrollar y comparar modelos que aprendan cuándo cada modo es más probable, superando el desempeño de baselines estándar bajo Recall@10 y nDCG@10.

---

## 2. Estado actual del repositorio

| Componente | Estado | Ubicación |
|---|---|---|
| Pipeline de preprocesamiento del TSV Last.fm 1K | ✅ Implementado y verificado end-to-end | [`notebooks/preprocessing.py`](notebooks/preprocessing.py) |
| Subset H1 (100 usuarios × 2 000 plays) en Parquet | ✅ Generado (200 000 filas, 4.4 MB) | `data/lastfm_100_users_h1_fixed.parquet` |
| Dataset completo limpio (992 usuarios) en Parquet | ✅ Generado (19 098 853 filas, 375 MB) | `data/lastfm_1k_complete_fixed.parquet` |
| Baselines repeat-aware (4 variantes) | ✅ Implementados con interfaz común | [`src/models/baselines.py`](src/models/baselines.py) |
| Pipeline de evaluación leave-one-out temporal | ✅ Implementado y reproducible | [`src/evaluation.py`](src/evaluation.py) |
| Resultados de baselines y modelos avanzados | ✅ Calculados sobre subset H1 | `data/baselines_results.csv`, `data/repeat_advanced_results.csv` |
| Figuras del EDA y correlación frecuencia ↔ reaparición (OE1) | ✅ Reproducible | [`notebooks/eda_figures.py`](notebooks/eda_figures.py), [`docs/figures/`](docs/figures/) |
| Informe H2 (Markdown) | ✅ Entregado (documento histórico) | [`docs/H2_Midterm.md`](docs/H2_Midterm.md) |
| **Paper Final H3** | ✅ Entregado | [`docs/Informe_Final.md`](docs/Informe_Final.md) |
| **Temporal BPR (T-BPR)** — ventana configurable | ✅ Implementado, con parámetros `window_low_pct`/`window_high_pct` | [`src/models/tbpr.py`](src/models/tbpr.py) |
| **Búsqueda de hiperparámetros T-BPR** (rolling validation) | ✅ Implementado | [`src/tune_tbpr.py`](src/tune_tbpr.py) |
| **Análisis de sensibilidad** sobre intervalo W_u | ✅ Implementado | [`src/sensitivity_analysis.py`](src/sensitivity_analysis.py) |
| **Evaluación escalada** (992 usuarios, submuestreo) | ✅ Implementado | [`src/run_scaled_evaluation.py`](src/run_scaled_evaluation.py) |
| **PISA** (aproximación reproducible, RecSys 2024) | ✅ Implementado (ACT-R + contexto + popularidad) | [`src/models/pisa.py`](src/models/pisa.py) |
| **RepeatNet** (aproximación reproducible, AAAI 2019) | ✅ Implementado (gate repeat/explore) | [`src/models/repeatnet.py`](src/models/repeatnet.py) |
| Runner consolidado con `run_id` compartido | ✅ Implementado | [`src/run_all_models.py`](src/run_all_models.py) |
| **Modelo Híbrido Repeat/Explore** | 🔮 Trabajo futuro — diseño formalizado en §7.1 del paper | `docs/Informe_Final.md` §7.1 |
| Preprocesamiento Amazon Grocery | 🔮 Trabajo futuro — §7.2 del paper | `docs/Informe_Final.md` §7.2 |

---

## 3. Hallazgos principales

### Del análisis descriptivo del subset H1 (100 usuarios, 200 000 plays)

* **Tasa global de repetición = 60.43 %** — más de la mitad de las reproducciones del dataset son re-escuchas. Cuestiona el supuesto fundacional del filtrado colaborativo clásico.
* **Distribución de intervalos entre repeticiones (n = 87 985):** mediana 27.04 h, p₂₅ 1.74 h, p₇₅ 160.67 h, media 175.05 h, máximo 13 503.65 h.
  * El 50 % de las repeticiones ocurre dentro de las primeras 27 h → **localidad temporal / ráfagas de escucha**.
  * La cola larga revela ítems "favoritos persistentes" que reaparecen meses después.
* **Heterogeneidad inter-usuario:** la tasa de repetición individual varía entre 3.2 % y 98.2 % (std 0.207). Coexisten usuarios "exploradores" y "loopers" en el mismo dataset.
* **Correlación frecuencia histórica ↔ probabilidad de reaparición (OE1):** Spearman ρ = 0.284 (p < 10⁻³⁰⁰, n = 64 018 pares (usuario, ítem) con split 50/50 temporal). La frecuencia *sí* es señal, pero moderada-débil: explica una fracción menor de la varianza del fenómeno de reaparición, lo que motiva combinar frecuencia con recencia en los modelos avanzados.

### De la evaluación de baselines bajo LOO temporal

Sobre el subset H1, train = 199 900 / test = 100 (la última interacción cronológica de cada usuario):

| Modelo | Recall@10 | nDCG@10 | MRR | Repeat Ratio | Hits |
|---|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.009 | 0/100 |
| MostPopular | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| SimpleRepeat-Freq | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| **SimpleRepeat-Recency** | **0.0900** | **0.0667** | **0.0597** | 1.000 | **9/100** |

Hallazgos críticos (detalle en [`docs/H2_Midterm.md`](docs/H2_Midterm.md) §4.3):

* Sólo el **33 %** de las últimas interacciones es un repeat (vs 60 % global) — los eventos terminales son sistemáticamente más exploratorios que la media.
* Cuando es repeat, el ítem ocupa una **posición mediana 221** en el ranking por frecuencia del historial → la frecuencia histórica **no es informativa** para predecir el próximo consumo.
* Cambiar el ranking de "frecuencia" a "recencia" eleva Recall@10 de 0.00 a 0.09 (de los 9 hits, 5 caen en rank 1). Confirmación empírica directa de la hipótesis de localidad temporal del análisis descriptivo.

---

## 4. Estructura del repositorio

```text
Analisis-Last.fm-1K-Users/
├── data/                                      # Datasets procesados y resultados (ignorado por git)
│   ├── lastfm_1k_complete_fixed.parquet       # Dataset completo (992 usuarios, 375 MB)
│   ├── lastfm_100_users_h1_fixed.parquet      # Subset H1 (100 usuarios, 4.4 MB)
│   ├── baselines_results.csv                  # Métricas baselines
│   ├── repeat_advanced_results.csv            # Métricas T-BPR, PISA, RepeatNet
│   ├── tbpr_tuning_results.csv                # Grid de hiperparámetros T-BPR
│   ├── tbpr_best_config.json                  # Mejor configuración T-BPR
│   ├── sensitivity_interval_results.csv       # Sensibilidad al intervalo W_u
│   ├── scaled_eval_per_round.csv              # Evaluación escalada por ronda
│   └── scaled_eval_summary.csv               # Resumen promedio evaluación escalada
├── notebooks/
│   ├── preprocessing.ipynb                    # Notebook exploratorio H1
│   ├── preprocessing.py                       # Pipeline chunked reproducible
│   └── eda_figures.py                         # Figuras EDA y correlación Spearman
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py                       # Random, MostPopular, SimpleRepeat-Freq/Recency
│   │   ├── tbpr.py                            # T-BPR con ventana configurable
│   │   ├── pisa.py                            # Aproximación reproducible de PISA
│   │   └── repeatnet.py                       # Aproximación reproducible de RepeatNet
│   ├── evaluation.py                          # temporal_loo_split + evaluate_recommender
│   ├── utils.py                               # recall@k, nDCG@k, MRR, repeat_ratio
│   ├── tbpr_config.py                         # Configuración por defecto de T-BPR
│   ├── run_baselines.py                       # Runner baselines
│   ├── run_repeat_advanced.py                 # Compara T-BPR vs PISA vs RepeatNet
│   ├── run_tbpr.py                            # Evalúa T-BPR standalone
│   ├── run_all_models.py                      # Orquesta todas las corridas
│   ├── tune_tbpr.py                           # Búsqueda de hiperparámetros rolling
│   ├── sensitivity_analysis.py                # Sensibilidad al intervalo W_u
│   └── run_scaled_evaluation.py              # Evaluación escalada 992 usuarios
├── docs/
│   ├── Informe_Final.md                       # Paper final H3
│   ├── H2_Midterm.md                          # Informe intermedio H2
│   ├── Poster RecSys.pdf                      # Póster H4
│   └── figures/                               # PNGs del EDA y análisis de sensibilidad
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 5. Datasets

### Last.fm 1K Users (dominio principal — alta frecuencia, ráfagas cortas)

Archivo crudo: `userid-timestamp-artid-artname-traid-traname.tsv` (~2.5 GB, 19 098 853 filas, 992 usuarios). Disponible en el repositorio oficial de Last.fm Research; no se versiona aquí por tamaño.

**Filas corruptas conocidas.** El TSV original contiene 9 filas con campos malformados (índices `2120260, 2446318, 11141081, 11152099, 11152402, 11882087, 12902539, 12935044, 17589539`). El pipeline las salta vía `skiprows`.

**Estrategia de identificación de ítems.** Más de 69 000 artistas carecen de identificador MusicBrainz (`artist_id` / `track_id` con `NaN`), lo que propagaría errores a los *embeddings*. Construimos una clave única de ítem como:

```python
item_id = f"{artist_name} - {track_name}"
```

Esta convención mantiene determinismo, evita *NaNs* y permite que dos pistas con el mismo nombre interpretadas por distintos artistas se distingan.

### Amazon Reviews — Grocery & Gourmet Food (2018) (dominio secundario — pendiente)

Dataset del McAuley Lab. Régimen de repeat consumption esperado: baja frecuencia, estacional (semanas a meses entre recompras). Se incorporará como track experimental paralelo (no como respaldo) según retroalimentación del corrector. Preprocesamiento previsto para semana 1 del cronograma a H3 (06–12 jun).

---

## 6. Modelos

### Baselines implementados (`src/models/baselines.py`)

Todos cumplen la interfaz común:

```python
class Recommender:
    def fit(self, train_df: pd.DataFrame) -> None: ...
    def recommend(self, user_id: str, user_history: Set[str], k: int = 10) -> List[str]: ...
```

* **`RandomRecommender`** — muestreo uniforme sobre el catálogo completo (con semilla fija para reproducibilidad).
* **`MostPopularRecommender`** — ranking global por número de reproducciones en el set de entrenamiento.
* **`SimpleRepeatRecommender`** — *repeat-aware* por **frecuencia**: rankea el historial del usuario por número de reproducciones; *fallback* a popularidad global si el usuario tiene menos de K ítems únicos.
* **`SimpleRepeatRecencyRecommender`** — *repeat-aware* por **recencia**: rankea el historial del usuario en orden cronológico inverso (último ítem único primero). Incorporado en H2 para verificar empíricamente la hipótesis de localidad temporal.

**Nota metodológica.** Bajo el paradigma repeat-aware **no se excluye** el historial del usuario del espacio de candidatos. Esta decisión es deliberada: filtrar el historial penaliza por construcción a cualquier modelo que sugiera repeticiones, distorsionando la evaluación (es la causa por la cual `SimpleRepeat` puntuaba 0 en el setup exploratorio de H1).

### Modelos avanzados (estado actual)

* **Temporal BPR (T-BPR)** — adaptación del Bayesian Personalized Ranking [Rendle et al. 2009] mediante un esquema de muestreo negativo dependiente del tiempo: los ítems repetidos dentro de la ventana temporal típica del usuario se penalizan con peso $\alpha < 1$; los repetidos fuera de su ventana con peso $\beta > 1$. La ventana $W_u = (a_u, b_u)$ se calibra por usuario a partir de los percentiles 25 y 75 de su distribución empírica de intervalos. Implementado en `src/models/tbpr.py` y evaluable con `src/run_tbpr.py`.
* **Modelo Híbrido Repeat/Explore** — clasificador de segundo nivel (regresión logística) que decide entre delegar al sub-modelo *repeat* (`SimpleRepeat-Recency`) o al sub-modelo *explore* (filtrado colaborativo iALS), usando como features la tasa de repetición histórica, el tiempo desde la última reproducción, el log-intervalo medio entre repeticiones y la actividad en ventana de 30 días. Diseño documentado, implementación aún pendiente.
* **RepeatNet** [Ren et al. 2019] — en este repositorio se incluye una aproximación reproducible con dos ramas explícitas (*repeat* y *explore*) y una compuerta por usuario para mezclarlas.
* **PISA** [Tran et al. 2024, RecSys] — en este repositorio se incluye una aproximación reproducible que combina activación temporal tipo ACT-R, contexto de sesión reciente y prior de popularidad.

---

## 7. Protocolo de evaluación

**Split temporal leave-one-out (LOO):** para cada usuario, la última interacción cronológica se reserva como ground truth de test; el resto del historial conforma el train. Sobre el subset H1: train = 199 900 filas, test = 100 filas (una por usuario).

**Métricas (K = 10):**

* **Recall@10** — ¿el ítem relevante apareció en algún lugar del top-10? Equivalente a HitRate@10 bajo LOO.
* **nDCG@10** — penaliza por posición: hit en rank 1 vale 1.00, en rank 10 vale 0.29. Crítica en repeat-aware porque queremos el ítem repetido en los primeros lugares, no sólo "presente".
* **MRR** — recíproco de la posición del primer (y único) hit.
* **Repeat Ratio** — fracción del top-10 que pertenece al historial del usuario. **Métrica diagnóstica**, no de calidad: indica el sesgo del modelo en el eje *explore-only* (Repeat Ratio ≈ 0) ↔ *repeat-only* (Repeat Ratio ≈ 1). La calibración objetivo en Last.fm es ≈ 0.33 (la tasa empírica de repetición observada en la última interacción).

Definiciones formales y discusión en [`docs/H2_Midterm.md`](docs/H2_Midterm.md) §2.

---

## 8. Setup y reproducción de resultados

**Requisitos:** Python 3.10+, ~ 4 GB RAM disponibles.

**Instalación (PowerShell en Windows):**

```powershell
git clone https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users.git
cd Analisis-Last.fm-1K-Users
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Instalación (bash en Linux/macOS/WSL):**

```bash
git clone https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users.git
cd Analisis-Last.fm-1K-Users
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Obtener el dataset.** Los datos no se versionan por tamaño (`data/` y `*.parquet` están en `.gitignore`). Descarga el tarball oficial de Last.fm 1K Users (`lastfm-dataset-1K.tar.gz`, ~600 MB) desde el [repositorio oficial](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html), extráelo y coloca el TSV crudo en `data/` con su nombre original:

```
data/userid-timestamp-artid-artname-traid-traname.tsv   (~2.5 GB, 19 098 853 filas)
```

**Paso 1 — Preprocesamiento (≈ 70 s, < 1 GB RAM):**

```bash
python notebooks/preprocessing.py
```

Genera en `data/` los **dos parquets que consumen todos los scripts siguientes** (prerequisito obligatorio):
- `lastfm_1k_complete_fixed.parquet` (≈ 375 MB, 19 098 853 filas, 992 usuarios) — requerido por el Paso 6.
- `lastfm_100_users_h1_fixed.parquet` (≈ 4.4 MB, 200 000 filas, top-100 usuarios × tail 2 000) — requerido por los Pasos 2–5.

**Paso 2 — Evaluación de baselines (≈ 15 segundos):**

```bash
python src/run_baselines.py
```

**Paso 3 — Comparación avanzada (T-BPR vs PISA vs RepeatNet):**

```bash
python src/run_repeat_advanced.py
```

**Paso 4 — Búsqueda de hiperparámetros de T-BPR (rolling validation) — ≈ 70 min en CPU:**

```bash
python src/tune_tbpr.py
```

- **Requiere:** `data/lastfm_100_users_h1_fixed.parquet` (Paso 1).
- **Genera:** `data/tbpr_tuning_results.csv` (grid completo, 6 configs) y `data/tbpr_best_config.json` (config ganadora, consumida por el Paso 6).
- **Tiempo:** evalúa 6 configuraciones × 3 folds rolling (offsets {2,3,4}); cada entrenamiento de T-BPR tarda ~150–320 s según `factors`/`epochs`, para un total de ≈ 70 min. Acepta `--max-configs N` y `--fold-offsets a,b,c` para acotar la corrida.

**Paso 5 — Análisis de sensibilidad sobre el intervalo W_u — ≈ 4 h en CPU (barrido completo):**

```bash
python src/sensitivity_analysis.py
```

- **Requiere:** `data/lastfm_100_users_h1_fixed.parquet` (Paso 1).
- **Genera:** `data/sensitivity_interval_results.csv` y (si matplotlib está disponible) `docs/figures/fig_sensitivity_interval.png`.
- **Tiempo:** entrena T-BPR con la config por defecto en las 49 combinaciones válidas `(p_low, p_high)`, a ~300 s cada una (≈ 4 h en total). Es el paso más costoso; ejecutar solo si se necesita el mapa de calor de §5.3.

**Paso 6 — Evaluación escalada sobre los 992 usuarios — ≈ 30–60 min en CPU:**

```bash
python src/run_scaled_evaluation.py
```

- **Requiere:** `data/lastfm_1k_complete_fixed.parquet` (Paso 1) **y** `data/tbpr_best_config.json` (Paso 4; si falta, usa la config por defecto de `tbpr_config.py`).
- **Genera:** `data/scaled_eval_per_round.csv` y `data/scaled_eval_summary.csv`.
- **Tiempo:** 5 rondas × 100 usuarios muestreados × 5 modelos; el costo lo domina T-BPR (5 reentrenamientos). Ajustable vía las constantes `N_SAMPLE`, `N_ROUNDS` en el script.

**Paso 7 (opcional) — Corrida consolidada con `run_id` compartido:**

```bash
python src/run_all_models.py
```

**Paso 8 — Regenerar figuras del EDA:**

```bash
python notebooks/eda_figures.py
```

---

## 9. Documentos del curso

* **H1 — Propuesta (entregada, 80/100):** PDF entregado vía Canvas (no versionado).
* **H2 — Informe Intermedio (entregado, 68/70, 2026-06-05):** [`docs/H2_Midterm.md`](docs/H2_Midterm.md).
* **H3 — Paper Final (entregado, 2026-07-07):** [`docs/Informe_Final.md`](docs/Informe_Final.md). Incluye comparación de modelos, búsqueda de hiperparámetros, análisis de sensibilidad y evaluación escalada.
* **H4 — Póster (sesión presencial 2026-07-02):** [`docs/Poster RecSys.pdf`](docs/Poster%20RecSys.pdf). Nota: el diagrama del modelo híbrido incluido en el poster fue generado con asistencia de IA generativa.

---

## 10. Cronograma y entregables

| Hito | Fecha | Peso | Estado |
|---|---|---|---|
| H1 Propuesta | 2026-05-08 | 15 % | ✅ Entregada (80/100) |
| H2 Midterm | 2026-06-05 | 25 % | ✅ Entregada (68/70) |
| H3 Paper | 2026-07-07 | 50 % | ✅ Entregado |
| H4 Sesión Pósters | 2026-07-02 | 10 % | ✅ Presentado |

---

## 11. Bibliografía clave

* Tran, V.-A., Salha-Galvan, G., Sguerra, B., & Hennequin, R. (2024). *Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation.* **RecSys 2024**. Código: <https://github.com/deezer/recsys24-pisa>.
* Ren, P., Chen, Z., Li, J., Ren, Z., Ma, J., & de Rijke, M. (2019). *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation.* **AAAI 2019**.
* Benson, A. R., Kumar, R., & Tomkins, A. (2016). *Modeling User Consumption Sequences.* **WWW 2016**.
* Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* **UAI 2009**.
* Zhao, W. X., et al. (2021). *RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms.* **CIKM 2021**.

---

*Pontificia Universidad Católica de Chile · Escuela de Ingeniería · Departamento de Ciencia de la Computación · IIC3633 Sistemas Recomendadores 2026-1*
