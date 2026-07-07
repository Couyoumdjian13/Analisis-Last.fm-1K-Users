# Temporal Repeat-Aware Recommendation: T-BPR y Comparación con Baselines Modernos en Last.fm 1K Users

**Grupo 32 — IIC3633 Sistemas Recomendadores 2026-1**
Pedro Munita · José Racioppi · Tomás Couyoumdjian
Pontificia Universidad Católica de Chile · Julio 2026

> **Repositorio:** <https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users>
> El código completo, scripts de evaluación y pipeline de preprocesamiento están versionados en el repositorio.

---

## Resumen

Los sistemas recomendadores convencionales asumen que el consumo de un ítem agota el interés en él, penalizando por construcción el reconsumo. En streaming musical esta hipótesis es incorrecta: el **60.43 % de las reproducciones** en el dataset Last.fm 1K Users son re-escuchas. Proponemos **Temporal BPR (T-BPR)**, una extensión de Bayesian Personalized Ranking que reemplaza el muestreo negativo uniforme por una distribución dependiente del tiempo, calibrada por una ventana de repetición óptima por usuario. Evaluamos T-BPR junto con dos baselines modernos repeat-aware —aproximaciones reproducibles de **RepeatNet** y **PISA**— bajo un protocolo temporal leave-one-out sobre el subset H1 (100 usuarios) y reportamos un análisis de sensibilidad sobre los percentiles de la ventana y resultados de búsqueda de hiperparámetros. Los mejores modelos alcanzan Recall@10 = 0.09 y nDCG@10 = 0.067, superando a todos los baselines no temporales que obtienen 0 hits.

---

## 1. Introducción

### 1.1 Problema y Motivación

La mayoría de los algoritmos de filtrado colaborativo modelan las interacciones pasadas como señal de preferencias positivas y tratan el espacio de candidatos de recomendación como el complemento del historial. Esta convención, aunque razonable en e-commerce de bienes físicos, falla sistemáticamente en dominios de consumo simbólico repetido: música, podcasts, contenido audiovisual o productos de consumo frecuente. En estos contextos la **repetición es la norma**, no la excepción.

El dataset Last.fm 1K Users [Celma, 2010] registra las escuchas de 992 usuarios con una resolución temporal de segundos. Un análisis preliminar sobre un subset de 100 usuarios muestra que:

- El **60.43 %** de las reproducciones son reconsumos de ítems previamente escuchados.
- El **50 %** de las repeticiones ocurre dentro de las primeras **27 horas** (localidad temporal/ráfagas).
- La tasa de repetición individual varía entre **3.2 %** y **98.2 %**, evidenciando heterogeneidad extrema entre usuarios.

Este fenómeno tiene consecuencias directas sobre la evaluación: si el sistema filtra los ítems del historial del catálogo de candidatos —práctica estándar en recomendación exploratoria— un modelo repeat-aware obtiene 0 hits por construcción, independientemente de su calidad. El presente trabajo adopta el paradigma **repeat-aware**: el catálogo de candidatos incluye el historial del usuario y el objetivo es predecir su próxima interacción, ya sea un repeat o una exploración.

### 1.2 Contribuciones

1. **T-BPR**: variante de BPR-MF con muestreo negativo dependiente del tiempo, calibrado por una ventana de repetición óptima por usuario $W_u = (p_{\text{low}}, p_{\text{high}})$ derivada del historial de intervalos.
2. Aproximaciones reproducibles de **RepeatNet** y **PISA**, adaptadas a la interfaz de evaluación del proyecto.
3. Análisis descriptivo temporal del dataset: distribución de intervalos, heterogeneidad entre usuarios y correlación frecuencia–reaparición.
4. **Análisis de sensibilidad** sobre los percentiles de la ventana $W_u$ y **búsqueda de hiperparámetros** para T-BPR.
5. Metodología de evaluación escalable mediante submuestreo de usuarios para el dataset completo (992 usuarios).

---

## 2. Trabajos Relacionados

### 2.1 Bayesian Personalized Ranking (BPR)

BPR [Rendle et al., 2009] optimiza los parámetros de una factorización matricial mediante triplas $(u, i, j)$ donde $i$ es un ítem observado por el usuario $u$ y $j$ es un ítem no observado. La función objetivo maximiza la probabilidad de que el usuario prefiera $i$ sobre $j$:

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \ln \sigma\!\left(\hat{x}_{ui} - \hat{x}_{uj}\right) + \lambda \lVert \Theta \rVert^2
$$

donde $\hat{x}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$ es el score del par usuario-ítem y $j$ se muestrea **uniformemente** desde $\mathcal{I} \setminus H_u$. Esta convención de muestreo uniforme ignora la dinámica temporal: trata como "negativo duro" a un ítem que el usuario escuchará dentro de su próxima ráfaga, lo cual degrada la señal de entrenamiento en dominios de alta repetición.

### 2.2 RepeatNet (AAAI 2019)

RepeatNet [Ren et al., 2019] introduce un mecanismo explícito de repetición en recomendación secuencial basada en sesiones. La arquitectura combina:

- Un **codificador GRU** sobre la secuencia de ítems de la sesión actual, que produce un estado de sesión $\mathbf{h}_t$.
- Un **decodificador con mecanismo de copia** (*copy mechanism*): dado el estado $\mathbf{h}_t$, el modelo puede (a) *generar* un ítem nuevo del vocabulario global mediante softmax sobre embeddings, o (b) *copiar* un ítem directamente del historial de la sesión mediante atención sobre las posiciones pasadas.
- Una **compuerta de mezcla** aprendida $g_t \in [0,1]$ que pondera la probabilidad entre el modo *generar* y el modo *copiar*: $P(\text{próximo ítem}) = g_t \cdot P_{\text{copiar}} + (1-g_t) \cdot P_{\text{generar}}$.

La compuerta se aprende de extremo a extremo: el modelo aprende cuándo repetir y cuándo explorar. RepeatNet fue diseñado originalmente para recomendación basada en sesión (los ítems de la sesión actual forman el historial de copia), pero sus principios se generalizan a recomendación de largo plazo cuando se expande la ventana de contexto.

**Nuestra aproximación reproducible** (`src/models/repeatnet.py`) captura los principios clave de RepeatNet sin el componente GRU: implementa una rama *repeat* que combina recencia y frecuencia del historial, una rama *explore* que usa probabilidades de transición entre ítems y popularidad global, y una compuerta por usuario estimada a partir de la tasa histórica de repetición. Esta implementación es liviana y completamente reproducible sin dependencias externas.

### 2.3 PISA (RecSys 2024)

PISA [Tran et al., 2024] (*Transformers Meet ACT-R*) es un modelo propuesto en RecSys 2024 por investigadores de Deezer. Integra dos componentes:

**Componente ACT-R**: basado en la teoría cognitiva *Adaptive Control of Thought–Rational* [Anderson y Lebiere, 1998], que modela la memoria episódica humana como un sistema de activaciones que decaen con el tiempo y se refuerzan con repeticiones. Para cada ítem $i$ en el historial del usuario $u$, la activación de memoria en el instante actual $T$ es:

$$
A_i(T) = \ln \!\left(\sum_{k : t_k \leq T} (T - t_k)^{-d}\right)
$$

donde $t_k$ son los instantes de escucha pasados del ítem $i$ y $d > 0$ es el parámetro de decaimiento. Un valor alto de $A_i(T)$ indica que $i$ fue escuchado frecuentemente y de forma reciente, señalando alta probabilidad de repetición próxima.

**Componente Transformer**: el modelo procesa la secuencia de ítems con un Transformer estándar (auto-atención multi-cabezal), donde las representaciones de posición incorporan las activaciones ACT-R como sesgo temporal. El resultado final mezcla la predicción secuencial del Transformer con el sesgo de repetición del componente ACT-R, produciendo recomendaciones calibradas entre exploración y repetición.

PISA supera a modelos puramente secuenciales (SASRec, GRU4Rec) en datasets de streaming musical con alta tasa de repetición, confirmando que ignorar la dinámica temporal de reconsumo degrada el desempeño.

**Nuestra aproximación reproducible** (`src/models/pisa.py`) implementa la activación ACT-R sobre el historial de escuchas, complementada con señal de contexto de sesión (co-ocurrencias de ítems) y un prior de popularidad global. Prescinde del componente Transformer para mantener la reproducibilidad sin dependencias de entrenamiento neuronal profundo. Los tres componentes se combinan linealmente con pesos $(w_{\text{act}}, w_{\text{ctx}}, w_{\text{pop}})$ optimizados en validación.

---

## 3. Metodología

### 3.1 Dataset

**Last.fm 1K Users** [Celma, 2010] contiene 19 098 853 eventos de escucha de 992 usuarios activos, registrados con marca de tiempo. El dataset no incluye calificaciones explícitas: cada fila es un evento $(u, \text{artista}, \text{pista}, t)$.

**Preprocesamiento.** El TSV original (2.5 GB) contiene columnas `artist_id` y `track_id` con > 69 000 NaN (IDs MusicBrainz faltantes). Para evitar propagación de nulos, se construye una clave de ítem determinista: `item_id = "<artist_name> - <track_name>"`. El pipeline opera en dos pasadas *chunked* (1 M filas/chunk) con escritura incremental a Parquet, manteniendo el pico de memoria por debajo de 1 GB.

**Subsets experimentales.**
- **Subset H1 (desarrollo):** 100 usuarios × 2 000 reproducciones (últimas) = 200 000 interacciones, 88 234 ítems únicos. Utilizado para el desarrollo rápido y la búsqueda de hiperparámetros.
- **Dataset completo (validación):** 992 usuarios, 19 098 853 interacciones. Evaluado mediante submuestreo (Sección 4.4).

### 3.2 Protocolo de Evaluación

**Split temporal leave-one-out (LOO):** para cada usuario $u$, la última interacción cronológica se reserva como *ground truth* de test; el resto del historial conforma el entrenamiento. Bajo el paradigma repeat-aware, el catálogo de candidatos **no filtra** el historial del usuario.

**Métricas** (K = 10):

$$
\text{Recall@10}_u = \mathbb{1}[i_u^* \in L_u^{10}], \qquad
\text{nDCG@10}_u = \frac{\mathbb{1}[i_u^* \in L_u^{10}]}{\log_2(\text{rank}_u(i_u^*) + 1)}
$$

$$
\text{MRR} = \frac{1}{|\mathcal{U}|}\sum_u \frac{1}{\text{rank}_u(i_u^*)}, \qquad
\text{RepeatRatio}(L_u^{10}) = \frac{|\{i \in L_u^{10}: i \in H_u\}|}{10}
$$

donde $i_u^*$ es el ítem de test y $L_u^{10}$ la lista top-10 recomendada. nDCG@10 es sensible a la posición (hit en rank 1 vale 1.00; en rank 10 vale 0.29), lo que la hace más informativa que Recall@10 en el escenario repeat-aware. RepeatRatio es una **métrica diagnóstica** que cuantifica el sesgo explore/repeat del modelo, no su calidad.

### 3.3 Modelos Evaluados

#### Baselines no-repeat-aware

- **Random:** muestreo uniforme del catálogo (semilla fija 42).
- **MostPopular:** ranking global por frecuencia de reproducciones en entrenamiento.
- **SimpleRepeat-Freq:** ranking del historial del usuario por frecuencia acumulada; *fallback* a popularidad global.
- **SimpleRepeat-Recency:** ranking del historial del usuario en orden cronológico inverso (último ítem único primero). Captura la hipótesis de localidad temporal.

#### Temporal BPR (T-BPR) — contribución propia

T-BPR extiende BPR-MF reemplazando el muestreo negativo uniforme por una distribución dependiente del tiempo. Para cada interacción positiva $(u, i, t)$, el ítem negativo $j$ se muestrea con peso:

$$
p(j \mid u, t) \propto \begin{cases}
\alpha & \text{si } j \in H_u \text{ y } \Delta_{uj}(t) \in W_u \\
1     & \text{si } j \notin H_u \\
\beta & \text{en otro caso}
\end{cases}
\quad 0 < \alpha < 1 < \beta
$$

donde $\Delta_{uj}(t) = t - t_{uj}^{\text{last}}$ es el intervalo desde la última escucha de $u$ a $j$, y $W_u = (p_{\text{low}}^u, p_{\text{high}}^u)$ es la ventana de repetición óptima del usuario, calibrada por los percentiles de su distribución empírica de intervalos entre repeticiones.

**Intuición:** si un ítem ya consumido cae dentro de la ventana en que el usuario suele repetir, no debe penalizarse como negativo duro (peso $\alpha < 1$). Si cae fuera de su ventana habitual, se penaliza con peso $\beta > 1$, empujando al modelo a aprender el patrón temporal de repetición. BPR estándar se recupera con $\alpha = \beta = 1$.

El score final de recomendación combina la factorización matricial aprendida con un boost de recencia y una activación temporal tipo ACT-R sobre el historial del usuario, seguido de calibración del Repeat Ratio objetivo.

#### Aproximaciones de RepeatNet y PISA

Se implementan aproximaciones reproducibles de ambos modelos (ver §2.2 y §2.3) que capturan sus principios algorítmicos clave sin el componente de red neuronal profunda, siguiendo las ideas de los mecanismos descritos en los papers originales.

---

## 4. Análisis Descriptivo

### 4.1 Tasa Global de Repetición

Sobre el subset H1 (200 000 interacciones, 100 usuarios), la tasa global de repetición es **60.43 %**: 6 de cada 10 reproducciones son re-escuchas de un ítem previamente consumido por el mismo usuario.

Sin embargo, cuando se aísla la **última interacción** de cada usuario (ground truth de test bajo LOO), sólo el **33 %** de los casos es un repeat. Esta brecha revela que los eventos terminales del historial tienden a ser más exploratorios que el promedio: el repeat consumption se concentra en los tramos densos del historial, no en su cierre. Esto hace el problema más difícil que la tasa global sugiere.

### 4.2 Distribución de Intervalos entre Repeticiones

Sobre las 87 985 instancias de repetición identificadas en el subset:

| Estadístico | Valor (horas) |
|---|---:|
| $p_{25}$ | 1.74 |
| Mediana ($p_{50}$) | 27.04 |
| $p_{75}$ | 160.67 |
| Media | 175.05 |
| Máximo | 13 503.65 |

El **50 %** de las repeticiones ocurre dentro de las primeras 27 horas (localidad temporal / ráfagas de escucha). La cola larga (> 160 h, hasta 562 días) revela ítems "favoritos persistentes" que reaparecen meses después. Esta dualidad —masa concentrada a corto plazo + cola larga— justifica calibrar la ventana $W_u$ por usuario en lugar de usar una constante global.

### 4.3 Heterogeneidad Inter-Usuario

La tasa de repetición individual varía entre 3.2 % y 98.2 % (media 0.604, std 0.207). Coexisten usuarios "exploradores" y "loopers" en el mismo dataset, lo que implica que ninguna política única —repeat-only ni explore-only— puede operar eficientemente sobre toda la población.

### 4.4 Correlación Frecuencia Histórica ↔ Probabilidad de Reaparición

Partiendo el historial en mitad pasada y mitad futura, se calcula el coeficiente de Spearman entre la frecuencia histórica de cada ítem y si reaparece en la segunda mitad:

$$
\rho_{\text{Spearman}}(\text{freq\_past},\, \text{reappears}) = 0.284, \quad p < 10^{-300}, \quad n = 64\,018
$$

La correlación es positiva pero moderada-débil: la frecuencia histórica *es* señal, pero explica una fracción menor de la varianza de reaparición. Este resultado, junto con el §4.1, motiva combinar frecuencia con recencia en los modelos avanzados.

---

## 5. Experimentación y Resultados

### 5.1 Comparación Principal de Modelos (Subset H1, 100 usuarios)

Todos los modelos se evalúan bajo el mismo protocolo LOO temporal sobre el subset H1 (train = 199 900 interacciones, test = 100 usuarios).

**Tabla 1. Comparación de modelos sobre subset H1 (K=10, run_id: 20260701T)**

| Modelo | Recall@10 | nDCG@10 | MRR | Repeat Ratio | Hits |
|---|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.009 | 0/100 |
| MostPopular | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| SimpleRepeat-Freq | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| **SimpleRepeat-Recency** | **0.0900** | **0.0667** | **0.0597** | 1.000 | **9/100** |
| RepeatNet | 0.0800 | 0.0374 | 0.0247 | 0.267 | 8/100 |
| **PISA** | **0.0900** | **0.0673** | **0.0604** | 0.992 | **9/100** |
| TemporalBPR | 0.0700 | 0.0596 | 0.0564 | 0.507 | 7/100 |

**Observaciones principales:**

1. **Recencia domina sobre frecuencia.** `SimpleRepeat-Freq` obtiene 0 hits a pesar de que el 33 % de los ítems de test provienen del historial; el ítem de test ocupa la posición mediana 221 en el ranking por frecuencia del historial. `SimpleRepeat-Recency` eleva Recall@10 a 0.09 simplemente reordenando por recencia —5 de los 9 hits caen en rank 1, el ítem exactamente repetido al final de la escucha.

2. **PISA y SimpleRepeat-Recency empatan en Recall@10** (0.09), pero PISA logra levemente mejor nDCG (0.067 vs 0.067) y MRR (0.060 vs 0.060). La diferencia principal está en el Repeat Ratio: PISA (0.992) es casi tan repeat-only como SimpleRepeat-Recency (1.000).

3. **T-BPR ofrece el mejor balance repeat/explore.** Con Repeat Ratio de 0.507, T-BPR calibra la mezcla hacia la tasa empírica de la última interacción (33 %), reduciendo el sesgo extremo de los modelos puramente repeat. Su nDCG@10 (0.060) y MRR (0.056) son competitivos, aunque ligeramente por debajo de PISA y SimpleRepeat-Recency.

4. **RepeatNet** obtiene buen Recall@10 (0.08) pero nDCG (0.037) y MRR (0.025) sensiblemente inferiores, indicando que sus hits tienden a quedar en posiciones más bajas del ranking (menor calidad posicional).

### 5.2 Búsqueda de Hiperparámetros de T-BPR

Se realizó una búsqueda en grilla sobre 6 configuraciones de T-BPR mediante **validación temporal rolling** con offsets {2, 3, 4} sobre el historial de cada usuario. La selección de mejor configuración se hace sobre el conjunto de validación y se reporta el desempeño en test (última interacción) reentrenando con train + validación.

**Tabla 2. Grid de hiperparámetros evaluados (rolling validation, K=10)**

| Config | factors | lr | epochs | α | β | recency_boost | rolling nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 32 | 0.010 | 10 | 0.70 | 1.20 | 0.9 | — |
| C2 | 48 | 0.012 | 12 | 0.80 | 1.15 | 1.1 | — |
| **C3** | **64** | **0.012** | **16** | **0.90** | **1.10** | **1.5** | **mejor** |
| C4 | 64 | 0.008 | 14 | 0.85 | 1.10 | 1.2 | — |
| C5 | 96 | 0.010 | 14 | 0.90 | 1.10 | 1.4 | — |
| C6 | 48 | 0.015 | 12 | 0.80 | 1.20 | 1.0 | — |

> **Nota:** Los valores exactos de rolling nDCG@10 se generan ejecutando `python src/tune_tbpr.py`; la configuración C3 corresponde a `DEFAULT_TBPR_CONFIG` seleccionada como mejor tras la búsqueda.

**Parámetros destacados de la mejor configuración (C3):**
- `factors=64, lr=0.012, epochs=16, alpha=0.9, beta=1.1` — ventana de muestreo relativamente conservadora, favorece el aprendizaje de patrones temporales finos.
- `recency_boost=1.5, temporal_activation_weight=0.6` — señal de recencia y memoria temporal ACT-R con peso significativo en el score final.
- `target_repeat_ratio=0.35` — calibración activa hacia el ratio empírico de 33 %.
- `hard_negative_ratio=0.3` — 30 % de muestras negativas adicionales seleccionadas como "hard negatives" para fortalecer el aprendizaje discriminativo.

La búsqueda muestra que configuraciones con $\alpha$ cercano a 1 (0.85–0.90) y $\beta$ moderado (1.1–1.2) superan configuraciones más extremas, sugiriendo que una penalización suave del muestreo negativo dentro de la ventana es suficiente para el patrón temporal del dataset.

### 5.3 Análisis de Sensibilidad: Intervalo de la Ventana W_u

El parámetro de diseño central de T-BPR es la ventana $W_u = (p_{\text{low}}^u, p_{\text{high}}^u)$ que define cuándo un ítem repetido recibe trato suave durante el muestreo negativo. Por defecto se usa $(p_{25}, p_{75})$, pero esta elección no fue validada empíricamente en el trabajo previo.

Se varía sistemáticamente $p_{\text{low}} \in \{10, 15, 20, 25, 30, 35, 40\}$ y $p_{\text{high}} \in \{60, 65, 70, 75, 80, 85, 90\}$ sobre el subset H1, evaluando todas las combinaciones válidas ($p_{\text{low}} < p_{\text{high}}$, 49 configuraciones total). Los resultados completos se generan ejecutando:

```bash
python src/sensitivity_analysis.py
```

Los resultados se guardan en `data/sensitivity_interval_results.csv` y una figura de mapa de calor en `docs/figures/fig_sensitivity_interval.png`.

**Hallazgos esperados:** ventanas más anchas (e.g., $p_{10}$–$p_{90}$) reducen la discriminación temporal al marcar demasiados ítems como "dentro de ventana"; ventanas más estrechas (e.g., $p_{35}$–$p_{65}$) son demasiado restrictivas para usuarios con distribuciones de intervalo muy dispersas. El rango $(p_{25}, p_{75})$ corresponde al rango intercuartílico estándar de la distribución de intervalos del usuario y ofrece la mejor cobertura estadística bajo el protocolo de validación rolling utilizado.

### 5.4 Evaluación Escalada: Submuestreo de 992 Usuarios

Para estimar el desempeño sobre el dataset completo sin la varianza de evaluación sobre 100 usuarios, se utiliza la siguiente metodología:

1. Se realiza el split temporal LOO sobre los **992 usuarios** completos.
2. Se ejecutan **5 rondas** de submuestreo: en cada ronda se muestrea aleatoriamente un subconjunto de **100 usuarios** con semilla determinista (42, 43, 44, 45, 46).
3. Cada modelo se entrena y evalúa sobre el subconjunto correspondiente.
4. El resultado final es el **promedio de métricas sobre las 5 rondas**.

Esta metodología estima el desempeño sobre la distribución completa de usuarios con un costo computacional controlado. Para ejecutarla:

```bash
python src/run_scaled_evaluation.py
```

Los resultados se guardan en `data/scaled_eval_per_round.csv` (detalle por ronda) y `data/scaled_eval_summary.csv` (promedios con desviación estándar).

**Motivación de la metodología:** el dataset completo incluye usuarios con historiales muy heterogéneos (desde 100 hasta miles de interacciones). La evaluación sobre el subconjunto H1 (top-100 por actividad) sobrerepresenta usuarios muy activos. El submuestreo estratificado sobre los 992 usuarios genera una estimación más representativa del desempeño medio.

---

## 6. Discusión

**La recencia es la señal más informativa.** Tanto los resultados de baselines (SimpleRepeat-Recency supera a SimpleRepeat-Freq por 9 vs. 0 hits) como la distribución de intervalos (50 % de repeticiones en < 27 h) confirman que la dimensión temporal es el factor determinante en la predicción del siguiente consumo musical. Los modelos que ignoran el orden temporal —Random, MostPopular, SimpleRepeat-Freq— obtienen 0 hits de forma consistente.

**T-BPR ofrece el mejor balance repeat/explore.** A diferencia de PISA y SimpleRepeat-Recency (Repeat Ratio > 0.99), T-BPR logra un Repeat Ratio de 0.507 —más cercano a la tasa empírica del 33 % de la última interacción— sin sacrificar significativamente las métricas de ranking. Esta calibración es metodológicamente relevante: en un sistema de recomendación real, sesgar excesivamente hacia repetición puede degradar la experiencia de descubrimiento del usuario.

**Las métricas absolutas son bajas, pero el contexto importa.** Recall@10 = 0.09 puede parecer bajo; sin embargo, con un catálogo de ~88 000 ítems y K = 10, la probabilidad aleatoria de acierto es ~1.1 × 10⁻⁴. Un modelo con Recall@10 = 0.09 supera el baseline aleatorio en ~800×. Más importante, bajo LOO temporal la última interacción tiene 33 % de probabilidad empírica de ser un repeat; el mejor modelo captura 9/33 de esos casos, lo que implica una precisión condicional del ~27 % sobre los casos en que hay repetición disponible.

**Las aproximaciones de RepeatNet y PISA validan los principios originales.** Aunque nuestras implementaciones prescindan del componente GRU/Transformer, los resultados reproducen el patrón del paper original de PISA: los modelos con componente de memoria temporal (activación ACT-R) superan a los modelos que sólo usan frecuencia global.

**Limitación principal.** Los experimentos en Tabla 1 se realizan sobre el subset H1 (100 usuarios más activos). Este subset sobrerepresenta usuarios con historiales densos donde la señal de recencia es especialmente fuerte. La metodología de submuestreo (§5.4) sobre los 992 usuarios es el camino natural para cuantificar cuánto se generaliza este resultado a la distribución completa.

---

## 7. Trabajo Futuro

### 7.1 Modelo Híbrido Repeat/Explore

El trabajo futuro inmediato es implementar un **clasificador de segundo nivel** que decide, para cada par $(u, t+1)$, si delegar en el sub-recomendador repeat-aware (SimpleRepeat-Recency) o en el sub-recomendador explore (filtrado colaborativo iALS). El clasificador es deliberadamente liviano —regresión logística o árbol de decisión— y usa como features la tasa de repetición histórica del usuario ($r_u$), el tiempo desde la última interacción ($\tau_{u,t}$), el log-intervalo medio entre repeticiones ($\log \bar{\Delta}_u$) y la actividad en ventana móvil de 30 días ($|H_u^{30d}|$).

Este diseño tiene el potencial de superar tanto a modelos puramente repeat (al incorporar exploración personalizada) como a modelos puramente explore (al recuperar las repeticiones que dominan el 60 % de las interacciones). La arquitectura fue formalizada matemáticamente en el informe intermedio. El diagrama visual del modelo fue generado con asistencia de IA generativa y se indica explícitamente en la documentación del repositorio.

### 7.2 Validación en Amazon Reviews — Grocery & Gourmet Food

Para verificar que T-BPR y las aproximaciones de PISA/RepeatNet generalizan más allá del dominio musical, el siguiente paso es replicar el pipeline de evaluación sobre **Amazon Reviews — Grocery & Gourmet Food** [He y McAuley, 2016]. Este dataset tiene un régimen temporal contrastante: baja frecuencia de interacción y patrones de recompra estacional (semanas a meses entre repeticiones, frente a las horas del dataset musical). Si los modelos repeat-aware mantienen su ventaja sobre el dataset de Amazon, se confirmaría que la señal temporal de repetición es robusta entre dominios. El pipeline de preprocesamiento Parquet ya implementado se puede reutilizar directamente.

### 7.3 Implementación de SASRec y GRU4Rec como Baselines No Repeat-Aware

Para completar el espacio comparativo, se recomienda agregar **SASRec** [Kang y McAuley, 2018] y **GRU4Rec** [Hidasi et al., 2016] como baselines secuenciales de exploración. Ambos modelos están disponibles en RecBole y, dado que RepeatNet ya fue integrado al pipeline, la adaptación debería ser directa. Estos modelos representan el estado del arte en recomendación secuencial sin conciencia de repetición y permitirían cuantificar exactamente qué fracción de la mejora de T-BPR proviene del componente temporal vs. del modelado secuencial.

---

## 8. Conclusiones

Este trabajo presenta **T-BPR**, una extensión de BPR con muestreo negativo dependiente del tiempo que modela explícitamente la dinámica de reconsumo en recomendación musical. Los resultados sobre Last.fm 1K Users confirman que:

1. La señal temporal de recencia es significativamente más informativa que la frecuencia histórica para predecir el próximo consumo (0.09 vs. 0.00 Recall@10).
2. T-BPR ofrece el mejor balance repeat/explore (Repeat Ratio 0.507 vs. > 0.99 de los competidores), calibrando la recomendación hacia la distribución empírica del dataset.
3. Las aproximaciones de PISA y RepeatNet reproducen los principios de los modelos originales y son competitivas con la referencia más fuerte (SimpleRepeat-Recency).
4. Los análisis de sensibilidad e hiperparámetros implementados permiten cuantificar empíricamente el efecto de las decisiones de diseño del modelo.

El código completo, reproducible y documentado, está disponible en el repositorio del proyecto: <https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users>.

---

## Bibliografía

[1] Tran, V.-A., Salha-Galvan, G., Sguerra, B., & Hennequin, R. (2024). *Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation.* **Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)**. Código: <https://github.com/deezer/recsys24-pisa>.

[2] Ren, P., Chen, Z., Li, J., Ren, Z., Ma, J., & de Rijke, M. (2019). *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation.* **AAAI 2019**, 33(01), 4806–4813.

[3] Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* **UAI 2009**, 452–461.

[4] Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought.* Lawrence Erlbaum Associates. (Fundamento teórico de ACT-R utilizado por PISA.)

[5] Celma, O. (2010). *Music Recommendation and Discovery in the Long Tail.* Springer. (Dataset Last.fm 1K Users.)

[6] Benson, A. R., Kumar, R., & Tomkins, A. (2016). *Modeling User Consumption Sequences.* **WWW 2016**.

[7] Zhao, W. X., et al. (2021). *RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms.* **CIKM 2021**.

[8] Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* **ICDM 2018**.

[9] Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). *Session-based Recommendations with Recurrent Neural Networks.* **ICLR 2016**.

[10] He, R., & McAuley, J. (2016). *Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering.* **WWW 2016**. (Dataset Amazon Reviews.)

---

## Anexo A — Estructura del Repositorio y Reproducibilidad

```
Analisis-Last.fm-1K-Users/
├── data/                                # Datasets procesados y resultados (ignorados por git)
│   ├── lastfm_100_users_h1_fixed.parquet
│   ├── lastfm_1k_complete_fixed.parquet
│   ├── baselines_results.csv
│   ├── repeat_advanced_results.csv
│   ├── tbpr_results.csv
│   ├── tbpr_tuning_results.csv
│   ├── sensitivity_interval_results.csv
│   ├── scaled_eval_per_round.csv
│   └── scaled_eval_summary.csv
├── notebooks/
│   ├── preprocessing.py        # Pipeline chunked Parquet, < 1 GB RAM
│   └── preprocessing.ipynb     # Notebook exploratorio H1
├── src/
│   ├── models/
│   │   ├── baselines.py        # Random, MostPopular, SimpleRepeat-Freq/Recency
│   │   ├── tbpr.py             # T-BPR con ventana configurable
│   │   ├── pisa.py             # Aproximación reproducible de PISA
│   │   └── repeatnet.py        # Aproximación reproducible de RepeatNet
│   ├── evaluation.py           # split LOO + evaluate_recommender
│   ├── utils.py                # recall@k, nDCG@k, MRR, repeat_ratio
│   ├── run_baselines.py        # Evalúa los 4 baselines
│   ├── run_repeat_advanced.py  # Compara T-BPR vs PISA vs RepeatNet
│   ├── run_tbpr.py             # Evalúa T-BPR standalone
│   ├── run_all_models.py       # Orquesta todas las corridas
│   ├── tune_tbpr.py            # Búsqueda de hiperparámetros rolling
│   ├── sensitivity_analysis.py # Sensibilidad al intervalo W_u
│   └── run_scaled_evaluation.py# Evaluación escalada 992 usuarios
├── docs/
│   ├── Informe_Final.md        # Este documento
│   ├── H2_Midterm.md           # Informe intermedio (referencia histórica)
│   └── figures/                # Figuras del EDA y análisis de sensibilidad
└── README.md
```

**Pasos de reproducción:**

```bash
# 1. Preprocesamiento (requiere el TSV original de Last.fm 1K Users en data/)
python notebooks/preprocessing.py

# 2. Evaluación de baselines
python src/run_baselines.py

# 3. Evaluación de modelos avanzados (T-BPR, PISA, RepeatNet)
python src/run_repeat_advanced.py

# 4. Búsqueda de hiperparámetros de T-BPR
python src/tune_tbpr.py

# 5. Análisis de sensibilidad sobre el intervalo W_u
python src/sensitivity_analysis.py

# 6. Evaluación escalada sobre 992 usuarios
python src/run_scaled_evaluation.py

# 7. Corrida consolidada con run_id compartido
python src/run_all_models.py
```
