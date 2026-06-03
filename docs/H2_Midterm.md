# H2 — Informe Intermedio
## Recomendación de Ítems Repetidos (Repeat-Aware Recommendation)

**Grupo 32 · IIC3633 Sistemas Recomendadores 2026-1**
Pedro Munita · José Racioppi · Tomás Couyoumdjian
Pontificia Universidad Católica de Chile · 5 de junio de 2026

> **Código y artefactos reproducibles:** <https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users>
> El pipeline de preprocesamiento, los scripts de baselines y los resultados intermedios se encuentran versionados en el repositorio. Las figuras del análisis descriptivo de H1 se regeneran ejecutando `notebooks/preprocessing.py` seguido del notebook de EDA. Se busca evitar la inclusión de capturas de pantalla de código, en línea con la retroalimentación recibida en H1.

---

## 1. Progreso en el Desarrollo de la Solución

### 1.1 Infraestructura de datos y reproducibilidad

Como paso previo a la fase de modelado, se reescribió el pipeline de preprocesamiento del dataset Last.fm 1K Users con el objetivo de hacerlo ejecutable de forma autónoma y eficiente en memoria. El TSV original (2.5 GB, 19 098 853 filas) saturaba la RAM en máquinas con menos de 16 GB. La nueva implementación (`notebooks/preprocessing.py`) opera en dos pasadas: (i) lectura *chunked* del TSV con `pandas`, descartando los campos `artist_id` y `track_id` afectados por *NaNs* de MusicBrainz, y escritura incremental de un archivo Parquet comprimido con `pyarrow`; (ii) lectura selectiva (*predicate pushdown*) del Parquet completo para construir el subset H1 de los 100 usuarios más activos. El tiempo total de ejecución sobre los 19 M de eventos es ≈ 67 s y el *peak* de memoria se mantiene por debajo de 1 GB.

Para subsanar la ausencia de identificadores MusicBrainz —presentes en más de 69 000 artistas— se construye una clave única de ítem como `item_id = "<artist_name> - <track_name>"`. Esta convención evita propagación de *NaNs* en los *embeddings* posteriores y se documenta explícitamente en el repositorio para garantizar la replicabilidad solicitada por la pauta.

### 1.2 Temporal BPR: muestreo negativo dependiente del tiempo

El método de referencia para nuestra adaptación es **Bayesian Personalized Ranking (BPR)** [Rendle et al., 2009], cuya función objetivo, para una interacción positiva $(u, i)$ y un ítem negativo $j \notin H_u$, es:

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \ln \sigma\!\left(\hat{x}_{ui} - \hat{x}_{uj}\right) + \lambda \lVert \Theta \rVert^2,
$$

donde $\hat{x}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$ es el *score* del par usuario-ítem, $\sigma$ la sigmoide y $\Theta$ los parámetros del modelo. En el BPR estándar, $j$ se muestrea **uniformemente** desde el complemento del historial $H_u$.

Nuestra adaptación, denominada **T-BPR**, reemplaza ese muestreo uniforme por una distribución dependiente del tiempo. Sea $t$ el instante de la interacción positiva $(u, i, t)$ y $\Delta_{uj}(t) = t - t_{uj}^{\text{last}}$ el intervalo desde la última vez que $u$ consumió $j$ (definido como $+\infty$ si $j \notin H_u$). Definimos una *ventana óptima de repetición* específica por usuario $W_u = (a_u, b_u)$, calibrada a partir de los percentiles 25 y 75 de la distribución empírica de intervalos entre repeticiones de $u$. La probabilidad de muestrear $j$ como negativo se modula entonces como:

$$
p(j \mid u, t) \;\propto\; \begin{cases}
\;\alpha & \text{si } \Delta_{uj}(t) \in W_u \\
\;1     & \text{si } j \notin H_u \\
\;\beta & \text{en otro caso,}
\end{cases}
\quad \text{con } 0 < \alpha < 1 < \beta.
$$

La intuición es directa: si un ítem ya consumido cae dentro de la ventana en que el usuario suele repetir, **no debe penalizarse** como si fuera negativo "duro" (de ahí $\alpha < 1$); en cambio, ítems repetidos fuera de su ventana habitual se penalizan con peso $\beta > 1$, empujando al modelo a aprender el patrón temporal de repetición. El BPR estándar se recupera con $\alpha = \beta = 1$.

### 1.3 Modelo Híbrido: ruteo Repeat / Explore

El segundo método propuesto es un **clasificador de segundo nivel** que decide, para cada par $(u, t+1)$, si emplear el sub-recomendador *repeat-aware* (`SimpleRepeat`) o el sub-recomendador *explore* (filtrado colaborativo, e.g. iALS). El clasificador es deliberadamente liviano —regresión logística o un árbol de decisión de profundidad acotada— para preservar interpretabilidad y permitir el análisis de variables.

Para cada interacción potencial se construye un vector de *features* $\mathbf{f}_{u,t} \in \mathbb{R}^4$:

* $f_1 = r_u = \dfrac{|\{i \in H_u : c_{ui} > 1\}|}{|H_u|}$ : tasa de repetición histórica del usuario.
* $f_2 = \log\!\left(1 + \tau_{u,t}\right)$ con $\tau_{u,t}$ el tiempo (en horas) desde la última interacción registrada de $u$.
* $f_3 = \log\!\left(1 + \overline{\Delta}_u\right)$ : log-intervalo medio entre repeticiones del usuario.
* $f_4 = |H_u^{\text{30d}}|$ : número de interacciones en la ventana móvil de 30 días.

El clasificador estima $P(\text{modo} = \text{repeat} \mid \mathbf{f}_{u,t})$; si supera un umbral $\theta$ (ajustado por validación cruzada para optimizar nDCG@10), se delega en `SimpleRepeat`, en caso contrario en el modelo colaborativo. La salida final es una lista top-K formada por la unión de ambos sub-rankings, ponderada por la probabilidad estimada. Este diseño está inspirado en arquitecturas de *gating* utilizadas en sistemas de recomendación sensibles al contexto.

### 1.4 Baseline moderno repeat-aware: PISA (RecSys 2024)

Por sugerencia explícita del corrector, se incorporará como baseline avanzado **PISA**, presentado por Tran et al. en RecSys 2024 [1]. A diferencia de RepeatNet, que combina un decodificador secuencial GRU con un mecanismo de copia, PISA articula **Transformers** con principios de la teoría cognitiva **ACT-R** (Adaptive Control of Thought—Rational): la *activación* de un ítem en la memoria del usuario decae con el tiempo y se refuerza con cada nueva exposición, mecánica que el trabajo acopla a la representación atencional del Transformer. Según la descripción provista por los autores [1], esta arquitectura busca capturar simultáneamente patrones secuenciales y la huella temporal de repetición, equilibrando recomendaciones repeat-aware con exploración para mitigar el sesgo conjunto de popularidad y repetición.

Su inclusión nos permitirá situar el desempeño de T-BPR y del modelo híbrido frente a una arquitectura publicada en una venue del estado del arte (RecSys 2024), satisfaciendo el requisito de incorporar un *baseline* repeat-aware moderno. La implementación de referencia provista por los autores reduce el riesgo de bugs de reimplementación.

### 1.5 RepeatNet vía RecBole

RepeatNet [Ren et al., 2019] se mantiene como punto de comparación clásico, pero —siguiendo la sugerencia del corrector— se utilizará la implementación de la biblioteca **RecBole**, evitando los riesgos de una reimplementación desde cero y liberando esfuerzo de ingeniería para el modelo híbrido propio.

---

## 2. Definiciones Formales de Métricas

Sea $\mathcal{U}$ el conjunto de usuarios y $\mathcal{I}$ el catálogo de ítems. Para cada $u$, $H_u \subset \mathcal{I}$ es su historial de entrenamiento e $i_u^{\star}$ es el único ítem reservado como ground-truth bajo el **protocolo temporal leave-one-out** (la última interacción cronológica de $u$). Denotamos por $L_u^K$ la lista ordenada de $K=10$ ítems recomendados a $u$, y por $\text{rank}_u(i)$ la posición de $i$ en $L_u^K$ (con $\text{rank}_u(i) = +\infty$ si $i \notin L_u^K$).

**Recall@K.** Bajo LOO temporal $|R_u| = 1$, por lo que:

$$
\text{Recall@K}_u = \mathbb{1}\!\left[i_u^{\star} \in L_u^K\right], \qquad \text{Recall@K} = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \text{Recall@K}_u.
$$

Equivalente a HitRate@K: simplemente indica si el ítem relevante apareció en algún lugar de los top-K.

**nDCG@K.** Para un único ítem relevante:

$$
\text{nDCG@K}_u = \frac{1}{\log_2(\text{rank}_u(i_u^{\star}) + 1)} \cdot \mathbb{1}\!\left[i_u^{\star} \in L_u^K\right].
$$

A diferencia de Recall@K, **nDCG@K es sensible a la posición**: un acierto en el puesto 1 vale 1.00, en el puesto 3 vale 0.50, en el 10 vale 0.29. Esta sensibilidad es crítica en el escenario *repeat-aware*: no basta con que el ítem repetido aparezca, sino que debe quedar en los primeros lugares para impactar efectivamente la experiencia del usuario (en streaming, un ítem en la posición 10 rara vez es reproducido).

**MRR.** Media de los recíprocos del rango del primer (y único) ítem relevante:

$$
\text{MRR} = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u(i_u^{\star})}.
$$

**Repeat Ratio.** Mide el sesgo del recomendador hacia ítems ya consumidos por el usuario:

$$
\text{RepeatRatio}(L_u^K) = \frac{|\{i \in L_u^K : i \in H_u\}|}{K}.
$$

Es una métrica **diagnóstica**, no de calidad: un Repeat Ratio cercano a 0 indica que el modelo sólo recomienda exploración (RepeatNet, BPR estándar); cercano a 1 indica que sólo recomienda repetición (SimpleRepeat). Un modelo Repeat-Aware ideal debería **calibrar** su Repeat Ratio promedio al de la distribución empírica del dataset (60.43 % en Last.fm 1K, ver §3).

---

## 3. Análisis Descriptivo Avanzado y Discusión de Patrones Temporales

### 3.1 Tasa global de repetición

El análisis sobre el subset H1 (100 usuarios × 200 000 interacciones) arroja una tasa global de repetición del **60.43 %**: aproximadamente 6 de cada 10 reproducciones del dataset corresponden a un *track* ya escuchado previamente por el mismo usuario. Esta cifra cuestiona directamente el supuesto fundacional del filtrado colaborativo clásico —"ítems consumidos = preferencias agotadas"— y cuantifica el costo de oportunidad de modelos que penalizan repeticiones: están descartando como ruido más de la mitad de la señal observada. En términos de negocio para una plataforma de streaming musical, esto implica que un sistema que sugiera activamente el ítem correcto a re-escuchar puede capturar una fracción significativa de las reproducciones que hoy se generan por búsqueda manual del usuario.

> **Observación metodológica.** La tasa del 60.43 % es global y se calcula sobre el conjunto completo de interacciones. Cuando se aísla específicamente la **última interacción** de cada usuario —el ítem que sirve de *ground truth* bajo LOO temporal—, la tasa de repetición desciende al **33 %** (33/100 usuarios, ver §4.3). Esta brecha entre repeat-rate global y repeat-rate de la última interacción es metodológicamente relevante: indica que los eventos terminales del historial tienden a ser más exploratorios que el promedio y que el problema repeat-aware no se resuelve trivialmente decidiendo "repetir" cuando la tasa global es alta.

### 3.2 Distribución de intervalos entre repeticiones

Sobre las 87 985 instancias de repetición identificadas en el subset, los intervalos temporales entre repeticiones sucesivas (en horas) presentan una distribución fuertemente asimétrica:

| Estadístico | Valor (h) |
|---|---|
| Mediana ($p_{50}$) | 27.04 |
| $p_{25}$ | 1.74 |
| $p_{75}$ | 160.67 |
| Media | 175.05 |
| Máximo | 13 503.65 |

El **50 % de las repeticiones ocurre dentro de las primeras 27 horas** y el **25 % ocurre en menos de 1.74 horas**. Esta concentración de masa en intervalos cortos evidencia un fenómeno de **localidad temporal** —el usuario tiende a re-escuchar dentro de la misma sesión o del mismo día— característico de las "ráfagas" de consumo musical. Simultáneamente, el cuartil superior (> 160 h) y el máximo (562 días) revelan una **cola larga** de ítems "favoritos persistentes" que reaparecen meses después, comportamiento típico de canciones de alta valencia emocional o asociadas a contextos específicos.

Esta dualidad —masa concentrada a corto plazo + cola larga— es la justificación empírica directa de la decisión de diseño tomada en §1.2: la ventana óptima de repetición $W_u$ del esquema de muestreo negativo de T-BPR **no puede ser una constante global** (ej. 24 h fijas), porque ignoraría tanto la heterogeneidad entre usuarios (§3.3) como la coexistencia de los dos modos. La parametrización por percentiles individuales $(a_u, b_u) = (p_{25}(u), p_{75}(u))$ adapta dinámicamente la ventana a cada perfil temporal.

### 3.3 Heterogeneidad inter-usuario

La distribución de la tasa de repetición individual sobre los 100 usuarios del subset (media $0.604$, std $0.207$, mínimo $0.032$, máximo $0.982$) confirma que el balance repeat / explore es **fuertemente heterogéneo**. Existen usuarios "exploradores" (3 % de repeticiones) coexistiendo con usuarios "loopers" (98 % de repeticiones). Ningún modelo de política única —ni un colaborativo puro, ni un SimpleRepeat puro— podría operar eficientemente sobre esta variabilidad. Este hallazgo es, a su vez, la justificación empírica del **modelo híbrido con ruteo aprendido** propuesto en §1.3: la decisión repeat vs. explore debe condicionarse en *features* del usuario, no asumirse global.

---

## 4. Experimentación Preliminar y Resultados Intermedios

### 4.1 Setup experimental

A la fecha de este informe se han implementado, entrenado y evaluado **cuatro baselines** sobre el subset H1 (200 000 interacciones, 100 usuarios, 88 234 ítems únicos) **bajo el protocolo temporal leave-one-out** definido en §2. El pipeline completo está versionado en el repositorio bajo `src/`:

* `src/models/baselines.py` — implementaciones con interfaz común `fit` / `recommend`.
* `src/evaluation.py` — split LOO temporal y agregación de métricas.
* `src/run_baselines.py` — *runner* reproducible que genera `data/baselines_results.csv`.

El split produce `train` = 199 900 interacciones, `test` = 100 (una por usuario). El recomendador propone una lista top-10 sobre el catálogo completo, **sin excluir el historial del usuario** (a diferencia del setup exploratorio de H1, donde los ítems del historial se filtraban del espacio de candidatos: ese filtro es incompatible con el paradigma repeat-aware porque penaliza por construcción cualquier modelo que sugiera repeticiones).

A los tres baselines comprometidos en H1 se añade una cuarta variante, **`SimpleRepeat-Recency`**, que rankea el historial por orden cronológico inverso (último ítem único primero) en lugar de por frecuencia. Esta variante se incorpora para verificar empíricamente la hipótesis de localidad temporal levantada en §3.2.

### 4.2 Resultados bajo LOO temporal

| Modelo | Recall@10 | nDCG@10 | MRR | Repeat Ratio | Hits |
|---|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.009 | 0/100 |
| MostPopular | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| SimpleRepeat-Freq | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| **SimpleRepeat-Recency** | **0.0900** | **0.0667** | **0.0597** | 1.000 | **9/100** |

### 4.3 Lectura crítica

**Hallazgo 1 — La última interacción no es la pieza más frecuente.** Análisis directo del split revela que sólo el **33 %** de los ítems de test (33/100) aparece en el historial de entrenamiento de su usuario, sustancialmente menor al 60.43 % de repetición global reportado en §3.1. Esta brecha se explica porque el último evento cronológico tiende a ser más exploratorio que la media: el repeat consumption se concentra en los tramos densos del historial, no necesariamente en su cierre. De los 33 casos repetidos, el ítem ocupa la **posición mediana 221** en el ranking por frecuencia del historial del usuario (máximo 1 563, mínimo 8). Sólo uno de ellos cae en el top-10 por frecuencia. Esto explica de forma directa por qué `SimpleRepeat-Freq` reporta 0 hits: la métrica más obvia de "ítem favorito" (frecuencia acumulada) **no es informativa para predecir el siguiente consumo**.

**Hallazgo 2 — La recencia sí captura el siguiente consumo.** Reemplazando el ranking por frecuencia por un ranking por recencia, `SimpleRepeat-Recency` alcanza Recall@10 = **0.09** y nDCG@10 = **0.067**. Más relevante aún: de los 9 hits, **5 caen en el rango 1** (detalle por usuario disponible en `data/baselines_hits.csv`), es decir, en cinco de cada cien usuarios la siguiente canción es exactamente la última que escuchó. Este resultado constituye la **confirmación experimental** de la hipótesis de localidad temporal articulada en §3.2 a partir de la distribución de intervalos: el 25 % de las repeticiones ocurre dentro de 1.74 h, y este patrón aflora con fuerza cuando se rankea por recencia.

**Hallazgo 3 — El Repeat Ratio expone el sesgo de cada modelo.** Los Repeat Ratios reportados son una métrica diagnóstica útil: Random arroja 0.9 % (los 10 ítems aleatorios casi nunca pertenecen al historial), `MostPopular` y `SimpleRepeat-Freq` arrojan 5.5 % (coincidencia incidental entre los más populares y los historiales más activos), y `SimpleRepeat-Recency` arroja 100 % (por construcción, sólo recomienda historial). Esto cuantifica el espectro completo de comportamiento *explore-only ↔ repeat-only* y dibuja el espacio en el que los modelos avanzados (§1) deben operar: un Repeat Ratio cercano al 33 % empírico sería el objetivo de calibración para la siguiente etapa.

**Hallazgo 4 — Los baselines de exploración (Random, MostPopular) no son competitivos.** Ambos puntúan 0 hits. Con un catálogo de ~88 K ítems y K = 10, la probabilidad a priori de un acierto bajo Random es ~1.1 × 10⁻⁴ por usuario; obtener 0 hits sobre 100 usuarios es coherente. MostPopular sufre porque los ítems globalmente más reproducidos son de un puñado de artistas mainstream, sesgo que se diluye en un dataset de "power users" con catálogos muy personalizados. Esta observación motiva la inclusión de un filtrado colaborativo personalizado (iALS) en el sub-modelo Explore del híbrido propuesto en §1.3.

### 4.4 Compromiso para la entrega final

Para H3 se entrenarán y evaluarán los modelos avanzados (T-BPR, Modelo Híbrido, RepeatNet vía RecBole y PISA) sobre los dos datasets (§5.2), reportando las mismas cuatro métricas más intervalos de confianza al 95 % por *bootstrap* sobre usuarios. El nivel de `SimpleRepeat-Recency` (Recall@10 = 0.09) constituye el piso a superar para que el aporte de los modelos avanzados sea metodológicamente significativo.

---

## 5. Problemas Identificados y Revisión del Plan

### 5.1 Consolidación del pipeline de evaluación

En la etapa exploratoria de H1, los baselines se evaluaron con un split aleatorio 80/20 por usuario y se filtraban los ítems del historial del catálogo de candidatos. Ambas decisiones fueron incompatibles con el paradigma repeat-aware: el split aleatorio diluye el orden temporal y el filtro de historial penaliza por construcción a los modelos que sugieren repeticiones (la razón por la cual `SimpleRepeat` puntuaba 0 en H1). Estas dos inconsistencias se resolvieron en esta etapa migrando la rutina de evaluación a un módulo independiente (`src/evaluation.py`), reproducible y compartido por todos los modelos del proyecto. La diferencia operativa explica la ausencia de comparación directa con los números reportados en H1.

### 5.2 Promoción de Amazon Reviews — Grocery & Gourmet Food a track paralelo

En H1, el dataset de Amazon Grocery fue declarado como **alternativa de respaldo**. Atendiendo a la observación del corrector y reconociendo que el aporte metodológico principal del proyecto reside en el **contraste de dominios**, se promueve a track experimental paralelo:

| Dataset | Régimen temporal dominante | Mediana intervalo repetición |
|---|---|---|
| Last.fm 1K | Alta frecuencia, ráfagas cortas | ≈ 27 h (verificado, §3.2) |
| Amazon Grocery | Baja frecuencia, recompra estacional | semanas a meses (esperado) |

El objetivo es demostrar que los modelos repeat-aware propuestos (T-BPR, Híbrido, PISA) generalizan a ambos extremos del espectro temporal de repetición, no únicamente al caso musical en el que se diseñaron. Music4All, sugerido por el corrector, queda como dataset de validación opcional si los tiempos lo permiten.

### 5.3 Cuello de botella computacional y reproducibilidad

La carga del TSV original de 2.5 GB en pandas saturaba la memoria RAM en máquinas con menos de 16 GB, impidiendo iteración rápida durante el EDA. Se diagnosticó como un problema de representación: (i) inferencia de dtypes `object` para columnas string sin necesidad, (ii) carga de columnas innecesarias (`artist_id`, `track_id`). La solución implementada combina lectura *chunked* en dos pasadas con escritura incremental a Parquet (`pyarrow`), reduciendo el *peak* de memoria a < 1 GB y el tiempo de carga posterior (Parquet vs. TSV) en aproximadamente dos órdenes de magnitud. El pipeline completo, junto con `requirements.txt` y el `README`, se encuentra modularizado y versionado en el repositorio enlazado.

### 5.4 Cronograma actualizado hacia H3

| Sem | Fechas | Actividad | Responsable |
|---|---|---|---|
| 1 | 06–12 Jun | Replicación del pipeline de preprocesamiento y evaluación sobre Amazon Grocery; agregar intervalos de confianza vía *bootstrap* | T. Couyoumdjian |
| 2 | 13–19 Jun | Implementación de T-BPR (extensión sobre `implicit`); experimentos de sensibilidad a $\alpha, \beta$ y a la parametrización de $W_u$ | J. Racioppi |
| 3 | 20–26 Jun | Modelo Híbrido: entrenamiento del clasificador de ruteo, integración con iALS y `SimpleRepeat-Recency`; ablación de *features* | P. Munita |
| 4 | 27 Jun – 02 Jul | Ejecución de RepeatNet (RecBole) y PISA sobre ambos datasets; tabla comparativa final con intervalos de confianza al 95 % | Todos |
| 5 | 03 – 06 Jul | Redacción del paper H3 (formato ACM, máx. 8 páginas); preparación del póster para la sesión presencial | Todos |

### 5.5 Riesgos asumidos

* **PISA y RepeatNet** son los componentes con mayor incertidumbre de implementación. El uso de RecBole para RepeatNet y de la implementación de referencia de Deezer para PISA mitiga este riesgo, pero ambos requieren adaptación al pipeline de evaluación propio.
* **Amazon Grocery** requiere replicar todo el pipeline de preprocesamiento. Se prevé reutilizar la arquitectura *chunked* + Parquet documentada en §5.3 para acelerar esta etapa.

---

## 6. Bibliografía

[1] Tran, V.-A., Salha-Galvan, G., Sguerra, B., & Hennequin, R. (2024). *Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation.* In **Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)**. Código disponible en: <https://github.com/deezer/recsys24-pisa>

[2] Ren, P., Chen, Z., Li, J., Ren, Z., Ma, J., & de Rijke, M. (2019). *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation.* **Proceedings of the AAAI Conference on Artificial Intelligence**, 33(01), 4806–4813. <https://doi.org/10.1609/aaai.v33i01.33014806>

[3] Benson, A. R., Kumar, R., & Tomkins, A. (2016). *Modeling User Consumption Sequences.* In **Proceedings of the 25th International Conference on World Wide Web (WWW '16)**. <https://www.cs.cornell.edu/~arb/papers/sequences-www2016.pdf>

[4] Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* In **Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI '09)**, 452–461.

[5] Zhao, W. X., et al. (2021). *RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms.* In **CIKM '21**.

[6] Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought.* Lawrence Erlbaum Associates. (Fundamento teórico de ACT-R utilizado por PISA.)
