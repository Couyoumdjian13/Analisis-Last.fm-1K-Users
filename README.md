# [cite_start]Repeat-Aware Recommendation - IIC3633 (Grupo 32) [cite: 5]

[cite_start]Este repositorio contiene el código fuente, los scripts de preprocesamiento y el pipeline de evaluación experimental para el proyecto final del curso **IIC3633 Sistemas Recomendadores (2026-1)** [cite: 5] [cite_start]en la **Pontificia Universidad Católica de Chile**[cite: 2].

[cite_start]El proyecto, titulado **"Recomendación de Ítems Repetidos (Repeat-Aware Recommendation)"** [cite: 109][cite_start], está desarrollado por Pedro Munita, José Racioppi y Tomás Couyoumdjian[cite: 109].

## 📝 Descripción del Proyecto

[cite_start]Los sistemas recomendadores convencionales suelen tratar el historial de interacciones como evidencia de preferencias positivas, asumiendo erróneamente que el interés en un ítem se agota tras su primer consumo[cite: 111]. [cite_start]Este proyecto aborda de manera explícita la coexistencia de dos modos de consumo en el comportamiento de los usuarios[cite: 118]:
* [cite_start]**Repeat Consumption:** Influenciado por el intervalo temporal transcurrido y la frecuencia histórica de repetición del ítem[cite: 116].
* [cite_start]**Explore Consumption:** El enfoque de filtrado colaborativo clásico orientado al descubrimiento y exploración de nuevos ítems[cite: 117].

[cite_start]El objetivo general es desarrollar un modelo Repeat-Aware que aprenda, a partir del historial temporal de interacciones, cuándo un usuario tiene mayor probabilidad de repetir el consumo de un ítem ya visitado[cite: 130].

---

## 📊 Datasets Utilizados

[cite_start]La evaluación empírica de los modelos se realiza bajo un protocolo de contraste en dos dominios con dinámicas temporales y de recompra distintas[cite: 143]:

1. [cite_start]**Last.fm 1K Users Dataset:** Contiene aproximadamente 19 millones de eventos de escucha de 992 usuarios[cite: 137, 138]. [cite_start]Se caracteriza por una alta tasa de *repeat consumption* global del 60.43%[cite: 139, 177].
2. [cite_start]**Amazon Reviews - Grocery & Gourmet Food (2018):** Dataset de e-commerce utilizado para evaluar la generalización del modelo en un entorno de recompra periódica y estacional[cite: 143].

---

## 🛠️ Modelos Implementados

[cite_start]El repositorio se estructura para soportar una interfaz común de entrenamiento y recomendación (`fit` / `recommend`) [cite: 356] bajo un protocolo de evaluación temporal:

### Modelos de Referencia (Baselines)
* [cite_start]**RandomRecommender:** Genera recomendaciones aleatorias sobre el catálogo excluyendo los ítems de entrenamiento[cite: 254, 294].
* [cite_start]**MostPopularRecommender:** Recomienda los ítems globales más reproducidos o adquiridos en el dataset[cite: 255].
* [cite_start]**SimpleRepeatRecommender:** Baseline especializado que prioriza la repetición de los ítems del historial del usuario[cite: 256].

### Modelos Avanzados (Etapa Midterm / Final)
* [cite_start]**RepeatNet:** Arquitectura neuronal basada en sesiones que incorpora un mecanismo de copia sobre el historial para alternar entre los modos de repetición y exploración[cite: 125, 379].
* [cite_start]**Temporal BPR:** Adaptación del algoritmo Bayesian Personalized Ranking mediante un esquema de muestreo negativo basado en ventanas de tiempo[cite: 380, 393].
* [cite_start]**Modelo Híbrido:** Clasificador de segundo nivel acoplado a un recomendador colaborativo para optimizar la transición entre repetición y exploración[cite: 380].

---

## 📐 Protocolo Experimental y Métricas

[cite_start]Para cada usuario, el pipeline reserva cronológicamente la última interacción como conjunto de prueba (*ground truth*)[cite: 374]. [cite_start]Los modelos generan un ranking top-K (K=10) [cite: 375] [cite_start]evaluado mediante las siguientes métricas[cite: 134, 376]:
* [cite_start]**Recall@10:** Mide la presencia del ítem relevante dentro de la lista de recomendación[cite: 376].
* [cite_start]**nDCG@10:** Evalúa la ganancia acumulada descontada normalizada, ponderando la posición del ítem repetido[cite: 376].
* [cite_start]**MRR (Mean Reciprocal Rank):** Pondera el recíproco del rango del primer acierto[cite: 376].
* [cite_start]**Repeat Ratio:** Mide la proporción de recomendaciones emitidas que corresponden a ítems previamente vistos[cite: 376].

---

## 📂 Estructura del Repositorio

```text
├── data/                             # Directorio para los datasets crudos y procesados
├── notebooks/
│   └── preprocessing.ipynb           # Carga optimizada, corrección de IDs y exportación a Parquet
├── src/
│   ├── models/                       # Clases y scripts de entrenamiento de los recomendadores
│   ├── evaluation.py                 # Pipeline del protocolo temporal leave-one-out
│   └── utils.py                      # Funciones auxiliares de cálculo de métricas
├── README.md                         # Documentación principal del repositorio
└── requirements.txt                  # Dependencias del proyecto (pandas, numpy, pyarrow, etc.)
