# Repeat-Aware Recommendation - IIC3633 (Grupo 32)

Este repositorio contiene el código fuente, los scripts de preprocesamiento y el pipeline de evaluación experimental para el proyecto final del curso Sistemas Recomendadores (IIC3633) en la Escuela de Ingeniería de la Pontificia Universidad Católica de Chile.

El proyecto, titulado "Recomendación de Ítems Repetidos (Repeat-Aware Recommendation)", está desarrollado por Pedro Munita, José Racioppi y Tomás Couyoumdjian.

## Descripción del Proyecto

Los sistemas recomendadores convencionales suelen tratar el historial de interacciones como evidencia de preferencias positivas, asumiendo que el interés en un ítem se agota tras su primer consumo. Este proyecto aborda de manera explícita la coexistencia de dos modos de consumo en el comportamiento de los usuarios:

* **Repeat Consumption:** Influenciado por el intervalo temporal transcurrido y la frecuencia histórica de repetición del ítem.
* **Explore Consumption:** El enfoque de filtrado colaborativo clásico orientado al descubrimiento y exploración de nuevos ítems.

El objetivo general es desarrollar un modelo Repeat-Aware que aprenda, a partir del historial temporal de interacciones, cuándo un usuario tiene mayor probabilidad de repetir el consumo de un ítem ya visitado.

---

## Datasets Utilizados

La evaluación empírica de los modelos se realiza bajo un protocolo de contraste en dos dominios con dinámicas temporales y de recompra distintas:

1. **Last.fm 1K Users Dataset:** Contiene aproximadamente 19 millones de eventos de escucha de 992 usuarios. Se caracteriza por una alta tasa de repeat consumption global del 60.43%.
2. **Amazon Reviews - Grocery & Gourmet Food (2018):** Dataset de e-commerce utilizado para evaluar la generalización del modelo en un entorno de recompra periódica y estacional.

---

## Modelos Implementados

El repositorio se estructura para soportar una interfaz común de entrenamiento y recomendación (fit / recommend) bajo un protocolo de evaluación temporal:

### Modelos de Referencia (Baselines)
* **RandomRecommender:** Genera recomendaciones aleatorias sobre el catálogo excluyendo los ítems de entrenamiento.
* **MostPopularRecommender:** Recomienda los ítems globales más reproducidos o adquiridos en el dataset.
* **SimpleRepeatRecommender:** Baseline especializado que prioriza la repetición de los ítems del historial del usuario.

### Modelos Avanzados (Etapa Midterm / Final)
* **RepeatNet:** Arquitectura neuronal basada en sesiones que incorpora un mecanismo de copia sobre el historial para alternar entre los modos de repetición y exploración.
* **Temporal BPR:** Adaptación del algoritmo Bayesian Personalized Ranking mediante un esquema de muestreo negativo basado en ventanas de tiempo.
* **Modelo Híbrido:** Clasificador de segundo nivel acoplado a un recomendador colaborativo para optimizar la transición entre repetición y exploración.

---

## Protocolo Experimental y Métricas

Para cada usuario, el pipeline reserva cronológicamente la última interacción como conjunto de prueba. Los modelos generan un ranking top-K (K=10) evaluado mediante las siguientes métricas:

* **Recall@10:** Mide la presencia del ítem relevante dentro de la lista de recomendación.
* **nDCG@10:** Evalúa la ganancia acumulada descontada normalizada, ponderando la posición del ítem repetido.
* **MRR (Mean Reciprocal Rank):** Pondera el recíproco del rango del primer acierto.
* **Repeat Ratio:** Mide la proporción de recomendaciones emitidas que corresponden a ítems previamente vistos.

---

## Estructura del Repositorio

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
