# Quickstart: Ejecutar el Código desde Cero

Este tutorial te guía paso a paso para ejecutar todo el pipeline del proyecto **Repeat-Aware Recommendation** en tu máquina local.

## Requisitos Previos

- **Python 3.10+** instalado
- **Git** instalado
- ~4 GB de RAM disponibles
- ~500 MB de espacio en disco

## Paso 1: Clonar el Repositorio

```powershell
git clone https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users.git
cd Analisis-Last.fm-1K-Users
```

## Paso 2: Crear el Entorno Virtual (5 segundos)

```powershell
python -m venv .venv
```

Esto crea una carpeta `.venv/` con un Python aislado.

## Paso 3: Activar el Entorno Virtual

En **Windows PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

*(Si ejecutas desde CMD, usa `.venv\Scripts\activate.bat`)*

Deberías ver el nombre del entorno entre paréntesis en tu prompt:

```
(.venv) PS C:\...\Analisis-Last.fm-1K-Users>
```

## Paso 4: Instalar Dependencias (1-2 minutos)

```powershell
pip install -r requirements.txt
```

Instala: `pandas`, `numpy`, `pyarrow`, `matplotlib`, `scipy`.

## Paso 5: Verificar que el Dataset Existe

Los archivos Parquet (datos procesados) deben estar en `data/`:

```powershell
dir data\*.parquet
```

Deberías ver:
- `lastfm_100_users_h1_fixed.parquet` (4.4 MB, subset de 100 usuarios)
- `lastfm_1k_complete_fixed.parquet` (375 MB, dataset completo)

✅ Si existen, saltate al Paso 7.

## Paso 6: Preprocesar el Dataset Crudo (Opcional, ~1 min)

Si necesitas regenerar los Parquets desde el TSV original:

```powershell
python notebooks/preprocessing.py
```

**Nota:** Requiere que el archivo `userid-timestamp-artid-artname-traid-traname.tsv` esté en `data/`. 
Si no tienes este archivo, descárgalo desde el [repositorio oficial de Last.fm Research](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html).

## Paso 7: Ejecutar los Baselines (15 segundos)

Este paso evalúa 4 modelos recomendadores base bajo LOO temporal.

```powershell
python src/run_baselines.py
```

### Salida esperada:

```
Cargando c:\...\data\lastfm_100_users_h1_fixed.parquet
  200,000 filas | 100 usuarios | 88,234 items
Split temporal LOO -> train: 199,900 | test: 100

Evaluando Random ...
  recall@10     0.0000
  ndcg@10       0.0000
  mrr           0.0000
  repeat_ratio  0.0090
  hits          0/100  (0.37s)

Evaluando MostPopular ...
  recall@10     0.0000
  ndcg@10       0.0000
  mrr           0.0000
  repeat_ratio  0.0550
  hits          0/100  (0.39s)

Evaluando SimpleRepeat-Freq ...
  recall@10     0.0000
  ndcg@10       0.0000
  mrr           0.0000
  repeat_ratio  0.0550
  hits          0/100  (12.22s)

Evaluando SimpleRepeat-Recency ...
  recall@10     0.0900
  ndcg@10       0.0667
  mrr           0.0597
  repeat_ratio  1.0000
  hits          9/100  (1.0s)

=== Resumen ===
                  model  recall@10   ndcg@10       mrr  repeat_ratio
                 Random     0.0000     0.0000  0.0000         0.0090
             MostPopular     0.0000     0.0000  0.0000         0.0550
       SimpleRepeat-Freq     0.0000     0.0000  0.0000         0.0550
    SimpleRepeat-Recency     0.0900     0.0667  0.0597         1.0000
```

Los resultados se guardan en:
- `data/baselines_results.csv` — tabla resumen
- `data/baselines_hits.csv` — detalle de cada acierto

## Paso 8 (Opcional): Regenerar Figuras del EDA (10 segundos)

Genera gráficos del análisis exploratorio:

```powershell
python notebooks/eda_figures.py
```

Crea 3 PNGs en `docs/figures/` y calcula la correlación de Spearman entre frecuencia histórica y probabilidad de reaparición.

## Paso 9 (Opcional): Comparar T-BPR vs PISA vs RepeatNet

```powershell
python src/run_repeat_advanced.py
```

Genera:
- `data/repeat_advanced_results.csv`
- `data/repeat_advanced_hits.csv`

## Paso 9b (Opcional): Evaluar T-BPR standalone

```powershell
python src/run_tbpr.py
```

Genera:
- `data/tbpr_results.csv`
- `data/tbpr_hits.csv`

## Paso 10 (Recomendado): Ejecutar todo alineado en una sola corrida

Para evitar desalineaciones entre CSVs, corre todos los modelos con un `run_id` compartido:

```powershell
python src/run_all_models.py
```

Genera y actualiza en conjunto:
- `data/baselines_results.csv`
- `data/baselines_hits.csv`
- `data/repeat_advanced_results.csv`
- `data/repeat_advanced_hits.csv`
- `data/tbpr_results.csv`
- `data/tbpr_hits.csv`
- `data/all_models_results.csv`

---

## Interpretación de Resultados

### Las Métricas Explicadas

| Métrica | Rango | Interpretación |
|---------|-------|---|
| **Recall@10** | 0–1 | Fracción de usuarios para los que el modelo acertó el siguiente ítem en top-10 |
| **nDCG@10** | 0–1 | Calidad del ranking: penaliza si el acierto está lejos (eg. rank 10 > rank 1) |
| **MRR** | 0–1 | Recíproco de la posición del primer acierto (MRR=0.5 → acierto en posición 2) |
| **Repeat Ratio** | 0–1 | Proporción de recomendaciones que pertenecen al historial del usuario (sesgo hacia repetir) |
| **Hits** | 0–n_users | Número de usuarios en los que acertó |

### Por Qué SimpleRepeat-Recency Gana

```
Random, MostPopular, SimpleRepeat-Freq → Recall = 0.0000 (no aciertos)
SimpleRepeat-Recency                   → Recall = 0.0900 (9 aciertos / 100 usuarios)
```

**Razón:** En música streaming, los usuarios tienden a repetir canciones *recientemente* escuchadas en ráfagas cortas, no sus canciones más escuchadas ever. 

- **SimpleRepeat-Freq** rankea por "cuántas veces escuché esto" → ordena canciones antiguas/favoritas de larga data.
- **SimpleRepeat-Recency** rankea por "cuándo fue la última vez" → captura la ráfaga temporal.

### ¿Por Qué Repeat Ratio = 1.0?

SimpleRepeat-Recency **siempre** recomienda del historial del usuario (nunca sugiere nuevas canciones). 
Esto es por diseño: es un baseline puramente repeat-aware para validar que la recencia funciona.

El siguiente paso es un **modelo híbrido** que decida: "¿Este usuario quiere repetir o explorar?"

---

## Paso a Paso: Qué Pasa Internamente

### 1. **Carga de datos** (< 1 seg)
```python
df = pd.read_parquet("data/lastfm_100_users_h1_fixed.parquet")
# Resultado: 200,000 filas (reproducciones) de 100 usuarios sobre 88,234 canciones únicas
```

### 2. **Split temporal Leave-One-Out (LOO)**
```python
train_df, test_df = temporal_loo_split(df)
# Toma la ÚLTIMA reproducción cronológica de cada usuario como test (100 filas)
# El resto va a train (199,900 filas)
# Objetivo: predecir qué canción escuchará cada usuario DESPUÉS de todo su historial
```

### 3. **Fit de cada modelo** (entrena una vez)
```python
recommender.fit(train_df)  # Aprende patrones de 199,900 reproducciones
```

### 4. **Recomendación y Evaluación**
```python
for cada usuario en test:
    historial = reproducciones del usuario en train
    cancion_siguiente = test (ground truth)
    top_10 = recommender.recommend(user_id, k=10)
    
    if cancion_siguiente in top_10:
        hits += 1  # Acertamos
```

### 5. **Cálculo de métricas**
```python
recall@10 = hits / len(test_users)              # Fracción de aciertos
ndcg@10   = penalización_por_posición(top_10)  # Qué tan arriba estaba el acierto
mrr       = promedio(1 / rank_del_acierto)     # Posición promedio
repeat_ratio = fracción_del_top_10_en_historial
```

---

## Troubleshooting

### Error: `No such file or directory: '...lastfm_100_users_h1_fixed.parquet'`
→ El subset no existe. Ejecuta Paso 6 (preprocesamiento) primero.

### Error: `ModuleNotFoundError: No module named 'pandas'`
→ El entorno virtual no está activado. Ejecuta Paso 3 y luego Paso 4 de nuevo.

### Ejecución lenta en `SimpleRepeat-Freq`
→ Es normal: iteramos sobre 100 usuarios × 88,234 items = millones de comparaciones. Toma ~12 seg.

### ¿Puedo correr esto en Google Colab?
→ Sí, salta los Pasos 2–3 y ejecuta en las primeras celdas:
```python
!git clone https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users.git
%cd Analisis-Last.fm-1K-Users
!pip install -r requirements.txt
!python src/run_baselines.py
```

---

## Próximos Pasos

Una vez que tengas los resultados, el proyecto sugiere:

1. **Analizar hits vs misses por modelo:**
  - ¿Qué patrones aparecen en usuarios con aciertos de Recency?
  - ¿Dónde gana o pierde TemporalBPR frente a PISA y RepeatNet?

2. **Revisar consistencia entre corridas:**
  - Usar `run_id` para comparar resultados entre archivos CSV.
  - Validar que `all_models_results.csv` coincida con los agregados individuales.

3. **Extender el pipeline experimental:**
  - Implementar y evaluar el modelo híbrido Repeat/Explore.
  - Replicar protocolo en un segundo dominio (Amazon Grocery).

---

## Archivos Clave en el Proyecto

| Archivo | Propósito |
|---------|-----------|
| `src/run_baselines.py` | Runner principal (lo que ejecutaste) |
| `src/models/baselines.py` | Implementación de los 4 modelos |
| `src/evaluation.py` | Métricas y split temporal |
| `notebooks/preprocessing.py` | Limpieza del TSV original → Parquet |
| `notebooks/eda_figures.py` | Gráficos exploratorios |
| `data/baselines_results.csv` | Resultados que generaste |
| `docs/H2_Midterm.md` | Informe técnico detallado (formalismo matemático) |

---

## Contacto & Bibliografía

**Equipo:** Pedro Munita · José Racioppi · Tomás Couyoumdjian  
**Curso:** IIC3633 Sistemas Recomendadores (PUC Chile, 2026-1)  
**Papers referenciados:**
- Rendle et al. (2009): *BPR: Bayesian Personalized Ranking*
- Ren et al. (2019): *RepeatNet: Repeat-Aware Neural Recommendation Machine*
- Tran et al. (2024): *Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation* (RecSys)

---

Listo. El pipeline queda ejecutable end-to-end y con salidas alineadas.
