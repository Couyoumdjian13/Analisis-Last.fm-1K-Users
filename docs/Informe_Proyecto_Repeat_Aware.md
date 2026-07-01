# Informe del Proyecto

## Motivacion

Los sistemas recomendadores tradicionales suelen asumir que un item ya consumido pierde valor para recomendaciones futuras. Esta suposicion no representa bien escenarios reales como streaming musical, donde la repeticion es frecuente y estructural. En Last.fm 1K Users, una fraccion importante de reproducciones corresponde a reconsumo, por lo que modelar explicitamente el comportamiento repeat-aware puede mejorar la calidad de recomendacion respecto de enfoques puramente exploratorios.

## Problema

El problema central es predecir el siguiente item de cada usuario considerando dos modos de consumo que coexisten:

- Repeticion: volver a escuchar items ya consumidos.
- Exploracion: descubrir items no consumidos recientemente.

El desafio es que la senal de repeticion no depende solo de frecuencia historica. La dimension temporal (recencia e intervalos entre consumos) es critica para anticipar el siguiente evento. Por ello, el objetivo es evaluar modelos que integren esta dinamica bajo un protocolo temporal riguroso.

## Propuesta

Se propone comparar un conjunto de modelos repeat-aware y baselines clasicos bajo un mismo protocolo temporal leave-one-out.

Modelos evaluados:

- Baselines: Random, MostPopular, SimpleRepeat-Freq, SimpleRepeat-Recency.
- Avanzados: TemporalBPR, PISA (aproximacion reproducible), RepeatNet (aproximacion reproducible).

La hipotesis principal del proyecto es que modelos sensibles a recencia y dinamica temporal superan a alternativas que solo usan popularidad global o frecuencia acumulada.

## Metodologia

### Datos

- Dataset principal: Last.fm 1K Users.
- Subset experimental H1: 100 usuarios, 200,000 interacciones, 88,234 items unicos.
- Identificador de item: combinacion artista-cancion para evitar perdida de senal por IDs faltantes.

### Protocolo de evaluacion

- Split temporal leave-one-out por usuario.
- Train: 199,900 interacciones.
- Test: 100 interacciones (ultima de cada usuario).
- Top-K: K = 10.

### Metricas

- Recall@10
- nDCG@10
- MRR
- Repeat Ratio (metrica diagnostica de sesgo repeat/explore)
- Hits (aciertos sobre usuarios)

### Reproducibilidad

Scripts principales:

- src/run_baselines.py
- src/run_repeat_advanced.py
- src/run_tbpr.py
- src/run_all_models.py

Archivos de salida:

- data/baselines_results.csv
- data/repeat_advanced_results.csv
- data/tbpr_results.csv

## Resultados

### Baselines (run_id: 20260701T032930Z)

| Modelo | Recall@10 | nDCG@10 | MRR | Repeat Ratio | Hits |
|---|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.009 | 0/100 |
| MostPopular | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| SimpleRepeat-Freq | 0.0000 | 0.0000 | 0.0000 | 0.055 | 0/100 |
| SimpleRepeat-Recency | 0.0900 | 0.0667 | 0.0597 | 1.000 | 9/100 |

### Modelos avanzados (run_id: 20260701T030743Z)

| Modelo | Recall@10 | nDCG@10 | MRR | Repeat Ratio | Hits |
|---|---:|---:|---:|---:|---:|
| TemporalBPR | 0.0700 | 0.0596 | 0.0564 | 0.507 | 7/100 |
| PISA | 0.0900 | 0.0673 | 0.0604 | 0.992 | 9/100 |
| RepeatNet | 0.0800 | 0.0374 | 0.0247 | 0.267 | 8/100 |

## Discusion

1. La recencia aparece como una senal mas util que la frecuencia acumulada para predecir el siguiente consumo en este subset.
2. SimpleRepeat-Recency y PISA logran el mejor Recall@10 (0.09), superando claramente baselines no temporales.
3. TemporalBPR obtiene un equilibrio mas intermedio en Repeat Ratio (0.507), lo que sugiere una mezcla repeat/explore mas calibrada que modelos casi puramente repeat.
4. Repeat Ratio ayuda a interpretar el comportamiento del modelo: no indica por si solo calidad, pero permite entender si la estrategia esta sesgada hacia repetir o explorar.
5. El desempeno global aun deja espacio de mejora (recalls bajos en terminos absolutos), lo que valida seguir con ajuste de hiperparametros, calibracion de mezcla y evaluacion cruzada en dominios adicionales.

## Conclusiones

- El proyecto confirma que el fenomeno de repeticion debe modelarse explicitamente en recomendacion musical.
- Modelos con sensibilidad temporal (recencia y dinamica de repeticion) entregan mejores resultados que enfoques clasicos de popularidad o frecuencia historica aislada.
- En el estado actual, SimpleRepeat-Recency y PISA obtienen el mejor Recall@10 (0.09), mientras TemporalBPR muestra una alternativa con mejor balance repeat/explore.
- Como trabajo futuro inmediato, se recomienda consolidar una corrida unica con run_id compartido para todos los modelos y extender la validacion a un segundo dominio de datos para medir robustez.

## Referencias

1. Rendle, S., Freudenthaler, C., Gantner, Z., y Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI.
2. Ren, K., Qin, J., Zheng, L., Yang, Z., Zhang, W., Qiu, M., y Yu, Y. (2019). RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation. AAAI.
3. Tran, T., et al. (2024). PISA: Repeat-Aware Session Recommendation via ACT-R and Sequential Modeling. RecSys 2024.
4. Celma, O. (2010). Music Recommendation and Discovery in the Long Tail. Springer. (Dataset Last.fm 1K Users, material asociado).
5. Proyecto del curso IIC3633 Grupo 32: Repeat-Aware Recommendation. Repositorio: https://github.com/Couyoumdjian13/Analisis-Last.fm-1K-Users