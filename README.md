# On‑Time AI: Modelos predictivos para reducir retrasos logísticos

## Descripción del proyecto

Una empresa internacional de comercio electrónico especializada en productos tecnológicos presenta una tasa de retraso en sus envíos del **59.7%**. Este proyecto aplica técnicas de Machine Learning para predecir si un paquete llegará tarde **antes de que salga del almacén**, permitiendo al equipo de Atención al Cliente actuar de forma proactiva.

### Estrategia de negocio

Enviar un cupón de descuento **preventivo** al cliente antes de que se produzca el retraso, transformando una experiencia negativa en una oportunidad de retención.

### Enfoque dual

Se desarrollan dos líneas de análisis complementarias:

- **Modelo supervisado**: Clasificación binaria que predice si un envío específico llegará tarde
- **Modelo no supervisado**: Clustering K-means que identifica perfiles de envío con distinto nivel de riesgo estructural

---

## Objetivos

- Construir un modelo predictivo capaz de detectar retrasos antes de que ocurran
- Identificar las variables más determinantes del retraso y su impacto
- Segmentar los envíos en perfiles de riesgo accionables para operaciones
- Cuantificar el impacto en negocio con métricas traducibles a decisiones operativas
- Generar modelos escalables listos para producción

---

## Estructura del repositorio
```
ML_shipping/
├── README.md                        ← Este archivo
├── src/
│   ├── data/
│   │   ├── shipping_data.csv        ← Dataset original (10.999 muestras)
│   │   ├── train.csv                ← 80% entrenamiento (8.799 muestras)
│   │   └── test.csv                 ← 20% evaluación (2.200 muestras)
│   │
│   ├── model/
│   │   ├── production/
│   │   │   └── xgboost_final.pkl    ← Modelo elegido para producción
│   │   ├── arbol_decision.pkl
│   │   ├── knn.pkl
│   │   ├── random_forest.pkl
│   │   ├── regresion_logistica.pkl
│   │   └── xgboost.pkl
│   │
│   ├── notebooks/
│   │   ├── 01_exploración_y_eda.ipynb           ← Análisis exploratorio: "¿qué tenemos?"
│   │   ├── 02_preprocesamiento_y_modelado.ipynb ← Entrenamiento de modelos: "¿podemos predecir retrasos y cómo?"
│   │   └── 03_clustering_kmeans.ipynb           ← Análisis de segmentación: "¿qué grupos o patrones esconde el dataset?"
│   │
│   ├── presentacion/
│   │   ├── on_time_ai.pdf
│   │   └── on_time_ai.pptx
│   │
│   ├── resources/
│   │   └── img/                     ← Visualizaciones y gráficos
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metricas.py              ← Función evaluar_modelo()
│   │   └── preprocesamiento.py      ← Función cargar_y_limpiar()
│   │
│   ├── app.py                       ← Aplicación Streamlit
│   ├── .streamlit/
│   │   └── config.toml              ← Configuración de tema de la app
│   └── memoria.ipynb                ← Resumen ejecutivo del proyecto
```
## Dataset

**Tamaño**: 10,999 muestras por múltiples variables
**Clases**: Binaria (Retraso: Sí/No)
**Desbalance**: Clase positiva (retrasos): 59.7%

### Variables principales

- `Discount_offered`: Descuento aplicado al producto (%)
- `Weight_in_gms`: Peso del paquete en gramos
- `Mode_of_Shipment`: Modo de transporte (Ship, Flight, Road)
- `Warehouse_block`: Bloque del almacén de origen (A, B, C, D, F)

## Preprocesamiento

- **Eliminación**: columna `ID` (identificador sin valor predictivo)
- **Ordinal Encoding**: `Product_importance` → low=1, medium=2, high=3 (preserva jerarquía)
- **One-Hot Encoding**: `Warehouse_block`, `Mode_of_Shipment`, `Gender` con `drop_first=True` (evita multicolinealidad)
- **Conversión de tipos**: columnas booleanas a enteros para compatibilidad con sklearn
- **Escalado**: StandardScaler en Pipeline para Regresión Logística y KNN
- **Split estratificado**: 80% train / 20% test con `stratify=y` y `random_state=42`

---

## Hallazgos del EDA

El análisis exploratorio reveló tres patrones clave confirmados posteriormente por los modelos:

- **Efecto del descuento**: a partir del 15% de descuento, ningún envío llega a tiempo. El umbral crítico está entre el 10% y el 15%. Las campañas promocionales agresivas colapsan la capacidad logística del almacén de forma sistemática.
- **Segmentación por peso**: los paquetes ligeros (1-2 kg) tienen una alta tasa de retraso; los pesados (4-6 kg) un comportamiento más estable. Esta segmentación natural justifica el análisis de clustering.
- **Modo de envío**: el barco (Ship) concentra el mayor volumen de incidencias, tanto en términos absolutos como en proporción de retrasos.

---

## Resultados clave

### Modelo seleccionado: XGBoost

El modelo de **XGBoost** fue elegido para producción por su excepcional **Recall (98%)** — detecta 98 de cada 100 retrasos antes de que ocurran, optimizado mediante **GridSearchCV**:

- **Accuracy**: 59.8%
- **Precision**: 60% — de cada 10 alertas, 6 son retrasos reales
- **Recall**: **98%** ← **Métrica clave** para minimizar falsos negativos
- **F1-Score**: 74.4% — el más alto de todos los modelos
- **ROC-AUC**: 0.756 — mejor capacidad discriminativa

#### Los dos ajustes clave

**Ajuste 1 — scale_pos_weight**: debe ser `positivos/negativos` (≈1.48) y **NO** `negativos/positivos` (≈0.68). Con el ratio incorrecto el Recall era del 47%, con el correcto sube al **91.6%**. Un solo parámetro, 44 puntos de diferencia.

**Ajuste 2 — learning_rate**: optimizado mediante GridSearchCV explorando 8 combinaciones con CV=5. El valor óptimo es **0.05** frente al 0.1 inicial. El Recall sube del 91.6% al **98.0%**. Otro parámetro, 6 puntos más de Recall.

#### Variables más importantes

1. **Weight_in_gms** (≈28%) — el peso es el factor más determinante
2. **Discount_offered** (≈23%) — confirmado por el EDA
3. **Cost_of_the_Product** (≈17%) — productos caros = mayor riesgo

#### Matriz de confusión del modelo final (conjunto test)

|                          | Predicho: A tiempo | Predicho: Retraso |
| ------------------------ | ------------------ | ----------------- |
| **Real: A tiempo** | 28 (TN)            | 859 (FP)          |
| **Real: Retraso**  | 26 (FN)            | 1.287 (TP)       |

## Resultados del clustering K-Means

Se aplicó K-Means sobre las 7 variables numéricas del dataset. El número óptimo de clusters (K=3) fue determinado mediante el Método del Codo, el Índice de Silhouette (score: 0.2386) y el diagrama de Silhouette comparativo.

| Cluster   | Tamaño       | Tasa de retraso | Perfil dominante                          |
| --------- | ------------- | --------------- | ----------------------------------------- |
| Cluster 0 | 2.294 (20.9%) | **99.5%** | Descuento alto (media 40.1%)              |
| Cluster 1 | 6.097 (55.4%) | 47.9%           | Paquete pesado (media 4.801g)             |
| Cluster 2 | 2.608 (23.7%) | 52.3%           | Cliente fidelizado (compras previas: 5.1) |

El Cluster 0 es el hallazgo más impactante: el K-Means agrupa espontáneamente el 20.9% de los envíos en un perfil con una tasa de retraso del 99.5%, definido exclusivamente por el descuento alto. Confirma desde una metodología independiente el hallazgo central del EDA.

---

## Impacto de negocio

- **De cada 100 retrasos reales, detectamos 98** antes de que ocurran
- **1.287 clientes** reciben un cupón preventivo vs. **26 que se escapan**
- Un cupón de más es un coste asumible
- Un cliente que reclama sin atención es un daño de reputación
- Transformación de experiencia negativa en oportunidad de retención

---

## Demo interactiva — Streamlit

🔗 **[**Acceder a la app**](**https://mlshipping-ontimeai.streamlit.app/**)**

La aplicación incluye tres pestañas:

- **Predicción individual**: introduce los datos de un envío y obtén la predicción con probabilidad de retraso y recomendación de acción
- **Carga masiva**: sube un CSV con varios envíos y descarga el resultado enriquecido con predicciones para todos
- **Análisis**: resumen visual del proyecto con los hallazgos del EDA, comparativa de modelos y resultados del clustering

---

## Notas Importantes

- El dataset está balanceado en la split train/test (80/20)
- Se utilizó **Ordinal Encoding** para `Product_importance` (preserva jerarquía: low < medium < high)
- El **Recall es la métrica clave** para minimizar falsos negativos (envíos que no detectamos como retrasados)
- Todos los modelos están serializados en `.pkl` para reutilización

---

## Cómo ejecutar el proyecto

1. Clona el repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta los notebooks en orden desde `src/notebooks/`
4. El modelo final está disponible en `src/model/production/xgboost_final.pkl`
