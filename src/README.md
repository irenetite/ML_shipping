# Logística Predictiva: Optimizando la Entrega en E-commerce para Proteger la Reputación de Marca

## Descripción del proyecto

Una empresa internacional de comercio electrónico especializada en productos tecnológicos presenta una tasa de retraso en sus envíos del **59.7%**. Este proyecto aplica técnicas de Machine Learning para predecir si un paquete llegará tarde antes de que salga del almacén, permitiendo al equipo de Atención al Cliente actuar de forma proactiva: enviar un cupón de descuento preventivo antes de que el cliente reclame.

Se desarrollan dos líneas de análisis complementarias:

- **Modelo supervisado** (clasificación binaria): predice si un envío concreto llegará tarde
- **Modelo no supervisado** (clustering): identifica perfiles de envío con distinto nivel de riesgo estructural

---

## Objetivos

- Construir un modelo predictivo que detecte retrasos antes de que ocurran
- Identificar las variables más determinantes del retraso
- Segmentar los envíos en perfiles de riesgo accionables para el negocio
- Demostrar el impacto en negocio con métricas concretas y traducibles a decisiones operativas

---

## Estructura del repositorio

```
ML_SHIPPING/
└── src/
    ├── data/
    │   ├── shipping_data.csv        ← dataset original
    │   ├── train.csv                ← 80% para entrenamiento (8.799 muestras)
    │   └── test.csv                 ← 20% para evaluación (2.200 muestras)
    ├── model/
    │   ├── production/
    │   │   └── random_forest_final.pkl  ← modelo elegido para producción
    │   ├── arbol_decision.pkl
    │   ├── knn.pkl
    │   ├── random_forest.pkl
    │   ├── regresion_logistica.pkl
    │   └── xgboost.pkl
    ├── notebooks/
    │   ├── 01_exploración_y_eda.ipynb
    │   ├── 02_preprocesamiento_y_modelado.ipynb
    │   └── 03_clustering_kmeans.ipynb
    ├── resources/
    │   └── img/                     ← gráficos generados en los notebooks
    ├── utils/
    │   ├── __init__.py
    │   ├── metricas.py              ← función evaluar_modelo()
    │   └── preprocesamiento.py      ← función cargar_y_limpiar()
    ├── memoria.ipynb                ← resumen ejecutivo del proyecto
    └── README.md
```

---

## Dataset

| Campo                   | Detalle                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| **Origen**        | [Kaggle — E-Commerce Shipping Dataset](https://www.kaggle.com/datasets/prachi13/customer-analytics) |
| **Observaciones** | 10.999 registros                                                                                  |
| **Variables**     | 12 (7 numéricas, 4 categóricas, 1 target)                                                       |
| **Target**        | `Reached.on.Time_Y.N` (1=retraso, 0=a tiempo)                                                   |
| **Valores nulos** | Ninguno                                                                                           |

---

## Metodología

### Notebooks

| Notebook                                 | Contenido                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `01_exploración_y_eda.ipynb`          | Carga de datos, análisis estadístico, visualizaciones, hallazgos clave               |
| `02_preprocesamiento_y_modelado.ipynb` | Limpieza, encoding, train/test split, 5 modelos supervisados, evaluación y selección |
| `03_clustering_kmeans.ipynb`           | K-Means, elección del K óptimo, análisis de perfiles de riesgo                      |

### Modelos entrenados

| Modelo                    | Familia               | Recall          | ROC-AUC         |
| ------------------------- | --------------------- | --------------- | --------------- |
| Regresión Logística     | Lineal                | 67.4%           | 0.717           |
| Árbol de Decisión       | Reglas interpretables | 47.8%           | 0.736           |
| **Random Forest**  | Ensamble bagging      | **61.9%** | **0.735** |
| XGBoost                   | Ensamble boosting     | 47.0%           | 0.748           |
| KNN                       | Similitud             | 65.4%           | 0.691           |

---

## Modelo elegido: Random Forest

El **Random Forest** es el modelo con mejor equilibrio entre todas las métricas para nuestro objetivo de negocio.

| Métrica         | Resultado       |
| ---------------- | --------------- |
| Accuracy         | 65.9%           |
| Precision        | 76.4%           |
| **Recall** | **61.9%** |
| F1-Score         | 68.4%           |
| ROC-AUC          | 0.735           |

### Variables más importantes

1. `Weight_in_gms` (28%) — el peso es el factor más determinante
2. `Discount_offered` (23%) — confirmado por el EDA y el clustering
3. `Cost_of_the_Product` (17%) — productos caros implican mayor riesgo

---

## Hallazgo clave

El descuento alto como factor crítico aparece confirmado por **tres metodologías independientes**:

1. **EDA**: con descuento > 10.5%, el 100% de los envíos llegan tarde
2. **Árbol de Decisión**: el descuento es la primera pregunta del árbol (gini=0 en la rama de descuento alto)
3. **K-Means**: sin ver ninguna etiqueta de retraso, agrupa espontáneamente los envíos con descuento alto en un cluster con tasa de retraso del 99.5%

---

## Impacto en negocio

Sobre los 2.200 envíos del conjunto de test:

| Resultado                                       | Número       |
| ----------------------------------------------- | ------------- |
| Retrasos detectados (cupón preventivo enviado) | **813** |
| Retrasos no detectados                          | 500           |
| Falsas alarmas (cupones innecesarios)           | 251           |

El modelo detecta **6 de cada 10 retrasos** antes de que ocurran. Un cupón enviado de más es un coste asumible. Un cliente que reclama sin atención previa es un daño de reputación evitable.

---

## Futuros pasos

- GridSearchCV para optimizar hiperparámetros del Random Forest
- Ajustar el umbral de decisión de XGBoost (50% → 25-30%) y comparar Recall
- Añadir la variable Cluster como feature al modelo supervisado
- Explorar DBSCAN como alternativa al K-Means
- Enriquecer el dataset con datos de tráfico, clima o temporada
- Demo en Streamlit para el equipo de Atención al Cliente

---

## Requisitos

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
```

---

## Cómo ejecutar el proyecto

1. Clona el repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta los notebooks en orden desde `src/notebooks/`
4. El modelo final está disponible en `src/model/production/random_forest_final.pkl`

```python
import pickle

with open('src/model/production/random_forest_final.pkl', 'rb') as f:
    modelo = pickle.load(f)

prediccion = modelo.predict([[...]])
```
