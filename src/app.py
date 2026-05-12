import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="On-Time AI · Predictor de Retrasos",
    page_icon="📦",
    layout="wide"
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px !important;
}

label, .stSelectbox label, .stNumberInput label,
.stSlider label, p, li, td, th {
    font-size: 1.05rem !important;
}

.main { background-color: #f7f6f2; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.header-block {
    background: #1a6b6b;
    color: white;
    padding: 3rem 3.5rem;
    border-radius: 16px;
    margin-bottom: 2.5rem;
}

.header-block h1 {
    font-size: 3rem;
    margin: 0 0 0.5rem 0;
    color: white;
}

.header-block p {
    color: #ffffff;
    margin: 0;
    font-size: 1.15rem;
}

.result-card {
    padding: 2.5rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-retraso {
    background: #fff1f1;
    border: 2px solid #f46666;
}

.result-puntual {
    background: #f1fff4;
    border: 2px solid #60e368;
}

.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.6rem;
}

.prob-bar-container {
    background: #e8e8e8;
    border-radius: 50px;
    height: 18px;
    width: 100%;
    margin: 1.2rem 0;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.5s ease;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #1a1a2e;
    margin: 2.5rem 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #1a1a2e;
}

.info-box {
    background: #c5e8e8;
    border-left: 4px solid #1a1a2e;
    padding: 1.2rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1.2rem 0;
    font-size: 1.05rem;
    color: #333;
    line-height: 1.6;
}

.stButton > button {
    background: #1a1a2e !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.8rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100%;
    font-size: 1.1rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.batch-retraso { color: #e05050; font-weight: 600; }
.batch-puntual { color: #3ab54a; font-weight: 600; }

.stTabs [data-baseweb="tab"] {
    font-size: 1.05rem !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 700 !important;
    color: #0d2b2b !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #0d2b2b !important;
}

.stTabs [aria-selected="false"] {
    color: #0d2b2b !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #0d3d3d !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #0d3d3d !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: #0d3d3d !important;
}
div[data-testid="stThumbValue"] {
    color: #0d3d3d !important;
}
.stSlider > div > div > div > div {
    background: #0d3d3d !important;
}
.stImage figcaption, [data-testid="caption"],
div[data-testid="stCaptionContainer"] p,
small, .caption {
    color: #000000 !important;
    font-weight: 600 !important;
}
.stDataFrame td, .stDataFrame th {
    color: #0d2b2b !important;
    font-weight: 500 !important;
}
footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    rutas = [
        "src/model/production/xgboost_final.pkl",
        "model/production/xgboost_final.pkl",
        "../model/production/xgboost_final.pkl",
        "xgboost_final.pkl",
        "/mount/src/ml_shipping/src/model/production/xgboost_final.pkl",
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return pickle.load(f)
    return None

modelo = cargar_modelo()

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
    'Prior_purchases', 'Product_importance', 'Discount_offered',
    'Weight_in_gms', 'Warehouse_block_B', 'Warehouse_block_C',
    'Warehouse_block_D', 'Warehouse_block_F', 'Mode_of_Shipment_Road',
    'Mode_of_Shipment_Ship', 'Gender_M'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE_DIR, "resources/img")

def preparar_input(datos):
    """Convierte los datos del formulario en el vector de features."""
    fila = {f: 0 for f in FEATURES}
    fila['Customer_care_calls'] = datos['Customer_care_calls']
    fila['Customer_rating']     = datos['Customer_rating']
    fila['Cost_of_the_Product'] = datos['Cost_of_the_Product']
    fila['Prior_purchases']     = datos['Prior_purchases']
    fila['Discount_offered']    = datos['Discount_offered']
    fila['Weight_in_gms']       = datos['Weight_in_gms']

    imp_map = {'Baja': 1, 'Media': 2, 'Alta': 3}
    fila['Product_importance'] = imp_map[datos['Product_importance']]

    bloque = datos['Warehouse_block']
    if bloque in ['B', 'C', 'D', 'F']:
        fila[f'Warehouse_block_{bloque}'] = 1

    modo = datos['Mode_of_Shipment']
    if modo == 'Road':
        fila['Mode_of_Shipment_Road'] = 1
    elif modo == 'Ship':
        fila['Mode_of_Shipment_Ship'] = 1

    if datos['Gender'] == 'Hombre':
        fila['Gender_M'] = 1

    return pd.DataFrame([fila])[FEATURES]


def preparar_csv(df_raw):
    """Preprocesa un CSV subido por el usuario."""
    df = pd.DataFrame()
    df['Customer_care_calls'] = df_raw['Customer_care_calls']
    df['Customer_rating']     = df_raw['Customer_rating']
    df['Cost_of_the_Product'] = df_raw['Cost_of_the_Product']
    df['Prior_purchases']     = df_raw['Prior_purchases']
    df['Discount_offered']    = df_raw['Discount_offered']
    df['Weight_in_gms']       = df_raw['Weight_in_gms']

    imp_map = {'low': 1, 'medium': 2, 'high': 3}
    df['Product_importance'] = df_raw['Product_importance'].map(imp_map)

    for col in ['Warehouse_block_B', 'Warehouse_block_C',
                'Warehouse_block_D', 'Warehouse_block_F',
                'Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship', 'Gender_M']:
        df[col] = 0

    if 'Warehouse_block' in df_raw.columns:
        for bloque in ['B', 'C', 'D', 'F']:
            df[f'Warehouse_block_{bloque}'] = (df_raw['Warehouse_block'] == bloque).astype(int)

    if 'Mode_of_Shipment' in df_raw.columns:
        df['Mode_of_Shipment_Road'] = (df_raw['Mode_of_Shipment'] == 'Road').astype(int)
        df['Mode_of_Shipment_Ship'] = (df_raw['Mode_of_Shipment'] == 'Ship').astype(int)

    if 'Gender' in df_raw.columns:
        df['Gender_M'] = (df_raw['Gender'] == 'M').astype(int)

    return df[FEATURES]


def mostrar_resultado(prob_retraso):
    """Renderiza la tarjeta de resultado."""
    pred = prob_retraso >= 0.50
    color_barra = "#f46666" if pred else "#60e368"
    clase_card  = "result-retraso" if pred else "result-puntual"
    emoji       = "⚠️" if pred else "✅"
    titulo      = "Riesgo de retraso detectado" if pred else "Envío probablemente puntual"
    subtitulo   = (
        "Se recomienda enviar un cupón preventivo al cliente antes de que reclame."
        if pred else
        "No se anticipa retraso. No se requiere acción proactiva."
    )
    pct = int(prob_retraso * 100)

    st.markdown(f"""
    <div class="result-card {clase_card}">
        <div class="result-title">{emoji} {titulo}</div>
        <p style="color:#555; margin-bottom:0.5rem">{subtitulo}</p>
        <div class="prob-bar-container">
            <div class="prob-bar-fill" style="width:{pct}%; background:{color_barra};"></div>
        </div>
        <p style="font-size:0.85rem; color:#777; margin:0">
            Probabilidad de retraso: <strong>{pct}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>📦 On-Time AI</h1>
    <p>Predictor de retrasos logísticos · Modelo XGBoost · Recall 91.6%</p>
</div>
""", unsafe_allow_html=True)

if modelo is None:
    st.error("⚠️ No se encontró el modelo. Asegúrate de que `xgboost_final.pkl` está en `src/model/production/`.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predicción individual", "📂 Carga masiva (CSV)", "📊 Análisis"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Datos del envío</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Introduce las características del envío. El modelo analizará el riesgo de retraso
        y recomendará si es necesario enviar un cupón preventivo al cliente.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📦 Producto**")
        cost    = st.number_input("Coste del producto (USD)", min_value=0, max_value=1000, value=200, step=10)
        weight  = st.number_input("Peso del paquete (gramos)", min_value=500, max_value=8000, value=3000, step=100)
        imp     = st.selectbox("Importancia del producto", ["Baja", "Media", "Alta"])
        discount= st.slider("Descuento aplicado (%)", min_value=0, max_value=65, value=5)

    with col2:
        st.markdown("**🏭 Logística**")
        bloque  = st.selectbox("Bloque del almacén", ["A", "B", "C", "D", "F"])
        modo    = st.selectbox("Modo de envío", ["Flight", "Road", "Ship"])

    with col3:
        st.markdown("**👤 Cliente**")
        calls   = st.number_input("Llamadas a atención al cliente", min_value=0, max_value=10, value=2)
        rating  = st.slider("Valoración del cliente (1-5)", min_value=1, max_value=5, value=3)
        prior   = st.number_input("Compras anteriores", min_value=0, max_value=10, value=3)
        gender  = st.selectbox("Género", ["Hombre", "Mujer"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analizar riesgo de retraso"):
        datos = {
            'Customer_care_calls': calls,
            'Customer_rating':     rating,
            'Cost_of_the_Product': cost,
            'Prior_purchases':     prior,
            'Product_importance':  imp,
            'Discount_offered':    discount,
            'Weight_in_gms':       weight,
            'Warehouse_block':     bloque,
            'Mode_of_Shipment':    modo,
            'Gender':              gender
        }
        X = preparar_input(datos)
        prob = modelo.predict_proba(X)[0][1]
        mostrar_resultado(prob)

        with st.expander("Ver detalle de las features enviadas al modelo"):
            st.dataframe(X.T.rename(columns={0: "Valor"}), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CARGA MASIVA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Predicción por lotes</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            Sube un CSV con los datos de varios envíos. El modelo predecirá el riesgo de retraso
            para cada uno y podrás descargar el resultado enriquecido con las predicciones.
            El CSV debe tener las siguientes columnas:
            <code>Customer_care_calls</code>, <code>Customer_rating</code>,
            <code>Cost_of_the_Product</code>, <code>Prior_purchases</code>,
            <code>Product_importance</code> (low/medium/high), <code>Discount_offered</code>,
            <code>Weight_in_gms</code>, <code>Warehouse_block</code> (A/B/C/D/F),
            <code>Mode_of_Shipment</code> (Flight/Road/Ship), <code>Gender</code> (M/F).
        </div>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        df_raw = pd.read_csv(archivo)
        st.markdown(f"**{len(df_raw)} envíos cargados.** Vista previa:")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("Predecir para todos los envíos"):
            try:
                X_batch = preparar_csv(df_raw)
                probs   = modelo.predict_proba(X_batch)[:, 1]
                preds   = (probs >= 0.50).astype(int)

                df_resultado = df_raw.copy()
                df_resultado['Prob_retraso (%)'] = (probs * 100).round(1)
                df_resultado['Predicción']       = preds
                df_resultado['Acción recomendada'] = np.where(
                    preds == 1,
                    '⚠️ Enviar cupón preventivo',
                    '✅ Sin acción necesaria'
                )

                n_retraso  = preds.sum()
                n_puntual  = len(preds) - n_retraso
                pct_retraso= round(n_retraso / len(preds) * 100, 1)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Total de envíos", len(preds))
                m2.metric("Con riesgo de retraso", f"{n_retraso} ({pct_retraso}%)")
                m3.metric("Sin riesgo de retraso", n_puntual)

                st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)
                st.dataframe(
                    df_resultado[['Prob_retraso (%)', 'Predicción', 'Acción recomendada']
                                 + [c for c in df_raw.columns]].head(50),
                    use_container_width=True
                )

                csv_out = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar resultados completos",
                    data=csv_out,
                    file_name="predicciones_retrasos.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error al procesar el CSV: {e}")
                st.info("Comprueba que el CSV tiene las columnas correctas: Customer_care_calls, Customer_rating, Cost_of_the_Product, Prior_purchases, Product_importance, Discount_offered, Weight_in_gms, Warehouse_block, Mode_of_Shipment, Gender")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Introducción ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">El problema</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Una empresa internacional de e-commerce gestiona miles de envíos diarios desde un almacén
        central. Su problema no es la capacidad de venta, sino la de entrega: <strong>el 59.7% de
        los paquetes llegan tarde</strong>. Este proyecto construye un modelo capaz de anticipar
        esos retrasos antes de que ocurran, para que la empresa pueda actuar de forma proactiva.
    </div>
    """, unsafe_allow_html=True)

    # ── EDA ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Análisis exploratorio</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribución del target**")
        st.image(f"{IMG}/20_distribución_de_entregas.png", use_container_width=True)
        st.caption("6 de cada 10 envíos llegan tarde. El problema es sistémico, no puntual.")

    with col2:
        st.markdown("**El efecto del descuento**")
        st.image(f"{IMG}/21_descuentos_retrasos.png", use_container_width=True)
        st.caption("A partir del 15% de descuento la barra verde desaparece: ningún envío llega a tiempo.")

    st.markdown("""
    <div class="info-box">
        <strong>Hallazgo clave:</strong> el umbral crítico está entre el 10% y el 15% de descuento.
        Las campañas promocionales agresivas saturan la capacidad logística del almacén hasta el punto
        de colapsar la puntualidad de forma sistemática. Este patrón será confirmado después por el
        árbol de decisión (nodo raíz: <code>Discount_offered &lt;= 10.5</code>) y por el clustering
        (Cluster 0: descuento medio 40%, tasa de retraso 99.5%).
    </div>
    """, unsafe_allow_html=True)

    # ── Modelo elegido ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Modelo elegido: XGBoost</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Se entrenaron 5 modelos de familias distintas (Regresión Logística, Árbol de Decisión,
        Random Forest, XGBoost y KNN). La métrica prioritaria fue el <strong>Recall</strong>:
        de todos los retrasos reales, ¿cuántos detecta el modelo? Un retraso no detectado es
        un cliente que reclama sin haber recibido ninguna atención proactiva.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Comparativa de los 5 modelos**")
    st.image(f"{IMG}/08_comparativa_5_modelos.png", use_container_width=True)
    st.caption("La barra de Recall del XGBoost (0.92) destaca sobre todas las demás. Es el modelo que más retrasos detecta.")

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown("**Métricas del modelo final**")
        metricas = {
            "Métrica": ["Accuracy", "Precision", "Recall ⭐", "F1-Score", "ROC-AUC"],
            "Resultado": ["62.6%", "62.8%", "91.6%", "74.5%", "0.746"]
        }
        st.dataframe(pd.DataFrame(metricas), use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
            El ajuste clave fue el parámetro <code>scale_pos_weight ≈ 1.48</code>
            (positivos/negativos). Con el ratio incorrecto el Recall era del 47%.
            Con el ratio correcto sube al 91.6%. Un solo parámetro, 44 puntos de diferencia.
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("**Matriz de confusión — XGBoost**")
        st.image(f"{IMG}/07_matriz_confusion_xgb.png", use_container_width=True)
        st.caption("Solo 110 retrasos se escapan de 1.313 reales. Los 713 falsos positivos son cupones enviados de más: un coste asumible.")

    # ── Clustering ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Clustering K-Means (K=3)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Complementando el modelo supervisado, se aplicó K-Means sobre las variables numéricas
        del dataset. El modelo encontró espontáneamente, <strong>sin ver ninguna etiqueta de
        retraso</strong>, los mismos patrones que los modelos supervisados habían identificado.
    </div>
    """, unsafe_allow_html=True)

    st.image(f"{IMG}/15_resultados_clustering_kmeans_k3.png", use_container_width=True)

    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown("""
        <div class="result-card result-retraso">
            <div class="result-title">🟣 Cluster 0</div>
            <p style="margin:0.3rem 0"><strong>99.5% de retraso</strong></p>
            <p style="color:#777; font-size:0.85rem; margin:0">
                2.294 envíos (20.9%)<br>Descuento alto (media 40.1%)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="result-card result-puntual">
            <div class="result-title">🟢 Cluster 1</div>
            <p style="margin:0.3rem 0"><strong>47.9% de retraso</strong></p>
            <p style="color:#777; font-size:0.85rem; margin:0">
                6.097 envíos (55.4%)<br>Paquete pesado (media 4.801g)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown("""
        <div class="result-card" style="background:#f0f4ff; border: 2px solid #7090e0; padding:2rem; border-radius:16px; text-align:center;">
            <div class="result-title">🔵 Cluster 2</div>
            <p style="margin:0.3rem 0"><strong>52.3% de retraso</strong></p>
            <p style="color:#777; font-size:0.85rem; margin:0">
                2.608 envíos (23.7%)<br>Cliente fidelizado (5.1 compras prev.)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem">
        El Cluster 0 es el hallazgo más impactante: el 20.9% de los envíos tienen una tasa de
        retraso del 99.5%. Este grupo se define exclusivamente por el descuento alto, confirmando
        desde una metodología completamente independiente que las campañas promocionales agresivas
        son el factor de riesgo número uno de la cadena logística.
    </div>
    """, unsafe_allow_html=True)
