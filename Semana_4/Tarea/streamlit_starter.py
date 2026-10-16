"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"


@st.cache_data
def cargar():
    catalogo = pd.read_csv("data/catalogo.csv")
    hist = pd.read_csv("data/ordenes_historial.csv")
    hold = pd.read_csv("data/ordenes_holdout.csv")
    exp = pd.read_csv("data/experimento.csv")
    clientes = pd.read_csv("data/clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)   

def puntajes(metodo, x_vec, K):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # suma de las similitudes de cada producto con lo que el cliente ya compró
        if x_vec.sum() == 0:
            return popularidad.astype(float)  # Cold start fallback
        return x_vec @ S_it   
    #filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    if x_vec.sum() == 0:
        return popularidad.astype(float)
    sims = cosine_similarity(x_vec[None, :], X)[0]
    if fila_idx is not None:
        sims[fila_idx] = 0.0
    vecinos = np.argsort(-sims)[:K]
    return sims[vecinos] @ X[vecinos]       


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption("TODO: escriban aquí una frase sobre qué decisión de negocio ayuda a tomar esta app.")

# barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    metodo = st.selectbox(
        "Seleccione el método de recomendación:",
        ["Filtrado por ítem", "Filtrado por cliente", "Popularidad"]
    )
    K = st.slider("Número de vecinos (K) para filtrado por cliente:", min_value=5, max_value=100, value=30, step=5)
    st.markdown("---")
    st.markdown(
        "**Reglas de negocio aplicadas:**\n"
        "- Se excluyen productos ya comprados.\n"
        "- Se excluye el **Agua** (producto general sin complementariedad).\n"
        "- Clientes con 0 compras reciben respaldo por popularidad."
    )

tab1, tab2 = st.tabs(["Recomendar a un Cliente", "Comparar Métodos (Holdout Control)"])
with tab1:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    items_comprados = [productos[j] for j in np.where(x_vec > 0)[0]]

    with col_info:
        st.write(f"**Historial de compras ({len(items_comprados)} productos):**")
        if len(items_comprados) > 0:
            st.info(", ".join(items_comprados))
        else:
            st.warning("Cliente sin historial previo (Cold Start / 0 compras registradas).")

    # Recomendaciones con explicabilidad de negocio
    scores = puntajes(metodo, x_vec, K, fila_idx=fila)
    recs = top3(scores, x_vec)

    st.subheader(f"Top 3 Recomendaciones para el Cliente {cid} (Método: {metodo})")
    cols = st.columns(3)

    for rank, (col, prod_idx) in enumerate(zip(cols, recs), start=1):
        prod_nombre = productos[prod_idx]
        score_val = scores[prod_idx]
        
        # Lógica del "Por qué"
        if x_vec.sum() == 0:
            porque = "Es de los productos más vendidos en la tienda (respaldo por falta de historial)."
        elif metodo == "Popularidad":
            porque = f"Es de lo más vendido globalmente (comprado {int(score_val):,} veces)."
        elif metodo == "Filtrado por ítem":
            # Buscar cuál de los productos comprados aporta más similitud
            sims_con_comprados = {productos[j]: S_it[prod_idx, j] for j in np.where(x_vec > 0)[0] if j != COL_AGUA}
            if sims_con_comprados:
                max_item, max_sim = max(sims_con_comprados.items(), key=lambda t: t[1])
                porque = f"Porque compró **{max_item}** (similitud de {max_sim:.2f})."
            else:
                porque = "Basado en la afinidad con sus productos habituales."
        else:  # Filtrado por cliente
            sims = cosine_similarity(x_vec[None, :], X)[0]
            sims[fila] = 0.0
            vecinos = np.argsort(-sims)[:K]
            compraron = int(X[vecinos, prod_idx].sum())
            porque = f"**{compraron}** de sus **{K}** vecinos más parecidos lo compraron."

        with col:
            st.metric(label=f"#{rank} {prod_nombre}", value=f"Score: {score_val:.2f}")
            st.markdown(f"**¿Por qué?** {porque}")

with tab2:
    st.subheader("Evaluación Offline: Hit Rate@3 y Cobertura en Grupo Control")
    st.caption("Evaluado exclusivamente en clientes del grupo control que probaron productos nuevos (sin sesgo de recomendación).")

    # Filtro de clientes elegibles del grupo control
    elegibles = (NUEVO.sum(axis=1) > 0) & es_control
    total_elegibles = int(elegibles.sum())
    
    n_sample = st.slider("Cantidad de clientes del grupo control a evaluar:", min_value=100, max_value=total_elegibles, value=min(1000, total_elegibles), step=100)
    indices_eval = np.where(elegibles)[0][:n_sample]

    # Pre-cálculo para todos los métodos
    S_pop_full = np.tile(popularidad, (len(X), 1))
    S_item_full = X @ S_it

    # Filtrado por cliente para la muestra
    sims_control = cosine_similarity(X[indices_eval], X)
    for row_idx, real_idx in enumerate(indices_eval):
        sims_control[row_idx, real_idx] = 0.0
    
    vecinos_eval = np.argsort(-sims_control, axis=1)[:, :K]
    S_user_sample = np.zeros((len(indices_eval), len(productos)))
    for row_idx in range(len(indices_eval)):
        v = vecinos_eval[row_idx]
        S_user_sample[row_idx] = sims_control[row_idx, v] @ X[v]

    def evaluar_subset(scores_subset, indices):
        X_sub = X[indices]
        NUEVO_sub = NUEVO[indices]
        s = np.where(X_sub > 0, -np.inf, scores_subset).astype(float)
        s[:, COL_AGUA] = -np.inf
        top3_preds = np.argsort(-s, axis=1)[:, :3]

        aciertos = np.array([NUEVO_sub[u, top3_preds[u]].sum() > 0 for u in range(len(indices))])
        hit_rate = aciertos.mean()
        cobertura = len(np.unique(top3_preds))
        return hit_rate, cobertura

    hr_pop, cob_pop = evaluar_subset(S_pop_full[indices_eval], indices_eval)
    hr_item, cob_item = evaluar_subset(S_item_full[indices_eval], indices_eval)
    hr_user, cob_user = evaluar_subset(S_user_sample, indices_eval)

    df_comp = pd.DataFrame({
        "Método": ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"],
        "Hit Rate@3": [hr_pop, hr_item, hr_user],
        "Cobertura (de 28 productos)": [cob_pop, cob_item, cob_user]
    }).set_index("Método")

    col_tabla, col_graf = st.columns([1, 1])
    with col_tabla:
        st.dataframe(df_comp.style.format({"Hit Rate@3": "{:.1%}", "Cobertura (de 28 productos)": "{:d}"}))
        st.markdown(
            "> **Conclusión de negocio:**  \n"
            "> El **Filtrado por ítem** ofrece el mejor balance: alto hit rate, máxima cobertura de catálogo y alta explicabilidad sin el costo computacional de buscar vecinos en tiempo real."
        )

    with col_graf:
        st.bar_chart(df_comp["Hit Rate@3"])
