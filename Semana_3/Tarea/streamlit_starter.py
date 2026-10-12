"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    ruta_csv = BASE_DIR / "data" / "uplift_campaign.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta_csv}. "
            "Verifica que esté subido al repositorio de GitHub."
        )

    return pd.read_csv(ruta_csv)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000)
    modelo_tratado.fit(train_t[FEATURES],train_t["responded_60d"])   # <-- reemplacen esto
    modelo_control = LogisticRegression(max_iter=1000)
    modelo_control.fit(train_c[FEATURES],train_c["responded_60d"])   # <-- reemplacen esto

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)

def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col,ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt = n_tratados_acum[i - 1]
        nc = n_control_acum[i - 1]

        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0

        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)

# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("Herramienta para determinar el porcentaje de contacto de clientes para la campaña Strabucks Reward")

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider("Porcentaje de clientes a contactar",min_value=10,max_value=100,value=30,step=10)   # <-- reemplacen esto: st.slider(...)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---
xs_uplift, ys_uplift = curva_qini(test,"uplift_estimado")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(xs_uplift * 100,ys_uplift,color=GREEN,linewidth=2.5,label="T-learner (Regresión Logística)")

# Referencia aleatoria
ax.plot(xs_uplift * 100,np.linspace(0, ys_uplift[-1], len(xs_uplift)),color=GRAY,linestyle="--",label="Selección aleatoria")

ax.axhline(0, color=GRAY, linewidth=0.8)
ax.set_title("Curva Qini — T-learner")
ax.set_xlabel("Porcentaje de clientes priorizados (%)")
ax.set_ylabel("Ganancia incremental acumulada")
ax.legend()
ax.grid(alpha=0.2)

st.subheader("1. Curva Qini")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "La curva ayuda a evaluar si ordenar a los clientes por uplift "
    "estimado permite identificar grupos con mayor respuesta incremental."
)

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---

st.subheader(f"2. Resultado para contactar al {pct_contactar}%")

n_contactar = max(1,int(len(test) * pct_contactar / 100))

seleccionados = (test.sort_values("uplift_estimado",ascending=False).head(n_contactar).index)

valor_cliente, n = valor_de_la_politica(test,seleccionados)

col1, col2, col3 = st.columns(3)

col1.metric("Clientes seleccionados",f"{n:,}")
col2.metric("Valor por cliente ($)",f"${valor_cliente:,.2f}")

col3.metric("Valor total estimado ($)",f"${valor_cliente * n:,.2f}")

st.info(
    "El porcentaje seleccionado prioriza a los clientes con mayor "
    "uplift estimado. Compara el valor por cliente y el valor total "
    "antes de recomendar una política."
)

#st.info("Reemplacen cada TODO con su propio código. Revisen el notebook del laboratorio para la lógica exacta de cada pieza.")
