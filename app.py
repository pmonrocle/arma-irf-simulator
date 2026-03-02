import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")
st.title("Simulador AR / MA / ARMA e IRF")

# -----------------------------
# Sidebar (controles)
# -----------------------------
with st.sidebar:
    st.header("Controles")

    model_type = st.selectbox("Modelo", ["AR", "MA", "ARMA"], index=2)

    # Órdenes (compacto)
    if model_type == "AR":
        p = st.slider("Orden AR (p)", 0, 5, 2)
        q = 0
    elif model_type == "MA":
        p = 0
        q = st.slider("Orden MA (q)", 0, 5, 1)
    else:
        p = st.slider("Orden AR (p)", 0, 5, 2)
        q = st.slider("Orden MA (q)", 0, 5, 1)

    # Pestañas para que no sea una lista eterna
    tab1, tab2 = st.tabs(["Parámetros", "Simulación"])

    with tab1:
        # Parámetros dinámicos (solo se muestran los que aplican)
        ar_params = []
        ma_params = []

        if model_type in ["AR", "ARMA"] and p > 0:
            st.subheader("AR (φ)")
            for i in range(1, p + 1):
                ar_params.append(st.slider(f"φ{i}", -0.99, 0.99, 0.3 if i == 1 else 0.0, 0.01))

        if model_type in ["MA", "ARMA"] and q > 0:
            st.subheader("MA (θ)")
            for i in range(1, q + 1):
                ma_params.append(st.slider(f"θ{i}", -0.99, 0.99, 0.3 if i == 1 else 0.0, 0.01))

    with tab2:
        sigma = st.slider("σ (desv. del shock)", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Tamaño de muestra", 50, 2000, 300, 10)
        burnin = st.slider("Burn-in (descartar al inicio)", 0, 2000, 200, 10)
        steps = st.slider("Horizonte IRF", 1, 60, 20, 1)
        seed = st.number_input("Semilla", min_value=0, max_value=999999, value=1234, step=1)

# -----------------------------
# Cálculo (se actualiza solo)
# -----------------------------
np.random.seed(int(seed))

ar_poly = np.r_[1, -np.array(ar_params)] if len(ar_params) else np.array([1.0])
ma_poly = np.r_[1,  np.array(ma_params)] if len(ma_params) else np.array([1.0])

proc = ArmaProcess(ar_poly, ma_poly)
y = proc.generate_sample(nsample=n + burnin, scale=sigma)[burnin:]

pe = p if model_type != "MA" else 0
qe = q if model_type != "AR" else 0
res = ARIMA(y, order=(pe, 0, qe)).fit()

if hasattr(res, "impulse_responses"):
    irf = res.impulse_responses(steps)
else:
    irf = res.impulse_response(steps=steps)
irf = np.asarray(irf).reshape(-1)

# -----------------------------
# Layout principal: gráficos grandes
# -----------------------------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Serie simulada")
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax1.plot(y, linewidth=1.6)
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Valor de la serie")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)

with right:
    st.subheader("IRF (modelo estimado)")
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax2.stem(range(len(irf)), irf, basefmt=" ")
    ax2.set_xlabel("Horizonte (pasos)")
    ax2.set_ylabel("Respuesta al impulso")
    ax2.grid(True, alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)


