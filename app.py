import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")
st.title("Simulador AR / MA / ARMA e IRF")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    model_type = st.selectbox("Modelo", ["AR", "MA", "ARMA"], index=2)

    if model_type == "AR":
        p = st.slider("Orden AR (p)", 1, 5, 2)
        q = 0
    elif model_type == "MA":
        p = 0
        q = st.slider("Orden MA (q)", 1, 5, 1)
    else:
        p = st.slider("Orden AR (p)", 1, 5, 2)
        q = st.slider("Orden MA (q)", 1, 5, 1)

    tab1, tab2, tab3 = st.tabs(["Parámetros", "Simulación", "IRF"])

    with tab1:
        ar_params = []
        ma_params = []

        if model_type in ["AR", "ARMA"]:
            for i in range(1, p + 1):
                ar_params.append(st.slider(f"φ{i}", -0.99, 0.99, 0.4 if i == 1 else 0.0, 0.01))

        if model_type in ["MA", "ARMA"]:
            for i in range(1, q + 1):
                ma_params.append(st.slider(f"θ{i}", -0.99, 0.99, 0.4 if i == 1 else 0.0, 0.01))

    with tab2:
        sigma = st.slider("σ", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Tamaño muestral", 100, 3000, 1000, 50)
        burnin = st.slider("Burn-in", 0, 2000, 300, 10)
        seed = st.number_input("Semilla", 0, 999999, 1234, 1)

    with tab3:
        steps = st.slider("Horizonte IRF", 1, 60, 20, 1)
        show_estimated = st.checkbox("Mostrar IRF estimada", value=True)

# -------------------------------------------------
# Preparación
# -------------------------------------------------
np.random.seed(int(seed))

ar_params = np.array(ar_params) if len(ar_params) else np.array([])
ma_params = np.array(ma_params) if len(ma_params) else np.array([])

ar_poly = np.r_[1.0, -ar_params] if len(ar_params) else np.array([1.0])
ma_poly = np.r_[1.0, ma_params] if len(ma_params) else np.array([1.0])

proc = ArmaProcess(ar_poly, ma_poly)

# Diagnóstico
is_stationary = proc.isstationary
is_invertible = proc.isinvertible

# Simulación
y = proc.generate_sample(nsample=n + burnin, scale=sigma)[burnin:]

# IRF teórica
irf_theoretical = proc.arma2ma(lags=steps)

# IRF estimada
irf_estimated = None
try:
    res = ARIMA(y, order=(p, 0, q), trend="n").fit()
    if hasattr(res, "impulse_responses"):
        irf_estimated = np.asarray(res.impulse_responses(steps=steps)).reshape(-1)
    else:
        irf_estimated = np.asarray(res.impulse_response(steps=steps)).reshape(-1)
except:
    pass

# -------------------------------------------------
# Resumen
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Modelo", model_type)
c2.metric("AR estacionario", "Sí" if is_stationary else "No")
c3.metric("MA invertible", "Sí" if is_invertible else "No")

# -------------------------------------------------
# Gráficos
# -------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Serie simulada")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(y)
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with right:
    st.subheader("IRF")
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    h = np.arange(len(irf_theoretical))
    ax2.stem(h, irf_theoretical, basefmt=" ", label="Teórica")

    if show_estimated and irf_estimated is not None:
        ax2.plot(np.arange(len(irf_estimated)), irf_estimated, marker="o", label="Estimada")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Horizonte")
    ax2.set_ylabel("Respuesta")
    ax2.grid(alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)



