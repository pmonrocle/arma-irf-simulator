import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")
st.title("Simulador AR / MA / ARMA e IRF")

# =========================================================
# SIDEBAR
# =========================================================
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
        st.subheader("Coeficientes")

        ar_params = []
        ma_params = []

        if model_type in ["AR", "ARMA"] and p > 0:
            st.markdown("**Parte AR**")
            for i in range(1, p + 1):
                ar_params.append(
                    st.slider(f"φ{i}", -0.99, 0.99, 0.4 if i == 1 else 0.0, 0.01)
                )

        if model_type in ["MA", "ARMA"] and q > 0:
            st.markdown("**Parte MA**")
            for i in range(1, q + 1):
                ma_params.append(
                    st.slider(f"θ{i}", -0.99, 0.99, 0.4 if i == 1 else 0.0, 0.01)
                )

    with tab2:
        st.subheader("Opciones de simulación")
        sigma = st.slider("σ (desv. típica de la innovación)", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Tamaño de muestra", 100, 3000, 1000, 50)
        burnin = st.slider("Burn-in", 0, 2000, 300, 10)
        seed = st.number_input("Semilla", min_value=0, max_value=999999, value=1234, step=1)

    with tab3:
        st.subheader("Opciones de IRF")
        steps = st.slider("Horizonte", 1, 60, 20, 1)
        show_estimated = st.checkbox("Mostrar IRF estimada", value=True)

# =========================================================
# PREPARACIÓN
# =========================================================
np.random.seed(int(seed))

ar_params = np.array(ar_params, dtype=float) if len(ar_params) else np.array([])
ma_params = np.array(ma_params, dtype=float) if len(ma_params) else np.array([])

# Convención de statsmodels:
# AR: 1 - φ1 L - φ2 L^2 - ...
# MA: 1 + θ1 L + θ2 L^2 + ...
ar_poly = np.r_[1.0, -ar_params] if len(ar_params) else np.array([1.0])
ma_poly = np.r_[1.0, ma_params] if len(ma_params) else np.array([1.0])

proc = ArmaProcess(ar_poly, ma_poly)

is_stationary = proc.isstationary
is_invertible = proc.isinvertible

ar_roots = np.roots(ar_poly) if len(ar_poly) > 1 else np.array([])
ma_roots = np.roots(ma_poly) if len(ma_poly) > 1 else np.array([])

# =========================================================
# SIMULACIÓN
# =========================================================
y = proc.generate_sample(nsample=n + burnin, scale=sigma)[burnin:]

# =========================================================
# IRF TEÓRICA (shock unitario)
# =========================================================
irf_theoretical = proc.arma2ma(lags=steps)

# =========================================================
# ESTIMACIÓN
# =========================================================
fit_ok = True
fit_msg = ""
irf_estimated = None
res = None

try:
    pe = p if model_type in ["AR", "ARMA"] else 0
    qe = q if model_type in ["MA", "ARMA"] else 0

    res = ARIMA(y, order=(pe, 0, qe), trend="n").fit()

    if hasattr(res, "impulse_responses"):
        irf_estimated = np.asarray(res.impulse_responses(steps=steps)).reshape(-1)
    else:
        irf_estimated = np.asarray(res.impulse_response(steps=steps)).reshape(-1)

except Exception as e:
    fit_ok = False
    fit_msg = str(e)

# =========================================================
# RESUMEN
# =========================================================
st.subheader("Resumen")

c1, c2, c3 = st.columns(3)
c1.metric("Modelo", model_type)
c2.metric("AR estacionario", "Sí" if is_stationary else "No")
c3.metric("MA invertible", "Sí" if is_invertible else "No")

if not is_stationary and model_type in ["AR", "ARMA"]:
    st.warning("La parte AR no cumple la condición de estacionariedad.")

if not is_invertible and model_type in ["MA", "ARMA"]:
    st.warning("La parte MA no cumple la condición de invertibilidad.")

# =========================================================
# GRÁFICOS
# =========================================================
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Serie simulada")
    fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax1.plot(y, linewidth=1.3)
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Valor")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)

with right:
    st.subheader("IRF: teórica vs estimada")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=120)

    h = np.arange(len(irf_theoretical))
    markerline1, stemlines1, baseline1 = ax2.stem(
        h, irf_theoretical, basefmt=" ", label="Teórica"
    )
    plt.setp(stemlines1, linewidth=1.8)
    plt.setp(markerline1, markersize=5)

    if show_estimated and irf_estimated is not None:
        h2 = np.arange(len(irf_estimated))
        ax2.plot(h2, irf_estimated, marker="o", linewidth=1.5, label="Estimada")

    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("Horizonte")
    ax2.set_ylabel("Respuesta al impulso")
    ax2.grid(True, alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

# =========================================================
# DETALLE TÉCNICO PLEGABLE
# =========================================================
with st.expander("Ver detalle técnico"):
    st.markdown("**Coeficientes introducidos**")

    rows = []
    for i, val in enumerate(ar_params, start=1):
        rows.append({"Tipo": "AR", "Coeficiente": f"φ{i}", "Valor": val})
    for i, val in enumerate(ma_params, start=1):
        rows.append({"Tipo": "MA", "Coeficiente": f"θ{i}", "Valor": val})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    colr1, colr2 = st.columns(2)

    with colr1:
        st.markdown("**Raíces AR**")
        if len(ar_roots) > 0:
            st.dataframe(
                pd.DataFrame({
                    "Raíz": ar_roots.astype(complex),
                    "Módulo": np.abs(ar_roots)
                }),
                use_container_width=True
            )
        else:
            st.write("No aplica.")

    with colr2:
        st.markdown("**Raíces MA**")
        if len(ma_roots) > 0:
            st.dataframe(
                pd.DataFrame({
                    "Raíz": ma_roots.astype(complex),
                    "Módulo": np.abs(ma_roots)
                }),
                use_container_width=True
            )
        else:
            st.write("No aplica.")

    st.markdown("**Estimación ARIMA sobre la muestra simulada**")
    if fit_ok and res is not None:
        st.code(res.summary().as_text())
    else:
        st.error("No se pudo estimar el modelo.")
        st.code(fit_msg)

# =========================================================
# INTERPRETACIÓN
# =========================================================
st.subheader("Interpretación")
st.write("La IRF representa la respuesta dinámica de la serie ante una innovación inicial de tamaño 1.")
st.write(
    "La respuesta teórica procede del modelo AR / MA / ARMA fijado por los parámetros introducidos. "
    "La respuesta estimada procede del modelo ajustado sobre la muestra simulada, por lo que puede diferir "
    "de la teórica por error muestral y de estimación."
)



