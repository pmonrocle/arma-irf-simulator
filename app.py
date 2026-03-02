import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")
st.title("Simulador AR / MA / ARMA e IRF")

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Controles")

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

        if model_type in ["AR", "ARMA"] and p > 0:
            st.subheader("Coeficientes AR (φ)")
            for i in range(1, p + 1):
                default_val = 0.4 if i == 1 else 0.0
                ar_params.append(
                    st.slider(f"φ{i}", -0.99, 0.99, default_val, 0.01)
                )

        if model_type in ["MA", "ARMA"] and q > 0:
            st.subheader("Coeficientes MA (θ)")
            for i in range(1, q + 1):
                default_val = 0.4 if i == 1 else 0.0
                ma_params.append(
                    st.slider(f"θ{i}", -0.99, 0.99, default_val, 0.01)
                )

    with tab2:
        sigma = st.slider("σ (desv. típica de la innovación)", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Tamaño de muestra", 50, 3000, 300, 10)
        burnin = st.slider("Burn-in", 0, 3000, 300, 10)
        seed = st.number_input("Semilla", min_value=0, max_value=999999, value=1234, step=1)

    with tab3:
        steps = st.slider("Horizonte IRF", 1, 60, 20, 1)
        shock_mode = st.radio(
            "Escala del impulso",
            ["Shock unitario", "Shock = σ"],
            index=0
        )
        show_estimated = st.checkbox("Mostrar IRF estimada", value=True)
        show_theoretical = st.checkbox("Mostrar IRF teórica", value=True)

# =========================================================
# Preparación del proceso
# =========================================================
np.random.seed(int(seed))

ar_params = np.array(ar_params, dtype=float) if len(ar_params) else np.array([])
ma_params = np.array(ma_params, dtype=float) if len(ma_params) else np.array([])

# Convención de statsmodels:
# AR: 1 - φ1 L - φ2 L^2 - ...
# MA: 1 + θ1 L + θ2 L^2 + ...
ar_poly = np.r_[1.0, -ar_params] if len(ar_params) else np.array([1.0])
ma_poly = np.r_[1.0,  ma_params] if len(ma_params) else np.array([1.0])

proc = ArmaProcess(ar_poly, ma_poly)

# =========================================================
# Chequeo de estacionariedad e invertibilidad
# =========================================================
is_stationary = proc.isstationary
is_invertible = proc.isinvertible

# Raíces
ar_roots = np.roots(ar_poly) if len(ar_poly) > 1 else np.array([])
ma_roots = np.roots(ma_poly) if len(ma_poly) > 1 else np.array([])

# =========================================================
# Simulación
# =========================================================
y = proc.generate_sample(nsample=n + burnin, scale=sigma)[burnin:]

# =========================================================
# IRF teórica
# =========================================================
# arma2ma devuelve los coeficientes psi_j de la representación MA(inf)
# Incluye psi_0 = 1
irf_theoretical = proc.arma2ma(lags=steps)

if shock_mode == "Shock = σ":
    irf_theoretical = sigma * irf_theoretical

# =========================================================
# Estimación ARIMA
# =========================================================
irf_estimated = None
fit_ok = True
fit_msg = ""

try:
    pe = p if model_type in ["AR", "ARMA"] else 0
    qe = q if model_type in ["MA", "ARMA"] else 0

    res = ARIMA(y, order=(pe, 0, qe), trend="n").fit()

    # En statsmodels modernos suele existir impulse_responses
    if hasattr(res, "impulse_responses"):
        irf_estimated = np.asarray(res.impulse_responses(steps=steps)).reshape(-1)
    else:
        irf_estimated = np.asarray(res.impulse_response(steps=steps)).reshape(-1)

    if shock_mode == "Shock = σ":
        irf_estimated = sigma * irf_estimated

except Exception as e:
    fit_ok = False
    fit_msg = str(e)

# =========================================================
# Información del modelo
# =========================================================
st.subheader("Diagnóstico del proceso introducido")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Modelo", model_type)

with col2:
    st.metric("Estacionario (AR)", "Sí" if is_stationary else "No")

with col3:
    st.metric("Invertible (MA)", "Sí" if is_invertible else "No")

if model_type in ["AR", "ARMA"]:
    if is_stationary:
        st.success("La parte AR cumple la condición de estacionariedad.")
    else:
        st.warning(
            "La parte AR NO cumple la condición de estacionariedad. "
            "Alguna raíz del polinomio AR no está fuera del círculo unidad."
        )

if model_type in ["MA", "ARMA"]:
    if is_invertible:
        st.success("La parte MA cumple la condición de invertibilidad.")
    else:
        st.warning(
            "La parte MA NO cumple la condición de invertibilidad. "
            "Alguna raíz del polinomio MA no está fuera del círculo unidad."
        )

# =========================================================
# Tabla de parámetros y raíces
# =========================================================
rows = []

for i, val in enumerate(ar_params, start=1):
    rows.append({"Tipo": "AR", "Coeficiente": f"φ{i}", "Valor": val})

for i, val in enumerate(ma_params, start=1):
    rows.append({"Tipo": "MA", "Coeficiente": f"θ{i}", "Valor": val})

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

root_cols = st.columns(2)

with root_cols[0]:
    st.markdown("**Raíces del polinomio AR**")
    if len(ar_roots) > 0:
        df_ar_roots = pd.DataFrame({
            "Raíz AR": ar_roots.astype(complex),
            "Módulo": np.abs(ar_roots)
        })
        st.dataframe(df_ar_roots, use_container_width=True)
    else:
        st.info("No hay parte AR.")

with root_cols[1]:
    st.markdown("**Raíces del polinomio MA**")
    if len(ma_roots) > 0:
        df_ma_roots = pd.DataFrame({
            "Raíz MA": ma_roots.astype(complex),
            "Módulo": np.abs(ma_roots)
        })
        st.dataframe(df_ma_roots, use_container_width=True)
    else:
        st.info("No hay parte MA.")

# =========================================================
# Gráficos principales
# =========================================================
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Serie simulada")
    fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax1.plot(y, linewidth=1.5, label="y_t")
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Valor")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

with right:
    st.subheader("IRF: teórica vs estimada")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=120)

    h = np.arange(len(irf_theoretical))

    if show_theoretical:
        markerline1, stemlines1, baseline1 = ax2.stem(
            h, irf_theoretical, linefmt="C0-", markerfmt="C0o", basefmt=" "
        )
        plt.setp(stemlines1, linewidth=1.8)
        plt.setp(markerline1, markersize=5)
        markerline1.set_label("IRF teórica")

    if show_estimated and (irf_estimated is not None):
        h2 = np.arange(len(irf_estimated))
        markerline2, stemlines2, baseline2 = ax2.stem(
            h2, irf_estimated, linefmt="C1--", markerfmt="C1s", basefmt=" "
        )
        plt.setp(stemlines2, linewidth=1.4)
        plt.setp(markerline2, markersize=4)
        markerline2.set_label("IRF estimada")

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
# Información adicional sobre la estimación
# =========================================================
st.subheader("Estimación del modelo sobre la serie simulada")

if fit_ok:
    st.success("La estimación ARIMA se ha realizado correctamente.")
    try:
        st.text(res.summary())
    except Exception:
        st.write("Resumen no disponible en formato texto.")
else:
    st.error("No se pudo estimar el modelo ARIMA sobre la muestra simulada.")
    st.code(fit_msg)

# =========================================================
# Interpretación breve
# =========================================================
st.subheader("Interpretación")

if shock_mode == "Shock unitario":
    st.write(
        "La IRF se interpreta como la respuesta dinámica de la serie ante una "
        "innovación de tamaño 1 en el instante inicial."
    )
else:
    st.write(
        "La IRF se interpreta como la respuesta dinámica de la serie ante una "
        "innovación de tamaño igual a σ, la desviación típica del shock."
    )

st.write(
    "La IRF teórica procede del proceso ARMA definido por los parámetros introducidos. "
    "La IRF estimada procede del modelo ajustado sobre la muestra simulada, por lo que "
    "puede diferir de la teórica debido al error muestral y de estimación."
)



