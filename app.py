import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")
st.title("Simulador AR / MA / ARMA e IRF (con checks de estabilidad)")

# -----------------------------
# Utils: raíces / checks
# -----------------------------
def _min_abs_root_outside_unit(poly: np.ndarray) -> float:
    """
    Devuelve min(|raiz|) para las raíces del polinomio.
    (Si poly es constante o no tiene raíces, devuelve +inf)
    """
    poly = np.asarray(poly, dtype=float)
    if poly.size <= 1:
        return np.inf
    r = np.roots(poly)
    if r.size == 0:
        return np.inf
    return float(np.min(np.abs(r)))

def check_stationarity_invertibility(ar_poly: np.ndarray, ma_poly: np.ndarray):
    """
    Para AR estacionario: raíces de ar_poly fuera del círculo unidad.
    Para MA invertible: raíces de ma_poly fuera del círculo unidad.
    """
    ar_min = _min_abs_root_outside_unit(ar_poly)
    ma_min = _min_abs_root_outside_unit(ma_poly)

    ar_ok = np.isinf(ar_min) or (ar_min > 1.0 + 1e-8)
    ma_ok = np.isinf(ma_min) or (ma_min > 1.0 + 1e-8)

    return ar_ok, ma_ok, ar_min, ma_min

# -----------------------------
# Sidebar (controles)
# -----------------------------
with st.sidebar:
    st.header("Controles")

    model_type = st.selectbox("Modelo", ["AR", "MA", "ARMA"], index=2)

    if model_type == "AR":
        p = st.slider("Orden AR (p)", 0, 5, 2)
        q = 0
    elif model_type == "MA":
        p = 0
        q = st.slider("Orden MA (q)", 0, 5, 1)
    else:
        p = st.slider("Orden AR (p)", 0, 5, 2)
        q = st.slider("Orden MA (q)", 0, 5, 1)

    tab1, tab2 = st.tabs(["Parámetros", "Simulación"])

    with tab1:
        ar_params = []
        ma_params = []

        if model_type in ["AR", "ARMA"] and p > 0:
            st.subheader("AR (φ)")
            for i in range(1, p + 1):
                ar_params.append(
                    st.slider(f"φ{i}", -0.99, 0.99, 0.3 if i == 1 else 0.0, 0.01)
                )

        if model_type in ["MA", "ARMA"] and q > 0:
            st.subheader("MA (θ)")
            for i in range(1, q + 1):
                ma_params.append(
                    st.slider(f"θ{i}", -0.99, 0.99, 0.3 if i == 1 else 0.0, 0.01)
                )

    with tab2:
        sigma = st.slider("σ (desv. del shock)", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Tamaño de muestra", 50, 2000, 300, 10)
        burnin = st.slider("Burn-in (descartar al inicio)", 0, 2000, 200, 10)
        steps = st.slider("Horizonte IRF", 1, 60, 20, 1)
        seed = st.number_input("Semilla", min_value=0, max_value=999999, value=1234, step=1)

# -----------------------------
# Construcción del proceso
# -----------------------------
np.random.seed(int(seed))

# statsmodels usa: ar = [1, -phi1, -phi2, ...], ma = [1, theta1, theta2, ...]
ar_poly = np.r_[1.0, -np.array(ar_params, dtype=float)] if len(ar_params) else np.array([1.0])
ma_poly = np.r_[1.0,  np.array(ma_params, dtype=float)] if len(ma_params) else np.array([1.0])

ar_ok, ma_ok, ar_min_root, ma_min_root = check_stationarity_invertibility(ar_poly, ma_poly)

# Mostrar estado
with st.sidebar:
    st.markdown("---")
    st.subheader("Checks")
    if model_type in ["AR", "ARMA"]:
        st.write(f"Estacionariedad AR: {'✅' if ar_ok else '❌'} (min |raíz| = {ar_min_root:.3f})")
    if model_type in ["MA", "ARMA"]:
        st.write(f"Invertibilidad MA: {'✅' if ma_ok else '❌'} (min |raíz| = {ma_min_root:.3f})")

# Si no cumple, cortamos ejecución para no simular/estimar algo inválido
if (model_type in ["AR", "ARMA"] and not ar_ok) or (model_type in ["MA", "ARMA"] and not ma_ok):
    st.error(
        "Parámetros no válidos: el AR no es estacionario y/o el MA no es invertible.\n\n"
        "Ajusta φ y/o θ para que las raíces del polinomio correspondiente queden fuera del círculo unidad (|raíz|>1)."
    )
    st.stop()

# -----------------------------
# Simulación y estimación
# -----------------------------
proc = ArmaProcess(ar_poly, ma_poly)

y = proc.generate_sample(nsample=n + burnin, scale=sigma)[burnin:]

pe = p if model_type != "MA" else 0
qe = q if model_type != "AR" else 0

# Ajuste ARIMA
res = ARIMA(y, order=(pe, 0, qe)).fit()

# IRF estimada
# (impulse_responses devuelve longitud steps+1 con el impacto en 0)
if hasattr(res, "impulse_responses"):
    irf_hat = res.impulse_responses(steps)
else:
    irf_hat = res.impulse_response(steps=steps)
irf_hat = np.asarray(irf_hat).reshape(-1)

# IRF teórica del DGP
irf_th = proc.impulse_response(steps=steps)
irf_th = np.asarray(irf_th).reshape(-1)

# Asegurar mismo largo
H = min(len(irf_hat), len(irf_th))
irf_hat = irf_hat[:H]
irf_th = irf_th[:H]
hgrid = np.arange(H)

# -----------------------------
# Layout principal
# -----------------------------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Serie simulada")
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax1.plot(y, linewidth=1.6, label="y_t")
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Valor de la serie")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)

with right:
    st.subheader("IRF: teórica vs estimada")
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), dpi=120)

    # Teórica (línea)
    ax2.plot(hgrid, irf_th, linewidth=2.0, label="IRF teórica (DGP)")

    # Estimada (stems)
    markerline, stemlines, baseline = ax2.stem(hgrid, irf_hat, basefmt=" ", label="IRF estimada (ARIMA)")
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=5)

    ax2.set_xlabel("Horizonte (pasos)")
    ax2.set_ylabel("Respuesta al impulso")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)

# Opcional: resumen rápido del ajuste
with st.expander("Resumen del modelo estimado"):
    st.text(res.summary().as_text())
