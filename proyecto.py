import streamlit as st
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
import scipy.stats as stats 
from io import BytesIO

# --- 1. CONSTANTES Y TABLA T-STUDENT ---
T_9545_TABLE = {
    1: 13.97, 2: 4.527, 3: 3.307, 4: 2.869, 5: 2.649, 6: 2.517, 7: 2.429,
    8: 2.366, 9: 2.320, 10: 2.284, 11: 2.255, 12: 2.231, 13: 2.212,
    14: 2.195, 15: 2.181, 16: 2.169, 17: 2.158, 18: 2.149, 19: 2.140,
    20: 2.133, 21: 2.126, 22: 2.120, 23: 2.115, 24: 2.110, 25: 2.105,
    26: 2.101, 27: 2.097, 28: 2.093, 29: 2.090, 30: 2.087, 35: 2.074,
    40: 2.064, 50: 2.051, 100: 2.025, 200: 2.016, 'inf': 2.000
}

def get_t_factor(v):
    if v == 'inf' or v >= 200: return T_9545_TABLE['inf']
    v = math.floor(v) 
    v_keys = sorted([k for k in T_9545_TABLE.keys() if isinstance(k, int)])
    for key_v in reversed(v_keys):
        if v >= key_v: return T_9545_TABLE[key_v]
    return T_9545_TABLE[1]

# --- 2. CONFIGURACIÓN DE LOS 6 CASOS DEL PROYECTO ---
def get_project_cases():
    return {
        "Manual (Usuario define todo)": None,
        "Caso 1: Resistencia Alta (R20)": {
            "vars": "V, I, RA, t, alpha",
            "eq": "(V/I - RA) * (1 + alpha * (20 - t))",
            "data": {} # Aquí irás agregando tus valores
        },
        "Caso 2: Resistencia Baja (R)": {
            "vars": "RV, V, I",
            "eq": "(RV * V) / (I * RV - V)",
            "data": {}
        },
        "Caso 3: Resistores en Serie": {
            "vars": "R1, R2",
            "eq": "R1 + R2",
            "data": {}
        },
        "Caso 4: Resistores en Paralelo": {
            "vars": "R1, R2",
            "eq": "(R1 * R2) / (R1 + R2)",
            "data": {}
        },
        "Caso 5: Potencia Disipada": {
            "vars": "V, I, RA",
            "eq": "V * I - (I**2 * RA)",
            "data": {}
        },
        "Caso 6: Aerogenerador (Potencia)": {
            "vars": "eta, rho, Cp, v, A",
            "eq": "(0.5 * eta * rho * Cp * v**3 * A) / 1000",
            "data": {
                "eta": {"val": 0.9, "b_type": "Especificación (Límites, Rectangular)", "b_params": {"limite_error": 0.009}},
                "rho": {"val": 1.225, "b_type": "Especificación (Límites, Rectangular)", "b_params": {"limite_error": 0.01225}},
                "Cp": {"val": 0.453, "b_type": "Especificación (Límites, Rectangular)", "b_params": {"limite_error": 0.00453}},
                "v": {
                    "val": 10.006, 
                    "data": "9.89, 10.00, 10.05, 10.11, 10.15, 10.00, 9.85, 9.93, 9.89, 10.23",
                    "b_type": "Digital (Exactitud, Rectangular)", 
                    "b_params": {"porcentaje": 0.5, "cuentas": 5, "alcance": 45.0, "max_cuentas": 4499}
                },
                "A": {"val": 6358.5, "b_type": "Especificación (Límites, Rectangular)", "b_params": {"limite_error": 6.3585}}
            }
        }
    }

# --- 3. FUNCIONES DE SOPORTE METROLÓGICO ---
def parse_measurements(data_str):
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*e[-+]?\d+|[-+]?\d+e[-+]?\d+|[-+]?\d+", data_str)
    return np.array([float(n) for n in numbers]) if numbers else np.array([])

def calculate_type_a_stats(data_array):
    n = len(data_array)
    if n < 2: return {'n': n, 'v': 0, 'mean': np.mean(data_array) if n==1 else None, 'var': 0, 'std': 0, 'u_a': 0}
    mean, std = np.mean(data_array), np.std(data_array, ddof=1)
    return {'n': n, 'v': n-1, 'mean': mean, 'var': std**2, 'std': std, 'u_a': std/np.sqrt(n)}

def calculate_type_b(b_type, params):
    try:
        lectura = params.get("lectura", 0.0)
        if b_type == "Analógico (Clase, Rectangular)":
            err = (params.get("clase", 0.0)/100.0) * params.get("alcance", 0.0)
            return err/np.sqrt(3), f"±{err:.4g}/√3"
        elif b_type == "Digital (Exactitud, Rectangular)":
            res = params.get("alcance", 0.0) / params.get("max_cuentas", 1)
            err = (params.get("porcentaje", 0.0)/100.0) * abs(lectura) + params.get("cuentas", 0) * res
            return err/np.sqrt(3), f"±{err:.4g}/√3"
        elif b_type == "Especificación (Límites, Rectangular)":
            lim = params.get("limite_error", 0.0)
            return lim/np.sqrt(3), f"±{lim:.4g}/√3"
        elif b_type == "Estándar Conocida (u, Normal)":
            u_std = params.get("u_std", 0.0)
            return u_std, f"u={u_std:.4g} (k=1)"
        return 0.0, ""
    except: return 0.0, "Error"
# --- 4. MOTOR DE EVALUACIÓN (ENGINE) ---
def create_function(var_names, func_expr):
    """Crea una función evaluable corrigiendo sintaxis de potencia."""
    # Corrección de ^ a ** para potencias de Python
    func_expr = func_expr.replace('^', '**')
    
    # Diccionario de funciones permitidas (Seguridad + Numpy para Monte Carlo)
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
    allowed_names['np'] = np
    allowed_names['math'] = math
    
    var_str = ", ".join(var_names)
    lambda_str = f"lambda {var_str}: {func_expr}"
    
    try:
        # Evaluamos el lambda en un entorno controlado
        f = eval(lambda_str, {"__builtins__": {}}, allowed_names)
        return f, None
    except Exception as e:
        return None, f"Error en la estructura de la ecuación: {e}"

def calculate_sensitivity(f, point_dict, var_to_diff):
    """Calcula la sensibilidad (c_i) mediante diferencias finitas centrales."""
    h_base = 1e-8
    p_plus = point_dict.copy()
    p_minus = point_dict.copy()
    
    val = point_dict[var_to_diff]
    h = h_base * max(1.0, abs(val))
    
    p_plus[var_to_diff] = val + h
    p_minus[var_to_diff] = val - h
    
    try:
        # c_i = [f(x+h) - f(x-h)] / 2h
        derivative = (f(**p_plus) - f(**p_minus)) / (2.0 * h)
        return derivative, None
    except Exception as e:
        return 0.0, f"Error calculando sensibilidad para {var_to_diff}: {e}"

# --- 5. LÓGICA DE INTERFAZ Y ESTADO ---
if 'variables' not in st.session_state:
    st.session_state.variables = {}
if 'current_case' not in st.session_state:
    st.session_state.current_case = "Manual (Usuario define todo)"

# BARRA LATERAL: Selección de Casos del Proyecto
with st.sidebar:
    st.header("Casos mediciones eléctricas")
    casos_disponibles = get_project_cases()
    seleccion = st.selectbox(
        "Seleccione un caso para cargar valores:",
        list(casos_disponibles.keys()),
        index=0
    )

    # Lógica de carga automática al cambiar selección
    if seleccion != st.session_state.current_case:
        st.session_state.current_case = seleccion
        if seleccion == "Manual (Usuario define todo)":
            st.session_state.variables = {}
            st.session_state.vars_manual = "V, I, RA"
            st.session_state.eq_manual = "V * I - (I**2 * RA)"
        else:
            caso_config = casos_disponibles[seleccion]
            st.session_state.vars_manual = caso_config["vars"]
            st.session_state.eq_manual = caso_config["eq"]
            # Pre-poblar variables
            st.session_state.variables = {}
            for v_name, v_info in caso_config["data"].items():
                st.session_state.variables[v_name] = {
                    "best_estimate": v_info["val"],
                    "data_str": v_info.get("data", ""),
                    "b_type": v_info.get("b_type", "Ninguna"),
                    "b_params": v_info.get("b_params", {}),
                    "u_a": 0.0, "u_b": 0.0, "u_total": 0.0,
                    "stats_a": calculate_type_a_stats(parse_measurements(v_info.get("data", "")))
                }
        st.rerun()

    st.divider()
    st.subheader("⚙️ Configuración Global")
    v_b_eff_sidebar = st.number_input(
        "Grados de Libertad (Tipo B)", 
        min_value=1, value=50, 
        help="v=50 se asume para una confiabilidad muy alta en fuentes de catálogo."
    )
    
    # Iteraciones según el PDF (N = 100 a 1,000,000)
    n_mc_sidebar = st.select_slider(
        "Número de Iteraciones (N)",
        options=[100, 1000, 10000, 100000, 1000000],
        value=1000000,
        help="El proyecto pide evaluar la convergencia desde N=100"
    )
# --- 6. PASO 1: DEFINICIÓN DEL MENSURANDO ---
st.markdown("---")
with st.container(border=True):
    st.header("1. Definición del Mensurando")
    col_v, col_e = st.columns([1, 2])
    
    with col_v:
        # Recupera las variables del estado de sesión (cargadas por el caso o manuales)
        default_vars = st.session_state.get("vars_manual", "V, I, RA")
        var_str_input = st.text_input("Variables de entrada (separadas por coma)", value=default_vars)
        var_list = [v.strip() for v in var_str_input.split(',') if v.strip()]
        var_names = list(dict.fromkeys(var_list)) # Evita duplicados preservando el orden
    
    with col_e:
        # Recupera la ecuación del estado de sesión
        default_eq = st.session_state.get("eq_manual", "V * I - (I**2 * RA)")
        func_expr = st.text_input("Ecuación y = f(...)", value=default_eq)

# Sincronización: Asegura que cada variable en la lista tenga una entrada en la memoria
for var in var_names:
    if var not in st.session_state.variables:
        st.session_state.variables[var] = {
            "best_estimate": 0.0, "data_str": "", "b_type": "Ninguna", 
            "b_params": {}, "u_a": 0.0, "u_b": 0.0, "u_total": 0.0, 
            "b_calc_str": "", "stats_a": calculate_type_a_stats(np.array([]))
        }

# --- 7. PASO 2: FUENTES DE INCERTIDUMBRE ---
st.header("2. Fuentes de Incertidumbre (Variables de Entrada)")
st.info("Configure los parámetros para cada variable. Si cargó un caso, los valores ya están pre-poblados.")

for var in var_names:
    with st.expander(f"**Configuración de Variable: {var}**", expanded=True):
        v = st.session_state.variables[var]
        
        # Valor Central (Mejor Estimación)
        v["best_estimate"] = st.number_input(
            f"Mejor Estimación (Valor Central) para {var}", 
            value=float(v["best_estimate"]), 
            format="%.8g", 
            key=f"est_{var}"
        )
        
        tab1, tab2 = st.tabs(["📊 Incertidumbre Tipo A (Estadística)", "🤖 Incertidumbre Tipo B (Instrumento)"])
        
        with tab1:
            v["data_str"] = st.text_area(
                f"Pega mediciones para {var} (separadas por coma o espacio)", 
                value=v["data_str"], 
                key=f"data_{var}", 
                height=100
            )
            # Recalcula estadísticas Tipo A en tiempo real
            data_array = parse_measurements(v["data_str"])
            v["stats_a"] = calculate_type_a_stats(data_array)
            v["u_a"] = v["stats_a"]['u_a']
            
            if v["stats_a"]['mean'] is not None:
                if st.button(f"Usar Media ({v['stats_a']['mean']:.7g}) como mejor estimación para {var}", key=f"btn_m_{var}"):
                    v["best_estimate"] = v["stats_a"]['mean']
                    st.rerun()
            
            # Estadísticos detallados para el reporte técnico
            c1, c2, c3 = st.columns(3)
            c1.metric("Muestras (n)", v["stats_a"]["n"])
            c2.metric("Media (x̄)", f"{v['stats_a']['mean']:.7g}" if v['stats_a']['mean'] is not None else "N/A")
            c3.metric("u_A (s/√n)", f"{v['u_a']:.4e}")
            
            c4, c5 = st.columns(2)
            c4.metric("Desv. Estándar (s)", f"{v['stats_a']['std']:.6e}")
            c5.metric("Varianza (s²)", f"{v['stats_a']['var']:.6e}")

        with tab2:
            b_options = ["Ninguna", "Analógico (Clase, Rectangular)", "Digital (Exactitud, Rectangular)", 
                         "Especificación (Límites, Rectangular)", "Resistencia (±1%, Rectangular)", "Estándar Conocida (u, Normal)"]
            
            # Selecciona el tipo B basándose en el estado cargado
            current_b_idx = b_options.index(v["b_type"]) if v["b_type"] in b_options else 0
            v["b_type"] = st.selectbox(f"Fuente de Incertidumbre B para {var}", options=b_options, index=current_b_idx, key=f"sb_{var}")
            
            v["b_params"]["lectura"] = v["best_estimate"]
            bp = v["b_params"]
            
            if "Analógico" in v["b_type"]:
                col_b1, col_b2 = st.columns(2)
                bp["clase"] = col_b1.number_input("Clase de Exactitud", value=bp.get("clase", 1.5), key=f"cl_{var}")
                bp["alcance"] = col_b2.number_input("Alcance del Instrumento", value=bp.get("alcance", 10.0), key=f"al_{var}")
            elif "Digital" in v["b_type"]:
                col_b1, col_b2 = st.columns(2)
                bp["porcentaje"] = col_b1.number_input("% de Lectura", value=bp.get("porcentaje", 0.5), key=f"pct_{var}")
                bp["cuentas"] = col_b2.number_input("Número de Cuentas (d)", value=bp.get("cuentas", 2), step=1, key=f"cnt_{var}")
                col_b3, col_b4 = st.columns(2)
                bp["alcance"] = col_b3.number_input("Alcance del Rango", value=bp.get("alcance", 20.0), key=f"rg_{var}")
                bp["max_cuentas"] = col_b4.number_input("Cuentas Máximas del Rango", value=bp.get("max_cuentas", 1999), step=1, key=f"mc_{var}")
            elif "Límites" in v["b_type"]:
                bp["limite_error"] = st.number_input("Límite de Error (±)", value=bp.get("limite_error", 0.1), key=f"lim_{var}")
            elif "Resistencia" in v["b_type"]:
                bp["resistencia"] = st.number_input("Valor de Resistencia (Ω)", value=bp.get("resistencia", v["best_estimate"]), key=f"rn_{var}")
            elif "Estándar" in v["b_type"]:
                bp["u_std"] = st.number_input("Incertidumbre Estándar (u)", value=bp.get("u_std", 0.01), key=f"ustd_{var}")

            v["u_b"], v["b_calc_str"] = calculate_type_b(v["b_type"], bp)
            st.metric("Incertidumbre Tipo B (u_B)", f"{v['u_b']:.4e}")
            if v["b_calc_str"]: st.caption(v["b_calc_str"])
        
        # Incertidumbre estándar combinada de la variable individual
        v["u_total"] = math.sqrt(v["u_a"]**2 + v["u_b"]**2)
        st.divider()
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric(f"u_A({var})", f"{v['u_a']:.4e}")
        res_c2.metric(f"u_B({var})", f"{v['u_b']:.4e}")
        res_c3.metric(f"u_std_total({var})", f"{v['u_total']:.4e}")
# --- 8. EJECUCIÓN DEL ANÁLISIS (PASO 3) ---
st.markdown("---")
st.header("3. Resultados del Análisis Metrológico")

if st.button("🚀 EJECUTAR ANÁLISIS COMPLETO (GUM + MONTE CARLO)", type="primary", use_container_width=True):
    # 1. PREPARACIÓN
    f, error = create_function(var_names, func_expr)
    if error:
        st.error(f"Error en la ecuación: {error}")
        st.stop()
    
    # 2. CÁLCULO ANALÍTICO (GUM)
    point_dict = {var: st.session_state.variables[var]["best_estimate"] for var in var_names}
    try:
        y_gum = f(**point_dict)
    except Exception as e:
        st.error(f"Error al evaluar la ecuación: {e}")
        st.stop()
        
    u_c_sq = 0.0
    veff_den = 0.0
    budget_data = []
    
    for var in var_names:
        v = st.session_state.variables[var]
        c_i, _ = calculate_sensitivity(f, point_dict, var)
        
        # Contribuciones (Ley de Propagación)
        u_i_y = abs(c_i * v["u_total"])
        u_c_sq += u_i_y**2
        
        # Welch-Satterthwaite
        v_eff_var = v["stats_a"]["v"] if v["stats_a"]["v"] > 0 else v_b_eff_sidebar
        veff_den += (u_i_y**4) / v_eff_var
            
        budget_data.append({
            "Variable": var, "x_i": v["best_estimate"], "c_i": c_i,
            "u(x_i)": v["u_total"], "Varianza u²": v["u_total"]**2,
            "Contribución u_i(y)": u_i_y, "Contrib %": 0.0
        })

    u_c_gum = math.sqrt(u_c_sq)
    v_eff_gum = (u_c_gum**4) / veff_den if veff_den > 0 else float('inf')
    k_gum = get_t_factor(v_eff_gum)
    U_gum = k_gum * u_c_gum

    # 3. BUCLE DE CONVERGENCIA MONTE CARLO (N variable)
    # Valores de N según el PDF 
    n_values = [n for n in [100, 1000, 10000, 100000, 1000000] if n <= n_mc_sidebar]
    mc_results = []
    y_mc_final_dist = None # Guardamos la distribución de N=1,000,000 para el histograma

    progress_bar = st.progress(0, text="Simulando convergencia...")
    for i, N in enumerate(n_values):
        mc_samples = {}
        for var in var_names:
            v = st.session_state.variables[var]
            # Ruido Tipo A: Normal
            noise_a = np.random.normal(0, v["u_a"], N) if v["u_a"] > 0 else 0
            # Ruido Tipo B: Rectangular o Normal
            if v["u_b"] > 0:
                if "Normal" in v["b_type"] or "Estándar" in v["b_type"]:
                    noise_b = np.random.normal(0, v["u_b"], N)
                else:
                    a = v["u_b"] * np.sqrt(3) # semi-ancho
                    noise_b = np.random.uniform(-a, a, N)
            else: noise_b = 0
            mc_samples[var] = v["best_estimate"] + noise_a + noise_b

        y_samples = f(**mc_samples)
        y_mean = np.mean(y_samples)
        u_std = np.std(y_samples)
        low, high = np.percentile(y_samples, [2.275, 97.725]) # 95.45%
        U_exp_mc = (high - low) / 2
        
        mc_results.append({"N": N, "Media": y_mean, "U_exp": U_exp_mc})
        if N == n_mc_sidebar: y_mc_final_dist = y_samples
        progress_bar.progress((i + 1) / len(n_values))

    # --- 9. DESPLIEGUE DE RESULTADOS ---
    st.subheader("📊 Comparativa de Métodos")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.info("📘 **Método Analítico (GUM)**")
        st.metric("Resultado (y)", f"{y_gum:.8g}")
        st.metric("u_c combinada", f"{u_c_gum:.4e}")
        st.metric("v_eff", f"{v_eff_gum:.1f}")
        st.success(f"**U_exp = ±{U_gum:.6g}** (k={k_gum:.3f})")

    with res_col2:
        final_mc = mc_results[-1]
        st.warning(f"🎲 **Monte Carlo (N={final_mc['N']:,})**")
        st.metric("Media simulada", f"{final_mc['Media']:.8g}", delta=f"{final_mc['Media'] - y_gum:.2e} diff")
        st.metric("U_exp (95.45%)", f"{final_mc['U_exp']:.6g}", delta=f"{final_mc['U_exp'] - U_gum:.2e} diff")

    # --- 10. GRÁFICAS DEL PROYECTO ---
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("📈 Convergencia de Monte Carlo")
        # Gráfica de N vs Media [cite: 14, 20, 25, 31, 37, 43]
        df_conv = pd.DataFrame(mc_results)
        fig_conv, ax_conv = plt.subplots()
        ax_conv.semilogx(df_conv["N"], df_conv["Media"], 'bo-', label="Monte Carlo")
        ax_conv.axhline(y_gum, color='r', linestyle='--', label="GUM (Analítico)")
        ax_conv.set_xlabel("Número de simulaciones (N)")
        ax_conv.set_ylabel("Mejor estimación (y)")
        ax_conv.set_title("Estabilidad de la media vs N")
        ax_conv.legend()
        st.pyplot(fig_conv)

    with col_g2:
        st.subheader("📊 Distribución del Mensurando")
        # Histograma [cite: 15, 21, 26, 32, 38, 44]
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(y_mc_final_dist, bins=100, density=True, alpha=0.6, color='skyblue', label="Simulación")
        # Overlay Normal GUM
        x_plot = np.linspace(y_gum - 4*u_c_gum, y_gum + 4*u_c_gum, 200)
        ax_hist.plot(x_plot, stats.norm.pdf(x_plot, y_gum, u_c_gum), 'r--', lw=2, label="GUM (Teórica)")
        ax_hist.set_title(f"Histograma de estimaciones (N={n_mc_sidebar:,})")
        ax_hist.legend()
        st.pyplot(fig_hist)

    # --- 11. TABLA DE PRESUPUESTO Y EXPORTACIÓN ---
    st.subheader("📐 Presupuesto de Incertidumbre Detallado")
    df_budget = pd.DataFrame(budget_data)
    df_budget["Contrib %"] = (df_budget["Contribución u_i(y)"]**2 / u_c_sq) * 100
    
    # Formateo científico para reporte técnico
    st.table(df_budget.style.format({
        "x_i": "{:.4g}", "c_i": "{:.4e}", "u(x_i)": "{:.4e}", 
        "Varianza u²": "{:.4e}", "Contribución u_i(y)": "{:.4e}", "Contrib %": "{:.2f}%"
    }))

    # Función para Excel (Definida localmente o al inicio)
    def to_excel_pro(df_b, df_mc):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_b.to_excel(writer, index=False, sheet_name="Presupuesto GUM")
            df_mc.to_excel(writer, index=False, sheet_name="Convergencia Monte Carlo")
        return output.getvalue()

    st.download_button(
        label="📥 Descargar Reporte Técnico (Excel)",
        data=to_excel_pro(df_budget, df_conv),
        file_name=f"Reporte_Metrologia_{st.session_state.current_case.replace(':', '').replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

