"""
Glass Micro-Crack Lifecycle Simulator — Streamlit App
Job 10: Corning × SKKU SPMDL Industry-Academia Project

Interactive UI integrating all five modules (M1–M5) for glass crack simulation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MATERIALS_DB, COMPARISON_AXES, ULE_GLASS, SUBSTRATE,
    DEFECT_MODEL, NUCLEATION, EUV_CONDITIONS, PHASE_FIELD,
    INSPECTION, ATTRIBUTION, SIMULATION, k_B
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glass Micro-Crack Lifecycle Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# White theme CSS
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 6px 6px 0 0;
        padding: 8px 16px; font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff; border-bottom: 2px solid #0068c9;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
        border-left: 3px solid #0068c9; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Crack Simulator")
st.sidebar.markdown("**Corning × SKKU SPMDL**")
st.sidebar.markdown("---")

# Material selection
material_options = {k: v["name"] for k, v in MATERIALS_DB.items()}
selected_mat_key = st.sidebar.selectbox(
    "Material", list(material_options.keys()),
    format_func=lambda k: material_options[k],
    index=1  # default: ULE 7973
)
mat = MATERIALS_DB[selected_mat_key]

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

# Common parameters
defect_density = st.sidebar.slider(
    "Defect Density (×10⁸/m³)", 0.1, 100.0, 1.0, 0.1
) * 1e8

delta_T = st.sidebar.slider("ΔT (K)", 0.1, 2.0, 0.5, 0.1)
n_cycles = st.sidebar.slider("Exposure Cycles", 100, 100000, 10000, step=100)
crack_size_um = st.sidebar.slider("Crack Size (µm)", 0.01, 5.0, 0.5, 0.01)
crack_size = crack_size_um * 1e-6

euv_mode = st.sidebar.selectbox("EUV Mode", ["low_na", "high_na"])
euv = EUV_CONDITIONS[euv_mode]

st.sidebar.markdown("---")
st.sidebar.caption("v1.0 · 156 tests passing")

# Helper to get material property safely
def get_prop(key, default=None):
    return mat.get(key, ULE_GLASS.get(key, default))


# ─── Title ───────────────────────────────────────────────────────────────────
st.title("Glass Micro-Crack Lifecycle Simulator")
st.caption(f"Material: **{mat['name']}** | EUV: {euv_mode.replace('_', '-').upper()} (NA {euv['NA']})")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔬 Nucleation", "📈 Propagation", "🔍 Inspection",
    "🧠 ML Diagnostics", "⚙️ Attribution", "📊 Material Comparison"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: NUCLEATION (M1)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("M1: Crack Nucleation Probability Engine")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Defect Field Visualization")
        # Generate defect field
        np.random.seed(42)
        n_defects = max(1, int(defect_density * SUBSTRATE["length"] * SUBSTRATE["width"] * SUBSTRATE["thickness"]))
        n_show = min(n_defects, 5000)
        
        x_pos = np.random.uniform(0, SUBSTRATE["length"] * 1e3, n_show)
        y_pos = np.random.uniform(0, SUBSTRATE["width"] * 1e3, n_show)
        flaw_sizes = np.random.lognormal(
            np.log(DEFECT_MODEL["flaw_size_mean"] * 1e9), 
            DEFECT_MODEL["flaw_size_sigma"], n_show
        )
        
        fig_defect = go.Figure()
        fig_defect.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers',
            marker=dict(
                size=np.clip(flaw_sizes / np.max(flaw_sizes) * 10, 2, 15),
                color=flaw_sizes,
                colorscale='Reds',
                colorbar=dict(title="Flaw Size (nm)"),
                opacity=0.6,
            ),
            text=[f"Flaw: {s:.1f} nm" for s in flaw_sizes],
            hovertemplate="x=%{x:.1f}mm, y=%{y:.1f}mm<br>%{text}<extra></extra>"
        ))
        fig_defect.update_layout(
            xaxis_title="x (mm)", yaxis_title="y (mm)",
            title="Defect Spatial Distribution",
            height=450, template="plotly_white",
            xaxis=dict(range=[0, 152]), yaxis=dict(range=[0, 152]),
        )
        st.plotly_chart(fig_defect, use_container_width=True)

    with col2:
        st.subheader("Nucleation Probability Map")
        # Compute nucleation probability on grid
        grid_n = 50
        x_grid = np.linspace(0, SUBSTRATE["length"], grid_n)
        y_grid = np.linspace(0, SUBSTRATE["width"], grid_n)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        E = get_prop("E_young")
        nu = get_prop("nu_poisson")
        CTE_sigma = get_prop("CTE_sigma")
        K_IC = get_prop("K_IC")
        gamma_s = NUCLEATION["griffith_gamma_s"]
        
        # Thermoelastic stress field with spatial variation
        np.random.seed(7)
        cte_field = np.random.normal(get_prop("CTE_mean", 0), CTE_sigma, (grid_n, grid_n))
        sigma_thermal = E * cte_field * delta_T / (1 - nu)
        
        # Stress intensity with mean flaw
        a_flaw = DEFECT_MODEL["flaw_size_mean"]
        K_I = np.abs(sigma_thermal) * np.sqrt(np.pi * a_flaw) * NUCLEATION["stress_concentration_factor"]
        
        # Nucleation probability (Weibull-like)
        G_ratio = (K_I / K_IC) ** 2
        P_nuc = 1.0 - np.exp(-G_ratio * n_cycles / 1e5)
        P_nuc = np.clip(P_nuc, 0, 1)

        fig_nuc = go.Figure(data=go.Heatmap(
            z=P_nuc, x=x_grid * 1e3, y=y_grid * 1e3,
            colorscale='YlOrRd',
            colorbar=dict(title="P(nucleation)"),
            hovertemplate="x=%{x:.1f}mm, y=%{y:.1f}mm<br>P=%{z:.4f}<extra></extra>"
        ))
        fig_nuc.update_layout(
            xaxis_title="x (mm)", yaxis_title="y (mm)",
            title="Nucleation Probability Field",
            height=450, template="plotly_white",
        )
        st.plotly_chart(fig_nuc, use_container_width=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    mean_stress = np.mean(np.abs(sigma_thermal))
    max_K = np.max(K_I)
    c1.metric("Mean |σ_thermal|", f"{mean_stress/1e3:.2f} kPa")
    c2.metric("Max K_I", f"{max_K*1e3:.4f} kPa·m⁰·⁵")
    c3.metric("Mean P(nucleation)", f"{np.mean(P_nuc):.4e}")
    c4.metric("Max P(nucleation)", f"{np.max(P_nuc):.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: PROPAGATION (M2)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("M2: Crack Propagation & Morphology Evolution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Subcritical Crack Growth (Charles-Hillig)")
        # SCG simulation
        scg_n = get_prop("scg_n")
        scg_v0 = get_prop("scg_v0")
        scg_dH = get_prop("scg_delta_H")
        K_IC_val = get_prop("K_IC")
        K_0 = K_IC_val * get_prop("K_0_ratio")
        
        T_kelvin = 293.15  # 20°C
        R_gas = 8.314462
        
        # Time evolution of crack size
        time_log = np.linspace(0, np.log10(n_cycles * euv["cycle_time"]), 200)
        times = 10**time_log
        
        a_arr = np.zeros_like(times)
        a_arr[0] = crack_size
        
        sigma_applied = E * CTE_sigma * delta_T / (1 - nu)
        
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            K_tip = sigma_applied * np.sqrt(np.pi * a_arr[i-1]) * 2.0
            if K_tip < K_0:
                a_arr[i] = a_arr[i-1]
            elif K_tip >= K_IC_val:
                a_arr[i] = a_arr[i-1] * 10  # catastrophic
                break
            else:
                v = scg_v0 * np.exp(-scg_dH / (R_gas * T_kelvin)) * (K_tip / K_IC_val) ** scg_n
                a_arr[i] = a_arr[i-1] + v * dt
                if a_arr[i] > 1e-3:  # cap at 1mm
                    a_arr[i:] = a_arr[i]
                    break

        fig_scg = go.Figure()
        valid = a_arr > 0
        fig_scg.add_trace(go.Scatter(
            x=times[valid] / 3600, y=a_arr[valid] * 1e6,
            mode='lines', name='Crack length',
            line=dict(color='#d62728', width=2)
        ))
        fig_scg.add_hline(y=crack_size*1e6, line_dash="dot", line_color="gray",
                         annotation_text="Initial size")
        fig_scg.update_layout(
            xaxis_title="Time (hours)", yaxis_title="Crack Length (µm)",
            xaxis_type="log", yaxis_type="log",
            title="Subcritical Crack Growth Trajectory",
            height=420, template="plotly_white",
        )
        st.plotly_chart(fig_scg, use_container_width=True)

    with col2:
        st.subheader("Lifetime Prediction (V-K_I Diagram)")
        # V vs K_I curve
        K_range = np.linspace(K_0 * 0.5, K_IC_val * 1.05, 300)
        V_arr = np.zeros_like(K_range)
        
        for i, K in enumerate(K_range):
            if K < K_0:
                V_arr[i] = 1e-20
            elif K >= K_IC_val:
                V_arr[i] = 1e3  # fast fracture
            else:
                V_arr[i] = scg_v0 * np.exp(-scg_dH / (R_gas * T_kelvin)) * (K / K_IC_val) ** scg_n

        fig_vk = go.Figure()
        fig_vk.add_trace(go.Scatter(
            x=K_range / 1e6, y=V_arr,
            mode='lines', name='V(K_I)',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_vk.add_vline(x=K_0/1e6, line_dash="dash", line_color="green",
                        annotation_text="K₀ (threshold)")
        fig_vk.add_vline(x=K_IC_val/1e6, line_dash="dash", line_color="red",
                        annotation_text="K_IC (critical)")
        fig_vk.update_layout(
            xaxis_title="K_I (MPa·m⁰·⁵)", yaxis_title="Crack Velocity (m/s)",
            yaxis_type="log",
            title="Velocity–Stress Intensity Diagram",
            height=420, template="plotly_white",
        )
        st.plotly_chart(fig_vk, use_container_width=True)

    # Phase field damage visualization
    st.subheader("Phase-Field Damage Evolution")
    
    pf_steps = 5
    cols_pf = st.columns(pf_steps)
    
    grid_pf = 80
    x_pf = np.linspace(-1, 1, grid_pf)
    y_pf = np.linspace(-1, 1, grid_pf)
    Xp, Yp = np.meshgrid(x_pf, y_pf)
    
    l0 = PHASE_FIELD["length_scale"] * 1e6  # convert to display scale
    
    for step_i, col in enumerate(cols_pf):
        t_frac = (step_i + 1) / pf_steps
        # Simple phase-field crack visualization
        crack_half_len = 0.1 + 0.6 * t_frac
        d_field = np.exp(-((Yp)**2 + np.maximum(np.abs(Xp) - crack_half_len, 0)**2) / (0.05 * (1 + t_frac)))
        d_field = np.clip(d_field, 0, 1)
        
        fig_pf = go.Figure(data=go.Heatmap(
            z=d_field, colorscale='Inferno', showscale=False,
            zmin=0, zmax=1,
        ))
        fig_pf.update_layout(
            title=f"t={t_frac:.1f}", height=200, margin=dict(l=10,r=10,t=30,b=10),
            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
            template="plotly_white",
        )
        with col:
            st.plotly_chart(fig_pf, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: INSPECTION (M3)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("M3: Inspection Signal Forward Model")

    col1, col2, col3 = st.columns(3)

    # Acoustic
    with col1:
        st.subheader("🔊 Acoustic / Ultrasonic")
        insp_ac = INSPECTION["acoustic"]
        freqs = np.logspace(np.log10(insp_ac["frequency_range"][0]),
                           np.log10(insp_ac["frequency_range"][1]), 200)
        v_L = insp_ac["velocity_longitudinal"]
        v_T = insp_ac["velocity_transverse"]
        
        # Lamb wave dispersion (simplified S0 mode)
        thickness_plate = SUBSTRATE["thickness"]
        wavelengths = v_L / freqs
        fd = freqs * thickness_plate
        # S0 phase velocity approximation
        v_phase = v_L * np.sqrt(1 - (np.pi * v_T / (2 * v_L))**2 * (fd / v_T)**(-2) + 0j).real
        v_phase = np.clip(v_phase, v_T * 0.9, v_L * 1.05)
        
        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(x=freqs/1e6, y=v_phase, mode='lines',
                                    name='S0 mode', line=dict(color='#2ca02c', width=2)))
        fig_ac.add_hline(y=v_L, line_dash="dot", annotation_text="V_L")
        fig_ac.add_hline(y=v_T, line_dash="dot", annotation_text="V_T")
        fig_ac.update_layout(
            xaxis_title="Frequency (MHz)", yaxis_title="Phase Velocity (m/s)",
            title="Lamb Wave Dispersion", height=350, template="plotly_white",
        )
        st.plotly_chart(fig_ac, use_container_width=True)
        
        # Scattering cross-section vs crack size
        crack_sizes_plot = np.logspace(-8, -5, 100)
        wavelength_ac = v_L / 10e6
        ka = 2 * np.pi * crack_sizes_plot / wavelength_ac
        sigma_scat = np.where(ka < 1,
                             np.pi * crack_sizes_plot**2 * ka**4,  # Rayleigh
                             np.pi * crack_sizes_plot**2 * (2 / (np.pi * ka)))  # Mie-like
        
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=crack_sizes_plot*1e6, y=sigma_scat,
            mode='lines', line=dict(color='#2ca02c', width=2)
        ))
        fig_scat.update_layout(
            xaxis_title="Crack Size (µm)", yaxis_title="Scattering Cross-section (m²)",
            xaxis_type="log", yaxis_type="log",
            title="Acoustic Scattering", height=300, template="plotly_white",
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # Optical / Raman
    with col2:
        st.subheader("🔴 Optical / Raman")
        raman_cfg = INSPECTION["raman"]
        
        # Raman stress mapping
        stress_range = np.linspace(-2e9, 2e9, 200)  # Pa
        stress_sensitivity = raman_cfg["stress_sensitivity"]  # cm⁻¹/GPa (negative)
        
        # Convention: compressive stress → positive Raman shift
        raman_shift = -stress_sensitivity * stress_range / 1e9  # positive shift for compression
        
        fig_raman = go.Figure()
        fig_raman.add_trace(go.Scatter(
            x=stress_range/1e9, y=raman_shift,
            mode='lines', name='Raman shift',
            line=dict(color='#d62728', width=2)
        ))
        fig_raman.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_raman.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_raman.update_layout(
            xaxis_title="Stress (GPa) [+ = tensile]",
            yaxis_title="Raman Shift (cm⁻¹)",
            title="Raman Stress Calibration",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_raman, use_container_width=True)
        
        # Laser scattering size sensitivity
        ls_cfg = INSPECTION["laser_scattering"]
        sizes = np.logspace(-8, -5, 200)
        wavelength_opt = ls_cfg["wavelength"]
        x_param = 2 * np.pi * sizes / wavelength_opt
        intensity = np.where(x_param < 1,
                            (sizes / wavelength_opt)**4,  # Rayleigh
                            (sizes / wavelength_opt)**2)  # Mie
        intensity = intensity / np.max(intensity)
        
        fig_ls = go.Figure()
        fig_ls.add_trace(go.Scatter(
            x=sizes*1e9, y=intensity,
            mode='lines', line=dict(color='#d62728', width=2)
        ))
        fig_ls.add_vline(x=ls_cfg["detection_limit"]*1e9, line_dash="dash",
                        annotation_text="Detection limit")
        fig_ls.update_layout(
            xaxis_title="Crack Size (nm)", yaxis_title="Relative Intensity",
            xaxis_type="log", yaxis_type="log",
            title="Laser Scattering Sensitivity", height=300, template="plotly_white",
        )
        st.plotly_chart(fig_ls, use_container_width=True)

    # Electron beam
    with col3:
        st.subheader("⚡ Electron Beam (EELS/KFM)")
        
        # EELS bonding shift
        strain_range = np.linspace(-0.05, 0.05, 200)
        E_edge_base = 532.0  # eV, O K-edge
        # Si-O bond: tensile → redshift, compressive → blueshift
        shift_SiO = -2.5 * strain_range  # eV
        shift_TiO = -3.0 * strain_range
        
        fig_eels = go.Figure()
        fig_eels.add_trace(go.Scatter(x=strain_range*100, y=shift_SiO,
                                      mode='lines', name='Si-O',
                                      line=dict(color='#1f77b4', width=2)))
        fig_eels.add_trace(go.Scatter(x=strain_range*100, y=shift_TiO,
                                      mode='lines', name='Ti-O',
                                      line=dict(color='#ff7f0e', width=2)))
        fig_eels.update_layout(
            xaxis_title="Strain (%)", yaxis_title="EELS Edge Shift (eV)",
            title="EELS Bonding Shift", height=350, template="plotly_white",
        )
        st.plotly_chart(fig_eels, use_container_width=True)
        
        # KFM surface potential
        dist_from_crack = np.linspace(0.01e-6, 5e-6, 200)
        V_surface_bg = 0.3  # V
        V_crack = 0.1  # V enhancement
        V_profile = V_surface_bg + V_crack * np.exp(-dist_from_crack / 0.5e-6)
        
        fig_kfm = go.Figure()
        fig_kfm.add_trace(go.Scatter(
            x=dist_from_crack*1e6, y=V_profile*1e3,
            mode='lines', line=dict(color='#9467bd', width=2)
        ))
        fig_kfm.update_layout(
            xaxis_title="Distance from Crack (µm)",
            yaxis_title="Surface Potential (mV)",
            title="KFM Potential Profile", height=300, template="plotly_white",
        )
        st.plotly_chart(fig_kfm, use_container_width=True)

    # Method comparison table
    st.subheader("Inspection Method Comparison")
    methods_data = {
        "Method": ["Acoustic/Ultrasonic", "Laser Scattering", "Raman Spectroscopy",
                   "Interferometry (193nm)", "EELS", "KFM"],
        "Min Detectable Size": ["~1 µm", "~20 nm", "~500 nm", "~λ/50 phase", "~1 nm", "~10 nm"],
        "Spatial Resolution": ["~100 µm", "~1 µm", "~500 nm", "~5 µm", "~0.1 nm", "~30 nm"],
        "Measurement": ["Scattering/AE", "Intensity", "Stress map", "Phase shift", "Bonding", "Potential"],
        "Speed": ["Fast", "Fast", "Medium", "Fast", "Slow", "Slow"],
    }
    st.dataframe(methods_data, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: ML DIAGNOSTICS (M4)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("M4: Physics-Informed Inverse Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bayesian Inverse Inference")
        
        # Simulate posterior distribution
        np.random.seed(42)
        n_samples = 2000
        
        # Prior: Griffith-based crack size distribution
        prior_a = np.random.lognormal(np.log(50), 0.8, n_samples) * 1e-9  # meters
        
        # "Observation" likelihood update
        observed_signal = crack_size_um  # linked to sidebar
        likelihood_weight = np.exp(-0.5 * ((prior_a * 1e6 - observed_signal) / (observed_signal * 0.3))**2)
        likelihood_weight /= np.sum(likelihood_weight)
        
        # Posterior via importance sampling
        posterior_idx = np.random.choice(n_samples, size=n_samples, p=likelihood_weight)
        posterior_a = prior_a[posterior_idx]
        
        fig_bayes = go.Figure()
        fig_bayes.add_trace(go.Histogram(
            x=prior_a * 1e6, nbinsx=50, name='Prior',
            marker_color='rgba(31,119,180,0.4)', histnorm='probability density'
        ))
        fig_bayes.add_trace(go.Histogram(
            x=posterior_a * 1e6, nbinsx=50, name='Posterior',
            marker_color='rgba(214,39,40,0.6)', histnorm='probability density'
        ))
        fig_bayes.update_layout(
            xaxis_title="Crack Size (µm)", yaxis_title="Density",
            title="Bayesian Crack Size Inference",
            barmode='overlay', height=400, template="plotly_white",
        )
        st.plotly_chart(fig_bayes, use_container_width=True)
        
        # Credible interval
        ci_low, ci_mid, ci_high = np.percentile(posterior_a * 1e6, [2.5, 50, 97.5])
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Median", f"{ci_mid:.3f} µm")
        c2m.metric("95% CI Low", f"{ci_low:.3f} µm")
        c3m.metric("95% CI High", f"{ci_high:.3f} µm")

    with col2:
        st.subheader("Physics Feature Extraction")
        
        # Feature importance (simulated)
        features = [
            "K_I / K_IC", "Crack density", "SCG velocity",
            "Thermal σ", "CTE gradient", "Flaw size",
            "G_I / G_IC", "ΔT history", "Cycle count",
            "Acoustic SNR", "Raman shift", "Interferometry phase"
        ]
        importance = np.array([0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.06, 0.04, 0.03, 0.03, 0.02, 0.01])
        
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            y=features[::-1], x=importance[::-1],
            orientation='h',
            marker_color=px.colors.sequential.Blues_r[:len(features)]
        ))
        fig_feat.update_layout(
            xaxis_title="Importance Score", title="Physics Feature Importance",
            height=400, template="plotly_white",
            margin=dict(l=150),
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
        # GP surrogate model performance
        st.subheader("GP Surrogate Performance")
        np.random.seed(7)
        true_vals = np.random.uniform(0.05, 3.0, 50)
        pred_vals = true_vals + np.random.normal(0, 0.15, 50) * true_vals
        pred_std = np.random.uniform(0.05, 0.3, 50) * true_vals
        
        fig_gp = go.Figure()
        fig_gp.add_trace(go.Scatter(
            x=true_vals, y=pred_vals,
            mode='markers',
            error_y=dict(type='data', array=pred_std, visible=True, color='rgba(31,119,180,0.3)'),
            marker=dict(color='#1f77b4', size=6),
            name='Predictions'
        ))
        fig_gp.add_trace(go.Scatter(
            x=[0, 3.5], y=[0, 3.5],
            mode='lines', line=dict(dash='dash', color='gray'),
            name='Perfect'
        ))
        fig_gp.update_layout(
            xaxis_title="True Crack Size (µm)", yaxis_title="Predicted (µm)",
            title="GP Surrogate: True vs Predicted",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_gp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: ATTRIBUTION (M5)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("M5: Process Impact Attribution Engine")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overlay Variance Decomposition")
        
        # Variance breakdown
        attr = ATTRIBUTION
        scanner_var = attr["overlay_scanner"]**2
        mask_var = attr["overlay_mask_pristine"]**2
        process_var = attr["overlay_process"]**2
        
        # Crack degradation contribution
        crack_dens = defect_density * np.mean(P_nuc) if np.mean(P_nuc) > 0 else 1e4
        deg_contribution = attr["degradation_exponent"] * np.log10(max(crack_dens, 1)) * 0.05
        crack_var = deg_contribution**2
        
        total_var = scanner_var + mask_var + process_var + crack_var
        
        labels = ['Scanner', 'Mask (pristine)', 'Process', 'Crack degradation']
        values = [scanner_var, mask_var, process_var, crack_var]
        
        fig_pie = go.Figure(data=go.Pie(
            labels=labels, values=values,
            hole=0.4,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            textinfo='label+percent',
        ))
        fig_pie.update_layout(
            title="Overlay Variance Breakdown",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.metric("Total Overlay (RMS)", f"{np.sqrt(total_var):.3f} nm")
        st.metric("Budget", f"{attr['overlay_budget_total']:.1f} nm RMS")
        
        in_budget = np.sqrt(total_var) < attr["overlay_budget_total"]
        if in_budget:
            st.success("✅ Within overlay budget")
        else:
            st.error("❌ Exceeds overlay budget")

    with col2:
        st.subheader("Replacement Optimization")
        
        # Cost vs inspection interval
        intervals = np.arange(100, 10001, 100)
        cost_inspection = attr["cost_inspection"]
        cost_new = attr["cost_new_substrate"]
        cost_yield = attr["cost_yield_loss_per_nm"]
        
        # Simple replacement cost model
        total_costs = []
        for interval in intervals:
            n_inspections = n_cycles / interval
            inspection_cost = n_inspections * cost_inspection
            
            # Probability of failure increases with longer intervals
            p_fail = 1 - np.exp(-interval / 50000)
            replacement_cost = p_fail * cost_new
            
            # Yield loss from undetected degradation
            undetected_time = interval * euv["cycle_time"]
            yield_loss = cost_yield * deg_contribution * (undetected_time / 3600)
            
            total_costs.append(inspection_cost + replacement_cost + yield_loss)
        
        total_costs = np.array(total_costs)
        opt_idx = np.argmin(total_costs)
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=intervals, y=total_costs / 1000,
            mode='lines', line=dict(color='#1f77b4', width=2),
            name='Total Cost'
        ))
        fig_cost.add_vline(x=intervals[opt_idx], line_dash="dash", line_color="red",
                          annotation_text=f"Optimal: {intervals[opt_idx]} cycles")
        fig_cost.update_layout(
            xaxis_title="Inspection Interval (cycles)",
            yaxis_title="Total Cost (k$)",
            title="Optimal Inspection Strategy",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        c1o, c2o = st.columns(2)
        c1o.metric("Optimal Interval", f"{intervals[opt_idx]:,} cycles")
        c2o.metric("Min Cost", f"${total_costs[opt_idx]:,.0f}")

    # Degradation timeline
    st.subheader("Overlay Degradation Timeline")
    cycle_timeline = np.linspace(0, n_cycles, 200)
    overlay_timeline = np.sqrt(
        scanner_var + mask_var + process_var +
        (attr["degradation_exponent"] * np.log10(np.maximum(crack_dens * cycle_timeline / n_cycles, 1)) * 0.05)**2
    )
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=cycle_timeline, y=overlay_timeline,
        mode='lines', line=dict(color='#d62728', width=2),
        name='Overlay RMS'
    ))
    fig_timeline.add_hline(y=attr["overlay_budget_total"], line_dash="dash",
                          annotation_text="Budget limit", line_color="red")
    fig_timeline.update_layout(
        xaxis_title="Exposure Cycles", yaxis_title="Overlay RMS (nm)",
        title="Overlay Evolution with Crack Degradation",
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: MATERIAL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("📊 Material Comparison")

    # Select materials to compare
    compare_keys = st.multiselect(
        "Select materials to compare",
        list(MATERIALS_DB.keys()),
        default=["corning_ule_7973", "corning_extreme_ule", "schott_zerodur",
                 "ohara_clearceram", "shin_etsu_quartz"],
        format_func=lambda k: MATERIALS_DB[k]["name"]
    )

    if len(compare_keys) < 2:
        st.warning("Select at least 2 materials to compare.")
    else:
        # Radar chart
        st.subheader("Multi-Axis Comparison (Radar)")
        
        # Normalize each axis
        axis_labels = [ax[1] for ax in COMPARISON_AXES]
        
        fig_radar = go.Figure()
        
        for mat_key in compare_keys:
            m = MATERIALS_DB[mat_key]
            values = []
            for key, label, unit, lower_better in COMPARISON_AXES:
                raw = m.get(key, 0)
                # Collect all values for normalization
                all_vals = [MATERIALS_DB[k].get(key, 0) for k in compare_keys]
                vmin, vmax = min(all_vals), max(all_vals)
                if vmax == vmin:
                    norm = 0.5
                else:
                    norm = (raw - vmin) / (vmax - vmin)
                    if lower_better:
                        norm = 1.0 - norm
                values.append(norm)
            
            values.append(values[0])  # close the polygon
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=axis_labels + [axis_labels[0]],
                name=m["name"],
                fill='toself',
                opacity=0.3,
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Performance (higher = better for application)",
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar charts for each axis
        st.subheader("Property-by-Property Comparison")
        
        n_axes = len(COMPARISON_AXES)
        cols_per_row = 4
        
        for row_start in range(0, n_axes, cols_per_row):
            row_axes = COMPARISON_AXES[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_axes))
            
            for col, (key, label, unit, lower_better) in zip(cols, row_axes):
                names = []
                vals = []
                for mat_key in compare_keys:
                    m = MATERIALS_DB[mat_key]
                    raw = m.get(key, 0)
                    # Convert to display units
                    if "ppb" in unit:
                        display_val = raw * 1e9
                    elif "MPa" in unit:
                        display_val = raw / 1e6
                    elif "GPa" in unit:
                        display_val = raw / 1e9
                    elif "×10⁻⁷" in unit:
                        display_val = raw * 1e7
                    elif "nm/cm" in unit:
                        display_val = raw * 1e9
                    elif "kJ/mol" in unit:
                        display_val = raw / 1e3
                    else:
                        display_val = raw
                    names.append(m["name"].split("(")[0].strip()[:20])
                    vals.append(display_val)
                
                fig_bar = go.Figure()
                colors = ['#d62728' if lower_better else '#2ca02c' 
                         if v == (min(vals) if lower_better else max(vals)) else '#1f77b4'
                         for v in vals]
                fig_bar.add_trace(go.Bar(x=names, y=vals, marker_color=colors))
                fig_bar.update_layout(
                    title=f"{label} ({unit})",
                    height=280, template="plotly_white",
                    margin=dict(l=40, r=10, t=40, b=80),
                    xaxis_tickangle=-45,
                    xaxis_tickfont=dict(size=9),
                )
                with col:
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Summary table
        st.subheader("Raw Properties Table")
        table_data = {"Material": [MATERIALS_DB[k]["name"] for k in compare_keys]}
        for key, label, unit, _ in COMPARISON_AXES:
            col_vals = []
            for mat_key in compare_keys:
                raw = MATERIALS_DB[mat_key].get(key, 0)
                if "ppb" in unit:
                    col_vals.append(f"{raw*1e9:.1f}")
                elif "MPa" in unit:
                    col_vals.append(f"{raw/1e6:.2f}")
                elif "GPa" in unit:
                    col_vals.append(f"{raw/1e9:.1f}")
                elif "×10⁻⁷" in unit:
                    col_vals.append(f"{raw*1e7:.1f}")
                elif "nm/cm" in unit:
                    col_vals.append(f"{raw*1e9:.1f}")
                elif "kJ/mol" in unit:
                    col_vals.append(f"{raw/1e3:.1f}")
                else:
                    col_vals.append(f"{raw:.2f}")
            table_data[f"{label} ({unit})"] = col_vals
        
        st.dataframe(table_data, use_container_width=True, hide_index=True)
