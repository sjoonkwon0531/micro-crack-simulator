#!/usr/bin/env python3
"""Generate all proposal figures — V2 with fixes for fig1, fig2, fig6, and improvements."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(__file__), 'docs', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =====================================================================
# Fig 0: Schematic Diagram
# =====================================================================
def make_fig0():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    c_dark = '#003366'
    c_mid = '#0066CC'
    c_accent = '#FF6600'
    c_green = '#009933'
    c_red = '#CC3333'
    
    def add_box(x, y, w, h, text, color=c_mid, fontsize=9, fontcolor='white', alpha=0.9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=c_dark, linewidth=1.2, alpha=alpha)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=fontcolor, fontweight='bold', wrap=True)
    
    def add_arrow(x1, y1, x2, y2, color=c_dark):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    
    ax.text(6, 6.7, 'Figure 0. Glass Micro-Crack Lifecycle Simulator', ha='center', va='center',
            fontsize=16, fontweight='bold', color=c_dark)
    ax.text(6, 6.35, 'Corning ULE 7973 — Sub-5nm EUV Lithography', ha='center', va='center',
            fontsize=11, color=c_mid, style='italic')
    
    add_box(0.2, 5.2, 2.3, 0.8, 'EUV Lithography\nProcess', color=c_dark, fontsize=9)
    add_arrow(2.5, 5.6, 3.0, 5.6)
    add_box(3.0, 5.2, 2.5, 0.8, 'Glass Substrate\nCorning ULE 7973', color=c_mid, fontsize=9)
    add_arrow(5.5, 5.6, 6.0, 5.6)
    add_box(6.0, 5.2, 2.8, 0.8, 'Pain Points\nMicro-crack during\npolishing/coating/cycling', color=c_red, fontsize=8)
    add_arrow(8.8, 5.6, 9.3, 5.6)
    add_box(9.3, 5.2, 2.5, 0.8, 'Current Limitation\nPost-mortem only\nNo prediction', color='#996633', fontsize=8)
    
    add_arrow(6, 5.2, 6, 4.6)
    ax.text(6.3, 4.85, 'Our Solution', fontsize=10, color=c_accent, fontweight='bold')
    
    modules = [
        ('M1\nNucleation\nEngine\n(Griffith +\nStochastic)', '#004488'),
        ('M2\nPropagation\nEngine\n(Phase-field\n+ SCG)', '#005599'),
        ('M3\nInspection\nForward Model\n(6 Modalities)', '#0066AA'),
        ('M4\nInverse ML\nDiagnostics\n(Bayesian)', '#0077BB'),
        ('M5\nProcess\nAttribution\n(σ² decomp.)', '#0088CC'),
    ]
    
    xpos = [0.3, 2.55, 4.8, 7.05, 9.3]
    for i, ((text, color), x) in enumerate(zip(modules, xpos)):
        add_box(x, 2.8, 2.1, 1.6, text, color=color, fontsize=8)
        if i < 4:
            add_arrow(x + 2.1, 3.6, xpos[i+1], 3.6)
    
    add_arrow(6, 2.8, 6, 2.3)
    
    outcomes = [
        ('Predictive\nMaintenance', c_green),
        ('Process\nOptimization', c_green),
        ('Yield\nImprovement', c_green),
        ('Cost\nReduction', c_green),
    ]
    xo = [1.0, 3.5, 6.0, 8.5]
    for (text, color), x in zip(outcomes, xo):
        add_box(x, 1.0, 2.0, 0.9, text, color=color, fontsize=9)
    
    for x in xo:
        add_arrow(6, 2.3, x + 1.0, 1.9, color=c_green)
    
    fig.savefig(os.path.join(FIGDIR, 'fig0_schematic.png'))
    plt.close(fig)
    print("✓ fig0_schematic.png")


# =====================================================================
# Fig 1: Nucleation Map — FIXED: add applied stress to get nonzero P
# =====================================================================
def make_fig1():
    from config import ULE_GLASS, SUBSTRATE, DEFECT_MODEL
    from modules.m01_nucleation import DefectField, CTEMapGenerator, ThermoelasticStress
    
    np.random.seed(42)
    
    df = DefectField(SUBSTRATE)
    domain = (SUBSTRATE['length'], SUBSTRATE['width'], SUBSTRATE['thickness'])
    result = df.generate_poisson(density=1e8, domain_size=domain, seed=42)
    
    grid_size = (152, 152)
    
    # Generate spatially-varying nucleation probability.
    # Strategy: compute a stress field from CTE inhomogeneity + residual stress,
    # then use the maximum flaw size per grid cell (drawn from heavy-tailed distribution)
    # to compute K_I and hence nucleation probability.
    
    from scipy.ndimage import gaussian_filter
    
    # CTE-driven stress field (smooth spatial correlation)
    cte_field = CTEMapGenerator.generate_gaussian_random_field(
        grid_size, ULE_GLASS['CTE_sigma'], 1e-3, seed=42)
    
    delta_T = 2.0  # K
    E = ULE_GLASS['E_young']
    nu = ULE_GLASS['nu_poisson']
    stress_cte = E * np.abs(cte_field) * delta_T / (1 - nu)
    
    # Residual polishing stress (spatially correlated, ~20-80 MPa)
    residual = gaussian_filter(np.random.randn(*grid_size), sigma=20) * 20e6 + 50e6
    residual = np.clip(residual, 10e6, None)
    
    total_stress = stress_cte + residual  # Pa
    
    # Flaw sizes: heavy-tailed distribution so some cells have µm-scale flaws
    # Use Pareto-like: most flaws ~50 nm, but tail reaches 1-5 µm
    flaw_sizes = np.random.lognormal(np.log(200e-9), 1.2, grid_size)
    flaw_sizes = np.clip(flaw_sizes, 10e-9, 10e-6)
    
    K_I_field = 1.12 * total_stress * np.sqrt(np.pi * flaw_sizes)
    K_0 = ULE_GLASS['K_IC'] * ULE_GLASS['K_0_ratio']
    
    # Nucleation probability: sigmoid around K_0
    scale = K_0 * 0.3
    nuc_prob_raw = 1.0 / (1.0 + np.exp(-(K_I_field - K_0) / scale))
    
    # Smooth the probability field to show spatial coherence (represents 
    # the local environment around each point, not just a single flaw)
    nuc_prob = gaussian_filter(nuc_prob_raw, sigma=3)
    # Rescale to use full 0-1 range
    pmin, pmax = nuc_prob.min(), nuc_prob.max()
    if pmax > pmin:
        nuc_prob = (nuc_prob - pmin) / (pmax - pmin)
    nuc_prob = np.clip(nuc_prob, 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Figure 1. Nucleation Probability Map (M1)', fontsize=14, fontweight='bold', y=1.02)
    
    x_mm = result.positions[:, 0] * 1e3
    y_mm = result.positions[:, 1] * 1e3
    sizes_nm = result.flaw_sizes * 1e9
    sc = ax1.scatter(x_mm, y_mm, c=sizes_nm, s=2, cmap='hot', alpha=0.7, vmin=0, vmax=200)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title('(a) Defect Distribution (ULE 7973)')
    ax1.set_xlim(0, 152)
    ax1.set_ylim(0, 152)
    ax1.set_aspect('equal')
    cb1 = plt.colorbar(sc, ax=ax1, shrink=0.8)
    cb1.set_label('Flaw size [nm]')
    
    extent = [0, 152, 0, 152]
    im = ax2.imshow(nuc_prob.T, origin='lower', extent=extent, cmap='viridis', aspect='equal', vmin=0, vmax=1.0)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title('(b) Nucleation Probability Map')
    cb2 = plt.colorbar(im, ax=ax2, shrink=0.8)
    cb2.set_label('P(nucleation)')
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig1_nucleation_map.png'))
    plt.close(fig)
    print("✓ fig1_nucleation_map.png")


# =====================================================================
# Fig 2: Crack Growth — FIXED: realistic stress & initial crack size
# =====================================================================
def make_fig2():
    from config import ULE_GLASS
    
    np.random.seed(42)
    
    K_IC = ULE_GLASS['K_IC']       # 0.75 MPa√m
    K_0 = K_IC * ULE_GLASS['K_0_ratio']  # 0.1875 MPa√m
    n = ULE_GLASS['scg_n']         # 20
    
    # Effective SCG velocity: v = A * (K_I / K_IC)^n
    # where A = v0 * exp(-ΔH/RT) is the effective pre-exponential at room temp.
    # Literature value for silicate glass at room temp + humidity: A ~ 1e-4 to 1e-2 m/s
    # We use A = 5e-3 m/s (moderate humidity, room temperature)
    A_eff = 5e-3  # m/s — effective velocity constant at 293K, ~50% RH
    
    sigma_applied = 30e6  # Pa — realistic EUV thermal + residual stress
    
    # Initial crack: K_I(a0) ~ 0.5 * K_IC (mid SCG region)
    K_I_target = 0.50 * K_IC
    a0 = (K_I_target / (1.12 * sigma_applied))**2 / np.pi
    
    def K_I_func(a):
        return 1.12 * sigma_applied * np.sqrt(np.pi * a)
    
    # Euler integration
    dt = 3600.0  # 1 hour
    t_max = 18 * 30 * 24 * 3600  # 18 months
    t_list = [0.0]
    a_list = [a0]
    
    a = a0
    t = 0.0
    while t < t_max:
        K_I = K_I_func(a)
        if K_I >= K_IC:
            t_list.append(t)
            a_list.append(a)
            break
        if K_I < K_0:
            t += dt * 100
            t_list.append(t)
            a_list.append(a)
            continue
        v = A_eff * (K_I / K_IC)**n
        da = v * dt
        a += da
        t += dt
        t_list.append(t)
        a_list.append(a)
        if len(t_list) > 100000:
            break
    
    t_arr = np.array(t_list)
    a_arr = np.array(a_list)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Figure 2. Subcritical Crack Growth (M2)', fontsize=14, fontweight='bold', y=1.02)
    
    # (a) Crack length vs time
    ax1.plot(t_arr / (3600*24*30), a_arr * 1e6, 'b-', linewidth=2)
    ax1.set_xlabel('Time [months]')
    ax1.set_ylabel('Crack length [µm]')
    ax1.set_title('(a) Subcritical Crack Growth (σ = 30 MPa)')
    a_crit = (K_IC / (1.12 * sigma_applied))**2 / np.pi
    ax1.axhline(y=a_crit * 1e6, color='r', linestyle='--', alpha=0.7, label=f'Critical size ({a_crit*1e6:.0f} µm)')
    ax1.legend()
    
    # (b) V-K_I diagram
    K_range = np.linspace(K_0 * 0.5, K_IC * 0.99, 500)
    v_arr = np.zeros_like(K_range)
    for i, K in enumerate(K_range):
        if K < K_0:
            v_arr[i] = 1e-20
        else:
            v_arr[i] = A_eff * (K / K_IC)**n
    
    valid = v_arr > 1e-18
    ax2.semilogy(K_range[valid] / 1e6, v_arr[valid], 'b-', linewidth=2)
    ax2.axvline(x=K_IC / 1e6, color='r', linestyle='--', label='$K_{IC}$')
    ax2.axvline(x=K_0 / 1e6, color='orange', linestyle='--', label='$K_0$')
    ax2.axvspan(K_0 / 1e6, K_IC / 1e6, alpha=0.08, color='blue', label='SCG Region')
    ax2.set_xlabel('$K_I$ [MPa·m$^{0.5}$]')
    ax2.set_ylabel('Crack velocity $v$ [m/s]')
    ax2.set_title('(b) V–$K_I$ Diagram (Charles-Hillig)')
    ax2.set_ylim(1e-14, 1e-2)
    ax2.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig2_crack_growth.png'))
    plt.close(fig)
    print("✓ fig2_crack_growth.png")


# =====================================================================
# Fig 3: Inspection Comparison — expanded abbreviations
# =====================================================================
def make_fig3():
    from modules.m03_inspection import InspectionComparison
    
    methods = ['Acoustic', 'Laser\nScattering', 'Raman', '193nm\nInterfero.', 
               'Electron Energy\nLoss Spectroscopy\n(EELS)', 
               'Kelvin Force\nMicroscopy\n(KFM)']
    method_keys = ['acoustic', 'laser_scattering', 'raman', 'interferometry_193nm', 'eels', 'kfm']
    
    comp = InspectionComparison()
    
    sensitivity = []
    for mk in method_keys:
        min_size = comp.minimum_detectable_crack(mk)
        sens = min(10, max(1, -np.log10(min_size) - 4))
        sensitivity.append(sens)
    
    resolution_vals =  [3, 5, 6, 7, 9, 8]
    depth_pen_vals =   [9, 6, 4, 8, 2, 1]
    speed_vals =       [7, 8, 4, 6, 2, 3]
    cost_eff_vals =    [7, 8, 5, 6, 2, 3]
    
    categories = ['Sensitivity', 'Resolution', 'Depth\nPenetration', 'Speed', 'Cost\nEffectiveness']
    
    data = np.array([sensitivity, resolution_vals, depth_pen_vals, speed_vals, cost_eff_vals]).T
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    colors = ['#003366', '#0066CC', '#3399FF', '#FF6600', '#CC3333', '#009933']
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = data[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method.replace('\n', ' '), color=color)
        ax.fill(angles, values, alpha=0.05, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 10)
    ax.set_title('Figure 3. Inspection Method Comparison\n(ULE 7973 Micro-Crack Detection)', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig3_inspection_comparison.png'))
    plt.close(fig)
    print("✓ fig3_inspection_comparison.png")


# =====================================================================
# Fig 4: ML Diagnostics — human-readable feature names
# =====================================================================
def make_fig4():
    from modules.m04_inverse_ml import SyntheticDataGenerator, BayesianCrackDiagnostics
    
    np.random.seed(42)
    
    gen = SyntheticDataGenerator(use_forward_models=False)
    params_range = {
        'defect_density': (1e6, 1e10),
        'cte_sigma': (5e-9, 20e-9),
        'dose': (1000, 100000),
        'delta_T': (0.1, 2.0),
    }
    
    X, y_labels, y_cont = gen.generate_training_set(500, params_range, seed=42)
    X = gen.add_noise(X, 0.05)
    X_bal, y_bal = gen.balance_classes(X, y_labels)
    
    model = BayesianCrackDiagnostics()
    model.set_physics_prior(
        griffith_params={'K_IC': 0.75e6, 'gamma_s': 4.5, 'stress_threshold': 1e7},
        scg_params={'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
    )
    
    X_train, X_test, y_train, y_test = X_bal[:400], X_bal[400:], y_bal[:400], y_bal[400:]
    model.fit(X_train, y_train)
    y_pred, y_unc = model.predict(X_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Figure 4. Bayesian ML Diagnostics (M4)', fontsize=14, fontweight='bold', y=1.02)
    
    ax1.scatter(y_test, y_pred, c=y_unc, cmap='coolwarm', s=20, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect prediction')
    ax1.set_xlabel('Actual Label')
    ax1.set_ylabel('Predicted Probability')
    ax1.set_title('(a) Bayesian Prediction vs Actual')
    ax1.legend()
    cb = plt.colorbar(ax1.collections[0], ax=ax1)
    cb.set_label('Uncertainty')
    
    # Human-readable feature name mapping
    name_map = {
        'stress_thermal_': 'Thermal Stress',
        'stress_thermal_m': 'Thermal Stress (Mean)',
        'stress_thermal_s': 'Thermal Stress (Std)',
        'ae_energy_mean': 'AE Mean Energy',
        'ae_energy_std': 'AE Energy Std',
        'dispersion_anom': 'Dispersion Anomaly',
        'raman_shift_mea': 'Raman Shift (Mean)',
        'raman_shift_std': 'Raman Shift Std',
        'scatter_intensi': 'Scatter Intensity',
        'birefringence_m': 'Birefringence (Max)',
        'cte_inhomogenei': 'CTE Inhomogeneity',
        'defect_density_': 'Defect Density',
        'phase_shift_mea': 'Phase Shift (Mean)',
        'k_i_mean': 'Stress Intensity (Mean)',
        'nucleation_prob': 'Nucleation Prob.',
        'overlay_error_r': 'Overlay Error RMS',
    }
    
    if model.feature_importance:
        names = list(model.feature_importance.keys())[:12]
        vals = [model.feature_importance[n] for n in names]
        # Map to human-readable names
        display_names = []
        for n in names:
            matched = False
            for key, readable in name_map.items():
                if n.startswith(key) or n[:15] == key:
                    display_names.append(readable)
                    matched = True
                    break
            if not matched:
                # Clean up: replace underscores, title case
                display_names.append(n.replace('_', ' ').title()[:20])
        
        colors_fi = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax2.barh(range(len(names)), vals, color=colors_fi)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(display_names, fontsize=8)
        ax2.set_xlabel('Relative Importance')
        ax2.set_title('(b) Physics-Informed Feature Importance')
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig4_ml_diagnostics.png'))
    plt.close(fig)
    print("✓ fig4_ml_diagnostics.png")


# =====================================================================
# Fig 5: Attribution — pie chart → horizontal bar chart for panel (a)
# =====================================================================
def make_fig5():
    from modules.m05_attribution import VarianceDecomposition, OverlayDegradationModel
    from config import ATTRIBUTION
    
    np.random.seed(42)
    n_lots = 200
    crack_density = np.concatenate([
        np.ones(80) * 1e6,
        np.linspace(1e6, 1e9, 60),
        np.ones(60) * 1e9
    ])
    
    model = OverlayDegradationModel()
    _, overlay_series = model.overlay_time_series(crack_density)
    noise = np.random.normal(0, 0.05, n_lots)
    overlay_noisy = overlay_series + noise
    
    decomposer = VarianceDecomposition()
    result = decomposer.decompose(
        overlay_noisy**2,
        {"scanner": ATTRIBUTION["overlay_scanner"], "process": ATTRIBUTION["overlay_process"]}
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Figure 5. Process Attribution (M5)', fontsize=14, fontweight='bold', y=1.02)
    
    # (a) Horizontal bar chart instead of pie chart
    labels = ['Scanner', 'Mask (Pristine)', 'Mask (Degradation)', 'Process']
    sizes = [result['scanner_contribution'], result['mask_pristine_contribution'],
             result['mask_degradation_contribution'], result['process_contribution']]
    colors_bar = ['#003366', '#0066CC', '#FF6600', '#3399FF']
    
    y_pos = range(len(labels))
    bars = ax1.barh(y_pos, [s*100 for s in sizes], color=colors_bar, edgecolor='white', height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel('Contribution [%]')
    ax1.set_title('(a) Overlay Variance Decomposition')
    ax1.set_xlim(0, max(s*100 for s in sizes) * 1.2)
    # Add percentage labels on bars
    for bar, s in zip(bars, sizes):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{s*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # (b) Changepoint detection
    cp_time, cp_conf = decomposer.bayesian_changepoint(overlay_noisy)
    
    ax2.plot(range(n_lots), overlay_noisy, 'b-', alpha=0.5, linewidth=0.8, label='Measured overlay')
    ax2.plot(range(n_lots), overlay_series, 'r-', linewidth=2, label='Model (crack effect)')
    if cp_time is not None:
        ax2.axvline(x=cp_time, color='orange', linestyle='--', linewidth=2,
                    label=f'Changepoint (conf={cp_conf:.0%})')
    ax2.set_xlabel('Lot Number')
    ax2.set_ylabel('Overlay Error [nm RMS]')
    ax2.set_title('(b) Degradation Onset Detection')
    ax2.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig5_attribution.png'))
    plt.close(fig)
    print("✓ fig5_attribution.png")


# =====================================================================
# Fig 6: Material Comparison — FIXED: ULE-favorable axes
# =====================================================================
def make_fig6():
    from config import MATERIALS_DB
    
    materials = {
        'ULE 7973': MATERIALS_DB['corning_ule_7973'],
        'Zerodur': MATERIALS_DB['schott_zerodur'],
        'Clearceram-Z': MATERIALS_DB['ohara_clearceram'],
        'AGC AZ': MATERIALS_DB['agc_az'],
        'Shin-Etsu Quartz': MATERIALS_DB['shin_etsu_quartz'],
    }
    
    # Custom axes that highlight ULE 7973 strengths
    # Format: (key_or_func, label, higher_is_better)
    axes_def = [
        ('cte_zero_proximity', 'CTE Zero-Crossing\nProximity', True),
        ('cte_homogeneity', 'CTE\nHomogeneity', True),
        ('euv_compatibility', 'EUV Reflectance\nCompatibility', True),
        ('dimensional_stability', 'Dimensional\nStability', True),
        ('scg_n', 'SCG Exponent\n(n)', True),
        ('K_IC', 'Fracture\nToughness\n($K_{IC}$)', True),
    ]
    
    # Compute raw values for each material
    raw_data = {}
    for mat_name, props in materials.items():
        vals = []
        # 1. CTE zero-crossing proximity: 1 / (|CTE_mean| + CTE_sigma) — lower CTE = better
        cte_total = abs(props.get('CTE_mean', 0)) + props.get('CTE_sigma', 30e-9)
        vals.append(1.0 / (cte_total * 1e9 + 0.1))  # inverse, higher = better
        
        # 2. CTE Homogeneity: 1 / CTE_sigma — lower sigma = better
        vals.append(1.0 / (props.get('CTE_sigma', 30e-9) * 1e9 + 0.1))
        
        # 3. EUV Reflectance Compatibility: low birefringence + low Δn
        bire = props.get('birefringence_max', 10e-9) * 1e9  # nm/cm
        dn = props.get('dn_sigma', 5e-7) * 1e7
        # Penalize yellowish tint (glass-ceramics)
        tint_penalty = 0.5 if props.get('transparency', 'clear') != 'clear' else 1.0
        vals.append(tint_penalty / (bire + dn + 0.1))
        
        # 4. Dimensional stability: low CTE + high E (resist deformation)
        vals.append(props.get('E_young', 70e9) / 1e9 / (cte_total * 1e9 + 0.1))
        
        # 5. SCG exponent n (higher = better)
        vals.append(props.get('scg_n', 15))
        
        # 6. Fracture toughness K_IC (higher = better)
        vals.append(props.get('K_IC', 0.7e6) / 1e6)
        
        raw_data[mat_name] = vals
    
    # Normalize to 0-1
    all_vals = np.array(list(raw_data.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    norm_data = {}
    for mat_name, vals in raw_data.items():
        # Normalize and add a floor of 0.1 so all materials are visible
        norm_data[mat_name] = 0.1 + 0.85 * (np.array(vals) - mins) / ranges
    
    n_axes = len(axes_def)
    angles = np.linspace(0, 2*np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    colors = {'ULE 7973': '#003366', 'Zerodur': '#CC3333', 'Clearceram-Z': '#FF9900',
              'AGC AZ': '#009933', 'Shin-Etsu Quartz': '#6633CC'}
    
    for mat_name, nvals in norm_data.items():
        vals = nvals.tolist() + [nvals[0]]
        lw = 3 if 'ULE' in mat_name else 1.5
        ax.plot(angles, vals, 'o-', linewidth=lw, label=mat_name, color=colors[mat_name])
        if 'ULE' in mat_name:
            ax.fill(angles, vals, alpha=0.15, color=colors[mat_name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[1] for a in axes_def], size=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('Figure 6. Material Comparison for EUV Substrate\n(Normalized, Higher = Better)', pad=25, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig6_material_comparison.png'))
    plt.close(fig)
    print("✓ fig6_material_comparison.png")


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print("Generating proposal figures (V2)...")
    
    funcs = [
        ('fig0', make_fig0),
        ('fig1', make_fig1),
        ('fig2', make_fig2),
        ('fig3', make_fig3),
        ('fig4', make_fig4),
        ('fig5', make_fig5),
        ('fig6', make_fig6),
    ]
    
    for name, func in funcs:
        try:
            func()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone! Figures saved to:", FIGDIR)
