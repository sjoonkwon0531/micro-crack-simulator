"""
Glass Micro-Crack Lifecycle Simulator — Configuration
Job 10: Corning × SKKU SPMDL Industry-Academia Project

Five-module architecture:
  M1: Crack Nucleation Probability Engine
  M2: Crack Propagation & Morphology Evolution
  M3: Inspection Signal Forward Model
  M4: Physics-Informed Inverse Diagnostics (ML)
  M5: Process Impact Attribution Engine
"""

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================
k_B = 1.380649e-23       # Boltzmann constant [J/K]
R_gas = 8.314462         # Universal gas constant [J/(mol·K)]
h_planck = 6.62607e-34   # Planck constant [J·s]

# =============================================================================
# ULE Glass Material Properties (TiO₂-SiO₂)
# =============================================================================
ULE_GLASS = {
    "name": "Corning ULE 7973",
    "composition": "TiO2-SiO2 (~7.4 wt% TiO2)",
    
    # Mechanical properties
    "E_young": 67.6e9,            # Young's modulus [Pa]
    "nu_poisson": 0.17,           # Poisson's ratio
    "rho": 2210.0,                # Density [kg/m³]
    "K_IC": 0.75e6,               # Critical fracture toughness [Pa·m^0.5]
    "K_0_ratio": 0.25,            # Subcritical threshold K_0/K_IC
    "hardness_vickers": 5.4e9,    # Vickers hardness [Pa]
    
    # Thermal properties
    "CTE_mean": 0.0e-9,           # Mean CTE [1/K] (zero-crossing at ~20°C)
    "CTE_sigma": 10.0e-9,         # CTE std dev [1/K] (ppb/K)
    "k_thermal": 1.31,            # Thermal conductivity [W/(m·K)]
    "cp_specific": 767.0,         # Specific heat [J/(kg·K)]
    "alpha_thermal": 7.7e-7,      # Thermal diffusivity [m²/s]
    "T_g": 1043.0,                # Glass transition temperature [°C]
    "T_anneal": 1000.0,           # Annealing point [°C]
    "T_strain": 890.0,            # Strain point [°C]
    
    # Optical properties (at 193 nm)
    "n_193nm": 1.5607,            # Refractive index at 193 nm
    "dn_sigma": 2.0e-7,           # Refractive index std dev (plane-removed)
    "birefringence_max": 5.0e-9,  # Max birefringence [m/cm] → 5 nm/cm
    
    # Subcritical crack growth (Charles-Hillig)
    "scg_n": 20.0,                # Stress corrosion exponent (typical for silicates)
    "scg_v0": 1.0e-3,             # Pre-exponential velocity [m/s]
    "scg_delta_H": 80.0e3,        # Activation energy [J/mol]
}

# =============================================================================
# Competitor Glass Materials Database
# =============================================================================
SCHOTT_ZERODUR = {
    "name": "Schott Zerodur",
    "type": "glass-ceramic",       # Li₂O-Al₂O₃-SiO₂ system
    "composition": "Li2O-Al2O3-SiO2 (LAS glass-ceramic, ~70% crystalline β-quartz ss)",
    "notes": "Two-phase: nanocrystalline β-quartz solid solution (negative CTE) + residual glass (positive CTE)",
    
    # Mechanical
    "E_young": 90.3e9,            # [Pa] — higher than ULE due to crystalline phase
    "nu_poisson": 0.24,
    "rho": 2530.0,                # [kg/m³]
    "K_IC": 0.90e6,               # [Pa·m^0.5] — slightly tougher than ULE
    "K_0_ratio": 0.25,
    "hardness_vickers": 6.2e9,    # [Pa] — Knoop ~620
    
    # Thermal
    "CTE_mean": 0.0e-9,           # [1/K] zero-crossing tunable (0±0.02 ppm/K)
    "CTE_sigma": 15.0e-9,         # [1/K] — typically wider spread than ULE
    "k_thermal": 1.46,            # [W/(m·K)]
    "cp_specific": 821.0,         # [J/(kg·K)]
    "T_g": None,                  # Glass-ceramic: no single Tg
    "T_max_use": 600.0,           # [°C] max continuous use
    
    # Optical (at 193 nm)
    "n_193nm": 1.547,             # approximate
    "dn_sigma": 5.0e-7,           # [1] — less uniform than ULE (grain boundaries)
    "birefringence_max": 10.0e-9, # [m/cm] — higher due to crystallites
    "transparency": "yellowish_tint",  # absorbs in blue/UV
    
    # Subcritical crack growth
    "scg_n": 25.0,                # higher than ULE (ceramic phase)
    "scg_v0": 5.0e-4,
    "scg_delta_H": 85.0e3,       # [J/mol]
    
    # Crack-specific: grain boundary effects
    "has_grain_boundaries": True,
    "grain_size_nm": 50.0,        # [nm] nanocrystallites ~50nm
    "crack_deflection_factor": 1.3,  # crack path tortuosity from grain boundaries
    "intergranular_weakness": 0.85,  # G_IC,GB / G_IC,bulk ratio
}

OHARA_CLEARCERAM_Z = {
    "name": "Ohara Clearceram-Z HS",
    "type": "glass-ceramic",       # similar LAS system to Zerodur
    "composition": "Li2O-Al2O3-SiO2 (transparent glass-ceramic)",
    "notes": "Three grades: Regular (wide T range), HS (near RT), EX (ultra-low CTE dependence 0-50°C)",
    
    # Mechanical
    "E_young": 92.0e9,            # [Pa]
    "nu_poisson": 0.25,
    "rho": 2550.0,                # [kg/m³]
    "K_IC": 0.88e6,               # [Pa·m^0.5]
    "K_0_ratio": 0.25,
    "hardness_vickers": 6.0e9,    # [Pa]
    
    # Thermal
    "CTE_mean": 0.0e-9,           # [1/K] — HS grade optimized near RT
    "CTE_sigma": 12.0e-9,         # [1/K]
    "k_thermal": 1.49,            # [W/(m·K)]
    "cp_specific": 800.0,         # [J/(kg·K)]
    "T_max_use": 550.0,           # [°C]
    
    # Optical
    "n_193nm": 1.550,             # approximate
    "dn_sigma": 4.0e-7,
    "birefringence_max": 8.0e-9,  # [m/cm]
    "transparency": "yellowish_tint",
    
    # Subcritical crack growth
    "scg_n": 23.0,
    "scg_v0": 7.0e-4,
    "scg_delta_H": 83.0e3,        # [J/mol]
    
    # Crack-specific
    "has_grain_boundaries": True,
    "grain_size_nm": 40.0,         # [nm]
    "crack_deflection_factor": 1.25,
    "intergranular_weakness": 0.88,
}

AGC_AZ = {
    "name": "AGC AZ (EUV substrate candidate)",
    "type": "synthetic_quartz_modified",
    "composition": "Modified synthetic quartz (SiO2-based, proprietary dopants)",
    "notes": "AGC primarily a blank maker using Corning ULE; own substrate R&D ongoing",
    
    # Mechanical — synthetic quartz baseline
    "E_young": 72.0e9,            # [Pa]
    "nu_poisson": 0.17,
    "rho": 2200.0,                # [kg/m³]
    "K_IC": 0.70e6,               # [Pa·m^0.5]
    "K_0_ratio": 0.20,
    "hardness_vickers": 5.5e9,    # [Pa]
    
    # Thermal
    "CTE_mean": 5.0e-7,           # [1/K] — 0.5 ppm/K (much higher than ULE!)
    "CTE_sigma": 8.0e-9,          # [1/K]
    "k_thermal": 1.38,            # [W/(m·K)]
    "cp_specific": 740.0,         # [J/(kg·K)]
    "T_max_use": 1000.0,          # [°C]
    
    # Optical
    "n_193nm": 1.560,
    "dn_sigma": 1.5e-7,           # excellent homogeneity (synthetic quartz strength)
    "birefringence_max": 2.0e-9,  # [m/cm] — very low (amorphous)
    "transparency": "clear",
    
    # Subcritical crack growth
    "scg_n": 18.0,                # lower than glass-ceramics
    "scg_v0": 2.0e-3,
    "scg_delta_H": 75.0e3,        # [J/mol]
    
    # Crack-specific
    "has_grain_boundaries": False,
    "crack_deflection_factor": 1.0,  # no deflection (amorphous)
    "intergranular_weakness": 1.0,   # N/A
}

SHIN_ETSU_QUARTZ = {
    "name": "Shin-Etsu Synthetic Quartz (AQ series)",
    "type": "synthetic_quartz",
    "composition": "High-purity synthetic SiO2 (flame hydrolysis)",
    "notes": "Primary DUV photomask substrate; EUV reference material",
    
    # Mechanical
    "E_young": 73.0e9,
    "nu_poisson": 0.17,
    "rho": 2200.0,
    "K_IC": 0.72e6,
    "K_0_ratio": 0.20,
    "hardness_vickers": 5.6e9,
    
    # Thermal
    "CTE_mean": 5.5e-7,           # [1/K] — 0.55 ppm/K (standard quartz)
    "CTE_sigma": 5.0e-9,
    "k_thermal": 1.40,
    "cp_specific": 746.0,
    "T_max_use": 1160.0,
    
    # Optical
    "n_193nm": 1.5607,
    "dn_sigma": 1.0e-7,           # best-in-class homogeneity
    "birefringence_max": 1.0e-9,
    "transparency": "clear",
    
    # Subcritical crack growth
    "scg_n": 16.0,
    "scg_v0": 3.0e-3,
    "scg_delta_H": 72.0e3,
    
    # Crack-specific
    "has_grain_boundaries": False,
    "crack_deflection_factor": 1.0,
    "intergranular_weakness": 1.0,
}

# =============================================================================
# Materials Comparison Registry
# =============================================================================
MATERIALS_DB = {
    "corning_ule": ULE_GLASS,
    "schott_zerodur": SCHOTT_ZERODUR,
    "ohara_clearceram": OHARA_CLEARCERAM_Z,
    "agc_az": AGC_AZ,
    "shin_etsu_quartz": SHIN_ETSU_QUARTZ,
}

# Comparison axes for benchmarking
COMPARISON_AXES = [
    # (key, label, unit, lower_is_better)
    ("CTE_sigma", "CTE Uniformity", "ppb/K", True),
    ("K_IC", "Fracture Toughness", "MPa·m⁰·⁵", False),
    ("E_young", "Young's Modulus", "GPa", False),
    ("dn_sigma", "Optical Homogeneity (Δn)", "×10⁻⁷", True),
    ("birefringence_max", "Birefringence", "nm/cm", True),
    ("k_thermal", "Thermal Conductivity", "W/(m·K)", False),
    ("scg_n", "SCG Exponent n", "-", False),  # higher n = more resistant
    ("scg_delta_H", "SCG Activation Energy", "kJ/mol", False),  # higher = more resistant
]

# =============================================================================
# EUV Exposure Conditions
# =============================================================================
EUV_CONDITIONS = {
    # Low-NA (NXE:3800)
    "low_na": {
        "NA": 0.33,
        "wavelength": 13.5e-9,     # [m]
        "dose_range": (25, 100),   # [mJ/cm²]
        "dose_typical": 45,        # [mJ/cm²]
        "delta_T_typical": 0.5,    # [K] typical reticle heating
        "delta_T_range": (0.3, 0.8),
        "cycle_time": 1.5,         # [s] per exposure field
        "demag": (4, 4),           # (x, y) demagnification
    },
    # High-NA (EXE:5200)
    "high_na": {
        "NA": 0.55,
        "wavelength": 13.5e-9,
        "dose_range": (30, 120),
        "dose_typical": 55,
        "delta_T_typical": 0.3,
        "delta_T_range": (0.2, 0.6),
        "cycle_time": 1.2,
        "demag": (8, 4),           # anamorphic
    },
}

# =============================================================================
# Substrate Geometry (EUV Photomask)
# =============================================================================
SUBSTRATE = {
    "length": 152.0e-3,     # [m] 152 mm (6" standard)
    "width": 152.0e-3,      # [m]
    "thickness": 6.35e-3,   # [m] 6.35 mm (standard)
    "active_area": (132e-3, 104e-3),  # [m] usable pattern area
    "grid_resolution": 0.5e-3,  # [m] default simulation grid (0.5 mm)
}

# =============================================================================
# Defect / Impurity Model Parameters
# =============================================================================
DEFECT_MODEL = {
    # Spatial distribution
    "distribution_type": "poisson",  # "poisson", "neyman_scott", "clustered"
    "density_range": (1e6, 1e10),    # [defects/m³] 
    "density_default": 1e8,          # [defects/m³]
    
    # Flaw size distribution (log-normal)
    "flaw_size_mean": 50e-9,         # [m] 50 nm mean flaw radius
    "flaw_size_sigma": 0.5,          # log-normal sigma
    "flaw_size_min": 5e-9,           # [m] minimum detectable
    "flaw_size_max": 5e-6,           # [m] maximum before rejection
    
    # Spatial correlation
    "correlation_length_range": (0.1e-3, 10.0e-3),  # [m]
    "correlation_length_default": 1.0e-3,             # [m]
    
    # Neyman-Scott cluster parameters
    "cluster_density": 1e4,          # [clusters/m³]
    "cluster_radius": 0.5e-3,        # [m]
    "points_per_cluster": 10,        # mean
}

# =============================================================================
# Crack Nucleation Parameters (Module 1)
# =============================================================================
NUCLEATION = {
    "mc_runs": 10000,                # Monte Carlo realizations
    "griffith_gamma_s": 4.5,         # Surface energy [J/m²] for ULE glass
    "stress_concentration_factor": 2.0,  # Kt for elliptical flaw
    "thermal_cycles_max": 1e6,       # Max exposure cycles to simulate
    "nucleation_threshold": 0.9,     # G_I/G_IC threshold for nucleation
}

# =============================================================================
# Phase-Field Fracture Parameters (Module 2)
# =============================================================================
PHASE_FIELD = {
    "length_scale": 2.0e-6,          # [m] regularization length l_0
    "energy_release_rate": 9.0,      # G_c [J/m²] critical energy release rate
    "degradation_function": "quadratic",  # (1-d)² or AT1/AT2
    "time_step": 1e-3,               # [s] (normalized)
    "max_iterations": 1000,
    "convergence_tol": 1e-6,
    "mesh_refinement_near_crack": 4,  # refinement level near crack tip
}

# =============================================================================
# Inspection Methods (Module 3)
# =============================================================================
INSPECTION = {
    "acoustic": {
        "frequency_range": (1e6, 50e6),    # [Hz] ultrasonic
        "velocity_longitudinal": 5970.0,    # [m/s] in ULE glass
        "velocity_transverse": 3764.0,      # [m/s]
        "attenuation_coeff": 0.5,           # [dB/cm/MHz]
    },
    "laser_scattering": {
        "wavelength": 532e-9,               # [m] green laser
        "power": 50e-3,                     # [W]
        "spot_size": 1e-6,                  # [m]
        "detection_limit": 20e-9,           # [m] min crack size
    },
    "raman": {
        "excitation_wavelength": 532e-9,
        "spectral_resolution": 1.0,         # [cm⁻¹]
        "stress_sensitivity": -4.0,         # [cm⁻¹/GPa] for SiO₂
        "spatial_resolution": 500e-9,       # [m]
    },
    "interferometry_193nm": {
        "wavelength": 193e-9,
        "phase_sensitivity": 0.01,          # [rad] minimum detectable
    },
}

# =============================================================================
# ML / Inverse Model Parameters (Module 4)
# =============================================================================
ML_CONFIG = {
    "model_type": "bayesian_gp",     # "bayesian_gp", "pinn", "ensemble"
    "synthetic_training_size": 50000,
    "validation_split": 0.2,
    "n_features_physics": 12,        # physics-informed features
    "confidence_level": 0.95,
    "prior_type": "griffith",        # physics prior for Bayesian
}

# =============================================================================
# Process Attribution Parameters (Module 5)
# =============================================================================
ATTRIBUTION = {
    "overlay_budget_total": 1.5,     # [nm] RMS total
    "overlay_scanner": 0.8,          # [nm] RMS scanner contribution
    "overlay_mask_pristine": 0.5,    # [nm] RMS pristine mask
    "overlay_process": 0.3,          # [nm] RMS process
    # Degradation model
    "degradation_rate_model": "power_law",  # σ_deg ∝ (crack_density)^α
    "degradation_exponent": 0.5,
    # Replacement cost model
    "cost_new_substrate": 50000,     # [USD]
    "cost_inspection": 500,          # [USD] per inspection
    "cost_yield_loss_per_nm": 10000, # [USD] per nm overlay degradation per lot
    "cost_downtime_per_hour": 5000,  # [USD]
}

# =============================================================================
# Simulation Grid & Numerical Parameters
# =============================================================================
SIMULATION = {
    "grid_2d": (304, 304),           # 152mm / 0.5mm = 304 grid points
    "grid_3d": (304, 304, 13),       # 13 points through 6.35mm thickness
    "time_steps": 1000,
    "random_seed": 42,
}
