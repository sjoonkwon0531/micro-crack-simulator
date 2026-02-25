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
