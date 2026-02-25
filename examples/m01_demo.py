#!/usr/bin/env python3
"""
M1 Crack Nucleation Engine - Demonstration Script

Example usage of the M1 nucleation engine for EUV glass substrate analysis.

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.m01_nucleation import (
    DefectField, ThermoelasticStress, NucleationEngine, CTEMapGenerator
)
from config import ULE_GLASS, SUBSTRATE, DEFECT_MODEL, EUV_CONDITIONS


def demo_defect_generation():
    """Demonstrate defect field generation."""
    print("=== Defect Field Generation Demo ===")
    
    substrate = SUBSTRATE.copy()
    defect_field = DefectField(substrate)
    
    domain_size = (substrate["length"], substrate["width"], substrate["thickness"])
    
    # Generate Poisson defects
    print("Generating Poisson defect field...")
    poisson_result = defect_field.generate_poisson(
        density=1e8, domain_size=domain_size, seed=42
    )
    
    print(f"Generated {len(poisson_result.positions)} defects")
    print(f"Actual density: {poisson_result.density:.2e} defects/m³")
    print(f"Mean flaw size: {np.mean(poisson_result.flaw_sizes)*1e9:.1f} nm")
    
    # Generate clustered defects
    print("\nGenerating clustered (Neyman-Scott) defect field...")
    cluster_result = defect_field.generate_neyman_scott(
        cluster_density=1e4, cluster_radius=2e-3, points_per_cluster=8,
        domain_size=domain_size, seed=42
    )
    
    print(f"Generated {len(cluster_result.positions)} clustered defects") 
    print(f"Actual density: {cluster_result.density:.2e} defects/m³")
    
    return poisson_result, cluster_result


def demo_stress_calculation():
    """Demonstrate thermoelastic stress calculation.""" 
    print("\n=== Stress Field Calculation Demo ===")
    
    stress_calc = ThermoelasticStress()
    cte_generator = CTEMapGenerator()
    substrate = SUBSTRATE.copy()
    
    # EUV exposure conditions
    dose = 50.0  # mJ/cm²
    delta_T = 1.0  # K
    
    print(f"EUV dose: {dose} mJ/cm²")
    print(f"Peak temperature rise: {delta_T} K")
    
    # Generate thermal field
    thermal_field = stress_calc.compute_thermal_field(dose, delta_T, substrate)
    print(f"Max thermal field: {np.max(thermal_field):.3f} K")
    
    # Generate CTE map (Gaussian random field)
    grid_size = (100, 100)  # Smaller grid for demo
    cte_map = cte_generator.generate_gaussian_random_field(
        grid_size, ULE_GLASS["CTE_sigma"], 1e-3, seed=42
    )
    
    # Resize thermal field to match
    from scipy.ndimage import zoom
    zoom_factor = (grid_size[0] / thermal_field.shape[0], 
                   grid_size[1] / thermal_field.shape[1])
    thermal_field_resized = zoom(thermal_field, zoom_factor)
    
    # Calculate stress field
    stress_field = stress_calc.compute_stress_field(
        thermal_field_resized, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
    )
    
    print(f"Max stress: {np.max(np.abs(stress_field))/1e3:.1f} kPa")
    print(f"CTE variation: {np.std(cte_map)*1e9:.1f} ppb")
    
    return thermal_field_resized, stress_field, cte_map


def demo_nucleation_analysis():
    """Demonstrate crack nucleation probability analysis."""
    print("\n=== Nucleation Analysis Demo ===")
    
    nucleation_engine = NucleationEngine()
    substrate = SUBSTRATE.copy()
    
    # Parameters for analysis
    defect_params = {
        "distribution_type": "poisson",
        "density": 5e7  # defects/m³
    }
    
    stress_params = {
        "dose": 60.0,      # mJ/cm²
        "delta_T": 1.2,    # K
        "correlation_length": 1e-3,
        "cte_type": "gaussian"
    }
    
    print("Running Monte Carlo nucleation simulation...")
    print(f"Defect density: {defect_params['density']:.1e} defects/m³")
    print(f"EUV dose: {stress_params['dose']} mJ/cm²")
    print(f"Temperature rise: {stress_params['delta_T']} K")
    
    # Run Monte Carlo simulation
    n_runs = 50  # Reduced for demo speed
    result = nucleation_engine.run_monte_carlo(
        n_runs, defect_params, stress_params, substrate
    )
    
    print(f"\nResults after {n_runs} Monte Carlo runs:")
    print(f"Nucleation probability: {result['nucleation_probability']:.3f}")
    print(f"Mean nucleation density: {result['mean_nucleation_density']:.2e} nucleations/m³")
    print(f"Std nucleation density: {result['std_nucleation_density']:.2e}")
    
    if result['time_to_first_crack_stats']['mean'] < np.inf:
        print(f"Mean time to first crack: {result['time_to_first_crack_stats']['mean']:.1f} (relative units)")
    else:
        print("No cracks nucleated in simulation")
        
    return result


def demo_thermal_cycling():
    """Demonstrate thermal cycling fatigue analysis."""
    print("\n=== Thermal Cycling Analysis Demo ===")
    
    nucleation_engine = NucleationEngine()
    substrate = SUBSTRATE.copy()
    
    n_cycles = 100
    cycle_params = {
        "dose": 45.0,
        "delta_T": 0.8,
        "correlation_length": 1e-3
    }
    
    defect_params = {
        "density": 2e8  # Higher density for cycling effects
    }
    
    print(f"Simulating {n_cycles} thermal cycles...")
    print(f"Cycle dose: {cycle_params['dose']} mJ/cm²")
    print(f"Cycle ΔT: {cycle_params['delta_T']} K")
    
    result = nucleation_engine.run_thermal_cycling(
        n_cycles, cycle_params, defect_params, substrate
    )
    
    print(f"Total nucleations after {n_cycles} cycles: {result['total_nucleations']}")
    print(f"Cycles to first nucleation: {result['cycles_to_first_nucleation']}")
    
    final_prob = result['nucleation_probability_vs_cycles'][-1]
    print(f"Final nucleation probability: {final_prob:.4f}")
    
    return result


def main():
    """Run all demonstration functions."""
    print("Glass Micro-Crack Lifecycle Simulator - M1 Nucleation Engine Demo")
    print("=" * 70)
    
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    try:
        # Run demonstrations
        defect_result = demo_defect_generation()
        stress_result = demo_stress_calculation()
        nucleation_result = demo_nucleation_analysis()
        cycling_result = demo_thermal_cycling()
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("All M1 nucleation engine components are working correctly.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)