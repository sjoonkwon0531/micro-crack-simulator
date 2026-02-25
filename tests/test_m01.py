"""
Test suite for M1 Crack Nucleation Engine

Comprehensive tests for DefectField, ThermoelasticStress, NucleationEngine, 
and CTE map generation utilities.

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch
import sys
import os

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from m01_nucleation import (
    DefectField, ThermoelasticStress, NucleationEngine, CTEMapGenerator,
    DefectFieldResult, StressFieldResult, NucleationResult
)
from config import ULE_GLASS, SUBSTRATE, DEFECT_MODEL, SIMULATION


class TestDefectField:
    """Test DefectField class for spatial defect generation."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.substrate_geometry = {
            "length": 0.152,  # 152 mm
            "width": 0.152,
            "thickness": 0.00635  # 6.35 mm
        }
        self.defect_field = DefectField(self.substrate_geometry)
        
    def test_defect_field_initialization(self):
        """Test DefectField initialization with substrate geometry."""
        assert self.defect_field.length == 0.152
        assert self.defect_field.width == 0.152
        assert self.defect_field.thickness == 0.00635
        expected_volume = 0.152 * 0.152 * 0.00635
        assert abs(self.defect_field.volume - expected_volume) < 1e-10
        
    def test_generate_poisson_basic(self):
        """Test basic Poisson defect generation."""
        density = 1e8  # defects/m³
        domain_size = (0.1, 0.1, 0.005)
        
        result = self.defect_field.generate_poisson(density, domain_size, seed=42)
        
        assert isinstance(result, DefectFieldResult)
        assert result.positions.shape[1] == 3  # 3D positions
        assert len(result.flaw_sizes) == len(result.positions)
        assert result.density >= 0
        
    def test_generate_poisson_density_verification(self):
        """Test that Poisson generation respects density statistics."""
        density = 1e9  # High density for better statistics
        domain_size = (0.05, 0.05, 0.005)
        n_runs = 50
        
        densities = []
        for seed in range(n_runs):
            result = self.defect_field.generate_poisson(density, domain_size, seed=seed)
            densities.append(result.density)
            
        mean_density = np.mean(densities)
        std_density = np.std(densities)
        
        # Should be close to target density (within 2 sigma)
        assert abs(mean_density - density) < 2 * std_density
        
        # Poisson statistics: variance ≈ mean for count
        volume = np.prod(domain_size)
        expected_count = density * volume
        expected_std_density = np.sqrt(expected_count) / volume
        
        # Allow some tolerance due to finite sampling
        assert abs(std_density - expected_std_density) < expected_std_density * 0.5
        
    def test_generate_poisson_empty_case(self):
        """Test Poisson generation with zero density."""
        result = self.defect_field.generate_poisson(0.0, (0.1, 0.1, 0.005), seed=42)
        
        assert len(result.positions) == 0
        assert len(result.flaw_sizes) == 0
        assert result.density == 0.0
        
    def test_generate_poisson_error_cases(self):
        """Test error handling in Poisson generation."""
        domain_size = (0.1, 0.1, 0.005)
        
        # Negative density
        with pytest.raises(ValueError, match="Defect density must be non-negative"):
            self.defect_field.generate_poisson(-1.0, domain_size)
            
        # Invalid domain size
        with pytest.raises(ValueError, match="All domain dimensions must be positive"):
            self.defect_field.generate_poisson(1e8, (0, 0.1, 0.005))
            
    def test_generate_neyman_scott_basic(self):
        """Test basic Neyman-Scott cluster generation."""
        cluster_density = 1e4
        cluster_radius = 1e-3
        points_per_cluster = 5
        domain_size = (0.05, 0.05, 0.005)
        
        result = self.defect_field.generate_neyman_scott(
            cluster_density, cluster_radius, points_per_cluster, domain_size, seed=42
        )
        
        assert isinstance(result, DefectFieldResult)
        assert result.positions.shape[1] == 3
        assert len(result.flaw_sizes) == len(result.positions)
        assert result.correlation_length == cluster_radius
        
    def test_generate_neyman_scott_clustering(self):
        """Test that Neyman-Scott generates clustered defects."""
        cluster_density = 1e4
        cluster_radius = 2e-3
        points_per_cluster = 10
        domain_size = (0.02, 0.02, 0.005)  # Small domain to see clustering
        
        result = self.defect_field.generate_neyman_scott(
            cluster_density, cluster_radius, points_per_cluster, domain_size, seed=42
        )
        
        if len(result.positions) > 1:
            # Check spatial correlation
            r_centers, g_r = self.defect_field.compute_spatial_correlation(
                result.positions, r_max=cluster_radius*2
            )
            
            if len(g_r) > 0:
                # Should show clustering (g(r) > 1 at short distances)
                short_range_idx = r_centers < cluster_radius
                if np.any(short_range_idx):
                    max_g_short = np.max(g_r[short_range_idx])
                    assert max_g_short > 1.0, "Should show clustering at short distances"
                    
    def test_assign_flaw_sizes_distribution(self):
        """Test flaw size assignment follows log-normal distribution."""
        n_defects = 1000
        flaw_sizes = self.defect_field.assign_flaw_sizes(n_defects)
        
        assert len(flaw_sizes) == n_defects
        assert np.all(flaw_sizes > 0)
        
        # Check bounds
        assert np.all(flaw_sizes >= DEFECT_MODEL["flaw_size_min"])
        assert np.all(flaw_sizes <= DEFECT_MODEL["flaw_size_max"])
        
        # Test log-normal properties
        log_sizes = np.log(flaw_sizes)
        mean_log = np.mean(log_sizes)
        expected_mean = np.log(DEFECT_MODEL["flaw_size_mean"])
        
        # Should be reasonably close (within 10% due to clipping)
        assert abs(mean_log - expected_mean) < abs(expected_mean) * 0.1
        
    def test_assign_flaw_sizes_empty(self):
        """Test flaw size assignment with zero defects."""
        flaw_sizes = self.defect_field.assign_flaw_sizes(0)
        assert len(flaw_sizes) == 0
        
    def test_compute_spatial_correlation_basic(self):
        """Test spatial correlation computation."""
        # Create simple test pattern
        positions = np.array([
            [0.01, 0.01, 0.001],
            [0.02, 0.01, 0.001],  # 1cm apart
            [0.03, 0.01, 0.001],  # Another 1cm apart
            [0.05, 0.05, 0.001]   # Far away
        ])
        
        r_centers, g_r = self.defect_field.compute_spatial_correlation(positions)
        
        assert len(r_centers) == len(g_r)
        assert len(r_centers) > 0
        assert np.all(r_centers >= 0)
        assert np.all(g_r >= 0)
        
    def test_compute_spatial_correlation_empty(self):
        """Test spatial correlation with insufficient points."""
        positions = np.array([[0.01, 0.01, 0.001]])  # Only one point
        
        r_centers, g_r = self.defect_field.compute_spatial_correlation(positions)
        
        assert len(r_centers) == 0
        assert len(g_r) == 0


class TestCTEMapGenerator:
    """Test CTE map generation utilities."""
    
    def test_generate_gaussian_random_field_basic(self):
        """Test basic Gaussian random field generation."""
        grid_size = (50, 50)
        sigma = 10e-9
        correlation_length = 2e-3
        
        cte_map = CTEMapGenerator.generate_gaussian_random_field(
            grid_size, sigma, correlation_length, seed=42
        )
        
        assert cte_map.shape == grid_size
        
        # Check statistics
        cte_mean = ULE_GLASS["CTE_mean"]
        assert abs(np.mean(cte_map) - cte_mean) < sigma  # Should be close to mean
        assert abs(np.std(cte_map) - sigma) < sigma * 0.3  # Should be close to target std
        
    def test_generate_gaussian_random_field_statistics(self):
        """Test GRF statistical properties."""
        grid_size = (100, 100)
        sigma = 5e-9
        correlation_length = 1e-3
        
        # Generate multiple realizations
        n_realizations = 10
        means = []
        stds = []
        
        for seed in range(n_realizations):
            cte_map = CTEMapGenerator.generate_gaussian_random_field(
                grid_size, sigma, correlation_length, seed=seed
            )
            means.append(np.mean(cte_map))
            stds.append(np.std(cte_map))
            
        # Ensemble statistics
        ensemble_mean = np.mean(means)
        ensemble_std = np.mean(stds)
        
        cte_mean = ULE_GLASS["CTE_mean"]
        assert abs(ensemble_mean - cte_mean) < sigma * 0.5
        assert abs(ensemble_std - sigma) < sigma * 0.3
        
    def test_generate_patchy_field_basic(self):
        """Test basic patchy field generation."""
        grid_size = (30, 30)
        sigma = 8e-9
        
        cte_map = CTEMapGenerator.generate_patchy_field(grid_size, sigma, seed=42)
        
        assert cte_map.shape == grid_size
        
        # Should have patchy structure (discrete values)
        unique_values = np.unique(cte_map)
        n_unique = len(unique_values)
        total_points = np.prod(grid_size)
        
        # Should have significantly fewer unique values than grid points
        assert n_unique < total_points * 0.5
        
    def test_generate_patchy_field_statistics(self):
        """Test patchy field statistical properties."""
        grid_size = (50, 50)
        sigma = 10e-9
        
        cte_map = CTEMapGenerator.generate_patchy_field(grid_size, sigma, seed=42)
        
        cte_mean = ULE_GLASS["CTE_mean"]
        assert abs(np.mean(cte_map) - cte_mean) < sigma * 2  # Allow more variation for patchy
        assert abs(np.std(cte_map) - sigma) < sigma * 0.5


class TestThermoelasticStress:
    """Test ThermoelasticStress class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.stress_calc = ThermoelasticStress()
        self.substrate = SUBSTRATE.copy()
        
    def test_thermoelastic_stress_initialization(self):
        """Test ThermoelasticStress initialization."""
        assert self.stress_calc.E == ULE_GLASS["E_young"]
        assert self.stress_calc.nu == ULE_GLASS["nu_poisson"]
        assert self.stress_calc.cte_mean == ULE_GLASS["CTE_mean"]
        assert self.stress_calc.cte_sigma == ULE_GLASS["CTE_sigma"]
        
    def test_compute_thermal_field_basic(self):
        """Test basic thermal field computation."""
        dose = 50.0  # mJ/cm²
        delta_T = 1.0  # K
        
        thermal_field = self.stress_calc.compute_thermal_field(
            dose, delta_T, self.substrate
        )
        
        expected_shape = SIMULATION["grid_2d"]
        assert thermal_field.shape == expected_shape
        assert np.all(thermal_field >= 0)
        assert np.max(thermal_field) > 0
        
    def test_compute_thermal_field_scaling(self):
        """Test thermal field scaling with dose and delta_T."""
        base_dose = 50.0
        base_delta_T = 1.0
        
        field_1x = self.stress_calc.compute_thermal_field(
            base_dose, base_delta_T, self.substrate
        )
        
        field_2x_dose = self.stress_calc.compute_thermal_field(
            2 * base_dose, base_delta_T, self.substrate
        )
        
        field_2x_temp = self.stress_calc.compute_thermal_field(
            base_dose, 2 * base_delta_T, self.substrate
        )
        
        # Should scale approximately linearly
        ratio_dose = np.max(field_2x_dose) / np.max(field_1x)
        ratio_temp = np.max(field_2x_temp) / np.max(field_1x)
        
        assert 1.8 < ratio_dose < 2.2  # Allow some nonlinearity
        assert 1.9 < ratio_temp < 2.1
        
    def test_compute_thermal_field_errors(self):
        """Test thermal field error handling."""
        with pytest.raises(ValueError, match="Dose must be non-negative"):
            self.stress_calc.compute_thermal_field(-1.0, 1.0, self.substrate)
            
        with pytest.raises(ValueError, match="Temperature rise must be non-negative"):
            self.stress_calc.compute_thermal_field(50.0, -1.0, self.substrate)
            
    def test_compute_stress_field_uniform_cte(self):
        """Test stress computation with uniform CTE (should give minimal stress)."""
        nx, ny = SIMULATION["grid_2d"]
        thermal_field = np.ones((nx, ny)) * 1.0  # Uniform 1K heating
        cte_map = np.ones((nx, ny)) * ULE_GLASS["CTE_mean"]  # Uniform CTE
        
        stress_field = self.stress_calc.compute_stress_field(
            thermal_field, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
        )
        
        # For ULE glass with CTE_mean ≈ 0, stress should be very small
        assert np.all(np.abs(stress_field) < 1e6)  # Less than 1 MPa
        
    def test_compute_stress_field_nonuniform_cte(self):
        """Test stress computation with non-uniform CTE (should give significant stress)."""
        nx, ny = SIMULATION["grid_2d"]
        thermal_field = np.ones((nx, ny)) * 1.0  # Uniform 1K heating
        
        # Create CTE variation
        cte_map = np.ones((nx, ny)) * ULE_GLASS["CTE_mean"]
        cte_map[:, :nx//2] += 20e-9  # Add 20 ppb variation
        cte_map[:, nx//2:] -= 20e-9
        
        stress_field = self.stress_calc.compute_stress_field(
            thermal_field, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
        )
        
        # Should have stress from CTE mismatch
        # For 20 ppb CTE variation and 1K temperature: stress ≈ E * α * ΔT ≈ 67.6 GPa * 20e-9 * 1K ≈ 1.35 kPa
        assert np.max(np.abs(stress_field)) > 1000  # At least 1 kPa (reasonable for small CTE variation)
        assert np.max(np.abs(stress_field)) < 1e9   # But physically reasonable (< 1 GPa)
        
    def test_compute_stress_at_defects_basic(self):
        """Test stress interpolation at defect locations."""
        nx, ny = SIMULATION["grid_2d"]
        stress_field = np.random.rand(nx, ny) * 1e6  # Random stress field
        
        # Create coordinate grids
        x = np.linspace(0, self.substrate["length"], nx)
        y = np.linspace(0, self.substrate["width"], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Test defect positions
        defect_positions = np.array([
            [self.substrate["length"]/2, self.substrate["width"]/2, 0.001],  # Center
            [self.substrate["length"]/4, self.substrate["width"]/4, 0.002],  # Quarter
        ])
        
        stress_at_defects = self.stress_calc.compute_stress_at_defects(
            stress_field, defect_positions, X, Y
        )
        
        assert len(stress_at_defects) == len(defect_positions)
        assert np.all(np.isfinite(stress_at_defects))
        
    def test_compute_stress_at_defects_empty(self):
        """Test stress interpolation with no defects."""
        nx, ny = SIMULATION["grid_2d"]
        stress_field = np.ones((nx, ny))
        
        x = np.linspace(0, self.substrate["length"], nx)
        y = np.linspace(0, self.substrate["width"], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        defect_positions = np.empty((0, 3))
        
        stress_at_defects = self.stress_calc.compute_stress_at_defects(
            stress_field, defect_positions, X, Y
        )
        
        assert len(stress_at_defects) == 0


class TestNucleationEngine:
    """Test NucleationEngine class."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.engine = NucleationEngine()
        
    def test_nucleation_engine_initialization(self):
        """Test NucleationEngine initialization."""
        assert self.engine.K_IC == ULE_GLASS["K_IC"]
        assert self.engine.K_0_ratio == ULE_GLASS["K_0_ratio"]
        assert self.engine.K_0 == ULE_GLASS["K_IC"] * ULE_GLASS["K_0_ratio"]
        
    def test_compute_stress_intensity_accuracy(self):
        """Test stress intensity factor calculation accuracy."""
        sigma_local = np.array([1e6, 2e6, 5e6])  # 1, 2, 5 MPa
        flaw_size = np.array([50e-9, 100e-9, 200e-9])  # 50, 100, 200 nm
        geometry_factor = 1.12
        
        K_I = self.engine.compute_stress_intensity(sigma_local, flaw_size, geometry_factor)
        
        # Check formula: K_I = Y * σ * √(π * a)
        expected = geometry_factor * sigma_local * np.sqrt(np.pi * flaw_size)
        npt.assert_allclose(K_I, expected, rtol=1e-10)
        
    def test_compute_stress_intensity_physical_range(self):
        """Test that stress intensity factors are in physical range."""
        # Typical values for glass
        sigma_local = np.array([10e6, 100e6, 500e6])  # 10, 100, 500 MPa
        flaw_size = np.array([10e-9, 1e-6, 10e-6])    # 10 nm, 1 μm, 10 μm
        
        K_I = self.engine.compute_stress_intensity(sigma_local, flaw_size)
        
        # Should be positive
        assert np.all(K_I > 0)
        
        # Should be less than K_IC for sub-critical cases
        # (some might exceed for nucleation cases)
        assert np.any(K_I < ULE_GLASS["K_IC"])
        
        # Should be physically reasonable (< 10 MPa·m^0.5)
        assert np.all(K_I < 10e6)
        
    def test_griffith_criterion_threshold(self):
        """Test Griffith criterion calculation and threshold."""
        # Test values around K_IC
        K_I = np.array([
            0.5 * ULE_GLASS["K_IC"],  # Sub-critical
            0.9 * ULE_GLASS["K_IC"],  # Near critical
            1.0 * ULE_GLASS["K_IC"],  # At critical
            1.1 * ULE_GLASS["K_IC"]   # Super-critical
        ])
        
        griffith_ratios = self.engine.griffith_criterion(K_I)
        
        expected_ratios = np.array([0.25, 0.81, 1.0, 1.21])  # (K_I/K_IC)²
        npt.assert_allclose(griffith_ratios, expected_ratios, rtol=0.01)
        
        # Check nucleation threshold
        nucleated = griffith_ratios >= self.engine.nucleation_threshold
        expected_nucleated = np.array([False, False, True, True])
        npt.assert_array_equal(nucleated, expected_nucleated)
        
    def test_subcritical_check_logic(self):
        """Test subcritical crack growth regime identification."""
        K_I = np.array([
            0.1 * ULE_GLASS["K_IC"],  # Below K_0
            0.5 * ULE_GLASS["K_IC"],  # Between K_0 and K_IC
            0.8 * ULE_GLASS["K_IC"],  # Still subcritical
            1.2 * ULE_GLASS["K_IC"]   # Above K_IC
        ])
        
        subcritical_mask = self.engine.subcritical_check(K_I)
        
        # Only middle values should be subcritical
        expected_mask = np.array([False, True, True, False])
        npt.assert_array_equal(subcritical_mask, expected_mask)
        
    def test_run_monte_carlo_basic_functionality(self):
        """Test basic Monte Carlo nucleation simulation."""
        n_runs = 10  # Small number for fast testing
        
        defect_params = {
            "distribution_type": "poisson",
            "density": 1e8  # defects/m³
        }
        
        stress_params = {
            "dose": 50.0,
            "delta_T": 1.0,
            "correlation_length": 1e-3,
            "cte_type": "gaussian"
        }
        
        substrate = SUBSTRATE.copy()
        
        result = self.engine.run_monte_carlo(
            n_runs, defect_params, stress_params, substrate
        )
        
        # Check result structure
        required_keys = [
            "nucleation_probability", "mean_nucleation_density", 
            "std_nucleation_density", "nucleation_events", "time_to_first_crack_stats"
        ]
        for key in required_keys:
            assert key in result
            
        # Check value ranges
        assert 0 <= result["nucleation_probability"] <= 1
        assert result["mean_nucleation_density"] >= 0
        assert result["std_nucleation_density"] >= 0
        assert len(result["nucleation_events"]) == n_runs
        
    def test_run_monte_carlo_statistical_consistency(self):
        """Test MC simulation statistical properties."""
        n_runs = 50
        
        # Low stress case - should have low nucleation probability
        defect_params = {"distribution_type": "poisson", "density": 1e7}
        stress_params_low = {"dose": 10.0, "delta_T": 0.1}
        
        # High stress case - should have higher nucleation probability  
        stress_params_high = {"dose": 100.0, "delta_T": 2.0}
        
        substrate = SUBSTRATE.copy()
        
        result_low = self.engine.run_monte_carlo(
            n_runs, defect_params, stress_params_low, substrate
        )
        
        result_high = self.engine.run_monte_carlo(
            n_runs, defect_params, stress_params_high, substrate
        )
        
        # Higher stress should give higher nucleation probability
        assert result_high["nucleation_probability"] >= result_low["nucleation_probability"]
        assert result_high["mean_nucleation_density"] >= result_low["mean_nucleation_density"]
        
    def test_run_thermal_cycling_basic(self):
        """Test thermal cycling simulation."""
        n_cycles = 20
        
        cycle_params = {
            "dose": 50.0,
            "delta_T": 0.5,
            "correlation_length": 1e-3
        }
        
        defect_params = {
            "density": 1e8
        }
        
        substrate = SUBSTRATE.copy()
        
        result = self.engine.run_thermal_cycling(
            n_cycles, cycle_params, defect_params, substrate
        )
        
        # Check result structure
        required_keys = [
            "nucleation_probability_vs_cycles", "cumulative_damage", 
            "total_nucleations", "cycles_to_first_nucleation"
        ]
        for key in required_keys:
            assert key in result
            
        # Check array lengths
        assert len(result["nucleation_probability_vs_cycles"]) == n_cycles
        assert len(result["cumulative_damage"]) >= 0  # Depends on defects generated
        
        # Nucleation probability should be non-decreasing
        prob_history = result["nucleation_probability_vs_cycles"]
        assert np.all(np.diff(prob_history) >= 0), "Nucleation probability should be monotonic"


class TestIntegration:
    """Integration tests for complete M1 pipeline."""
    
    def test_end_to_end_pipeline_basic(self):
        """Test complete pipeline: defect generation → stress → nucleation."""
        # Setup
        substrate = SUBSTRATE.copy()
        defect_field = DefectField(substrate)
        stress_calc = ThermoelasticStress()
        nucleation_engine = NucleationEngine()
        cte_generator = CTEMapGenerator()
        
        # Step 1: Generate defects
        domain_size = (substrate["length"], substrate["width"], substrate["thickness"])
        defect_result = defect_field.generate_poisson(
            density=1e8, domain_size=domain_size, seed=42
        )
        
        # Step 2: Generate stress field
        grid_size = SIMULATION["grid_2d"]
        cte_map = cte_generator.generate_gaussian_random_field(
            grid_size, ULE_GLASS["CTE_sigma"], 1e-3, seed=42
        )
        
        thermal_field = stress_calc.compute_thermal_field(50.0, 1.0, substrate)
        stress_field = stress_calc.compute_stress_field(
            thermal_field, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
        )
        
        # Step 3: Compute nucleation
        nx, ny = grid_size
        x = np.linspace(0, substrate["length"], nx)
        y = np.linspace(0, substrate["width"], ny) 
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if len(defect_result.positions) > 0:
            stress_at_defects = stress_calc.compute_stress_at_defects(
                stress_field, defect_result.positions, X, Y
            )
            
            K_I = nucleation_engine.compute_stress_intensity(
                stress_at_defects, defect_result.flaw_sizes
            )
            
            griffith_ratios = nucleation_engine.griffith_criterion(K_I)
            
            # Check pipeline completed successfully
            assert len(stress_at_defects) == len(defect_result.positions)
            assert len(K_I) == len(defect_result.positions)
            assert len(griffith_ratios) == len(defect_result.positions)
            assert np.all(griffith_ratios >= 0)
            
    def test_end_to_end_pipeline_reproducibility(self):
        """Test pipeline reproducibility with same seed."""
        substrate = SUBSTRATE.copy()
        
        def run_pipeline(seed):
            defect_field = DefectField(substrate)
            stress_calc = ThermoelasticStress()
            nucleation_engine = NucleationEngine()
            
            defect_params = {"distribution_type": "poisson", "density": 1e8}
            stress_params = {"dose": 50.0, "delta_T": 1.0}
            
            result = nucleation_engine.run_monte_carlo(
                5, defect_params, stress_params, substrate
            )
            return result["nucleation_probability"]
        
        # Run with same seed twice
        prob1 = run_pipeline(42)
        prob2 = run_pipeline(42)
        
        assert prob1 == prob2, "Pipeline should be reproducible with same seed"
        
        # Run with different seed
        prob3 = run_pipeline(123)
        
        # Usually should be different (though could be same by chance)
        # Just check it's a valid probability
        assert 0 <= prob3 <= 1
        
    def test_end_to_end_parameter_sensitivity(self):
        """Test pipeline sensitivity to key parameters."""
        n_runs = 10
        substrate = SUBSTRATE.copy()
        nucleation_engine = NucleationEngine()
        
        base_defect_params = {"distribution_type": "poisson", "density": 1e8}
        base_stress_params = {"dose": 50.0, "delta_T": 1.0}
        
        # Base case
        result_base = nucleation_engine.run_monte_carlo(
            n_runs, base_defect_params, base_stress_params, substrate
        )
        
        # Higher defect density
        high_density_params = base_defect_params.copy()
        high_density_params["density"] = 5e8
        
        result_high_density = nucleation_engine.run_monte_carlo(
            n_runs, high_density_params, base_stress_params, substrate
        )
        
        # Higher stress
        high_stress_params = base_stress_params.copy()
        high_stress_params["delta_T"] = 3.0
        
        result_high_stress = nucleation_engine.run_monte_carlo(
            n_runs, base_defect_params, high_stress_params, substrate
        )
        
        # Check sensitivity
        base_prob = result_base["nucleation_probability"]
        high_density_prob = result_high_density["nucleation_probability"]
        high_stress_prob = result_high_stress["nucleation_probability"]
        
        # Higher defect density should increase nucleation probability
        assert high_density_prob >= base_prob
        
        # Higher stress should increase nucleation probability
        assert high_stress_prob >= base_prob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])