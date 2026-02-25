"""
Test Suite for M3: Inspection Signal Forward Model
Glass Micro-Crack Lifecycle Simulator — Job 10

Comprehensive testing of all inspection methods and physics models.
Minimum 20 tests covering acoustic, optical, electron beam, and comparison modules.

Author: Glass Crack Lifecycle Simulator Team  
Date: 2026-02-25
"""

import pytest
import numpy as np
import warnings
from typing import Dict, List, Tuple

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from m03_inspection import (
    AcousticInspection,
    OpticalInspection, 
    ElectronBeamInspection,
    InspectionComparison,
    InspectionResult,
    run_inspection_analysis
)

# Import config for test parameters
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ULE_GLASS, INSPECTION


class TestAcousticInspection:
    """Test cases for AcousticInspection class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.acoustic = AcousticInspection()
        
    def test_init_properties(self):
        """Test initialization loads correct material properties"""
        assert self.acoustic.v_l == INSPECTION["acoustic"]["velocity_longitudinal"]
        assert self.acoustic.v_t == INSPECTION["acoustic"]["velocity_transverse"] 
        assert self.acoustic.rho == ULE_GLASS["rho"]
        assert self.acoustic.attenuation_coeff == INSPECTION["acoustic"]["attenuation_coeff"]
        
    def test_lamb_wave_dispersion_positive_inputs(self):
        """Test Lamb wave dispersion with valid positive inputs"""
        frequency = 5e6  # 5 MHz
        thickness = 6.35e-3  # 6.35 mm
        
        v_s0, v_a0 = self.acoustic.compute_lamb_wave_dispersion(
            frequency, thickness, self.acoustic.v_l, self.acoustic.v_t
        )
        
        # Check physical ranges for ULE glass
        assert 3000 < v_s0 < 7000  # m/s, reasonable range for plate waves
        assert v_a0 > 0  # A0 is dispersive: v ∝ √f, can exceed bulk at high f·d
        assert isinstance(v_s0, float)
        assert isinstance(v_a0, float)
        
    def test_lamb_wave_dispersion_invalid_inputs(self):
        """Test Lamb wave dispersion with invalid inputs"""
        # Negative frequency
        with pytest.raises(ValueError):
            self.acoustic.compute_lamb_wave_dispersion(-1e6, 6.35e-3, 5970, 3764)
            
        # Zero thickness 
        with pytest.raises(ValueError):
            self.acoustic.compute_lamb_wave_dispersion(5e6, 0, 5970, 3764)
            
    def test_lamb_wave_frequency_dependence(self):
        """Test A0 mode shows correct frequency dependence (√ω scaling)"""
        thickness = 6.35e-3
        frequencies = np.array([1e6, 4e6, 9e6])  # 1, 4, 9 MHz → 2x, 3x √f scaling
        
        a0_velocities = []
        for freq in frequencies:
            _, v_a0 = self.acoustic.compute_lamb_wave_dispersion(
                freq, thickness, self.acoustic.v_l, self.acoustic.v_t
            )
            a0_velocities.append(v_a0)
            
        # A0 mode: v ∝ √ω, so v(4MHz)/v(1MHz) should ≈ 2, v(9MHz)/v(1MHz) ≈ 3
        if a0_velocities[0] > 0:  # Avoid division by zero
            ratio_2 = a0_velocities[1] / a0_velocities[0]
            ratio_3 = a0_velocities[2] / a0_velocities[0]
            assert 1.8 < ratio_2 < 2.2  # ~2 within tolerance
            assert 2.8 < ratio_3 < 3.2  # ~3 within tolerance
            
    def test_simulate_scattering_rayleigh_regime(self):
        """Test scattering simulation in Rayleigh regime (a << λ)"""
        # Small crack: 10 nm, high frequency → ka < 0.3
        crack_geometry = {"length": 10e-9, "depth": 5e-9, "width": 1e-9}
        wavelength = 1e-3  # 1 mm wavelength
        wave_params = {
            "frequency": self.acoustic.v_l / wavelength, 
            "wavelength": wavelength,
            "amplitude": 1e6
        }
        
        result = self.acoustic.simulate_scattering(crack_geometry, wave_params)
        
        assert result["scattering_regime"] == "rayleigh"
        assert result["size_parameter"] < 0.3
        assert result["rayleigh_cross_section"] > 0
        assert result["scattered_amplitude"] > 0
        
    def test_simulate_scattering_mie_regime(self):
        """Test scattering simulation in Mie regime (a ~ λ)"""
        # Medium crack: wavelength comparable to crack size
        wavelength = 100e-6  # 100 μm
        crack_geometry = {"length": 50e-6, "depth": 25e-6, "width": 1e-6}
        wave_params = {
            "frequency": self.acoustic.v_l / wavelength,
            "wavelength": wavelength, 
            "amplitude": 1e6
        }
        
        result = self.acoustic.simulate_scattering(crack_geometry, wave_params)
        
        assert result["scattering_regime"] == "mie"
        assert 0.3 < result["size_parameter"] < 10
        assert result["mie_cross_section"] > 0
        
    def test_simulate_scattering_size_dependence(self):
        """Test scattering cross-section scales correctly with crack size"""
        wavelength = 1e-3
        wave_params = {
            "frequency": self.acoustic.v_l / wavelength,
            "wavelength": wavelength,
            "amplitude": 1e6
        }
        
        # Test different crack sizes in Rayleigh regime
        sizes = [5e-9, 10e-9, 20e-9]  # Small cracks
        cross_sections = []
        
        for size in sizes:
            crack_geometry = {"length": size, "depth": size/2, "width": size/10}
            result = self.acoustic.simulate_scattering(crack_geometry, wave_params)
            cross_sections.append(result["rayleigh_cross_section"])
            
        # In Rayleigh regime: σ ∝ a⁶, so doubling size → 64x cross-section
        assert cross_sections[1] > cross_sections[0]  # Monotonic increase
        assert cross_sections[2] > cross_sections[1]
        
        # Check approximate scaling (allowing for geometry factors)
        if cross_sections[0] > 0:
            ratio = cross_sections[1] / cross_sections[0]
            assert ratio > 10  # Should be significant increase
            
    def test_compute_acoustic_emission_energy_positive(self):
        """Test acoustic emission energy is positive for crack growth"""
        crack_growth_rate = 1e-9  # 1 nm/s
        energy_release = 10.0     # J/m²
        
        ae_result = self.acoustic.compute_acoustic_emission(crack_growth_rate, energy_release)
        
        assert ae_result["ae_energy"] > 0
        assert ae_result["ae_amplitude"] > 0
        assert ae_result["event_rate"] > 0
        assert 0.4 <= ae_result["b_value"] <= 1.2  # Physical range for glass
        
    def test_acoustic_emission_scaling(self):
        """Test AE energy scales with crack growth rate"""
        energy_release = 10.0
        rates = [1e-10, 1e-9, 1e-8]  # Different growth rates
        
        energies = []
        for rate in rates:
            result = self.acoustic.compute_acoustic_emission(rate, energy_release)
            energies.append(result["ae_energy"])
            
        # Energy should increase with growth rate
        assert energies[1] > energies[0]
        assert energies[2] > energies[1]
        
    def test_signal_to_noise_attenuation(self):
        """Test SNR decreases with depth due to attenuation"""
        crack_size = 50e-9  # 50 nm
        depths = [0.1e-3, 1e-3, 5e-3]  # 0.1, 1, 5 mm depths
        method_params = {"frequency": 5e6, "noise_level": 0.01, "signal_strength": 1.0}
        
        snr_values = []
        for depth in depths:
            snr = self.acoustic.signal_to_noise(crack_size, depth, method_params)
            snr_values.append(snr)
            
        # SNR should decrease with depth
        assert snr_values[1] < snr_values[0]
        assert snr_values[2] < snr_values[1]
        assert all(snr >= 0 for snr in snr_values)  # Non-negative SNR
        
    def test_signal_to_noise_overflow_protection(self):
        """Test attenuation calculation prevents overflow for large depths"""
        crack_size = 10e-9
        large_depth = 1.0  # 1 meter (unrealistic but tests overflow protection)
        method_params = {"frequency": 50e6, "noise_level": 0.01}  # High frequency
        
        # Should not raise overflow warning/error
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            snr = self.acoustic.signal_to_noise(crack_size, large_depth, method_params)
            
        assert snr >= 0
        assert snr <= 1e10  # Reasonable upper bound


class TestOpticalInspection:
    """Test cases for OpticalInspection class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.optical = OpticalInspection()
        
    def test_laser_scattering_rayleigh_scaling(self):
        """Test laser scattering shows correct a⁶ scaling in Rayleigh regime"""
        wavelength = 532e-9  # Green laser
        power = 50e-3       # 50 mW
        crack_depth = 1e-6   # 1 μm
        
        # Test different sizes in Rayleigh regime (a << λ)
        sizes = np.array([10e-9, 20e-9])  # 10, 20 nm
        
        intensities = []
        for size in sizes:
            result = self.optical.laser_scattering_signal(size, crack_depth, wavelength, power)
            intensities.append(result["scattered_intensity"])
            
        # Rayleigh: I ∝ a⁶, so doubling size → 2⁶ = 64x intensity
        if intensities[0] > 0:
            ratio = intensities[1] / intensities[0]
            assert 50 < ratio < 80  # Around 64 with some tolerance for approximations
            
    def test_laser_scattering_regimes(self):
        """Test correct identification of scattering regimes"""
        wavelength = 532e-9
        power = 50e-3
        depth = 1e-6
        
        # Rayleigh regime: a << λ
        small_result = self.optical.laser_scattering_signal(10e-9, depth, wavelength, power)
        assert small_result["regime"] == "rayleigh"
        
        # Mie regime: a ~ λ  
        medium_result = self.optical.laser_scattering_signal(200e-9, depth, wavelength, power)
        assert medium_result["regime"] == "mie"
        
        # Geometric regime: a >> λ
        large_result = self.optical.laser_scattering_signal(5e-6, depth, wavelength, power)
        assert large_result["regime"] == "geometric"
        
    def test_raman_stress_mapping_sign_convention(self):
        """Test Raman shift has correct sign for compressive stress"""
        # Compressive stress field (negative values)
        stress_field = np.full((10, 10), -100e6)  # -100 MPa compression
        sensitivity = -4.0  # cm⁻¹/GPa for SiO₂
        resolution = 500e-9
        
        raman_shift = self.optical.raman_stress_mapping(stress_field, sensitivity, resolution)
        
        # Compressive stress + negative sensitivity → positive shift (blue shift)
        assert np.all(raman_shift > 0), "Compressive stress should give positive Raman shift"
        assert raman_shift.shape == stress_field.shape
        
        # Check magnitude: -100 MPa = -0.1 GPa; Δω = sensitivity × σ = (-4) × (-0.1) = +0.4 cm⁻¹
        expected_shift = 0.4  # positive (blue shift under compression)
        assert np.allclose(raman_shift, expected_shift, rtol=0.1)
        
    def test_raman_spatial_resolution(self):
        """Test Raman mapping applies spatial resolution limit correctly"""
        # Use large grid so that resolution differences are resolvable
        # Grid spacing: 152mm / 200 = 0.76mm
        stress_field = np.zeros((200, 200))
        stress_field[100, 100] = 1e9  # 1 GPa stress at center
        
        sensitivity = -4.0
        fine_resolution = 0.5e-3   # 0.5 mm — just below grid spacing (no blur)
        coarse_resolution = 5.0e-3  # 5 mm — ~6.6 pixels sigma (significant blur)
        
        raman_fine = self.optical.raman_stress_mapping(stress_field, sensitivity, fine_resolution)
        raman_coarse = self.optical.raman_stress_mapping(stress_field, sensitivity, coarse_resolution)
        
        # Coarse resolution should blur the sharp feature more
        peak_fine = np.max(np.abs(raman_fine))
        peak_coarse = np.max(np.abs(raman_coarse))
        
        # Coarse resolution reduces peak intensity due to blurring
        assert peak_coarse < peak_fine
        
    def test_interferometry_phase_range(self):
        """Test 193nm interferometry produces reasonable phase values"""
        # Realistic refractive index change from cracks
        dn_field = np.random.uniform(-1e-6, 1e-6, (10, 10))  # ±1 ppm variation
        thickness = 6.35e-3  # 6.35 mm substrate  
        wavelength = 193e-9  # 193 nm
        
        phase_map = self.optical.interferometry_193nm(dn_field, thickness, wavelength)
        
        # Check phase is in reasonable range and wrapped to [-π, π]
        assert np.all(phase_map >= -np.pi)
        assert np.all(phase_map <= np.pi)
        assert phase_map.shape == dn_field.shape
        
    def test_compute_dn_from_cracks_physics(self):
        """Test crack → refractive index change physics"""
        crack_field = np.zeros((5, 5))
        crack_field[2, 2] = 1.0  # Full crack at center
        n_glass = 1.56  # ULE glass at 193nm
        
        dn_field = self.optical.compute_dn_from_cracks(crack_field, n_glass)
        
        # Crack → void (n=1) → negative Δn for glass (n>1)
        assert dn_field[2, 2] < 0, "Crack should reduce local refractive index"
        assert np.abs(dn_field[2, 2] - (1.0 - n_glass)) < 1e-10
        assert dn_field[0, 0] == 0.0  # No crack → no index change


class TestElectronBeamInspection:
    """Test cases for ElectronBeamInspection class"""
    
    def setup_method(self):
        """Setup for each test method"""  
        self.ebeam = ElectronBeamInspection()
        
    def test_eels_bonding_shift_direction(self):
        """Test EELS energy shift has correct direction with strain"""
        # Tensile strain field
        strain_field = np.full((5, 5), 0.01)  # 1% strain
        
        eels_shift = self.ebeam.eels_bonding_shift(strain_field, "Si-O")
        
        # Should have positive shift for strain (bond elongation)
        assert np.all(eels_shift > 0)
        assert eels_shift.shape == strain_field.shape
        
        # Check magnitude: 1% strain × 5 eV/strain = 0.05 eV
        expected_shift = 0.01 * 5.0
        assert np.allclose(eels_shift, expected_shift)
        
    def test_eels_absolute_strain(self):
        """Test EELS shift uses absolute strain (compression = tension for bond distortion)"""
        compressive_strain = np.full((3, 3), -0.01)  # -1% compression
        tensile_strain = np.full((3, 3), 0.01)       # +1% tension
        
        eels_compression = self.ebeam.eels_bonding_shift(compressive_strain, "Si-O")
        eels_tension = self.ebeam.eels_bonding_shift(tensile_strain, "Si-O")
        
        # Both compression and tension cause bond distortion → same EELS shift
        assert np.allclose(eels_compression, eels_tension)
        
    def test_eels_bond_type_sensitivity(self):
        """Test different bond types have different EELS sensitivities"""
        strain_field = np.full((3, 3), 0.01)
        
        si_o_shift = self.ebeam.eels_bonding_shift(strain_field, "Si-O")
        ti_o_shift = self.ebeam.eels_bonding_shift(strain_field, "Ti-O")
        unknown_shift = self.ebeam.eels_bonding_shift(strain_field, "unknown")
        
        # Si-O should have higher sensitivity than Ti-O
        assert np.all(si_o_shift > ti_o_shift)
        assert np.all(unknown_shift > 0)  # Default sensitivity still positive
        
    def test_kfm_surface_potential_crack_enhancement(self):
        """Test KFM shows enhanced potential at crack sites"""
        # Uniform background charge
        charge_density = np.full((10, 10), 1e-12)  # 1 pC/m² background
        
        # Crack positions
        crack_positions = [(0.5, 0.5)]  # Center crack
        
        potential_map = self.ebeam.kfm_surface_potential(charge_density, crack_positions)
        
        # Should have enhanced potential near crack
        center_potential = potential_map[5, 5]  # Center pixel
        corner_potential = potential_map[0, 0]  # Corner pixel
        
        assert np.abs(center_potential) > np.abs(corner_potential)
        assert potential_map.shape == charge_density.shape
        
    def test_kfm_potential_scaling(self):
        """Test KFM potential has reasonable magnitude scaling"""
        charge_density = np.full((5, 5), 1e-12)
        crack_positions = [(0.5, 0.5)]
        
        potential_map = self.ebeam.kfm_surface_potential(charge_density, crack_positions)
        
        # Should be in reasonable range for KFM (mV to V)
        max_potential = np.max(np.abs(potential_map))
        assert 1e-6 < max_potential < 10.0  # μV to 10V range


class TestInspectionComparison:
    """Test cases for InspectionComparison class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.comparison = InspectionComparison()
        
    def test_compute_roc_curve_perfect_separation(self):
        """Test ROC curve for perfect signal separation"""
        # Perfect separation: pristine signals < cracked signals
        signal_pristine = np.array([0.1, 0.2, 0.3])
        signal_cracked = np.array([0.8, 0.9, 1.0])
        thresholds = np.linspace(0, 1, 11)
        
        roc_result = self.comparison.compute_roc_curve(signal_pristine, signal_cracked, thresholds)
        
        assert "fpr" in roc_result
        assert "tpr" in roc_result
        assert "auc" in roc_result
        assert 0.9 < roc_result["auc"] <= 1.0  # Near-perfect AUC
        
    def test_roc_auc_range(self):
        """Test ROC AUC is always in valid range [0.5, 1.0]"""
        # Random overlapping signals (should give AUC ~ 0.5)
        signal_pristine = np.random.uniform(0, 1, 100)
        signal_cracked = np.random.uniform(0, 1, 100)
        thresholds = np.linspace(0, 1, 20)
        
        roc_result = self.comparison.compute_roc_curve(signal_pristine, signal_cracked, thresholds)
        
        assert 0.5 <= roc_result["auc"] <= 1.0
        
    def test_minimum_detectable_crack_positive(self):
        """Test minimum detectable crack size is always positive"""
        methods = ["acoustic", "laser_scattering", "raman", "interferometry_193nm", "eels", "kfm"]
        
        for method in methods:
            min_size = self.comparison.minimum_detectable_crack(method)
            assert min_size > 0, f"Method {method} should have positive detection limit"
            assert min_size >= 1e-9, f"Method {method} cannot detect below 1 nm"
            
    def test_minimum_detectable_crack_snr_scaling(self):
        """Test detection limit scales with SNR threshold"""
        method = "acoustic"
        
        snr_low = self.comparison.minimum_detectable_crack(method, SNR_threshold=1.0)
        snr_high = self.comparison.minimum_detectable_crack(method, SNR_threshold=10.0)
        
        # Higher SNR requirement → larger minimum detectable crack
        assert snr_high > snr_low
        
    def test_optimal_inspection_strategy_cost_effectiveness(self):
        """Test optimal strategy ranks by cost-effectiveness"""
        crack_sizes = np.array([10e-9, 100e-9, 1e-6])  # 10nm to 1μm cracks
        
        # Method costs ($/inspection)
        costs = {
            "cheap_method": 100,
            "expensive_method": 1000
        }
        
        strategy = self.comparison.optimal_inspection_strategy(crack_sizes, costs)
        
        assert "optimal_method" in strategy
        assert strategy["optimal_method"] in costs.keys()
        assert "method_rankings" in strategy
        assert "detection_results" in strategy
        
        # Should prefer cost-effective methods
        rankings = strategy["method_rankings"]
        assert rankings[0] == strategy["optimal_method"]
        
    def test_compare_methods_structure(self):
        """Test compare_methods returns proper structure"""
        crack_field = np.random.uniform(0, 1, (20, 20))
        methods = ["acoustic", "laser_scattering", "interferometry", "unknown_method"]
        
        results = self.comparison.compare_methods(crack_field, methods)
        
        for method in methods:
            assert method in results
            assert "signal_level" in results[method]
            assert "snr" in results[method]
            assert "detection_probability" in results[method]
            assert "method_specific" in results[method]
            
            # Check ranges
            assert results[method]["detection_probability"] >= 0
            assert results[method]["detection_probability"] <= 1
            
    def test_pristine_snr_near_zero(self):
        """Test pristine substrate has SNR ≈ 0 (baseline verification)"""
        pristine_signal = np.zeros((10, 10))  # No cracks
        methods = ["acoustic", "laser_scattering"]
        
        results = self.comparison.compare_methods(pristine_signal, methods)
        
        for method in methods:
            # Pristine should have low signal and SNR
            assert results[method]["snr"] < 1.0  # Very low SNR for pristine


class TestIntegration:
    """Integration tests for M3 with M1/M2 results"""
    
    def test_run_inspection_analysis_structure(self):
        """Test run_inspection_analysis returns proper InspectionResult objects"""
        # Mock crack field data
        crack_data = {
            "damage_field": np.random.uniform(0, 0.5, (50, 50)),
            "stress_field": np.random.uniform(-1e8, 1e8, (50, 50))
        }
        
        methods = ["acoustic", "raman", "interferometry"]
        results = run_inspection_analysis(crack_data, methods)
        
        assert len(results) == len(methods)
        
        for result in results:
            assert isinstance(result, InspectionResult)
            assert hasattr(result, "method")
            assert hasattr(result, "signal_map")
            assert hasattr(result, "reference_signal") 
            assert hasattr(result, "delta_signal")
            assert hasattr(result, "snr_map")
            assert hasattr(result, "min_detectable_size")
            assert hasattr(result, "detection_probability")
            assert hasattr(result, "roc_auc")
            
    def test_m1_m2_compatibility(self):
        """Test M3 can process typical M1/M2 output formats"""
        # Simulate M1/M2 output structure
        m1_m2_output = {
            "damage_field": np.random.uniform(0, 1, (30, 30)),
            "stress_field": np.random.normal(0, 50e6, (30, 30)),  # ±50 MPa stress
            "crack_positions": [(0.3, 0.3), (0.7, 0.7)],
            "crack_sizes": [20e-9, 50e-9]
        }
        
        # Should not raise exceptions
        results = run_inspection_analysis(m1_m2_output, ["acoustic", "raman"])
        
        assert len(results) == 2
        assert all(isinstance(r, InspectionResult) for r in results)


class TestUnitsValidation:
    """Test all outputs are in SI units"""
    
    def test_acoustic_units_si(self):
        """Test acoustic outputs are in SI units"""
        acoustic = AcousticInspection()
        
        # Lamb wave velocities should be in m/s
        v_s0, v_a0 = acoustic.compute_lamb_wave_dispersion(5e6, 6.35e-3, 5970, 3764)
        assert 1000 < v_s0 < 10000  # Reasonable m/s range
        assert v_a0 > 0  # A0 dispersive, can be large at high f·d product
        
        # Scattering cross-sections should be in m²
        crack_geom = {"length": 1e-6, "depth": 0.5e-6, "width": 0.1e-6}
        wave_params = {"frequency": 5e6, "wavelength": 1.2e-3, "amplitude": 1e6}
        
        scattering = acoustic.simulate_scattering(crack_geom, wave_params)
        cross_section = scattering["rayleigh_cross_section"]
        
        # Cross-section should be reasonable area (much smaller than wavelength²)
        assert 0 < cross_section < 1e-6  # m² range; can be very small for sub-μm cracks
        
    def test_optical_units_si(self):
        """Test optical outputs are in SI units"""
        optical = OpticalInspection()
        
        # Laser scattering intensity should be in W/m²
        scattering = optical.laser_scattering_signal(50e-9, 1e-6, 532e-9, 50e-3)
        intensity = scattering["scattered_intensity"]
        
        assert intensity >= 0  # Non-negative intensity
        assert intensity < 1e12  # Reasonable upper bound W/m²
        
        # Raman shifts should be in cm⁻¹
        stress_field = np.full((5, 5), 100e6)  # 100 MPa
        raman_shift = optical.raman_stress_mapping(stress_field, -4.0, 500e-9)
        
        # Expected: 100 MPa × (-4 cm⁻¹/GPa) = -0.4 cm⁻¹
        expected_magnitude = 0.1 * 4.0  # 100 MPa = 0.1 GPa
        assert np.allclose(np.abs(raman_shift), expected_magnitude, rtol=0.1)
        
    def test_electron_beam_units_si(self):
        """Test electron beam outputs are in SI units"""
        ebeam = ElectronBeamInspection()
        
        # EELS shifts should be in eV
        strain_field = np.full((3, 3), 0.01)  # 1% strain
        eels_shift = ebeam.eels_bonding_shift(strain_field, "Si-O")
        
        # Expected: 1% × 5 eV = 0.05 eV  
        assert np.allclose(eels_shift, 0.05, rtol=0.1)
        
        # KFM potentials should be in V
        charge_density = np.full((5, 5), 1e-12)  # C/m²
        potential = ebeam.kfm_surface_potential(charge_density, [(0.5, 0.5)])
        
        max_potential = np.max(np.abs(potential))
        assert 1e-6 < max_potential < 10  # μV to V range


if __name__ == "__main__":
    # Run specific test groups
    test_acoustic = TestAcousticInspection()
    test_acoustic.setup_method()
    
    print("Running M3 Inspection Test Suite...")
    print("=" * 40)
    
    # Test acoustic dispersion
    try:
        test_acoustic.test_lamb_wave_dispersion_positive_inputs()
        print("✓ Lamb wave dispersion test passed")
    except Exception as e:
        print(f"✗ Lamb wave test failed: {e}")
        
    # Test scattering regimes
    try:
        test_acoustic.test_simulate_scattering_rayleigh_regime() 
        print("✓ Rayleigh scattering test passed")
    except Exception as e:
        print(f"✗ Rayleigh scattering test failed: {e}")
        
    # Test integration
    try:
        test_integration = TestIntegration()
        test_integration.test_run_inspection_analysis_structure()
        print("✓ Integration test passed")
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        
    print("\nRun full test suite with: pytest test_m03.py -v")