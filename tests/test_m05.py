"""
Test Suite for Module M5: Process Attribution Engine

Tests for overlay degradation modeling, variance decomposition, replacement optimization,
process simulation, and attribution analysis.

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import Dict, List

# Import module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.m05_attribution import (
    OverlayDegradationModel, VarianceDecomposition, ReplacementOptimizer,
    ProcessSimulator, AttributionResult, ProcessMetrics, run_attribution_analysis
)
from config import ATTRIBUTION


class TestOverlayDegradationModel:
    """Test overlay degradation modeling functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = OverlayDegradationModel()
    
    def test_pristine_overlay_default_params(self):
        """Test pristine overlay computation with default parameters."""
        pristine = self.model.compute_pristine_overlay()
        
        # Should be RSS combination of config values
        expected = np.sqrt(
            ATTRIBUTION["overlay_scanner"]**2 + 
            ATTRIBUTION["overlay_mask_pristine"]**2 + 
            ATTRIBUTION["overlay_process"]**2
        )
        
        assert abs(pristine - expected) < 1e-6
        assert pristine > 0
    
    def test_pristine_overlay_custom_params(self):
        """Test pristine overlay with custom parameters."""
        scanner = 1.0
        mask = 0.6  
        process = 0.4
        
        pristine = self.model.compute_pristine_overlay(scanner, mask, process)
        expected = np.sqrt(scanner**2 + mask**2 + process**2)
        
        assert abs(pristine - expected) < 1e-6
    
    def test_pristine_overlay_negative_values(self):
        """Test error handling for negative sigma values."""
        with pytest.raises(ValueError, match="must be non-negative"):
            self.model.compute_pristine_overlay(scanner_sigma=-1.0)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            self.model.compute_pristine_overlay(mask_pristine_sigma=-0.5)
    
    def test_degraded_overlay_zero_crack(self):
        """Test that zero crack density gives pristine overlay."""
        pristine = self.model.compute_pristine_overlay()
        degraded = self.model.compute_degraded_overlay(0.0, pristine)
        
        # Should be essentially equal (within numerical precision)
        assert abs(degraded - pristine) < 1e-6
    
    def test_degraded_overlay_monotonic_increase(self):
        """Test that overlay degradation increases monotonically with crack density."""
        pristine = self.model.compute_pristine_overlay()
        crack_densities = np.logspace(6, 10, 10)  # 10^6 to 10^10 m^-2
        
        overlay_values = [self.model.compute_degraded_overlay(cd, pristine) for cd in crack_densities]
        
        # Should be monotonically increasing
        for i in range(1, len(overlay_values)):
            assert overlay_values[i] >= overlay_values[i-1]
        
        # First should be close to pristine (allowing for small degradation at low crack density)
        assert overlay_values[0] <= pristine * 1.05  # Within 5% of pristine
    
    def test_degraded_overlay_array_input(self):
        """Test degraded overlay computation with array input."""
        pristine = self.model.compute_pristine_overlay()
        crack_array = np.array([0.0, 1e8, 5e8, 1e9])
        
        degraded_array = self.model.compute_degraded_overlay(crack_array, pristine)
        
        assert len(degraded_array) == len(crack_array)
        assert np.all(degraded_array >= pristine)  # All should be >= pristine
        assert degraded_array[0] <= pristine + 1e-6  # Zero crack should be pristine
    
    def test_degraded_overlay_negative_crack(self):
        """Test error handling for negative crack density."""
        pristine = self.model.compute_pristine_overlay()
        
        with pytest.raises(ValueError, match="must be non-negative"):
            self.model.compute_degraded_overlay(-1e6, pristine)
    
    def test_overlay_time_series(self):
        """Test time series generation."""
        crack_history = np.array([0.0, 1e6, 5e6, 1e7, 5e7])
        time_points = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        
        times, overlay_series = self.model.overlay_time_series(crack_history, time_points)
        
        assert len(times) == len(overlay_series) == len(crack_history)
        npt.assert_array_equal(times, time_points)
        
        # Should be monotonically increasing (or at least non-decreasing)
        for i in range(1, len(overlay_series)):
            assert overlay_series[i] >= overlay_series[i-1] - 1e-6
    
    def test_overlay_time_series_default_times(self):
        """Test time series with default time points."""
        crack_history = np.array([0.0, 1e6, 5e6])
        
        times, overlay_series = self.model.overlay_time_series(crack_history)
        
        expected_times = np.array([0.0, 1.0, 2.0])
        npt.assert_array_equal(times, expected_times)
        assert len(overlay_series) == 3


class TestVarianceDecomposition:
    """Test variance decomposition and changepoint detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.decomposer = VarianceDecomposition()
    
    def test_decompose_basic(self):
        """Test basic variance decomposition."""
        # Create synthetic variance series
        total_variance_series = np.array([2.0, 2.1, 2.2, 2.5, 3.0]) # nm^2
        known_components = {"scanner": 0.8, "process": 0.3}
        
        result = self.decomposer.decompose(total_variance_series, known_components)
        
        # Check that components sum to 100%
        total_contribution = (result["scanner_contribution"] + 
                            result["mask_pristine_contribution"] +
                            result["mask_degradation_contribution"] + 
                            result["process_contribution"])
        
        assert abs(total_contribution - 100.0) < 1e-6
        
        # All contributions should be non-negative
        assert result["scanner_contribution"] >= 0
        assert result["mask_pristine_contribution"] >= 0
        assert result["mask_degradation_contribution"] >= 0
        assert result["process_contribution"] >= 0
    
    def test_decompose_pristine_case(self):
        """Test decomposition for pristine substrate (no degradation)."""
        # Variance series consistent with pristine substrate
        pristine_var = (ATTRIBUTION["overlay_scanner"]**2 + 
                       ATTRIBUTION["overlay_mask_pristine"]**2 + 
                       ATTRIBUTION["overlay_process"]**2)
        total_variance_series = np.full(10, pristine_var)
        
        known_components = {
            "scanner": ATTRIBUTION["overlay_scanner"],
            "process": ATTRIBUTION["overlay_process"]
        }
        
        result = self.decomposer.decompose(total_variance_series, known_components)
        
        # Degradation contribution should be nearly zero
        assert result["mask_degradation_contribution"] < 1.0  # Less than 1%
    
    def test_decompose_empty_series(self):
        """Test error handling for empty variance series."""
        with pytest.raises(ValueError, match="cannot be empty"):
            self.decomposer.decompose(np.array([]), {"scanner": 0.8})
    
    def test_decompose_negative_variance(self):
        """Test error handling for negative variance values."""
        with pytest.raises(ValueError, match="must be non-negative"):
            self.decomposer.decompose(np.array([1.0, -0.5, 2.0]), {"scanner": 0.8})
    
    def test_bayesian_changepoint_no_change(self):
        """Test changepoint detection on stable time series."""
        # Perfectly stable time series (no changepoint)
        stable_series = np.full(50, 1.0) + np.random.normal(0, 0.01, 50)  # Very small noise
        
        changepoint_time, confidence = self.decomposer.bayesian_changepoint(stable_series)
        
        # Should not detect changepoint or have low confidence
        assert changepoint_time is None or confidence < 0.7  # Allow for some noise effects
    
    def test_bayesian_changepoint_clear_change(self):
        """Test changepoint detection with clear step change."""
        # Create series with clear changepoint at t=25
        series_before = np.random.normal(1.0, 0.1, 25)
        series_after = np.random.normal(2.0, 0.1, 25)  # Clear step up
        step_series = np.concatenate([series_before, series_after])
        
        changepoint_time, confidence = self.decomposer.bayesian_changepoint(step_series)
        
        # Should detect changepoint near t=25 with reasonable confidence
        if changepoint_time is not None:
            assert abs(changepoint_time - 25) < 10  # Within 10 points
            assert confidence > 0.3  # Some confidence in detection
    
    def test_bayesian_changepoint_short_series(self):
        """Test changepoint detection on short series."""
        short_series = np.array([1.0, 1.1])
        
        changepoint_time, confidence = self.decomposer.bayesian_changepoint(short_series)
        
        # Should return no changepoint for very short series
        assert changepoint_time is None
        assert confidence == 0.0
    
    def test_attribute_excursion_no_excursion(self):
        """Test excursion attribution when no excursion present."""
        # Stable process data (deterministic, no excursion)
        stable_data = np.full(20, 1.0)  # Perfectly stable at 1.0 nm
        process_data = {"overlay": stable_data}
        crack_state = 1e6  # Low crack density
        
        result = self.decomposer.attribute_excursion(process_data, crack_state)
        
        assert result["crack_attribution"] == 0.0
        assert result["confidence"] > 0.5
    
    def test_attribute_excursion_with_excursion(self):
        """Test excursion attribution when excursion is present."""
        # Process data with excursion
        baseline = np.random.normal(1.0, 0.1, 15)
        excursion = np.array([3.0])  # Clear excursion
        process_data = {"overlay": np.concatenate([baseline, excursion])}
        
        crack_state = 1e9  # High crack density
        
        result = self.decomposer.attribute_excursion(process_data, crack_state)
        
        assert result["crack_attribution"] > 0.0
        assert result["confidence"] > 0.5


class TestReplacementOptimizer:
    """Test substrate replacement optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = ReplacementOptimizer()
    
    def test_compute_cost_continued_use_zero_crack(self):
        """Test continued use cost with zero crack density."""
        cost = self.optimizer.compute_cost_continued_use(
            crack_state=0.0, 
            n_lots=100
        )
        
        # Should be zero or very low cost (no degradation)
        assert cost >= 0.0
        assert cost < 1000.0  # Minimal cost for pristine substrate
    
    def test_compute_cost_continued_use_high_crack(self):
        """Test continued use cost with high crack density."""
        cost_low = self.optimizer.compute_cost_continued_use(crack_state=1e6, n_lots=10)
        cost_high = self.optimizer.compute_cost_continued_use(crack_state=1e9, n_lots=10)
        
        # Higher crack density should cost more
        assert cost_high > cost_low
        assert cost_low >= 0.0
        assert cost_high >= 0.0
    
    def test_compute_cost_continued_use_negative_inputs(self):
        """Test error handling for negative inputs."""
        with pytest.raises(ValueError, match="must be non-negative"):
            self.optimizer.compute_cost_continued_use(-1e6, 10)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            self.optimizer.compute_cost_continued_use(1e6, -5)
    
    def test_compute_cost_replacement_basic(self):
        """Test basic replacement cost computation."""
        downtime_hours = 8.0
        cost = self.optimizer.compute_cost_replacement(downtime_hours)
        
        # Should include substrate + inspection + downtime costs
        expected_min_cost = (ATTRIBUTION["cost_new_substrate"] + 
                           ATTRIBUTION["cost_inspection"])
        
        assert cost > expected_min_cost
        assert cost >= 0.0
    
    def test_compute_cost_replacement_zero_downtime(self):
        """Test replacement cost with zero downtime."""
        cost = self.optimizer.compute_cost_replacement(0.0)
        
        expected_cost = (ATTRIBUTION["cost_new_substrate"] + 
                        ATTRIBUTION["cost_inspection"])
        
        assert abs(cost - expected_cost) < 1e-6
    
    def test_compute_cost_replacement_negative_downtime(self):
        """Test error handling for negative downtime."""
        with pytest.raises(ValueError, match="must be non-negative"):
            self.optimizer.compute_cost_replacement(-1.0)
    
    def test_optimal_replacement_time_basic(self):
        """Test optimal replacement time calculation."""
        # Create crack trajectory that grows over time
        crack_trajectory = np.linspace(1e6, 1e9, 100)
        
        optimal_time, min_cost = self.optimizer.optimal_replacement_time(crack_trajectory)
        
        # Optimal time should be within valid range
        assert optimal_time is not None
        assert 0 <= optimal_time < len(crack_trajectory)
        assert min_cost >= 0.0
    
    def test_optimal_replacement_time_empty_trajectory(self):
        """Test error handling for empty trajectory."""
        with pytest.raises(ValueError, match="cannot be empty"):
            self.optimizer.optimal_replacement_time(np.array([]))
    
    def test_sensitivity_analysis_basic(self):
        """Test sensitivity analysis functionality."""
        params_ranges = {
            "substrate_cost": (30000, 70000),
            "downtime_rate": (3000, 7000),
            "yield_loss_rate": (5000, 15000)
        }
        
        crack_trajectory = np.linspace(1e6, 5e8, 50)
        
        results = self.optimizer.sensitivity_analysis(params_ranges, crack_trajectory, n_samples=5)
        
        # Should have results for each parameter
        assert len(results) == 3
        for param_name in params_ranges.keys():
            assert param_name in results
            assert "param_values" in results[param_name]
            assert "optimal_times" in results[param_name]
            assert "optimal_costs" in results[param_name]
            assert "sensitivity" in results[param_name]
            
            # Sensitivity should be non-negative
            assert results[param_name]["sensitivity"] >= 0.0


class TestProcessSimulator:
    """Test process simulation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ProcessSimulator()
    
    def test_simulate_lithography_process_low_na(self):
        """Test Low-NA EUV process simulation."""
        substrate_state = {"crack_density": 1e7}
        
        metrics = self.simulator.simulate_lithography_process(
            substrate_state, "low_na_euv", 25
        )
        
        assert isinstance(metrics, ProcessMetrics)
        assert metrics.process_type == "low_na_euv"
        assert metrics.overlay_total > 0
        assert metrics.cdu > 0
        assert metrics.epe > 0
        assert 0 <= metrics.yield_loss <= 1.0
        
        # X and Y overlay should be roughly equal
        assert abs(metrics.overlay_x - metrics.overlay_y) < 0.1
    
    def test_simulate_lithography_process_high_na(self):
        """Test High-NA EUV process simulation."""
        substrate_state = {"crack_density": 1e7}
        
        metrics = self.simulator.simulate_lithography_process(
            substrate_state, "high_na_euv", 25
        )
        
        assert metrics.process_type == "high_na_euv"
        assert metrics.overlay_total > 0
    
    def test_process_crack_sensitivity_comparison(self):
        """Test that High-NA is more sensitive to cracks than Low-NA."""
        substrate_state_clean = {"crack_density": 0.0}
        substrate_state_cracked = {"crack_density": 1e8}
        
        # Low-NA metrics
        low_na_clean = self.simulator.simulate_lithography_process(
            substrate_state_clean, "low_na_euv", 25
        )
        low_na_cracked = self.simulator.simulate_lithography_process(
            substrate_state_cracked, "low_na_euv", 25
        )
        
        # High-NA metrics
        high_na_clean = self.simulator.simulate_lithography_process(
            substrate_state_clean, "high_na_euv", 25
        )
        high_na_cracked = self.simulator.simulate_lithography_process(
            substrate_state_cracked, "high_na_euv", 25
        )
        
        # Calculate degradation for each process
        low_na_degradation = low_na_cracked.overlay_total - low_na_clean.overlay_total
        high_na_degradation = high_na_cracked.overlay_total - high_na_clean.overlay_total
        
        # High-NA should be more sensitive (larger degradation)
        assert high_na_degradation > low_na_degradation
    
    def test_simulate_lithography_process_invalid_type(self):
        """Test error handling for invalid process type."""
        substrate_state = {"crack_density": 1e7}
        
        with pytest.raises(ValueError, match="Unknown process type"):
            self.simulator.simulate_lithography_process(
                substrate_state, "invalid_process", 25
            )
    
    def test_simulate_lithography_process_zero_wafers(self):
        """Test error handling for zero wafers."""
        substrate_state = {"crack_density": 1e7}
        
        with pytest.raises(ValueError, match="must be positive"):
            self.simulator.simulate_lithography_process(
                substrate_state, "low_na_euv", 0
            )
    
    def test_simulate_lot_sequence(self):
        """Test lot-by-lot simulation."""
        n_lots = 10
        
        # Linear crack growth function
        def crack_evolution(lot_num):
            return 1e6 + lot_num * 1e7
        
        lot_metrics = self.simulator.simulate_lot_sequence(n_lots, crack_evolution)
        
        assert len(lot_metrics) == n_lots
        
        # Check that overlay generally increases with lot number (crack growth)
        overlay_values = [m.overlay_total for m in lot_metrics]
        
        # Should show increasing trend (allowing for some numerical noise)
        trend_slope = np.polyfit(range(n_lots), overlay_values, 1)[0]
        assert trend_slope >= -0.01  # Should be non-decreasing (allowing small tolerance)
    
    def test_simulate_lot_sequence_zero_lots(self):
        """Test error handling for zero lots."""
        def dummy_crack_evolution(lot_num):
            return 1e6
            
        with pytest.raises(ValueError, match="must be positive"):
            self.simulator.simulate_lot_sequence(0, dummy_crack_evolution)


class TestAttributionIntegration:
    """Test integrated attribution analysis."""
    
    def test_run_attribution_analysis_basic(self):
        """Test complete attribution analysis workflow."""
        # Create synthetic data
        crack_history = np.linspace(1e6, 1e8, 50)
        overlay_measurements = np.linspace(1.0, 2.5, 50)  # nm RMS
        
        result = run_attribution_analysis(crack_history, overlay_measurements)
        
        # Check result structure
        assert isinstance(result, AttributionResult)
        
        # Check that contributions sum to 100%
        total_contribution = (result.scanner_contribution + 
                            result.mask_pristine_contribution +
                            result.mask_degradation_contribution + 
                            result.process_contribution)
        assert abs(total_contribution - 100.0) < 1e-6
        
        # All contributions should be non-negative
        assert result.scanner_contribution >= 0
        assert result.mask_pristine_contribution >= 0  
        assert result.mask_degradation_contribution >= 0
        assert result.process_contribution >= 0
        
        # Cost savings should be non-negative
        assert result.cost_savings >= 0
        
        # Total overlay should be positive
        assert result.total_overlay > 0
    
    def test_run_attribution_analysis_pristine_substrate(self):
        """Test attribution analysis for pristine substrate."""
        # Zero crack density throughout
        crack_history = np.zeros(20)
        
        # Overlay measurements consistent with pristine performance
        pristine_overlay = np.sqrt(
            ATTRIBUTION["overlay_scanner"]**2 + 
            ATTRIBUTION["overlay_mask_pristine"]**2 + 
            ATTRIBUTION["overlay_process"]**2
        )
        overlay_measurements = np.random.normal(pristine_overlay, 0.05, 20)
        
        result = run_attribution_analysis(crack_history, overlay_measurements)
        
        # Degradation contribution should be minimal for pristine substrate
        assert result.mask_degradation_contribution < 5.0  # Less than 5%
    
    def test_run_attribution_analysis_empty_inputs(self):
        """Test error handling for empty input arrays."""
        with pytest.raises(ValueError, match="cannot be empty"):
            run_attribution_analysis(np.array([]), np.array([1.0, 2.0]))
        
        with pytest.raises(ValueError, match="cannot be empty"):
            run_attribution_analysis(np.array([1e6, 1e7]), np.array([]))
    
    def test_edge_case_very_high_crack_density(self):
        """Test behavior with extremely high crack densities."""
        model = OverlayDegradationModel()
        pristine = model.compute_pristine_overlay()
        
        # Very high crack density
        extremely_high_crack = 1e12  # m^-2
        degraded = model.compute_degraded_overlay(extremely_high_crack, pristine)
        
        # Should be significantly degraded but finite
        assert degraded > pristine
        assert np.isfinite(degraded)
        assert degraded < 100.0  # Should not be unreasonably large


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_crack_density_throughout(self):
        """Test system behavior with zero crack density."""
        model = OverlayDegradationModel()
        
        # Zero crack density
        pristine = model.compute_pristine_overlay()
        degraded = model.compute_degraded_overlay(0.0, pristine)
        
        assert abs(degraded - pristine) < 1e-10
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero in various calculations."""
        decomposer = VarianceDecomposition()
        
        # Very small variance series
        tiny_variance = np.array([1e-15, 1e-15, 1e-15])
        result = decomposer.decompose(tiny_variance, {"scanner": 0.0, "process": 0.0})
        
        # Should not crash and should give reasonable results
        assert np.isfinite(result["scanner_contribution"])
        assert np.isfinite(result["mask_pristine_contribution"])
    
    def test_empty_array_handling(self):
        """Test handling of empty arrays in various methods."""
        model = OverlayDegradationModel()
        
        with pytest.raises(ValueError):
            model.overlay_time_series(np.array([]))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])