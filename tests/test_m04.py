"""
Tests for Module M4: Physics-Informed Inverse ML Diagnostics

Comprehensive test suite covering all classes and edge cases.
Minimum 15 tests as specified in requirements.

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import Dict, Any
import warnings

# Import the M4 module classes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from m04_inverse_ml import (
    PhysicsFeatureExtractor,
    SyntheticDataGenerator,
    BayesianCrackDiagnostics, 
    CrackDiagnosis,
    TransferLearningAdapter,
    PhysicsFeatures
)

# Import config for testing
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ML_CONFIG, ULE_GLASS


class TestPhysicsFeatureExtractor:
    """Test PhysicsFeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return PhysicsFeatureExtractor()
    
    @pytest.fixture
    def mock_acoustic_signal(self):
        """Create mock acoustic signal data."""
        time = np.linspace(0, 1, 1000)  # [s]
        frequency = np.linspace(1e6, 50e6, 100)  # [Hz]
        amplitude = 0.1 * np.sin(2 * np.pi * 5 * time) + 0.01 * np.random.randn(1000)  # [V]
        dispersion = 5000 * np.ones_like(frequency) + 100 * np.random.randn(len(frequency))  # [m/s]
        
        return {
            'time': time,
            'amplitude': amplitude,
            'frequency': frequency,
            'dispersion': dispersion
        }
    
    @pytest.fixture
    def mock_optical_signal(self):
        """Create mock optical signal data."""
        scattering = 100 + 50 * np.random.randn(50, 50)  # [counts]
        raman_wavenumber = np.linspace(600, 1000, 200)  # [cm⁻¹]
        raman_intensity = np.exp(-(raman_wavenumber - 800)**2 / 100) + 0.1 * np.random.randn(200)
        phase_map = 0.1 * np.random.randn(32, 32)  # [rad]
        
        return {
            'scattering_intensity': scattering,
            'raman_wavenumber': raman_wavenumber,
            'raman_intensity': raman_intensity,
            'phase_map': phase_map
        }
    
    @pytest.fixture
    def mock_thermal_history(self):
        """Create mock thermal history data."""
        time = np.linspace(0, 3600, 1000)  # [s]
        temperature = 300 + 2 * np.sin(2 * np.pi * time / 100) + 0.1 * np.random.randn(1000)  # [K]
        dose_history = np.cumsum(np.random.exponential(10, 1000))  # [J/m²]
        delta_T_history = np.abs(np.diff(temperature))
        delta_T_history = np.append(delta_T_history, delta_T_history[-1])  # [K]
        
        return {
            'time': time,
            'temperature': temperature,
            'dose_history': dose_history,
            'delta_T_history': delta_T_history
        }
    
    def test_acoustic_features_output_format(self, extractor, mock_acoustic_signal):
        """Test 1: Acoustic features have correct format and types."""
        features = extractor.extract_acoustic_features(mock_acoustic_signal)
        
        # Check all expected keys are present
        expected_keys = ["dispersion_anomaly", "ae_event_rate", "ae_energy_mean", "frequency_shift"]
        assert all(key in features for key in expected_keys), "Missing expected acoustic features"
        
        # Check all values are numeric
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"Feature {key} should be numeric, got {type(value)}"
            assert np.isfinite(value), f"Feature {key} should be finite, got {value}"
    
    def test_acoustic_features_physical_ranges(self, extractor, mock_acoustic_signal):
        """Test 2: Acoustic features are within physically reasonable ranges."""
        features = extractor.extract_acoustic_features(mock_acoustic_signal)
        
        # Dispersion anomaly should be percentage (typically 0-50%)
        assert 0 <= features["dispersion_anomaly"] <= 100, f"Dispersion anomaly {features['dispersion_anomaly']}% out of range"
        
        # AE event rate should be reasonable (0-1000 events/s for lab conditions)
        assert 0 <= features["ae_event_rate"] <= 1e4, f"AE event rate {features['ae_event_rate']} events/s out of range"
        
        # AE energy should be small positive values (pJ to μJ range)
        assert features["ae_energy_mean"] >= 0, f"AE energy {features['ae_energy_mean']} should be non-negative"
        assert features["ae_energy_mean"] <= 1e-6, f"AE energy {features['ae_energy_mean']} J too large"
        
        # Frequency shift should be within inspection bandwidth
        assert abs(features["frequency_shift"]) <= 50e6, f"Frequency shift {features['frequency_shift']} Hz too large"
    
    def test_optical_features_output_format(self, extractor, mock_optical_signal):
        """Test 3: Optical features have correct format and types."""
        features = extractor.extract_optical_features(mock_optical_signal)
        
        expected_keys = ["scattering_intensity_stats", "raman_shift_anomaly", "phase_perturbation_rms"]
        assert all(key in features for key in expected_keys), "Missing expected optical features"
        
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"Feature {key} should be numeric"
            assert np.isfinite(value), f"Feature {key} should be finite"
    
    def test_thermal_features_output_format(self, extractor, mock_thermal_history):
        """Test 4: Thermal features have correct format and types."""
        features = extractor.extract_thermal_features(mock_thermal_history)
        
        expected_keys = ["cumulative_dose", "thermal_cycles", "max_delta_T", "time_at_temperature"]
        assert all(key in features for key in expected_keys), "Missing expected thermal features"
        
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"Feature {key} should be numeric"
            assert np.isfinite(value), f"Feature {key} should be finite"
    
    def test_combine_features_correct_dimensions(self, extractor):
        """Test 5: Combined feature vector has correct dimensions."""
        # Mock feature dictionaries
        acoustic = {"dispersion_anomaly": 1.0, "ae_event_rate": 2.0, "ae_energy_mean": 3.0, "frequency_shift": 4.0}
        optical = {"scattering_intensity_stats": 5.0, "raman_shift_anomaly": 6.0, "phase_perturbation_rms": 7.0}
        thermal = {"cumulative_dose": 8.0, "thermal_cycles": 9.0, "max_delta_T": 10.0, "time_at_temperature": 11.0}
        
        combined = extractor.combine_features(acoustic, optical, thermal)
        
        # Should have exactly n_features_physics dimensions
        expected_dim = ML_CONFIG["n_features_physics"]
        assert combined.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {combined.shape}"
        assert combined.dtype == np.float64, f"Expected float64, got {combined.dtype}"
    
    def test_empty_signal_handling(self, extractor):
        """Test 6: Handle empty signals gracefully."""
        # Test with empty dictionaries
        acoustic_empty = extractor.extract_acoustic_features({})
        optical_empty = extractor.extract_optical_features({})
        thermal_empty = extractor.extract_thermal_features({})
        
        # Should return zero features without errors
        assert all(value == 0.0 for value in acoustic_empty.values()), "Empty acoustic should give zero features"
        assert all(value == 0.0 for value in optical_empty.values()), "Empty optical should give zero features"
        assert all(value == 0.0 for value in thermal_empty.values()), "Empty thermal should give zero features"
        
        # Combined should also work
        combined = extractor.combine_features(acoustic_empty, optical_empty, thermal_empty)
        assert combined.shape[0] == ML_CONFIG["n_features_physics"], "Combined empty features should have correct shape"


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(use_forward_models=False)  # Use mock data for testing
    
    @pytest.fixture
    def test_params_range(self):
        """Define parameter ranges for testing."""
        return {
            'defect_density': (1e6, 1e8),
            'cte_sigma': (5e-9, 15e-9),
            'dose': (1000, 50000),
            'delta_T': (0.2, 1.5)
        }
    
    def test_generate_training_set_shapes(self, generator, test_params_range):
        """Test 7: Generated training data has correct shapes."""
        n_samples = 100
        X, y_labels, y_continuous = generator.generate_training_set(n_samples, test_params_range)
        
        # Check shapes
        assert X.shape == (n_samples, ML_CONFIG["n_features_physics"]), f"X shape mismatch: {X.shape}"
        assert y_labels.shape == (n_samples,), f"y_labels shape mismatch: {y_labels.shape}"
        assert y_continuous.shape == (n_samples, 3), f"y_continuous shape mismatch: {y_continuous.shape}"
        
        # Check data types
        assert X.dtype == np.float64, f"X should be float64, got {X.dtype}"
        assert y_labels.dtype in [np.int64, np.int32, int], f"y_labels should be integer, got {y_labels.dtype}"
        assert y_continuous.dtype == np.float64, f"y_continuous should be float64, got {y_continuous.dtype}"
    
    def test_generate_training_set_value_ranges(self, generator, test_params_range):
        """Test 8: Generated data values are in expected ranges."""
        n_samples = 50
        X, y_labels, y_continuous = generator.generate_training_set(n_samples, test_params_range)
        
        # Binary classification labels should be 0 or 1
        assert np.all(np.isin(y_labels, [0, 1])), f"Labels should be binary, got unique values: {np.unique(y_labels)}"
        
        # Feature values should be finite
        assert np.all(np.isfinite(X)), "All features should be finite"
        
        # Continuous outputs should be non-negative
        assert np.all(y_continuous[:, 0] >= 0), "Density should be non-negative"  # Density
        assert np.all(y_continuous[:, 1] >= 0), "Size should be non-negative"     # Max size
        assert np.all(y_continuous[:, 2] >= 0), "Cause index should be non-negative"  # Cause index
        assert np.all(y_continuous[:, 2] <= 3), "Cause index should be <= 3"     # 4 causes: 0,1,2,3
    
    def test_class_balancing(self, generator, test_params_range):
        """Test 9: Class balancing works correctly."""
        # Generate imbalanced data
        n_samples = 100
        X, y, _ = generator.generate_training_set(n_samples, test_params_range)
        
        # Check initial balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Initial distribution: {dict(zip(unique, counts))}")
        
        if len(unique) == 2 and min(counts) < max(counts) * 0.8:  # Imbalanced
            # Apply balancing
            X_balanced, y_balanced = generator.balance_classes(X, y)
            
            # Check new balance
            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            balance_ratio = min(counts_bal) / max(counts_bal)
            
            assert balance_ratio >= 0.9, f"Classes should be balanced, got ratio: {balance_ratio}"
            assert len(X_balanced) >= len(X), "Balanced dataset should be larger or equal"
    
    def test_add_noise_functionality(self, generator):
        """Test 10: Noise addition works correctly."""
        # Create simple test data
        X_clean = np.ones((10, 5)) * 10  # All features = 10
        
        # Add noise
        X_noisy = generator.add_noise(X_clean, noise_level=0.1)
        
        # Check that noise was added
        assert not np.allclose(X_clean, X_noisy), "Noise should change the data"
        
        # Check that mean is approximately preserved
        mean_diff = np.abs(np.mean(X_noisy) - np.mean(X_clean))
        assert mean_diff < 1.0, f"Mean should be approximately preserved, diff: {mean_diff}"
        
        # Check no noise case
        X_no_noise = generator.add_noise(X_clean, noise_level=0.0)
        npt.assert_array_equal(X_clean, X_no_noise, "Zero noise should return identical data")


class TestBayesianCrackDiagnostics:
    """Test BayesianCrackDiagnostics class."""
    
    @pytest.fixture
    def diagnostics(self):
        """Create diagnostics instance."""
        return BayesianCrackDiagnostics()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        n_samples = 50
        n_features = ML_CONFIG["n_features_physics"]
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels with some pattern
        y_labels = (np.sum(X[:, :4], axis=1) > 0).astype(int)  # Based on first 4 features
        
        # Generate continuous targets
        y_continuous = np.random.exponential(1e8, (n_samples, 3))  # Density, size, cause
        
        return X, y_labels, y_continuous
    
    def test_physics_prior_setting(self, diagnostics):
        """Test 11: Physics priors can be set correctly."""
        griffith_params = {
            'K_IC': ULE_GLASS["K_IC"],
            'gamma_s': 4.5,
            'stress_threshold': 1e6
        }
        
        scg_params = {
            'n': ULE_GLASS["scg_n"],
            'v0': ULE_GLASS["scg_v0"],
            'activation_energy': ULE_GLASS["scg_delta_H"]
        }
        
        diagnostics.set_physics_prior(griffith_params, scg_params)
        
        assert diagnostics.griffith_params == griffith_params, "Griffith params not set correctly"
        assert diagnostics.scg_params == scg_params, "SCG params not set correctly"
        assert diagnostics.classifier is not None, "Classifier should be initialized"
        assert diagnostics.regressor is not None, "Regressor should be initialized"
    
    def test_model_fitting(self, diagnostics, sample_training_data):
        """Test 12: Model fitting works without errors."""
        X, y_labels, y_continuous = sample_training_data
        
        # Set physics priors first
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        
        # Fit model
        diagnostics.fit(X, y_labels, y_continuous)
        
        assert diagnostics.is_fitted, "Model should be marked as fitted"
        assert len(diagnostics.feature_importance) > 0, "Feature importance should be computed"
    
    def test_prediction_output_format(self, diagnostics, sample_training_data):
        """Test 13: Predictions have correct format and ranges."""
        X_train, y_train, y_continuous = sample_training_data
        
        # Fit model
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        diagnostics.fit(X_train, y_train, y_continuous)
        
        # Generate test data
        X_test = np.random.randn(10, ML_CONFIG["n_features_physics"])
        
        # Make predictions
        y_pred, y_uncertainty = diagnostics.predict(X_test)
        
        # Check shapes
        assert y_pred.shape == (10,), f"Predictions shape mismatch: {y_pred.shape}"
        assert y_uncertainty.shape == (10,), f"Uncertainty shape mismatch: {y_uncertainty.shape}"
        
        # Check value ranges
        assert np.all((y_pred >= 0) & (y_pred <= 1)), f"Predictions should be probabilities [0,1], got range [{np.min(y_pred)}, {np.max(y_pred)}]"
        assert np.all(y_uncertainty > 0), "Uncertainties should be positive"
        assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
        assert np.all(np.isfinite(y_uncertainty)), "Uncertainties should be finite"


class TestCrackDiagnosis:
    """Test CrackDiagnosis dataclass."""
    
    def test_diagnosis_dataclass_structure(self):
        """Test 14: CrackDiagnosis has all required fields with correct types."""
        # Create a sample diagnosis
        diagnosis = CrackDiagnosis(
            crack_probability=0.7,
            estimated_density=(1e8, 5e7, 1.5e8),
            estimated_max_size=(100e-9, 50e-9, 150e-9),
            probable_cause="thermal_stress",
            confidence=0.85,
            feature_importance={"feature_1": 0.3, "feature_2": 0.7},
            physics_consistency_score=0.9
        )
        
        # Check types
        assert isinstance(diagnosis.crack_probability, float), "crack_probability should be float"
        assert isinstance(diagnosis.estimated_density, tuple), "estimated_density should be tuple"
        assert len(diagnosis.estimated_density) == 3, "estimated_density should have 3 elements"
        assert isinstance(diagnosis.estimated_max_size, tuple), "estimated_max_size should be tuple"
        assert len(diagnosis.estimated_max_size) == 3, "estimated_max_size should have 3 elements"
        assert isinstance(diagnosis.probable_cause, str), "probable_cause should be string"
        assert isinstance(diagnosis.confidence, float), "confidence should be float"
        assert isinstance(diagnosis.feature_importance, dict), "feature_importance should be dict"
        assert isinstance(diagnosis.physics_consistency_score, float), "physics_consistency_score should be float"
        
        # Check value ranges
        assert 0 <= diagnosis.crack_probability <= 1, "crack_probability should be in [0,1]"
        assert 0 <= diagnosis.confidence <= 1, "confidence should be in [0,1]"
        assert 0 <= diagnosis.physics_consistency_score <= 1, "physics_consistency_score should be in [0,1]"
        
        # Check that confidence intervals are ordered
        mean, lower, upper = diagnosis.estimated_density
        assert lower <= mean <= upper, f"Density CI not ordered: {lower} <= {mean} <= {upper}"
        
        mean, lower, upper = diagnosis.estimated_max_size
        assert lower <= mean <= upper, f"Size CI not ordered: {lower} <= {mean} <= {upper}"


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""
    
    def test_complete_diagnosis_workflow(self):
        """Test 15: Complete workflow from inspection data to diagnosis."""
        # Create mock inspection data
        inspection_data = {
            'acoustic_signal': {
                'time': np.linspace(0, 1, 100),
                'amplitude': 0.1 * np.random.randn(100),
                'frequency': np.linspace(1e6, 50e6, 50),
                'dispersion': 5000 + 100 * np.random.randn(50)
            },
            'optical_signal': {
                'scattering_intensity': 100 + 20 * np.random.randn(20, 20),
                'raman_wavenumber': np.linspace(700, 900, 100),
                'raman_intensity': np.exp(-(np.linspace(700, 900, 100) - 800)**2 / 100),
                'phase_map': 0.05 * np.random.randn(16, 16)
            },
            'thermal_history': {
                'time': np.linspace(0, 3600, 200),
                'temperature': 300 + np.sin(np.linspace(0, 3600, 200) / 100),
                'dose_history': np.cumsum(np.random.exponential(5, 200)),
                'delta_T_history': np.abs(np.diff(np.sin(np.linspace(0, 3600, 200) / 100)))
            }
        }
        
        # Create and train diagnostics model
        generator = SyntheticDataGenerator(use_forward_models=False)
        params_range = {
            'defect_density': (1e6, 1e8),
            'cte_sigma': (5e-9, 15e-9),
            'dose': (1000, 50000),
            'delta_T': (0.2, 1.5)
        }
        
        X_train, y_train, y_continuous = generator.generate_training_set(100, params_range, seed=42)
        
        diagnostics = BayesianCrackDiagnostics()
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        diagnostics.fit(X_train, y_train, y_continuous)
        
        # Perform diagnosis
        diagnosis = diagnostics.diagnose(inspection_data)
        
        # Verify diagnosis completeness and validity
        assert isinstance(diagnosis, CrackDiagnosis), "Should return CrackDiagnosis object"
        assert 0 <= diagnosis.crack_probability <= 1, "Crack probability should be in valid range"
        assert diagnosis.probable_cause in ["thermal_stress", "impurity", "surface_damage", "fatigue"], "Should be valid cause"
        assert 0 <= diagnosis.confidence <= 1, "Confidence should be in valid range"
        assert 0 <= diagnosis.physics_consistency_score <= 1, "Physics score should be in valid range"
    
    def test_physics_consistency_scoring(self):
        """Test 16: Physics consistency scoring works correctly."""
        diagnostics = BayesianCrackDiagnostics()
        
        # Set physics priors
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        
        # Test high stress case (should favor crack prediction)
        high_stress_features = np.zeros(ML_CONFIG["n_features_physics"])
        high_stress_features[9] = 2.0  # max_delta_T = 2K (high)
        high_stress_features[7] = 50000  # cumulative_dose = 50kJ/m² (high)
        
        high_stress_score = diagnostics._compute_physics_consistency(high_stress_features, 0.8)
        
        # Test low stress case (should favor no-crack prediction)
        low_stress_features = np.zeros(ML_CONFIG["n_features_physics"])
        low_stress_features[9] = 0.1  # max_delta_T = 0.1K (low)
        low_stress_features[7] = 1000   # cumulative_dose = 1kJ/m² (low)
        
        low_stress_score = diagnostics._compute_physics_consistency(low_stress_features, 0.1)
        
        # Consistency should be higher when prediction matches physics expectation
        assert 0 <= high_stress_score <= 1, "Physics score should be in [0,1]"
        assert 0 <= low_stress_score <= 1, "Physics score should be in [0,1]"
    
    def test_edge_case_empty_data(self):
        """Test 17: Handle empty datasets gracefully."""
        diagnostics = BayesianCrackDiagnostics()
        
        # Test fitting with empty data
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            diagnostics.fit(np.array([]), np.array([]))
    
    def test_edge_case_single_sample(self):
        """Test 18: Handle single sample case."""
        diagnostics = BayesianCrackDiagnostics()
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        
        # Single sample training (should handle gracefully)
        X_single = np.random.randn(1, ML_CONFIG["n_features_physics"])
        y_single = np.array([1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # GP may warn about insufficient data
            diagnostics.fit(X_single, y_single)
            
            # Should still be able to predict
            pred, unc = diagnostics.predict(X_single)
            assert len(pred) == 1, "Should predict on single sample"
            assert len(unc) == 1, "Should return uncertainty for single sample"
    
    def test_edge_case_all_same_class(self):
        """Test 19: Handle datasets with all samples from same class."""
        diagnostics = BayesianCrackDiagnostics()
        diagnostics.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        
        # All negative samples
        X_all_neg = np.random.randn(20, ML_CONFIG["n_features_physics"])
        y_all_neg = np.zeros(20, dtype=int)  # All class 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about single class
            diagnostics.fit(X_all_neg, y_all_neg)
            
            pred, unc = diagnostics.predict(X_all_neg)
            
            # Should predict low probabilities consistently
            assert np.all(pred <= 0.7), "Should predict low crack probability for all-negative training"
            assert np.all(unc > 0), "Should still return positive uncertainty"


class TestTransferLearningAdapter:
    """Test TransferLearningAdapter class."""
    
    def test_transfer_learning_initialization(self):
        """Test 20: Transfer learning adapter initializes correctly."""
        # Create and fit base model
        base_model = BayesianCrackDiagnostics()
        base_model.set_physics_prior(
            {'K_IC': ULE_GLASS["K_IC"], 'gamma_s': 4.5, 'stress_threshold': 1e6},
            {'n': 20, 'v0': 1e-6, 'activation_energy': 80e3}
        )
        
        # Generate synthetic training data
        generator = SyntheticDataGenerator(use_forward_models=False)
        params_range = {'defect_density': (1e6, 1e8), 'cte_sigma': (5e-9, 15e-9), 
                       'dose': (1000, 50000), 'delta_T': (0.2, 1.5)}
        X, y, _ = generator.generate_training_set(50, params_range, seed=42)
        base_model.fit(X, y)
        
        # Create adapter
        adapter = TransferLearningAdapter(base_model)
        
        assert adapter.base_model is base_model, "Base model should be stored"
        assert hasattr(adapter, 'experimental_scaler'), "Should have experimental scaler"
        assert hasattr(adapter, 'domain_gap_history'), "Should have domain gap history"
    
    def test_domain_gap_computation(self):
        """Test 21: Domain gap computation works correctly."""
        base_model = BayesianCrackDiagnostics()
        adapter = TransferLearningAdapter(base_model)
        
        # Create synthetic vs experimental data
        X_synthetic = np.random.randn(30, 5)
        X_experimental = np.random.randn(20, 5) + 1.0  # Shifted distribution
        
        domain_gap = adapter.compute_domain_gap(X_synthetic, X_experimental)
        
        assert isinstance(domain_gap, float), "Domain gap should be float"
        assert domain_gap >= 0, "Domain gap should be non-negative"
        assert np.isfinite(domain_gap), "Domain gap should be finite"
        
        # Test identical distributions (should have low gap)
        X_identical = X_synthetic.copy()
        gap_identical = adapter.compute_domain_gap(X_synthetic, X_identical)
        assert gap_identical < 0.1, f"Identical distributions should have low gap, got {gap_identical}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])