"""
Module M4: Physics-Informed Inverse ML Diagnostics

Inverse ML diagnostics for crack distribution, density, and root cause attribution
from experimental inspection data. Uses physics-informed features and Bayesian
inference with Griffith/Charles-Hillig physics as priors.

Classes:
    PhysicsFeatureExtractor: Extract physics-informed features from inspection signals
    SyntheticDataGenerator: Generate synthetic training data using M1+M2+M3
    BayesianCrackDiagnostics: Bayesian inference with physics priors
    CrackDiagnosis: Result container for crack diagnosis
    TransferLearningAdapter: Adapt to experimental data

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import numpy as np
import numpy.typing as npt
from scipy import stats, interpolate, signal
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, mean_squared_error
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import pickle

# Import configuration parameters
from config import ULE_GLASS, ML_CONFIG, INSPECTION, SIMULATION

# Import forward models for synthetic data generation
try:
    from .m01_nucleation import NucleationEngine, DefectField, ThermoelasticStress, CTEMapGenerator
    from .m02_propagation import PropagationEngine  # Placeholder - will be implemented later
    from .m03_inspection import InspectionForwardModel  # Placeholder - will be implemented later
except ImportError:
    # Fallback for testing - will use mock data
    warnings.warn("Forward models M1/M2/M3 not available - using mock data for testing")


@dataclass
class CrackDiagnosis:
    """Result container for crack diagnosis from inverse ML."""
    crack_probability: float  # P(crack exists) ∈ [0,1]
    estimated_density: Tuple[float, float, float]  # (mean, lower_CI, upper_CI) [/m³]
    estimated_max_size: Tuple[float, float, float]  # (mean, lower_CI, upper_CI) [m]
    probable_cause: str  # "thermal_stress" | "impurity" | "surface_damage" | "fatigue"
    confidence: float  # Overall diagnosis confidence ∈ [0,1]
    feature_importance: Dict[str, float]  # Feature contribution weights
    physics_consistency_score: float  # Physics prior consistency ∈ [0,1]


@dataclass
class PhysicsFeatures:
    """Container for physics-informed feature vectors."""
    acoustic_features: Dict[str, float]
    optical_features: Dict[str, float]  
    thermal_features: Dict[str, float]
    combined_vector: npt.NDArray[np.float64]  # Shape: (n_features_physics,)


class PhysicsFeatureExtractor:
    """
    Extract physics-informed features from inspection signals.
    
    Converts raw acoustic, optical, and thermal inspection data into
    12-dimensional feature vectors for ML inference.
    """
    
    def __init__(self):
        """Initialize with inspection method parameters from config."""
        self.acoustic_params = INSPECTION["acoustic"]
        self.optical_params = INSPECTION["laser_scattering"]
        self.raman_params = INSPECTION["raman"]
        self.n_features = ML_CONFIG["n_features_physics"]
        
    def extract_acoustic_features(self, acoustic_signal: Dict[str, npt.NDArray[np.float64]]
                                ) -> Dict[str, float]:
        """
        Extract acoustic emission and ultrasonic features.
        
        Args:
            acoustic_signal: Dictionary containing:
                - 'time': time array [s]
                - 'amplitude': AE amplitude [V] 
                - 'frequency': frequency array [Hz]
                - 'dispersion': velocity dispersion data [m/s]
                
        Returns:
            Dictionary of acoustic features:
                - dispersion_anomaly: RMS deviation from theoretical dispersion [%]
                - ae_event_rate: AE event detection rate [events/s]
                - ae_energy_mean: Mean AE energy [J]
                - frequency_shift: Mean frequency shift from baseline [Hz]
        """
        if not acoustic_signal or len(acoustic_signal) == 0:
            return {
                "dispersion_anomaly": 0.0,
                "ae_event_rate": 0.0,
                "ae_energy_mean": 0.0,
                "frequency_shift": 0.0
            }
        
        # Extract time series data
        time = acoustic_signal.get('time', np.array([]))
        amplitude = acoustic_signal.get('amplitude', np.array([]))
        frequency = acoustic_signal.get('frequency', np.array([]))
        dispersion = acoustic_signal.get('dispersion', np.array([]))
        
        # 1. Dispersion anomaly analysis
        if len(dispersion) > 0 and len(frequency) > 0:
            # Theoretical Lamb wave dispersion for thin plate
            v_L = self.acoustic_params["velocity_longitudinal"]  # [m/s]
            v_T = self.acoustic_params["velocity_transverse"]   # [m/s]
            thickness = 6.35e-3  # ULE substrate thickness [m]
            
            # A0 mode approximate dispersion (low frequency limit)
            theoretical_velocity = np.zeros_like(frequency)
            valid_mask = frequency > 0
            if np.any(valid_mask):
                # Simplified A0 mode: v ≈ √(ω·h·√3·v_T²/(2·v_L))
                omega = 2 * np.pi * frequency[valid_mask]
                theoretical_velocity[valid_mask] = np.sqrt(
                    omega * thickness * np.sqrt(3) * v_T**2 / (2 * v_L)
                )
            
            # Compute RMS deviation
            if len(theoretical_velocity) > 0 and np.any(theoretical_velocity > 0):
                relative_error = np.abs(dispersion - theoretical_velocity) / (theoretical_velocity + 1e-12)
                dispersion_anomaly = 100.0 * np.sqrt(np.mean(relative_error**2))  # [%]
            else:
                dispersion_anomaly = 0.0
        else:
            dispersion_anomaly = 0.0
        
        # 2. AE event detection and rate calculation
        if len(amplitude) > 0 and len(time) > 0:
            # Simple threshold-based event detection
            threshold = 3.0 * np.std(amplitude)  # 3-sigma threshold
            events = amplitude > threshold
            n_events = np.sum(events)
            
            total_time = time[-1] - time[0] if len(time) > 1 else 1.0
            ae_event_rate = n_events / max(total_time, 1e-6)  # [events/s]
            
            # Estimate energy of detected events
            if n_events > 0:
                event_amplitudes = amplitude[events]
                # Simplified energy: E ∝ A² (acoustic power)
                event_energies = event_amplitudes**2 * 1e-12  # Convert to Joules (rough scaling)
                ae_energy_mean = np.mean(event_energies)  # [J]
            else:
                ae_energy_mean = 0.0
        else:
            ae_event_rate = 0.0
            ae_energy_mean = 0.0
        
        # 3. Frequency shift analysis
        if len(frequency) > 0 and len(amplitude) > 0:
            # Simple mean frequency (not weighted due to dimension mismatch)
            mean_frequency = np.mean(frequency)
            
            # Reference frequency (center of inspection band)
            f_min, f_max = self.acoustic_params["frequency_range"]
            f_ref = (f_min + f_max) / 2  # [Hz]
            
            frequency_shift = mean_frequency - f_ref  # [Hz]
        else:
            frequency_shift = 0.0
        
        return {
            "dispersion_anomaly": float(dispersion_anomaly),
            "ae_event_rate": float(ae_event_rate), 
            "ae_energy_mean": float(ae_energy_mean),
            "frequency_shift": float(frequency_shift)
        }
    
    def extract_optical_features(self, optical_signal: Dict[str, npt.NDArray[np.float64]]
                               ) -> Dict[str, float]:
        """
        Extract optical scattering and Raman spectroscopy features.
        
        Args:
            optical_signal: Dictionary containing:
                - 'scattering_intensity': 2D intensity map [counts]
                - 'raman_wavenumber': wavenumber array [cm⁻¹]
                - 'raman_intensity': Raman spectrum [counts]
                - 'phase_map': interferometric phase map [rad]
                
        Returns:
            Dictionary of optical features:
                - scattering_intensity_stats: combined statistics [counts]
                - raman_shift_anomaly: peak shift from reference [cm⁻¹]
                - phase_perturbation_rms: phase variation RMS [rad]
        """
        if not optical_signal or len(optical_signal) == 0:
            return {
                "scattering_intensity_stats": 0.0,
                "raman_shift_anomaly": 0.0,
                "phase_perturbation_rms": 0.0
            }
        
        # Extract optical data
        scattering = optical_signal.get('scattering_intensity', np.array([]))
        raman_wavenumber = optical_signal.get('raman_wavenumber', np.array([]))
        raman_intensity = optical_signal.get('raman_intensity', np.array([]))
        phase_map = optical_signal.get('phase_map', np.array([]))
        
        # 1. Scattering intensity statistics
        if len(scattering) > 0:
            mean_intensity = np.mean(scattering)
            std_intensity = np.std(scattering)
            max_intensity = np.max(scattering)
            
            # Skewness as measure of asymmetry (cracks create high-intensity tails)
            if std_intensity > 0:
                skewness = stats.skew(scattering.flatten())
            else:
                skewness = 0.0
            
            # Combine statistics into single feature (weighted sum)
            # Normalize by typical values to make dimensionless
            scattering_stats = (mean_intensity + std_intensity + 0.1*max_intensity + 10*abs(skewness)) / 4
        else:
            scattering_stats = 0.0
        
        # 2. Raman shift anomaly
        if len(raman_wavenumber) > 0 and len(raman_intensity) > 0:
            # Find main Si-O peak (around 800 cm⁻¹ for silica)
            ref_peak_wavenumber = 800.0  # [cm⁻¹] reference for ULE glass
            
            # Find actual peak position
            peak_idx = np.argmax(raman_intensity)
            if peak_idx < len(raman_wavenumber):
                measured_peak = raman_wavenumber[peak_idx]
                raman_shift_anomaly = measured_peak - ref_peak_wavenumber  # [cm⁻¹]
            else:
                raman_shift_anomaly = 0.0
        else:
            raman_shift_anomaly = 0.0
        
        # 3. Phase perturbation analysis
        if len(phase_map) > 0:
            # Remove linear trends (piston and tilt)
            if phase_map.ndim == 2:
                # 2D phase map
                ny, nx = phase_map.shape
                x_coords = np.arange(nx)
                y_coords = np.arange(ny)
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Fit plane: z = a + b*x + c*y
                coords = np.column_stack([np.ones(nx*ny), X.ravel(), Y.ravel()])
                phase_flat = phase_map.ravel()
                
                try:
                    plane_coeffs = np.linalg.lstsq(coords, phase_flat, rcond=None)[0]
                    plane_fit = coords @ plane_coeffs
                    phase_detrended = phase_flat - plane_fit
                    phase_perturbation_rms = np.sqrt(np.mean(phase_detrended**2))  # [rad]
                except np.linalg.LinAlgError:
                    phase_perturbation_rms = np.sqrt(np.var(phase_map))
            else:
                # 1D case - just remove mean
                phase_perturbation_rms = np.sqrt(np.var(phase_map))  # [rad]
        else:
            phase_perturbation_rms = 0.0
        
        return {
            "scattering_intensity_stats": float(scattering_stats),
            "raman_shift_anomaly": float(raman_shift_anomaly),
            "phase_perturbation_rms": float(phase_perturbation_rms)
        }
    
    def extract_thermal_features(self, thermal_history: Dict[str, npt.NDArray[np.float64]]
                               ) -> Dict[str, float]:
        """
        Extract thermal exposure and cycling features.
        
        Args:
            thermal_history: Dictionary containing:
                - 'time': time array [s]
                - 'temperature': temperature history [K]
                - 'dose_history': cumulative EUV dose [J/m²]
                - 'delta_T_history': temperature excursion history [K]
                
        Returns:
            Dictionary of thermal features:
                - cumulative_dose: total EUV dose exposure [J/m²]
                - thermal_cycles: number of thermal cycles
                - max_delta_T: maximum temperature excursion [K]
                - time_at_temperature: time-weighted temperature exposure [K·s]
        """
        if not thermal_history or len(thermal_history) == 0:
            return {
                "cumulative_dose": 0.0,
                "thermal_cycles": 0.0,
                "max_delta_T": 0.0,
                "time_at_temperature": 0.0
            }
        
        # Extract thermal data
        time = thermal_history.get('time', np.array([]))
        temperature = thermal_history.get('temperature', np.array([]))
        dose_history = thermal_history.get('dose_history', np.array([]))
        delta_T_history = thermal_history.get('delta_T_history', np.array([]))
        
        # 1. Cumulative EUV dose
        if len(dose_history) > 0:
            cumulative_dose = float(np.max(dose_history))  # [J/m²]
        else:
            cumulative_dose = 0.0
        
        # 2. Thermal cycle counting
        if len(temperature) > 1:
            # Count cycles using peak detection
            # Find local maxima that exceed a threshold above baseline
            baseline_temp = np.median(temperature)
            threshold_temp = baseline_temp + 0.1  # 0.1 K threshold
            
            # Smooth signal to reduce noise
            if len(temperature) > 5:
                window_size = min(5, len(temperature)//3)
                temp_smooth = signal.savgol_filter(temperature, window_size, 2)
            else:
                temp_smooth = temperature
            
            peaks, _ = signal.find_peaks(temp_smooth, height=threshold_temp, distance=3)
            thermal_cycles = float(len(peaks))
        else:
            thermal_cycles = 0.0
        
        # 3. Maximum temperature excursion
        if len(delta_T_history) > 0:
            max_delta_T = float(np.max(delta_T_history))  # [K]
        elif len(temperature) > 0:
            # Fallback: compute from temperature history
            max_delta_T = float(np.max(temperature) - np.min(temperature))  # [K]
        else:
            max_delta_T = 0.0
        
        # 4. Time-weighted temperature exposure
        if len(time) > 1 and len(temperature) > 1:
            # Integrate temperature over time
            dt = np.diff(time)  # [s]
            temp_avg = (temperature[1:] + temperature[:-1]) / 2  # Average between points
            time_at_temperature = float(np.sum(dt * temp_avg))  # [K·s]
        else:
            time_at_temperature = 0.0
        
        return {
            "cumulative_dose": cumulative_dose,
            "thermal_cycles": thermal_cycles,
            "max_delta_T": max_delta_T,
            "time_at_temperature": time_at_temperature
        }
    
    def combine_features(self, acoustic_features: Dict[str, float],
                        optical_features: Dict[str, float],
                        thermal_features: Dict[str, float]) -> npt.NDArray[np.float64]:
        """
        Combine individual feature dictionaries into single feature vector.
        
        Args:
            acoustic_features: Acoustic feature dictionary (4 features)
            optical_features: Optical feature dictionary (3 features)
            thermal_features: Thermal feature dictionary (4 features)
            
        Returns:
            Combined feature vector of length n_features_physics (12)
        """
        # Order: acoustic (4) + optical (3) + thermal (4) + derived (1) = 12 features
        feature_vector = np.zeros(self.n_features)
        
        # Acoustic features (indices 0-3)
        feature_vector[0] = acoustic_features.get("dispersion_anomaly", 0.0)
        feature_vector[1] = acoustic_features.get("ae_event_rate", 0.0)
        feature_vector[2] = acoustic_features.get("ae_energy_mean", 0.0)
        feature_vector[3] = acoustic_features.get("frequency_shift", 0.0)
        
        # Optical features (indices 4-6)
        feature_vector[4] = optical_features.get("scattering_intensity_stats", 0.0)
        feature_vector[5] = optical_features.get("raman_shift_anomaly", 0.0)
        feature_vector[6] = optical_features.get("phase_perturbation_rms", 0.0)
        
        # Thermal features (indices 7-10)
        feature_vector[7] = thermal_features.get("cumulative_dose", 0.0)
        feature_vector[8] = thermal_features.get("thermal_cycles", 0.0)
        feature_vector[9] = thermal_features.get("max_delta_T", 0.0)
        feature_vector[10] = thermal_features.get("time_at_temperature", 0.0)
        
        # Derived physics feature (index 11): Stress-thermal coupling
        # Combine thermal and mechanical indicators
        stress_thermal_coupling = (
            acoustic_features.get("dispersion_anomaly", 0.0) * 
            thermal_features.get("max_delta_T", 0.0) / 1000.0  # Normalize
        )
        feature_vector[11] = stress_thermal_coupling
        
        return feature_vector


class SyntheticDataGenerator:
    """
    Generate synthetic training data using M1+M2+M3 forward models.
    
    Creates realistic crack inspection data by running physics simulations
    with varied parameters and adding experimental noise.
    """
    
    def __init__(self, use_forward_models: bool = True):
        """
        Initialize synthetic data generator.
        
        Args:
            use_forward_models: If True, use actual M1+M2+M3 models; 
                               If False, use simplified mock data
        """
        self.use_forward_models = use_forward_models
        self.feature_extractor = PhysicsFeatureExtractor()
        
        # Try to import forward models
        if self.use_forward_models:
            try:
                self.nucleation_engine = NucleationEngine()
                self.defect_field = DefectField(SIMULATION)
                # self.propagation_engine = PropagationEngine()  # TODO: when M2 available
                # self.inspection_model = InspectionForwardModel()  # TODO: when M3 available
            except (ImportError, NameError):
                warnings.warn("Forward models not available - falling back to mock data")
                self.use_forward_models = False
    
    def generate_training_set(self, n_samples: int, 
                            params_range: Dict[str, Tuple[float, float]],
                            seed: Optional[int] = None) -> Tuple[npt.NDArray[np.float64], 
                                                               npt.NDArray[np.int64],
                                                               npt.NDArray[np.float64]]:
        """
        Generate synthetic training dataset.
        
        Args:
            n_samples: Number of training samples to generate
            params_range: Dictionary of parameter ranges:
                - 'defect_density': (min, max) [defects/m³]
                - 'cte_sigma': (min, max) [1/K] 
                - 'dose': (min, max) [J/m²]
                - 'delta_T': (min, max) [K]
                
        Returns:
            Tuple of (X_features, y_labels, y_continuous) where:
                - X_features: (n_samples, n_features) feature matrix
                - y_labels: (n_samples,) crack presence labels (0/1)
                - y_continuous: (n_samples, 3) [density, max_size, cause_idx]
        """
        if seed is not None:
            np.random.seed(seed)
        
        X_features = np.zeros((n_samples, ML_CONFIG["n_features_physics"]))
        y_labels = np.zeros(n_samples, dtype=int)
        y_continuous = np.zeros((n_samples, 3))  # density, max_size, cause_index
        
        # Parameter sampling ranges
        defect_density_range = params_range.get('defect_density', (1e6, 1e10))
        cte_sigma_range = params_range.get('cte_sigma', (5e-9, 20e-9))
        dose_range = params_range.get('dose', (1000, 100000))  # [J/m²]
        delta_T_range = params_range.get('delta_T', (0.1, 2.0))  # [K]
        
        for i in range(n_samples):
            # Sample parameters
            defect_density = np.random.uniform(*defect_density_range)
            cte_sigma = np.random.uniform(*cte_sigma_range)
            dose = np.random.uniform(*dose_range)
            delta_T = np.random.uniform(*delta_T_range)
            
            if self.use_forward_models:
                # Use actual forward models (when available)
                features, labels, continuous = self._generate_sample_forward_models(
                    defect_density, cte_sigma, dose, delta_T, i
                )
            else:
                # Use mock data generation
                features, labels, continuous = self._generate_sample_mock(
                    defect_density, cte_sigma, dose, delta_T, i
                )
            
            X_features[i, :] = features
            y_labels[i] = labels
            y_continuous[i, :] = continuous
        
        return X_features, y_labels, y_continuous
    
    def _generate_sample_forward_models(self, defect_density: float, cte_sigma: float,
                                      dose: float, delta_T: float, sample_idx: int
                                      ) -> Tuple[npt.NDArray[np.float64], int, npt.NDArray[np.float64]]:
        """Generate sample using actual M1+M2+M3 forward models."""
        # TODO: Implement when M1+M2+M3 are fully available
        # For now, fall back to mock data
        return self._generate_sample_mock(defect_density, cte_sigma, dose, delta_T, sample_idx)
    
    def _generate_sample_mock(self, defect_density: float, cte_sigma: float,
                            dose: float, delta_T: float, sample_idx: int
                            ) -> Tuple[npt.NDArray[np.float64], int, npt.NDArray[np.float64]]:
        """Generate mock sample for testing/demonstration."""
        # Simple physics-based relationships for mock data
        
        # Crack probability based on Griffith-like scaling with enhanced variation
        stress_factor = delta_T * cte_sigma * ULE_GLASS["E_young"] / (1 - ULE_GLASS["nu_poisson"])
        K_I_factor = stress_factor * np.sqrt(50e-9)  # Assume 50nm defects
        
        # Add more variation to ensure both classes are represented
        random_factor = 0.5 + np.random.uniform(0, 1.0)  # Random multiplier
        crack_probability = min(1.0, max(0.0, (K_I_factor / ULE_GLASS["K_IC"])**2 * random_factor))
        
        # Enhance probability to ensure class balance
        if crack_probability < 0.3:
            crack_probability = crack_probability + 0.2  # Boost low probabilities
        
        # Generate crack label
        has_crack = np.random.random() < crack_probability
        y_label = int(has_crack)
        
        # Mock inspection signals based on crack presence
        if has_crack:
            # Cracked sample - elevated features
            acoustic_features = {
                "dispersion_anomaly": np.random.normal(5.0, 2.0),  # [%]
                "ae_event_rate": np.random.exponential(10.0),  # [events/s]
                "ae_energy_mean": np.random.lognormal(-25, 1),  # [J] 
                "frequency_shift": np.random.normal(0, 1000)  # [Hz]
            }
            
            optical_features = {
                "scattering_intensity_stats": np.random.lognormal(5, 1),  # [counts]
                "raman_shift_anomaly": np.random.normal(0, 2.0),  # [cm⁻¹]
                "phase_perturbation_rms": np.random.exponential(0.1)  # [rad]
            }
            
            # Continuous outputs for cracked sample
            estimated_density = defect_density * crack_probability  # [/m³]
            estimated_max_size = np.random.lognormal(np.log(100e-9), 0.5)  # [m]
            cause_index = np.random.choice([0, 1, 2, 3])  # thermal, impurity, surface, fatigue
            
        else:
            # Pristine sample - low features
            acoustic_features = {
                "dispersion_anomaly": np.random.normal(0.5, 0.2),
                "ae_event_rate": np.random.exponential(1.0),
                "ae_energy_mean": np.random.lognormal(-30, 0.5),
                "frequency_shift": np.random.normal(0, 100)
            }
            
            optical_features = {
                "scattering_intensity_stats": np.random.lognormal(2, 0.5),
                "raman_shift_anomaly": np.random.normal(0, 0.5),
                "phase_perturbation_rms": np.random.exponential(0.01)
            }
            
            # Continuous outputs for pristine sample
            estimated_density = 0.0
            estimated_max_size = 0.0
            cause_index = 0
        
        # Add thermal history
        thermal_features = {
            "cumulative_dose": dose + np.random.normal(0, dose*0.1),  # [J/m²]
            "thermal_cycles": max(1, int(dose / 1000)),  # Rough scaling
            "max_delta_T": delta_T + np.random.normal(0, delta_T*0.1),  # [K]
            "time_at_temperature": dose * delta_T / 1000  # [K·s] rough scaling
        }
        
        # Combine features
        feature_vector = self.feature_extractor.combine_features(
            acoustic_features, optical_features, thermal_features
        )
        
        # Continuous outputs
        y_continuous = np.array([estimated_density, estimated_max_size, float(cause_index)])
        
        return feature_vector, y_label, y_continuous
    
    def add_noise(self, X: npt.NDArray[np.float64], noise_level: float = 0.1
                 ) -> npt.NDArray[np.float64]:
        """
        Add Gaussian noise to feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            noise_level: Relative noise level (std = noise_level * signal_std)
            
        Returns:
            Noisy feature matrix
        """
        if noise_level <= 0:
            return X.copy()
        
        X_noisy = X.copy()
        for feature_idx in range(X.shape[1]):
            feature_std = np.std(X[:, feature_idx])
            # Use absolute noise level if relative is too small
            if feature_std > 0:
                noise_std = noise_level * feature_std
            else:
                # Use absolute noise for constant features
                noise_std = noise_level * np.abs(np.mean(X[:, feature_idx])) + 0.01
            
            noise = np.random.normal(0, noise_std, X.shape[0])
            X_noisy[:, feature_idx] += noise
        
        return X_noisy
    
    def balance_classes(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int64]
                       ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Balance binary classes using SMOTE-like oversampling.
        
        Args:
            X: Feature matrix
            y: Binary labels (0/1)
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) != 2:
            warnings.warn("Expected binary classification, returning original data")
            return X, y
        
        # Find minority class
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]
        
        minority_mask = y == minority_class
        majority_mask = y == majority_class
        
        n_majority = np.sum(majority_mask)
        n_minority = np.sum(minority_mask)
        
        if n_majority <= n_minority:
            return X, y  # Already balanced or minority is larger
        
        # Oversample minority class
        n_oversample = n_majority - n_minority
        minority_indices = np.where(minority_mask)[0]
        
        # Simple random oversampling (could be improved with SMOTE)
        oversample_indices = np.random.choice(minority_indices, n_oversample, replace=True)
        
        # Add noise to oversampled examples to create variation
        X_oversample = X[oversample_indices].copy()
        X_oversample = self.add_noise(X_oversample, 0.05)  # Small noise
        y_oversample = np.full(n_oversample, minority_class)
        
        # Combine original and oversampled data
        X_balanced = np.vstack([X, X_oversample])
        y_balanced = np.concatenate([y, y_oversample])
        
        # Shuffle the balanced dataset
        X_balanced, y_balanced = shuffle(X_balanced, y_balanced)
        
        return X_balanced, y_balanced


class BayesianCrackDiagnostics:
    """
    Bayesian inference for crack diagnosis with physics priors.
    
    Uses Gaussian Process models with physics-informed priors from
    Griffith fracture mechanics and Charles-Hillig subcritical growth.
    """
    
    def __init__(self):
        """Initialize Bayesian models and physics priors."""
        self.classifier = None  # Will be GaussianProcessClassifier
        self.regressor = None   # Will be GaussianProcessRegressor
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Fallback for single class cases
        self.fallback_ridge = None
        self.use_fallback = False
        self.single_class_label = None
        
        # Physics prior parameters
        self.griffith_params = {}
        self.scg_params = {}
        
        # Feature importance (learned from data)
        self.feature_importance = {}
        
    def set_physics_prior(self, griffith_params: Dict[str, float],
                         scg_params: Dict[str, float]) -> None:
        """
        Set physics-based priors for Bayesian inference.
        
        Args:
            griffith_params: Griffith criterion parameters
                - 'K_IC': Critical stress intensity factor [Pa·m^0.5]
                - 'gamma_s': Surface energy [J/m²]
                - 'stress_threshold': Nucleation stress threshold [Pa]
            scg_params: Subcritical crack growth parameters
                - 'n': Stress corrosion exponent [-]
                - 'v0': Pre-exponential velocity [m/s]
                - 'activation_energy': Activation energy [J/mol]
        """
        self.griffith_params = griffith_params.copy()
        self.scg_params = scg_params.copy()
        
        # Set GP kernel parameters based on physics
        # Length scales from physical correlation lengths
        self._setup_physics_informed_kernels()
    
    def _setup_physics_informed_kernels(self) -> None:
        """Setup GP kernels with physics-informed length scales."""
        n_features = ML_CONFIG["n_features_physics"]
        
        # Physics-informed length scales for different feature types
        length_scales = np.ones(n_features)
        
        # Acoustic features (0-3): correlation over inspection bandwidth
        length_scales[0:4] = 2.0  # Acoustic correlation
        
        # Optical features (4-6): correlation over optical spot size  
        length_scales[4:7] = 1.5  # Optical correlation
        
        # Thermal features (7-10): correlation over thermal cycling
        length_scales[7:11] = 3.0  # Thermal correlation
        
        # Derived feature (11): coupling between thermal and mechanical
        length_scales[11] = 2.5
        
        # RBF kernel with physics-informed length scales
        kernel_base = C(1.0, (1e-3, 1e3)) * RBF(length_scales, (0.1, 10.0))
        
        # Add white noise kernel for measurement uncertainty
        kernel_classifier = kernel_base + WhiteKernel(1e-2, (1e-5, 1))
        kernel_regressor = kernel_base + WhiteKernel(1e-2, (1e-5, 1))
        
        # Initialize GP models
        self.classifier = GaussianProcessClassifier(
            kernel=kernel_classifier,
            n_restarts_optimizer=3
        )
        
        self.regressor = GaussianProcessRegressor(
            kernel=kernel_regressor,
            n_restarts_optimizer=3,
            alpha=1e-6  # Numerical stability
        )
    
    def fit(self, X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.int64],
           y_continuous: Optional[npt.NDArray[np.float64]] = None) -> None:
        """
        Fit Bayesian models to training data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels for classification (n_samples,)
            y_continuous: Training targets for regression (n_samples, n_targets)
        """
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Setup kernels if not already done
        if self.classifier is None:
            self._setup_physics_informed_kernels()
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Check for single class issue
        n_classes = len(np.unique(y_train))
        if n_classes == 1:
            warnings.warn("Only one class present in training data. Using BayesianRidge as fallback.")
            # Use BayesianRidge for single class case
            self.fallback_ridge = BayesianRidge()
            self.fallback_ridge.fit(X_train_scaled, y_train.astype(float))
            self.single_class_label = y_train[0]
            self.use_fallback = True
        else:
            # Fit classification model normally
            self.classifier.fit(X_train_scaled, y_train)
            self.use_fallback = False
        
        # Fit regression model if continuous targets provided
        if y_continuous is not None:
            # Only fit on positive samples for regression
            positive_mask = y_train > 0
            if np.any(positive_mask):
                X_positive = X_train_scaled[positive_mask]
                y_positive = y_continuous[positive_mask]
                
                # Fit separate regressors for each target (density, size, cause)
                # For simplicity, use the first target (density) for now
                if not self.use_fallback:  # Only if GP is available
                    self.regressor.fit(X_positive, y_positive[:, 0])
        
        # Compute feature importance from kernel parameters
        if not self.use_fallback:
            self._compute_feature_importance(X_train_scaled, y_train)
        else:
            # Use uniform importance for fallback case
            n_features = X_train_scaled.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            self.feature_importance = {name: 1.0/n_features for name in feature_names}
        
        self.is_fitted = True
    
    def _compute_feature_importance(self, X: npt.NDArray[np.float64], 
                                  y: npt.NDArray[np.int64]) -> None:
        """Compute feature importance from GP kernel parameters."""
        if self.classifier.kernel_ is not None:
            # Extract length scales from fitted RBF kernel
            try:
                # Get the RBF component of the composite kernel
                kernel = self.classifier.kernel_
                if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k2'):  # C * RBF + White
                    rbf_kernel = kernel.k1.k2
                elif hasattr(kernel, 'k2'):  # RBF + White  
                    rbf_kernel = kernel.k1
                else:
                    rbf_kernel = kernel
                
                if hasattr(rbf_kernel, 'length_scale'):
                    length_scales = rbf_kernel.length_scale
                    # Importance inversely related to length scale
                    importance = 1.0 / (length_scales + 1e-6)
                    importance = importance / np.sum(importance)  # Normalize
                    
                    # Feature names for interpretation
                    feature_names = [
                        "dispersion_anomaly", "ae_event_rate", "ae_energy_mean", "frequency_shift",
                        "scattering_stats", "raman_shift", "phase_perturbation", 
                        "cumulative_dose", "thermal_cycles", "max_delta_T", "time_at_temp",
                        "stress_thermal_coupling"
                    ]
                    
                    self.feature_importance = dict(zip(feature_names, importance))
                else:
                    # Fallback: uniform importance
                    self.feature_importance = {f"feature_{i}": 1.0/X.shape[1] for i in range(X.shape[1])}
            except:
                # Fallback for any kernel extraction errors
                self.feature_importance = {f"feature_{i}": 1.0/X.shape[1] for i in range(X.shape[1])}
    
    def predict(self, X_test: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], 
                                                              npt.NDArray[np.float64]]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            Tuple of (y_pred, y_uncertainty) where:
                - y_pred: Predictions (classification probs or regression values)
                - y_uncertainty: Prediction uncertainties (standard deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        if hasattr(self, 'use_fallback') and self.use_fallback:
            # Use BayesianRidge fallback for single class case
            y_pred_reg, y_std_reg = self.fallback_ridge.predict(X_test_scaled, return_std=True)
            # Convert to probabilities
            y_pred = np.clip(y_pred_reg, 0.0, 1.0)
            y_uncertainty = y_std_reg + 0.1  # Add baseline uncertainty
        else:
            # Classification predictions (probabilities)
            y_pred_proba = self.classifier.predict_proba(X_test_scaled)
            crack_probabilities = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
            
            # Uncertainty from GP prediction variance (not directly available for classification)
            # Use entropy as uncertainty measure
            epsilon = 1e-15  # Avoid log(0)
            y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1-epsilon)
            uncertainty_classification = -np.sum(y_pred_proba_safe * np.log(y_pred_proba_safe), axis=1)
            
            # Regression predictions (if regressor is fitted)
            if self.regressor is not None:
                try:
                    y_pred_reg, y_std_reg = self.regressor.predict(X_test_scaled, return_std=True)
                    
                    # Combine classification and regression uncertainties
                    y_pred = crack_probabilities
                    y_uncertainty = np.sqrt(uncertainty_classification**2 + (y_std_reg/10)**2)  # Scale regression uncertainty
                except:
                    # Fallback if regression fails
                    y_pred = crack_probabilities
                    y_uncertainty = uncertainty_classification
            else:
                y_pred = crack_probabilities
                y_uncertainty = uncertainty_classification
        
        return y_pred, y_uncertainty
    
    def diagnose(self, inspection_data: Dict[str, Any]) -> CrackDiagnosis:
        """
        Perform complete crack diagnosis from inspection data.
        
        Args:
            inspection_data: Dictionary containing:
                - 'acoustic_signal': acoustic inspection results
                - 'optical_signal': optical inspection results  
                - 'thermal_history': thermal exposure history
                
        Returns:
            CrackDiagnosis object with complete analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnosis")
        
        # Extract features
        feature_extractor = PhysicsFeatureExtractor()
        
        acoustic_features = feature_extractor.extract_acoustic_features(
            inspection_data.get('acoustic_signal', {})
        )
        optical_features = feature_extractor.extract_optical_features(
            inspection_data.get('optical_signal', {})
        )
        thermal_features = feature_extractor.extract_thermal_features(
            inspection_data.get('thermal_history', {})
        )
        
        feature_vector = feature_extractor.combine_features(
            acoustic_features, optical_features, thermal_features
        )
        
        # Make prediction
        X_test = feature_vector.reshape(1, -1)
        y_pred, y_uncertainty = self.predict(X_test)
        
        crack_probability = float(y_pred[0])
        prediction_uncertainty = float(y_uncertainty[0])
        
        # Estimate crack density and size (simplified)
        if crack_probability > 0.5:
            # Use thermal dose and features to estimate density
            dose = thermal_features.get('cumulative_dose', 0)
            ae_rate = acoustic_features.get('ae_event_rate', 0)
            
            # Rough scaling based on physics
            base_density = min(1e9, dose * ae_rate / 1000)  # [/m³]
            density_uncertainty = base_density * prediction_uncertainty
            
            estimated_density = (
                base_density,
                max(0, base_density - 1.96 * density_uncertainty),  # 95% CI
                base_density + 1.96 * density_uncertainty
            )
            
            # Estimate maximum crack size
            scattering_stats = optical_features.get('scattering_intensity_stats', 0)
            base_size = min(1e-6, scattering_stats * 1e-9)  # [m]
            size_uncertainty = base_size * prediction_uncertainty
            
            estimated_max_size = (
                base_size,
                max(0, base_size - 1.96 * size_uncertainty),  # 95% CI
                base_size + 1.96 * size_uncertainty
            )
        else:
            estimated_density = (0.0, 0.0, 0.0)
            estimated_max_size = (0.0, 0.0, 0.0)
        
        # Determine probable cause
        thermal_cycles = thermal_features.get('thermal_cycles', 0)
        max_delta_T = thermal_features.get('max_delta_T', 0)
        dispersion_anomaly = acoustic_features.get('dispersion_anomaly', 0)
        raman_shift = optical_features.get('raman_shift_anomaly', 0)
        
        if max_delta_T > 1.0 and thermal_cycles > 100:
            probable_cause = "thermal_stress"
        elif abs(raman_shift) > 2.0:
            probable_cause = "impurity"
        elif dispersion_anomaly > 3.0:
            probable_cause = "surface_damage"
        elif thermal_cycles > 1000:
            probable_cause = "fatigue"
        else:
            probable_cause = "thermal_stress"  # Default
        
        # Overall confidence
        base_confidence = 1.0 - prediction_uncertainty
        confidence = max(0.0, min(1.0, base_confidence))
        
        # Physics consistency score
        physics_score = self._compute_physics_consistency(feature_vector, crack_probability)
        
        return CrackDiagnosis(
            crack_probability=crack_probability,
            estimated_density=estimated_density,
            estimated_max_size=estimated_max_size,
            probable_cause=probable_cause,
            confidence=confidence,
            feature_importance=self.feature_importance.copy(),
            physics_consistency_score=physics_score
        )
    
    def _compute_physics_consistency(self, features: npt.NDArray[np.float64], 
                                   crack_prob: float) -> float:
        """
        Compute consistency between ML prediction and physics priors.
        
        Args:
            features: Feature vector
            crack_prob: Predicted crack probability
            
        Returns:
            Physics consistency score ∈ [0,1]
        """
        if not self.griffith_params:
            return 0.5  # Neutral if no physics priors set
        
        # Extract relevant features
        max_delta_T = features[9] if len(features) > 9 else 0.0
        cumulative_dose = features[7] if len(features) > 7 else 0.0
        
        # Estimate stress from thermal features
        cte = ULE_GLASS["CTE_sigma"]  # Use typical value
        E = ULE_GLASS["E_young"]
        nu = ULE_GLASS["nu_poisson"]
        
        estimated_stress = E * cte * max_delta_T / (1 - nu)
        
        # Griffith criterion check
        K_IC = self.griffith_params.get('K_IC', ULE_GLASS["K_IC"])
        typical_flaw_size = 50e-9  # [m] typical defect size
        
        # Simplified K_I calculation
        K_I = 1.12 * estimated_stress * np.sqrt(np.pi * typical_flaw_size)
        griffith_ratio = K_I / K_IC
        
        # Physics-based crack probability
        physics_prob = min(1.0, max(0.0, (griffith_ratio - 0.5) * 2))  # Sigmoid-like
        
        # Consistency score based on agreement
        prob_diff = abs(crack_prob - physics_prob)
        consistency_score = max(0.0, 1.0 - 2 * prob_diff)  # Linear penalty
        
        return float(consistency_score)


class TransferLearningAdapter:
    """
    Adapt synthetic model to experimental data using transfer learning.
    
    Handles domain adaptation when transitioning from synthetic training
    data to real experimental measurements.
    """
    
    def __init__(self, base_model: BayesianCrackDiagnostics):
        """Initialize with pre-trained base model."""
        self.base_model = base_model
        self.experimental_scaler = StandardScaler()
        self.domain_gap_history = []
        
    def update_with_experimental(self, X_exp: npt.NDArray[np.float64], 
                               y_exp: npt.NDArray[np.int64],
                               adaptation_strategy: str = "fine_tune") -> None:
        """
        Update model with experimental data.
        
        Args:
            X_exp: Experimental features
            y_exp: Experimental labels
            adaptation_strategy: "fine_tune", "domain_adaptation", or "ensemble"
        """
        if not self.base_model.is_fitted:
            raise ValueError("Base model must be fitted before adaptation")
        
        if len(X_exp) == 0:
            warnings.warn("No experimental data provided for adaptation")
            return
        
        if adaptation_strategy == "fine_tune":
            self._fine_tune_adaptation(X_exp, y_exp)
        elif adaptation_strategy == "domain_adaptation":
            self._domain_adaptation(X_exp, y_exp)
        elif adaptation_strategy == "ensemble":
            self._ensemble_adaptation(X_exp, y_exp)
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")
    
    def _fine_tune_adaptation(self, X_exp: npt.NDArray[np.float64], 
                            y_exp: npt.NDArray[np.int64]) -> None:
        """Fine-tune existing model with experimental data."""
        # Scale experimental data with existing scaler
        X_exp_scaled = self.base_model.scaler.transform(X_exp)
        
        # Retrain with experimental data (this overwrites previous training)
        # In practice, you'd want to combine synthetic + experimental data
        self.base_model.classifier.fit(X_exp_scaled, y_exp)
        
        # Update feature importance
        self.base_model._compute_feature_importance(X_exp_scaled, y_exp)
    
    def _domain_adaptation(self, X_exp: npt.NDArray[np.float64], 
                         y_exp: npt.NDArray[np.int64]) -> None:
        """Apply domain adaptation techniques."""
        # Compute domain gap
        domain_gap = self.compute_domain_gap(
            self.base_model.scaler.transform(X_exp), X_exp  # Synthetic vs experimental 
        )
        self.domain_gap_history.append(domain_gap)
        
        # Apply domain correction (simplified approach)
        # In practice, this could use techniques like CORAL, DANN, etc.
        correction_factor = min(1.0, 2.0 / (1 + domain_gap))
        
        # Adjust kernel parameters based on domain gap
        if self.base_model.classifier is not None:
            # This is a simplified approach - real domain adaptation would be more sophisticated
            X_exp_corrected = X_exp * correction_factor
            X_exp_scaled = self.base_model.scaler.transform(X_exp_corrected)
            
            # Retrain with corrected experimental data
            self.base_model.classifier.fit(X_exp_scaled, y_exp)
    
    def _ensemble_adaptation(self, X_exp: npt.NDArray[np.float64], 
                           y_exp: npt.NDArray[np.int64]) -> None:
        """Create ensemble of synthetic and experimental models."""
        # Train a new model on experimental data only
        exp_classifier = GaussianProcessClassifier(
            kernel=self.base_model.classifier.kernel,
            n_restarts_optimizer=3
        )
        
        X_exp_scaled = self.experimental_scaler.fit_transform(X_exp)
        exp_classifier.fit(X_exp_scaled, y_exp)
        
        # Store experimental classifier for ensemble prediction
        self.experimental_classifier = exp_classifier
    
    def compute_domain_gap(self, X_synthetic: npt.NDArray[np.float64],
                         X_experimental: npt.NDArray[np.float64]) -> float:
        """
        Compute domain gap between synthetic and experimental data.
        
        Uses Maximum Mean Discrepancy (MMD) as measure of distribution difference.
        
        Args:
            X_synthetic: Synthetic feature samples
            X_experimental: Experimental feature samples
            
        Returns:
            Domain gap measure (higher = more different)
        """
        if len(X_synthetic) == 0 or len(X_experimental) == 0:
            return float('inf')
        
        # Simplified MMD computation using RBF kernel
        def rbf_kernel(X1, X2, gamma=1.0):
            """RBF kernel computation."""
            dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dists)
        
        # Sample subset if datasets are large
        n_max = 200
        if len(X_synthetic) > n_max:
            idx_syn = np.random.choice(len(X_synthetic), n_max, replace=False)
            X_syn = X_synthetic[idx_syn]
        else:
            X_syn = X_synthetic
            
        if len(X_experimental) > n_max:
            idx_exp = np.random.choice(len(X_experimental), n_max, replace=False)
            X_exp = X_experimental[idx_exp]
        else:
            X_exp = X_experimental
        
        # Compute kernel matrices
        K_ss = rbf_kernel(X_syn, X_syn)
        K_ee = rbf_kernel(X_exp, X_exp)
        K_se = rbf_kernel(X_syn, X_exp)
        
        # MMD estimate
        n_syn = len(X_syn)
        n_exp = len(X_exp)
        
        mmd_squared = (
            np.sum(K_ss) / (n_syn ** 2) +
            np.sum(K_ee) / (n_exp ** 2) -
            2 * np.sum(K_se) / (n_syn * n_exp)
        )
        
        return float(np.sqrt(max(0, mmd_squared)))