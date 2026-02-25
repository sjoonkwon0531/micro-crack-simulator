"""
Module M3: Inspection Signal Forward Model
Glass Micro-Crack Lifecycle Simulator — Job 10

Physics-based simulation of non-destructive testing (NDT) signals
for cracked vs pristine ULE glass substrates under EUV lithography.

Classes:
    AcousticInspection: Ultrasonic/Lamb wave inspection simulation
    OpticalInspection: Laser scattering, Raman, interferometry methods  
    ElectronBeamInspection: EELS and Kelvin Force Microscopy
    InspectionComparison: Multi-method comparison and ROC analysis
    InspectionResult: Result container dataclass

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import numpy as np
import numpy.typing as npt
from scipy import ndimage, integrate, optimize, special
from scipy.signal import convolve2d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Import configuration parameters
from config import (
    ULE_GLASS, INSPECTION, SIMULATION,
    k_B, h_planck
)


@dataclass
class InspectionResult:
    """Result container for inspection signal analysis."""
    method: str
    signal_map: npt.NDArray[np.float64]     # 2D signal intensity map
    reference_signal: npt.NDArray[np.float64]  # Pristine substrate signal  
    delta_signal: npt.NDArray[np.float64]   # Difference: cracked - pristine
    snr_map: npt.NDArray[np.float64]        # Signal-to-noise ratio map
    min_detectable_size: float              # Minimum crack size detectable [m]
    detection_probability: float            # P(detection) for given crack field
    roc_auc: float                         # Area under ROC curve


class AcousticInspection:
    """
    Ultrasonic and Lamb wave inspection simulation for crack detection.
    
    Implements dispersion curve analysis, scattering simulation,
    and acoustic emission modeling for EUV substrate inspection.
    """
    
    def __init__(self):
        """Initialize with ULE glass acoustic properties from config."""
        self.v_l = INSPECTION["acoustic"]["velocity_longitudinal"]  # m/s
        self.v_t = INSPECTION["acoustic"]["velocity_transverse"]    # m/s
        self.rho = ULE_GLASS["rho"]                                # kg/m³
        self.attenuation_coeff = INSPECTION["acoustic"]["attenuation_coeff"]  # dB/cm/MHz
        
    def compute_lamb_wave_dispersion(self, frequency: float, thickness: float, 
                                   v_l: float, v_t: float) -> Tuple[float, float]:
        """
        Compute Lamb wave dispersion curves for symmetric (S0) and antisymmetric (A0) modes.
        
        For thin plate approximation (f·h << v_t), use first-order expansions:
        - S0 mode: v_phase ≈ v_l × (1 - ν²) / (1 - ν)  (extensional-like)
        - A0 mode: v_phase ≈ v_t × √(12) × √(v_t/(ω·h))  (flexural-like)
        
        Args:
            frequency: Frequency [Hz]
            thickness: Plate thickness [m] 
            v_l: Longitudinal velocity [m/s]
            v_t: Transverse velocity [m/s]
            
        Returns:
            Tuple of (v_phase_S0, v_phase_A0) phase velocities [m/s]
        """
        if frequency <= 0 or thickness <= 0:
            raise ValueError("Frequency and thickness must be positive")
            
        omega = 2 * np.pi * frequency
        fd_product = frequency * thickness  # Frequency-thickness product [Hz·m]
        
        # Poisson's ratio from velocities: ν = (v_l²-2v_t²)/(2(v_l²-v_t²))
        nu = (v_l**2 - 2*v_t**2) / (2*(v_l**2 - v_t**2))
        
        # S0 mode (symmetric, extensional-like)
        # For low frequencies: v_phase ≈ v_plate_extension  
        v_s0 = v_l * np.sqrt((1 - nu) / (1 + nu))  # Plate wave velocity
        
        # A0 mode (antisymmetric, flexural-like)
        # Phase velocity: v_A0 = (2π f h)^(1/2) × [E/(12ρ(1-ν²))]^(1/4)
        # This is dispersive: v_A0 ∝ √(f·h), always slower than bulk waves
        if fd_product < 1e-6:  # Avoid division by zero for very low frequencies
            v_a0 = 0.0
        else:
            # Bending stiffness parameter: [E/(12ρ(1-ν²))]^(1/4)  [m²/s]^(1/2)
            bending_param = (ULE_GLASS["E_young"] / (12 * self.rho * (1 - nu**2))) ** 0.25
            # v_A0 = √(ω·h) × bending_param^(1/2) — but correct formula:
            # v_phase = (D/ρh)^(1/4) × ω^(1/2) where D = Eh³/12(1-ν²)
            # = [Eh²/(12ρ(1-ν²))]^(1/4) × √ω
            v_a0 = ((ULE_GLASS["E_young"] * thickness**2 / 
                    (12 * self.rho * (1 - nu**2))) ** 0.25) * np.sqrt(omega)
            
        return v_s0, v_a0
    
    def simulate_scattering(self, crack_geometry: Dict[str, float], 
                          wave_params: Dict[str, float]) -> Dict[str, float]:
        """
        Simulate ultrasonic scattering from crack using Born approximation.
        
        Distinguishes between Rayleigh (a << λ) and Mie (a ~ λ) scattering regimes.
        
        Args:
            crack_geometry: Dict with 'length', 'depth', 'width' [m]
            wave_params: Dict with 'frequency', 'wavelength', 'amplitude' [Hz, m, Pa]
            
        Returns:
            Dict with scattering cross-sections and scattered amplitudes
        """
        crack_length = crack_geometry.get("length", 0.0)
        crack_depth = crack_geometry.get("depth", 0.0) 
        crack_width = crack_geometry.get("width", 1e-6)  # Default 1 μm width
        
        frequency = wave_params["frequency"]
        wavelength = wave_params["wavelength"]
        incident_amplitude = wave_params.get("amplitude", 1e6)  # Default 1 MPa
        
        if crack_length <= 0 or wavelength <= 0:
            return {
                "rayleigh_cross_section": 0.0,
                "mie_cross_section": 0.0, 
                "scattered_amplitude": 0.0,
                "scattering_regime": "none"
            }
        
        # Characteristic crack size (use length as primary dimension)
        a = crack_length
        
        # Dimensionless size parameter
        ka = 2 * np.pi * a / wavelength
        
        # Crack volume for scattering calculations
        crack_volume = crack_length * crack_depth * crack_width
        
        if ka < 0.3:  # Rayleigh regime: a << λ
            # Rayleigh scattering: σ_scat ∝ a⁶/λ⁴ for volume scatterer
            # For crack (cavity): contrast factor ≈ 1 (vacuum vs solid)
            rayleigh_cross_section = (9/4) * np.pi * (ka**4) * crack_volume**(2/3)
            mie_cross_section = rayleigh_cross_section  # Rayleigh limit
            regime = "rayleigh"
            
            # Scattered amplitude: A_scat ∝ A_incident × ka² × (crack_volume/λ³)
            scattered_amplitude = incident_amplitude * ka**2 * (crack_volume / wavelength**3)
            
        elif ka < 10:  # Mie regime: a ~ λ
            # Mie scattering: use geometric approximation
            # For slab-like crack: σ_geom ≈ crack_length × crack_depth
            geometric_cross_section = crack_length * crack_depth
            
            # Mie efficiency factor (approximation for thin crack)
            q_mie = 2.0 * (1 - np.cos(ka))  # Oscillatory efficiency
            mie_cross_section = q_mie * geometric_cross_section
            rayleigh_cross_section = (9/4) * np.pi * (ka**4) * crack_volume**(2/3)  # For comparison
            regime = "mie"
            
            # Use geometric optics limit
            scattered_amplitude = incident_amplitude * np.sqrt(mie_cross_section / 
                                                             (4 * np.pi * wavelength**2))
            
        else:  # Geometric regime: a >> λ
            geometric_cross_section = crack_length * crack_depth
            mie_cross_section = geometric_cross_section  # σ ≈ geometric area
            rayleigh_cross_section = 0.0  # Not applicable
            regime = "geometric"
            
            scattered_amplitude = incident_amplitude * 0.5  # ~50% reflection from crack
        
        return {
            "rayleigh_cross_section": rayleigh_cross_section,
            "mie_cross_section": mie_cross_section,
            "scattered_amplitude": scattered_amplitude,
            "scattering_regime": regime,
            "size_parameter": ka
        }
    
    def compute_acoustic_emission(self, crack_growth_rate: float, 
                                energy_release: float) -> Dict[str, float]:
        """
        Compute acoustic emission (AE) signal characteristics from crack growth.
        
        AE event energy ∝ crack area increment × G_c
        AE amplitude distribution follows Gutenberg-Richter analog: log N ∝ -b × log A
        
        Args:
            crack_growth_rate: da/dt [m/s]
            energy_release: Energy release rate G [J/m²]
            
        Returns:
            Dict with AE energy, amplitude, and event statistics
        """
        if crack_growth_rate <= 0 or energy_release <= 0:
            return {
                "ae_energy": 0.0,
                "ae_amplitude": 0.0, 
                "event_rate": 0.0,
                "b_value": 1.0
            }
        
        # Estimate crack area increment (assume penny-shaped crack)
        # dA/dt = 2π × a × (da/dt) for circular crack of radius a
        # Use minimum detectable crack size as reference
        a_ref = INSPECTION["laser_scattering"]["detection_limit"]  # 20 nm
        area_increment_rate = 2 * np.pi * a_ref * crack_growth_rate  # m²/s
        
        # AE energy from Griffith energy release
        G_c = ULE_GLASS["K_IC"]**2 / ULE_GLASS["E_young"]  # Critical energy release rate
        ae_energy_rate = energy_release * area_increment_rate  # J/s
        
        # Convert to per-event energy (assume 1 event per second at reference rate)
        ae_energy = ae_energy_rate  # J per event
        
        # AE amplitude from energy (empirical relation for glass)
        # A ∝ E^(1/2) typical for brittle fracture
        if ae_energy > 0:
            ae_amplitude = np.sqrt(ae_energy / 1e-12)  # Normalized to pJ reference
        else:
            ae_amplitude = 0.0
            
        # Event rate: higher for faster crack growth
        base_event_rate = 1.0  # events/s
        event_rate = base_event_rate * (crack_growth_rate / 1e-9)  # Scale by nm/s
        
        # b-value (Gutenberg-Richter exponent): typical range 0.5-2.0 for brittle materials
        # Lower b-value → more large events → more dangerous
        b_value = 0.8 + 0.4 * (1 - min(1.0, energy_release / G_c))  # 0.4-1.2 range
        
        return {
            "ae_energy": ae_energy,
            "ae_amplitude": ae_amplitude,
            "event_rate": event_rate,
            "b_value": b_value
        }
    
    def signal_to_noise(self, crack_size: float, depth: float, 
                       method_params: Dict[str, float]) -> float:
        """
        Calculate signal-to-noise ratio for crack detection.
        
        Includes attenuation: exp(-α × distance × frequency)
        
        Args:
            crack_size: Crack characteristic size [m]
            depth: Crack depth below surface [m] 
            method_params: Method-specific parameters
            
        Returns:
            Signal-to-noise ratio (dimensionless)
        """
        frequency = method_params.get("frequency", 5e6)  # Default 5 MHz
        noise_level = method_params.get("noise_level", 0.01)  # Default 1% noise
        signal_strength = method_params.get("signal_strength", 1.0)  # Default normalized
        
        if crack_size <= 0:
            return 0.0
            
        # Signal amplitude from scattering cross-section
        wavelength = self.v_l / frequency
        ka = 2 * np.pi * crack_size / wavelength
        
        # Use appropriate scattering regime
        if ka < 0.3:  # Rayleigh
            signal_amplitude = signal_strength * (ka**4)
        elif ka < 10:  # Mie
            signal_amplitude = signal_strength * (2 * (1 - np.cos(ka)))
        else:  # Geometric
            signal_amplitude = signal_strength
            
        # Attenuation with depth: exp(-α × depth × frequency)
        # Convert dB/cm/MHz to Np/m/Hz: 1 dB = ln(10)/20 Np ≈ 0.115 Np
        alpha_np = self.attenuation_coeff * 0.115 * 100  # Convert dB/cm to Np/m  
        attenuation = np.exp(-alpha_np * depth * frequency / 1e6)  # /MHz factor
        
        # Prevent overflow in attenuation calculation
        if alpha_np * depth * frequency / 1e6 > 50:  # exp(-50) ≈ 2e-22
            attenuation = 0.0
        
        signal_with_attenuation = signal_amplitude * attenuation
        
        # SNR calculation
        snr = signal_with_attenuation / (noise_level + 1e-12)  # Avoid division by zero
        
        return max(0.0, snr)


class OpticalInspection:
    """
    Optical inspection methods: laser scattering, Raman stress mapping,
    and 193nm interferometry for EUV substrate characterization.
    """
    
    def __init__(self):
        """Initialize with optical inspection parameters from config."""
        self.laser_wavelength = INSPECTION["laser_scattering"]["wavelength"]  # m
        self.laser_power = INSPECTION["laser_scattering"]["power"]           # W
        self.spot_size = INSPECTION["laser_scattering"]["spot_size"]         # m
        self.detection_limit = INSPECTION["laser_scattering"]["detection_limit"]  # m
        
        self.raman_wavelength = INSPECTION["raman"]["excitation_wavelength"]     # m
        self.raman_resolution = INSPECTION["raman"]["spectral_resolution"]       # cm⁻¹
        self.raman_sensitivity = INSPECTION["raman"]["stress_sensitivity"]       # cm⁻¹/GPa
        self.raman_spatial_resolution = INSPECTION["raman"]["spatial_resolution"] # m
        
        self.interferometry_wavelength = INSPECTION["interferometry_193nm"]["wavelength"]  # m
        self.phase_sensitivity = INSPECTION["interferometry_193nm"]["phase_sensitivity"]    # rad
        
    def laser_scattering_signal(self, crack_size: float, crack_depth: float,
                               wavelength: float, power: float) -> Dict[str, float]:
        """
        Compute laser scattering signal intensity from crack.
        
        Uses Rayleigh (I ∝ a⁶/λ⁴) and Mie scattering theories with depth attenuation.
        
        Args:
            crack_size: Crack characteristic size [m]
            crack_depth: Depth below surface [m]
            wavelength: Laser wavelength [m] 
            power: Incident laser power [W]
            
        Returns:
            Dict with scattered intensity and scattering regime info
        """
        if crack_size <= 0 or wavelength <= 0 or power <= 0:
            return {
                "scattered_intensity": 0.0,
                "scattering_efficiency": 0.0,
                "penetration_depth": 0.0,
                "regime": "none"
            }
            
        # Size parameter
        a = crack_size
        k = 2 * np.pi / wavelength
        ka = k * a
        
        # Penetration depth in ULE glass at given wavelength
        # For visible light (532nm): penetration depth ~ mm to cm
        # Empirical: δ_pen ≈ wavelength / (4π × k_imaginary)
        # For ULE glass: k_imag ~ 1e-6 at 532nm (very transparent)
        k_imaginary = 1e-6  # Absorption coefficient (very low for ULE)
        penetration_depth = wavelength / (4 * np.pi * k_imaginary)
        
        # Depth attenuation: Beer's law
        depth_attenuation = np.exp(-crack_depth / penetration_depth)
        
        # Scattering cross-section calculation
        if ka < 0.3:  # Rayleigh regime: a << λ
            # Rayleigh scattering: σ_scat = (8π/3) × ka⁴ × a²
            # For sphere: σ_R = (8π/3) × (ka)⁴ × π × a²
            rayleigh_cross_section = (8*np.pi/3) * (ka**4) * np.pi * a**2
            scattering_efficiency = rayleigh_cross_section / (np.pi * a**2)
            regime = "rayleigh"
            
        elif ka < 10:  # Mie regime: a ~ λ  
            # Mie scattering for spherical particle (simplified)
            # Q_scat ≈ 2 - (4/ka) × sin(ka) + (4/ka²) × (1-cos(ka))
            Q_mie = 2 - (4/ka) * np.sin(ka) + (4/ka**2) * (1 - np.cos(ka))
            Q_mie = max(0, Q_mie)  # Ensure non-negative
            scattering_efficiency = Q_mie
            regime = "mie"
            
        else:  # Geometric regime: a >> λ
            # Geometric scattering: Q ≈ 2 (extinction efficiency)
            scattering_efficiency = 2.0
            regime = "geometric"
            
        # Scattered intensity: I_scat = I_0 × (σ_scat / 4π × r²) × depth_attenuation
        # Use spot size as reference distance
        reference_distance = self.spot_size
        geometric_area = np.pi * a**2
        
        scattered_intensity = (power / (4 * np.pi * reference_distance**2)) * \
                             scattering_efficiency * geometric_area * depth_attenuation
        
        return {
            "scattered_intensity": scattered_intensity,
            "scattering_efficiency": scattering_efficiency, 
            "penetration_depth": penetration_depth,
            "regime": regime
        }
    
    def raman_stress_mapping(self, stress_field: npt.NDArray[np.float64],
                           sensitivity: float, resolution: float) -> npt.NDArray[np.float64]:
        """
        Generate Raman stress map from mechanical stress field.
        
        Δω = sensitivity × σ (cm⁻¹/GPa × Pa)
        For SiO₂: main peak ~440 cm⁻¹, compressive stress → blue shift (positive Δω)
        
        Args:
            stress_field: 2D stress field [Pa]
            sensitivity: Raman stress sensitivity [cm⁻¹/GPa]
            resolution: Spatial resolution limit [m]
            
        Returns:
            2D Raman shift map [cm⁻¹]
        """
        if stress_field.size == 0:
            return np.array([])
            
        # Convert stress from Pa to GPa
        stress_gpa = stress_field / 1e9
        
        # Raman shift: Δω = sensitivity × σ
        # Note: For compressive stress (negative), with negative sensitivity,
        # we get positive shift (blue shift) which is physically correct
        raman_shift = sensitivity * stress_gpa
        
        # Apply spatial resolution limit using Gaussian convolution
        # PSF (Point Spread Function) width in grid units
        grid_spacing = 152e-3 / stress_field.shape[0]  # Assume 152mm substrate
        sigma_pixels = resolution / grid_spacing  # PSF width in pixels
        
        if sigma_pixels > 0.5:  # Only convolve if resolution limit is significant
            raman_shift_blurred = ndimage.gaussian_filter(raman_shift, sigma=sigma_pixels)
        else:
            raman_shift_blurred = raman_shift
            
        return raman_shift_blurred
    
    def interferometry_193nm(self, dn_field: npt.NDArray[np.float64], 
                           thickness: float, wavelength: float) -> npt.NDArray[np.float64]:
        """
        Compute 193nm interferometry phase change map.
        
        ΔΦ = 2π/λ × Δn × t
        Crack → local refractive index change → phase perturbation
        
        Args:
            dn_field: 2D refractive index change field [dimensionless]
            thickness: Substrate thickness [m]
            wavelength: Interferometry wavelength [m]
            
        Returns:
            2D phase change map [radians]
        """
        if dn_field.size == 0 or wavelength <= 0:
            return np.array([])
            
        # Phase change formula: ΔΦ = 2π/λ × Δn × t
        phase_change = (2 * np.pi / wavelength) * dn_field * thickness
        
        # Phase wrapping: keep in [-π, π] for realistic measurement
        phase_change_wrapped = np.arctan2(np.sin(phase_change), np.cos(phase_change))
        
        return phase_change_wrapped
    
    def compute_dn_from_cracks(self, crack_field: npt.NDArray[np.float64],
                              n_glass: float) -> npt.NDArray[np.float64]:
        """
        Convert crack damage field to refractive index change.
        
        Crack → void (n=1) → local index reduction: Δn = -(n_glass-1) × crack_density
        
        Args:
            crack_field: 2D crack damage field [0=intact, 1=fully cracked]
            n_glass: Glass refractive index
            
        Returns:
            2D refractive index change field [dimensionless]
        """
        # Crack represents void: n_void = 1, so Δn = n_void - n_glass = 1 - n_glass
        delta_n_per_crack = 1.0 - n_glass  # Negative for typical glass (n>1)
        
        # Scale by crack density
        dn_field = delta_n_per_crack * crack_field
        
        return dn_field


class ElectronBeamInspection:
    """
    Electron beam inspection methods: EELS (Electron Energy Loss Spectroscopy)
    and KFM (Kelvin Force Microscopy) for crack characterization.
    """
    
    def __init__(self):
        """Initialize electron beam inspection parameters."""
        self.eels_energy_resolution = 0.1  # eV
        self.kfm_potential_sensitivity = 1e-3  # V
        
    def eels_bonding_shift(self, strain_field: npt.NDArray[np.float64],
                          bond_type: str = "Si-O") -> npt.NDArray[np.float64]:
        """
        Compute EELS energy shift from strain-induced bond distortion.
        
        Si-O bond: ~5 eV shift per unit strain
        Crack tip → bond distortion → peak broadening + shift
        
        Args:
            strain_field: 2D strain field [dimensionless]
            bond_type: Type of chemical bond
            
        Returns:
            2D EELS energy shift map [eV]
        """
        if strain_field.size == 0:
            return np.array([])
            
        # Bond-specific energy shift sensitivity
        if bond_type == "Si-O":
            shift_sensitivity = 5.0  # eV per unit strain
        elif bond_type == "Ti-O":  # For TiO₂ component in ULE
            shift_sensitivity = 3.0  # eV per unit strain
        else:
            shift_sensitivity = 1.0  # Default value
            
        # EELS energy shift: ΔE = sensitivity × |strain|
        # Use absolute strain since bond distortion occurs for both tension/compression
        eels_shift = shift_sensitivity * np.abs(strain_field)
        
        return eels_shift
    
    def kfm_surface_potential(self, charge_density: npt.NDArray[np.float64],
                            crack_positions: List[Tuple[float, float]]) -> npt.NDArray[np.float64]:
        """
        Compute Kelvin Force Microscopy surface potential map.
        
        Crack sites → charge trapping → surface potential change
        ΔCPD(x,y) (Contact Potential Difference) map generation
        
        Args:
            charge_density: 2D surface charge density [C/m²]
            crack_positions: List of (x,y) crack center coordinates [normalized 0-1]
            
        Returns:
            2D surface potential map [V]
        """
        if charge_density.size == 0:
            return np.zeros_like(charge_density)
            
        nx, ny = charge_density.shape
        
        # Create coordinate grids
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initialize potential map
        potential_map = np.zeros((nx, ny))
        
        # Add charge contribution (Coulomb potential)
        # V = k × Q / r, simplified for 2D surface
        k_coulomb = 9e9  # Coulomb constant [N⋅m²/C²]
        
        # Base potential from uniform charge distribution
        base_potential = k_coulomb * charge_density * 1e-9  # Scale to reasonable values
        
        # Add localized potential from crack sites
        for x_crack, y_crack in crack_positions:
            # Distance from each grid point to crack
            distance = np.sqrt((X - x_crack)**2 + (Y - y_crack)**2)
            
            # Avoid singularity at crack position
            distance = np.maximum(distance, 1.0/nx)  # Minimum distance = grid spacing
            
            # Enhanced charge trapping at crack sites
            crack_charge = 1e-12  # C (picocoulomb level)
            crack_potential = k_coulomb * crack_charge / distance
            
            potential_map += crack_potential
            
        # Total potential
        total_potential = base_potential + potential_map
        
        # Apply realistic potential scale (mV to V range)
        max_potential = np.max(np.abs(total_potential))
        if max_potential > 0:
            scaling_factor = 0.1 / max_potential  # Scale to ±100 mV
            total_potential *= scaling_factor
            
        return total_potential


class InspectionComparison:
    """
    Multi-method inspection comparison and statistical analysis.
    
    Provides ROC curve analysis, detection threshold optimization,
    and cost-benefit analysis for different inspection strategies.
    """
    
    def __init__(self):
        """Initialize comparison analysis tools."""
        pass
    
    def compute_roc_curve(self, signal_pristine: npt.NDArray[np.float64],
                         signal_cracked: npt.NDArray[np.float64],
                         thresholds: npt.NDArray[np.float64]) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Compute ROC (Receiver Operating Characteristic) curve.
        
        TPR = True Positive Rate (sensitivity)
        FPR = False Positive Rate (1 - specificity)
        
        Args:
            signal_pristine: Signal values for pristine substrate
            signal_cracked: Signal values for cracked substrate  
            thresholds: Detection threshold values
            
        Returns:
            Dict with 'fpr', 'tpr', 'thresholds', 'auc' arrays
        """
        if len(signal_pristine) == 0 or len(signal_cracked) == 0:
            return {
                "fpr": np.array([0, 1]), 
                "tpr": np.array([0, 1]),
                "thresholds": thresholds,
                "auc": 0.5
            }
            
        fpr_list = []
        tpr_list = []
        
        for threshold in thresholds:
            # Predictions: signal > threshold → "crack detected"
            fp = np.sum(signal_pristine > threshold)  # False positives
            tn = np.sum(signal_pristine <= threshold)  # True negatives
            tp = np.sum(signal_cracked > threshold)    # True positives
            fn = np.sum(signal_cracked <= threshold)   # False negatives
            
            # Calculate rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        
        # Compute AUC using trapezoidal integration
        auc = self._compute_auc(fpr_array, tpr_array)
        
        return {
            "fpr": fpr_array,
            "tpr": tpr_array, 
            "thresholds": thresholds,
            "auc": auc
        }
    
    def _compute_auc(self, fpr: npt.NDArray[np.float64], 
                     tpr: npt.NDArray[np.float64]) -> float:
        """
        Compute Area Under Curve (AUC) for ROC.
        
        Args:
            fpr: False positive rate array
            tpr: True positive rate array
            
        Returns:
            AUC value between 0.0 and 1.0
        """
        if len(fpr) < 2 or len(tpr) < 2:
            return 0.5
            
        # Sort by FPR for proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
        
        # Trapezoidal integration
        auc = integrate.trapezoid(tpr_sorted, fpr_sorted)
        
        # Ensure AUC is in valid range [0.5, 1.0]
        auc = np.clip(auc, 0.5, 1.0)
        
        return auc
    
    def minimum_detectable_crack(self, method: str, SNR_threshold: float = 3.0) -> float:
        """
        Calculate minimum detectable crack size for given method.
        
        Uses SNR > threshold criterion (typically SNR > 3).
        
        Args:
            method: Inspection method name
            SNR_threshold: Minimum required signal-to-noise ratio
            
        Returns:
            Minimum crack size [m]
        """
        if method == "acoustic":
            # For ultrasonic: limited by wavelength and attenuation
            freq = 5e6  # 5 MHz typical
            v_l = INSPECTION["acoustic"]["velocity_longitudinal"]
            wavelength = v_l / freq
            min_size = wavelength / 10  # λ/10 resolution limit
            
        elif method == "laser_scattering":
            # Rayleigh scattering detection limit
            min_size = INSPECTION["laser_scattering"]["detection_limit"]
            
        elif method == "raman":
            # Limited by spatial resolution
            min_size = INSPECTION["raman"]["spatial_resolution"]
            
        elif method == "interferometry_193nm":
            # Phase sensitivity limit
            min_phase = INSPECTION["interferometry_193nm"]["phase_sensitivity"]
            wavelength = INSPECTION["interferometry_193nm"]["wavelength"]
            # Δφ = 2π/λ × Δn × t → Δn = λ×Δφ/(2π×t)
            min_dn = wavelength * min_phase / (2 * np.pi * 6.35e-3)
            # Assume crack density ≈ 0.1 for detectable crack
            min_size = min_dn / 0.1  # Rough scaling
            
        elif method == "eels":
            # Electron beam resolution
            min_size = 1e-9  # 1 nm typical e-beam resolution
            
        elif method == "kfm":
            # KFM tip resolution
            min_size = 10e-9  # 10 nm typical tip resolution
            
        else:
            min_size = 1e-6  # 1 μm default
            
        # Apply SNR correction: smaller SNR_threshold → smaller detectable crack
        snr_factor = SNR_threshold / 3.0  # Normalize to SNR=3 baseline
        min_size_corrected = min_size * snr_factor
        
        return max(1e-9, min_size_corrected)  # Minimum 1 nm physical limit
    
    def optimal_inspection_strategy(self, crack_size_distribution: npt.NDArray[np.float64],
                                  cost_per_method: Dict[str, float]) -> Dict[str, float]:
        """
        Determine optimal inspection strategy based on cost-benefit analysis.
        
        Args:
            crack_size_distribution: Array of crack sizes to detect [m]
            cost_per_method: Dict of method costs [$/inspection]
            
        Returns:
            Dict with method rankings and optimal strategy
        """
        methods = list(cost_per_method.keys())
        
        # Compute detection capabilities
        detection_results = {}
        for method in methods:
            min_detectable = self.minimum_detectable_crack(method)
            detection_fraction = np.mean(crack_size_distribution >= min_detectable)
            cost = cost_per_method[method]
            
            # Cost-effectiveness: detection_fraction / cost
            cost_effectiveness = detection_fraction / (cost + 1e-6)  # Avoid division by zero
            
            detection_results[method] = {
                "min_detectable_size": min_detectable,
                "detection_fraction": detection_fraction,
                "cost": cost,
                "cost_effectiveness": cost_effectiveness
            }
        
        # Rank methods by cost-effectiveness
        sorted_methods = sorted(methods, key=lambda m: detection_results[m]["cost_effectiveness"], 
                              reverse=True)
        
        return {
            "optimal_method": sorted_methods[0] if sorted_methods else "none",
            "method_rankings": sorted_methods,
            "detection_results": detection_results
        }
    
    def compare_methods(self, crack_field: npt.NDArray[np.float64],
                       methods_list: List[str]) -> Dict[str, Dict]:
        """
        Compare multiple inspection methods on same crack field.
        
        Args:
            crack_field: 2D crack damage field
            methods_list: List of method names to compare
            
        Returns:
            Dict with comparison results for each method
        """
        comparison_results = {}
        
        # Initialize inspection modules
        acoustic = AcousticInspection()
        optical = OpticalInspection()
        ebeam = ElectronBeamInspection()
        
        for method in methods_list:
            if method == "acoustic":
                # Simulate acoustic inspection
                crack_density = np.mean(crack_field > 0.5)
                signal_level = crack_density * 100  # Arbitrary scaling
                snr = acoustic.signal_to_noise(crack_density * 1e-6, 1e-3, 
                                             {"frequency": 5e6, "noise_level": 0.01})
                
                comparison_results[method] = {
                    "signal_level": signal_level,
                    "snr": snr,
                    "detection_probability": min(1.0, snr / 3.0),
                    "method_specific": "ultrasonic"
                }
                
            elif method == "laser_scattering":
                # Simulate laser scattering
                max_crack_size = np.max(crack_field) * 1e-6  # Convert to meters
                scattering = optical.laser_scattering_signal(
                    max_crack_size, 1e-6, optical.laser_wavelength, optical.laser_power
                )
                
                comparison_results[method] = {
                    "signal_level": scattering["scattered_intensity"],
                    "snr": scattering["scattered_intensity"] / 1e-12,  # Rough SNR estimate
                    "detection_probability": min(1.0, max_crack_size / optical.detection_limit),
                    "method_specific": scattering["regime"]
                }
                
            elif method == "interferometry":
                # Simulate 193nm interferometry
                n_glass = ULE_GLASS["n_193nm"]
                dn_field = optical.compute_dn_from_cracks(crack_field, n_glass)
                phase_map = optical.interferometry_193nm(dn_field, 6.35e-3, 
                                                       optical.interferometry_wavelength)
                
                phase_rms = np.sqrt(np.mean(phase_map**2))
                
                comparison_results[method] = {
                    "signal_level": phase_rms,
                    "snr": phase_rms / optical.phase_sensitivity,
                    "detection_probability": min(1.0, phase_rms / optical.phase_sensitivity),
                    "method_specific": "phase_change"
                }
                
            else:
                # Default placeholder
                comparison_results[method] = {
                    "signal_level": 0.0,
                    "snr": 0.0, 
                    "detection_probability": 0.0,
                    "method_specific": "unknown"
                }
        
        return comparison_results


# =====================================================================
# Integration functions for module interoperability  
# =====================================================================

def run_inspection_analysis(crack_field_data: Dict[str, npt.NDArray[np.float64]], 
                          inspection_methods: List[str]) -> List[InspectionResult]:
    """
    Run comprehensive inspection analysis on crack field data.
    
    Args:
        crack_field_data: Dict with 'damage_field', 'stress_field', etc.
        inspection_methods: List of inspection methods to apply
        
    Returns:
        List of InspectionResult objects for each method
    """
    results = []
    
    damage_field = crack_field_data.get("damage_field", np.zeros((100, 100)))
    stress_field = crack_field_data.get("stress_field", np.zeros((100, 100)))
    
    # Initialize inspection modules
    acoustic = AcousticInspection()
    optical = OpticalInspection()
    ebeam = ElectronBeamInspection()
    comparison = InspectionComparison()
    
    for method in inspection_methods:
        
        if method == "acoustic":
            # Generate acoustic signal map
            signal_map = np.zeros_like(damage_field)
            for i in range(damage_field.shape[0]):
                for j in range(damage_field.shape[1]):
                    if damage_field[i, j] > 0.1:  # Crack present
                        crack_size = damage_field[i, j] * 1e-6  # Scale to meters
                        snr = acoustic.signal_to_noise(crack_size, 1e-3,
                                                     {"frequency": 5e6, "noise_level": 0.01})
                        signal_map[i, j] = snr
            
            reference_signal = np.zeros_like(signal_map)  # Pristine has no signal
            
        elif method == "raman":
            # Generate Raman stress map
            signal_map = optical.raman_stress_mapping(stress_field, -4.0, 500e-9)
            reference_signal = np.zeros_like(signal_map)  # Pristine has no stress
            
        elif method == "interferometry":
            # Generate phase change map
            n_glass = ULE_GLASS["n_193nm"]
            dn_field = optical.compute_dn_from_cracks(damage_field, n_glass)
            signal_map = optical.interferometry_193nm(dn_field, 6.35e-3, 193e-9)
            reference_signal = np.zeros_like(signal_map)
            
        else:
            # Default: use damage field as signal
            signal_map = damage_field.copy()
            reference_signal = np.zeros_like(signal_map)
        
        # Calculate metrics
        delta_signal = signal_map - reference_signal
        
        # SNR map (simplified)
        noise_level = 0.01 * np.max(np.abs(signal_map))
        snr_map = np.abs(delta_signal) / (noise_level + 1e-12)
        
        # Detection metrics
        min_detectable_size = comparison.minimum_detectable_crack(method)
        detection_probability = np.mean(snr_map > 3.0)  # SNR > 3 criterion
        
        # ROC analysis (simplified)
        if np.max(signal_map) > 0:
            thresholds = np.linspace(0, np.max(signal_map), 20)
            pristine_signals = reference_signal.flatten()
            cracked_signals = signal_map.flatten()
            
            roc_result = comparison.compute_roc_curve(pristine_signals, cracked_signals, thresholds)
            roc_auc = roc_result["auc"]
        else:
            roc_auc = 0.5
        
        # Create result object
        result = InspectionResult(
            method=method,
            signal_map=signal_map,
            reference_signal=reference_signal,
            delta_signal=delta_signal,
            snr_map=snr_map,
            min_detectable_size=min_detectable_size,
            detection_probability=detection_probability,
            roc_auc=roc_auc
        )
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Demonstration of M3 Inspection Forward Model
    print("M3 Inspection Signal Forward Model Demo")
    print("=" * 50)
    
    # Test AcousticInspection
    acoustic = AcousticInspection()
    
    # Lamb wave dispersion
    freq = 5e6  # 5 MHz
    thickness = 6.35e-3  # 6.35 mm
    v_s0, v_a0 = acoustic.compute_lamb_wave_dispersion(freq, thickness, 
                                                      acoustic.v_l, acoustic.v_t)
    print(f"Lamb wave velocities at {freq/1e6:.1f} MHz:")
    print(f"  S0 mode: {v_s0:.0f} m/s")
    print(f"  A0 mode: {v_a0:.0f} m/s")
    
    # Test OpticalInspection
    optical = OpticalInspection()
    
    # Laser scattering
    scattering = optical.laser_scattering_signal(50e-9, 1e-6, 532e-9, 50e-3)
    print(f"\nLaser scattering (50nm crack):")
    print(f"  Intensity: {scattering['scattered_intensity']:.2e} W/m²")
    print(f"  Regime: {scattering['regime']}")
    
    # Test InspectionComparison
    comparison = InspectionComparison()
    
    min_sizes = {}
    for method in ["acoustic", "laser_scattering", "raman", "interferometry_193nm"]:
        min_size = comparison.minimum_detectable_crack(method)
        min_sizes[method] = min_size
        print(f"  {method}: {min_size*1e9:.1f} nm")
    
    print("\nDemo completed successfully!")