"""
Module M1: Crack Nucleation Probability Engine

Physics-based simulation of micro-crack nucleation in ULE glass substrates
under EUV exposure thermal cycling.

Classes:
    DefectField: Generate and manage spatial defect distributions
    ThermoelasticStress: Compute thermal fields and stress distributions
    NucleationEngine: Monte Carlo nucleation probability calculation

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import numpy as np
import numpy.typing as npt
from scipy import ndimage, spatial
from scipy.special import gamma
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import configuration parameters
from config import (
    ULE_GLASS, EUV_CONDITIONS, SUBSTRATE, DEFECT_MODEL, 
    NUCLEATION, SIMULATION, k_B
)


@dataclass
class DefectFieldResult:
    """Result container for defect field generation."""
    positions: npt.NDArray[np.float64]  # Shape: (N_defects, 3) [x, y, z] in meters
    flaw_sizes: npt.NDArray[np.float64]  # Shape: (N_defects,) radius in meters
    density: float  # Actual defects/m³
    correlation_length: Optional[float] = None  # meters


@dataclass
class StressFieldResult:
    """Result container for stress field computation."""
    thermal_field: npt.NDArray[np.float64]  # Shape: (nx, ny) temperature in K
    stress_field: npt.NDArray[np.float64]   # Shape: (nx, ny) stress in Pa
    cte_map: npt.NDArray[np.float64]        # Shape: (nx, ny) CTE in 1/K
    grid_x: npt.NDArray[np.float64]         # x coordinates in meters
    grid_y: npt.NDArray[np.float64]         # y coordinates in meters


@dataclass
class NucleationResult:
    """Result container for nucleation analysis."""
    nucleation_probability: float
    nucleation_sites: List[Tuple[float, float, float]]  # (x, y, z) coordinates
    stress_intensity_factors: npt.NDArray[np.float64]   # K_I values
    griffith_ratios: npt.NDArray[np.float64]            # G_I/G_IC ratios
    time_to_first_crack: Optional[float] = None         # seconds
    nucleation_density: float = 0.0                     # nucleations/m³


class DefectField:
    """
    Generate spatial distributions of defects/impurities in glass substrate.
    
    Supports Poisson and Neyman-Scott cluster processes for realistic
    defect spatial patterns.
    """
    
    def __init__(self, substrate_geometry: Dict[str, float]):
        """Initialize with substrate dimensions."""
        self.length = substrate_geometry["length"]
        self.width = substrate_geometry["width"] 
        self.thickness = substrate_geometry["thickness"]
        self.volume = self.length * self.width * self.thickness
        
    def generate_poisson(self, density: float, domain_size: Tuple[float, float, float],
                        seed: Optional[int] = None) -> DefectFieldResult:
        """
        Generate defect positions using 3D Poisson point process.
        
        Args:
            density: Defect density [defects/m³]
            domain_size: (length, width, thickness) in meters
            seed: Random seed for reproducibility
            
        Returns:
            DefectFieldResult with positions and metadata
            
        Raises:
            ValueError: If density is negative or domain_size invalid
        """
        if density < 0:
            raise ValueError("Defect density must be non-negative")
        if any(d <= 0 for d in domain_size):
            raise ValueError("All domain dimensions must be positive")
            
        if seed is not None:
            np.random.seed(seed)
            
        volume = np.prod(domain_size)
        expected_count = density * volume
        actual_count = np.random.poisson(expected_count)
        
        if actual_count == 0:
            positions = np.empty((0, 3))
            flaw_sizes = np.empty(0)
        else:
            # Uniform random positions in domain
            positions = np.random.uniform(
                low=[0, 0, 0],
                high=domain_size,
                size=(actual_count, 3)
            )
            flaw_sizes = self.assign_flaw_sizes(actual_count)
            
        actual_density = actual_count / volume
        
        return DefectFieldResult(
            positions=positions,
            flaw_sizes=flaw_sizes,
            density=actual_density
        )
    
    def generate_neyman_scott(self, cluster_density: float, cluster_radius: float,
                             points_per_cluster: int, domain_size: Tuple[float, float, float],
                             seed: Optional[int] = None) -> DefectFieldResult:
        """
        Generate clustered defects using Neyman-Scott cluster process.
        
        Args:
            cluster_density: Cluster centers per m³
            cluster_radius: Cluster radius in meters
            points_per_cluster: Mean points per cluster
            domain_size: (length, width, thickness) in meters
            seed: Random seed
            
        Returns:
            DefectFieldResult with clustered defect distribution
        """
        if seed is not None:
            np.random.seed(seed)
            
        volume = np.prod(domain_size)
        n_clusters = np.random.poisson(cluster_density * volume)
        
        all_positions = []
        
        for _ in range(n_clusters):
            # Random cluster center
            center = np.random.uniform(low=[0, 0, 0], high=domain_size)
            
            # Number of points in this cluster
            n_points = np.random.poisson(points_per_cluster)
            
            for _ in range(n_points):
                # Generate point around cluster center
                # Use spherical coordinates for uniform distribution in sphere
                r = cluster_radius * np.random.random()**(1/3)
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.arccos(1 - 2*np.random.random())
                
                offset = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta), 
                    r * np.cos(phi)
                ])
                
                position = center + offset
                
                # Keep only points within domain
                if all(0 <= position[i] <= domain_size[i] for i in range(3)):
                    all_positions.append(position)
        
        if all_positions:
            positions = np.array(all_positions)
            flaw_sizes = self.assign_flaw_sizes(len(all_positions))
        else:
            positions = np.empty((0, 3))
            flaw_sizes = np.empty(0)
            
        actual_density = len(all_positions) / volume
        
        return DefectFieldResult(
            positions=positions,
            flaw_sizes=flaw_sizes,
            density=actual_density,
            correlation_length=cluster_radius
        )
    
    def assign_flaw_sizes(self, n_defects: int) -> npt.NDArray[np.float64]:
        """
        Assign initial flaw sizes using log-normal distribution.
        
        Args:
            n_defects: Number of defects
            
        Returns:
            Array of flaw sizes in meters
        """
        if n_defects == 0:
            return np.empty(0)
            
        mean = DEFECT_MODEL["flaw_size_mean"]
        sigma = DEFECT_MODEL["flaw_size_sigma"]
        min_size = DEFECT_MODEL["flaw_size_min"]
        max_size = DEFECT_MODEL["flaw_size_max"]
        
        # Log-normal parameters
        mu = np.log(mean)
        
        sizes = np.random.lognormal(mu, sigma, n_defects)
        
        # Clip to physical bounds
        sizes = np.clip(sizes, min_size, max_size)
        
        return sizes
    
    def compute_spatial_correlation(self, positions: npt.NDArray[np.float64],
                                  r_max: Optional[float] = None,
                                  n_bins: int = 50) -> Tuple[npt.NDArray[np.float64], 
                                                            npt.NDArray[np.float64]]:
        """
        Compute pair correlation function g(r).
        
        Args:
            positions: Defect positions array (N, 3)
            r_max: Maximum distance to consider (default: substrate diagonal/4)
            n_bins: Number of distance bins
            
        Returns:
            Tuple of (distances, g_r) arrays
        """
        if len(positions) < 2:
            return np.array([]), np.array([])
            
        if r_max is None:
            diagonal = np.sqrt(self.length**2 + self.width**2 + self.thickness**2)
            r_max = diagonal / 4
            
        # Compute all pairwise distances
        distances = spatial.distance.pdist(positions)
        
        # Create distance bins
        r_bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        dr = r_bins[1] - r_bins[0]
        
        # Histogram of distances
        hist, _ = np.histogram(distances, bins=r_bins)
        
        # Normalization for g(r)
        n_pairs = len(distances)
        density = len(positions) / self.volume
        
        # Volume of spherical shell
        shell_volumes = (4/3) * np.pi * ((r_bins[1:])**3 - (r_bins[:-1])**3)
        
        # Pair correlation function
        expected_pairs = density * shell_volumes * len(positions)
        g_r = hist / (expected_pairs + 1e-12)  # Avoid division by zero
        
        return r_centers, g_r


class CTEMapGenerator:
    """Utility class for generating coefficient of thermal expansion maps."""
    
    @staticmethod
    def generate_gaussian_random_field(grid_size: Tuple[int, int], sigma: float,
                                     correlation_length: float, seed: Optional[int] = None
                                     ) -> npt.NDArray[np.float64]:
        """
        Generate 2D Gaussian random field for CTE variations.
        
        Args:
            grid_size: (nx, ny) grid dimensions
            sigma: Standard deviation of CTE variation [1/K]
            correlation_length: Spatial correlation length [m]
            seed: Random seed
            
        Returns:
            2D array of CTE values centered at ULE_GLASS["CTE_mean"]
        """
        if seed is not None:
            np.random.seed(seed)
            
        nx, ny = grid_size
        
        # Convert correlation length to grid units
        dx = SUBSTRATE["length"] / nx
        dy = SUBSTRATE["width"] / ny
        sigma_x = correlation_length / dx
        sigma_y = correlation_length / dy
        
        # Generate white noise with adjusted amplitude to compensate for filtering
        # The Gaussian filter reduces the variance, so we scale up the input
        filter_reduction_factor = 1.0 / np.sqrt(2 * np.pi * sigma_x * sigma_y)
        adjusted_sigma = sigma / max(0.3, filter_reduction_factor)  # Prevent over-scaling
        
        white_noise = np.random.normal(0, adjusted_sigma, (nx, ny))
        
        # Apply Gaussian filter for spatial correlation
        grf = ndimage.gaussian_filter(white_noise, sigma=[sigma_x, sigma_y], mode='wrap')
        
        # Rescale to target standard deviation
        actual_std = np.std(grf)
        if actual_std > 0:
            grf = grf * (sigma / actual_std)
        
        # Add mean CTE value
        cte_mean = ULE_GLASS["CTE_mean"]
        return grf + cte_mean
    
    @staticmethod
    def generate_patchy_field(grid_size: Tuple[int, int], sigma: float,
                             seed: Optional[int] = None) -> npt.NDArray[np.float64]:
        """
        Generate patchy CTE field using Voronoi tessellation.
        
        Args:
            grid_size: (nx, ny) grid dimensions  
            sigma: Standard deviation of CTE patches [1/K]
            seed: Random seed
            
        Returns:
            2D array of CTE values with patchy structure
        """
        if seed is not None:
            np.random.seed(seed)
            
        nx, ny = grid_size
        
        # Number of Voronoi seed points (patches)
        n_patches = max(10, int(np.sqrt(nx * ny) / 4))
        
        # Random seed points
        seed_points = np.random.uniform(0, 1, (n_patches, 2))
        seed_points[:, 0] *= nx
        seed_points[:, 1] *= ny
        
        # Create grid for distance calculation
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        grid_points = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Find nearest seed point for each grid point
        distances = spatial.distance.cdist(grid_points, seed_points)
        nearest_seeds = np.argmin(distances, axis=1)
        
        # Assign random CTE values to each patch
        cte_mean = ULE_GLASS["CTE_mean"]
        patch_values = np.random.normal(cte_mean, sigma, n_patches)
        
        # Map patch values to grid
        cte_field = patch_values[nearest_seeds].reshape((nx, ny))
        
        return cte_field


class ThermoelasticStress:
    """
    Compute thermal fields and resulting stress distributions from EUV exposure.
    
    Handles both uniform and spatially-varying coefficient of thermal expansion.
    """
    
    def __init__(self):
        """Initialize with ULE glass material properties."""
        self.E = ULE_GLASS["E_young"]
        self.nu = ULE_GLASS["nu_poisson"]
        self.cte_mean = ULE_GLASS["CTE_mean"]
        self.cte_sigma = ULE_GLASS["CTE_sigma"]
        
    def compute_thermal_field(self, dose: float, delta_T: float,
                            substrate_geometry: Dict[str, float]) -> npt.NDArray[np.float64]:
        """
        Compute 2D temperature field from EUV slit heating profile.
        
        Args:
            dose: EUV dose [mJ/cm²]
            delta_T: Peak temperature rise [K]
            substrate_geometry: Dictionary with substrate dimensions
            
        Returns:
            2D temperature field array
        """
        if dose < 0:
            raise ValueError("Dose must be non-negative")
        if delta_T < 0:
            raise ValueError("Temperature rise must be non-negative")
            
        # Create spatial grid
        nx, ny = SIMULATION["grid_2d"]
        x = np.linspace(0, substrate_geometry["length"], nx)
        y = np.linspace(0, substrate_geometry["width"], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # EUV slit heating profile (Gaussian-like)
        # Slit is typically oriented along y-direction
        x_center = substrate_geometry["length"] / 2
        slit_width = 0.020  # 20mm typical slit width
        
        # Gaussian profile in x-direction, uniform in y
        thermal_profile = np.exp(-((X - x_center) / (slit_width / 4))**2)
        
        # Scale by dose and delta_T
        dose_normalized = dose / 50.0  # Normalize to typical 50 mJ/cm²
        thermal_field = delta_T * dose_normalized * thermal_profile
        
        return thermal_field
    
    def compute_stress_field(self, thermal_field: npt.NDArray[np.float64],
                           cte_map: npt.NDArray[np.float64],
                           E: float, nu: float) -> npt.NDArray[np.float64]:
        """
        Compute thermoelastic stress field using plane stress approximation.
        
        Args:
            thermal_field: 2D temperature field [K]
            cte_map: 2D coefficient of thermal expansion map [1/K]
            E: Young's modulus [Pa]
            nu: Poisson's ratio
            
        Returns:
            2D stress field [Pa]
        """
        # Plane stress thermoelastic equation:
        # For constrained thermal expansion with spatial CTE variation:
        # σ = E/(1-ν) * [α(x,y) - α_mean] * ΔT
        # This gives stress from CTE mismatch rather than absolute expansion
        
        constraint_factor = 1.0  # Full constraint for substrate mounted in chuck
        
        # Calculate CTE variation from mean
        cte_mean = np.mean(cte_map)
        cte_variation = cte_map - cte_mean
        
        # Stress from mismatch strain
        stress_field = (E / (1 - nu)) * cte_variation * thermal_field * constraint_factor
        
        return stress_field
    
    def compute_stress_at_defects(self, stress_field: npt.NDArray[np.float64],
                                defect_positions: npt.NDArray[np.float64],
                                grid_x: npt.NDArray[np.float64],
                                grid_y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Interpolate stress values at defect positions.
        
        Args:
            stress_field: 2D stress field array
            defect_positions: Array of (x, y, z) defect coordinates
            grid_x: X coordinate grid
            grid_y: Y coordinate grid
            
        Returns:
            Array of stress values at defect locations
        """
        if len(defect_positions) == 0:
            return np.array([])
            
        # Use bilinear interpolation
        from scipy.interpolate import RegularGridInterpolator
        
        interpolator = RegularGridInterpolator(
            (grid_x[:, 0], grid_y[0, :]), 
            stress_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Only use x, y coordinates (ignore z for 2D field)
        xy_positions = defect_positions[:, :2]
        stress_at_defects = interpolator(xy_positions)
        
        return stress_at_defects


class NucleationEngine:
    """
    Monte Carlo engine for crack nucleation probability calculation.
    
    Implements Griffith criterion with stress intensity factor analysis.
    """
    
    def __init__(self):
        """Initialize with material properties and nucleation parameters."""
        self.K_IC = ULE_GLASS["K_IC"]
        self.K_0_ratio = ULE_GLASS["K_0_ratio"]
        self.K_0 = self.K_IC * self.K_0_ratio
        self.stress_concentration_factor = NUCLEATION["stress_concentration_factor"]
        self.nucleation_threshold = NUCLEATION["nucleation_threshold"]
        
    def compute_stress_intensity(self, sigma_local: npt.NDArray[np.float64],
                               flaw_size: npt.NDArray[np.float64],
                               geometry_factor: float = 1.12) -> npt.NDArray[np.float64]:
        """
        Compute Mode I stress intensity factor K_I.
        
        Args:
            sigma_local: Local stress at defect [Pa]
            flaw_size: Flaw radius [m]
            geometry_factor: Geometric factor Y (default for penny-shaped crack)
            
        Returns:
            Array of K_I values [Pa·m^0.5]
        """
        # K_I = Y * σ * √(π * a)
        K_I = geometry_factor * sigma_local * np.sqrt(np.pi * flaw_size)
        return K_I
    
    def griffith_criterion(self, K_I: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply Griffith energy criterion for nucleation.
        
        Args:
            K_I: Stress intensity factors [Pa·m^0.5]
            
        Returns:
            Array of G_I/G_IC ratios
        """
        # Energy release rate: G_I = K_I² / E'
        # For plane stress: E' = E
        # For plane strain: E' = E/(1-ν²)
        # Use plane stress for thin substrate
        
        E_prime = ULE_GLASS["E_young"]
        G_I = K_I**2 / E_prime
        
        # Critical energy release rate
        K_IC = self.K_IC
        G_IC = K_IC**2 / E_prime
        
        return G_I / G_IC
    
    def subcritical_check(self, K_I: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Check if defects are in subcritical crack growth regime.
        
        Args:
            K_I: Stress intensity factors
            
        Returns:
            Boolean array indicating subcritical regime (K_0 < K_I < K_IC)
        """
        return (K_I > self.K_0) & (K_I < self.K_IC)
    
    def run_monte_carlo(self, n_runs: int, defect_params: Dict,
                       stress_params: Dict, substrate: Dict) -> Dict:
        """
        Run Monte Carlo simulation for nucleation probability.
        
        Args:
            n_runs: Number of MC realizations
            defect_params: Parameters for defect generation
            stress_params: Parameters for stress calculation  
            substrate: Substrate geometry parameters
            
        Returns:
            Dictionary with nucleation statistics
        """
        nucleation_events = []
        nucleation_densities = []
        first_crack_times = []
        
        defect_field = DefectField(substrate)
        stress_calculator = ThermoelasticStress()
        cte_generator = CTEMapGenerator()
        
        for run in range(n_runs):
            # Generate defects
            seed = SIMULATION["random_seed"] + run if SIMULATION.get("random_seed") else None
            
            domain_size = (substrate["length"], substrate["width"], substrate["thickness"])
            
            if defect_params.get("distribution_type") == "neyman_scott":
                defect_result = defect_field.generate_neyman_scott(
                    cluster_density=defect_params["cluster_density"],
                    cluster_radius=defect_params["cluster_radius"], 
                    points_per_cluster=defect_params["points_per_cluster"],
                    domain_size=domain_size,
                    seed=seed
                )
            else:
                defect_result = defect_field.generate_poisson(
                    density=defect_params["density"],
                    domain_size=domain_size,
                    seed=seed
                )
                
            if len(defect_result.positions) == 0:
                nucleation_events.append([])
                nucleation_densities.append(0.0)
                first_crack_times.append(np.inf)
                continue
                
            # Generate CTE map
            grid_size = SIMULATION["grid_2d"]
            if stress_params.get("cte_type") == "patchy":
                cte_map = cte_generator.generate_patchy_field(
                    grid_size, ULE_GLASS["CTE_sigma"], seed
                )
            else:
                cte_map = cte_generator.generate_gaussian_random_field(
                    grid_size, ULE_GLASS["CTE_sigma"], 
                    stress_params.get("correlation_length", 1e-3), seed
                )
                
            # Compute thermal and stress fields
            thermal_field = stress_calculator.compute_thermal_field(
                dose=stress_params["dose"],
                delta_T=stress_params["delta_T"],
                substrate_geometry=substrate
            )
            
            stress_field = stress_calculator.compute_stress_field(
                thermal_field, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
            )
            
            # Create coordinate grids
            nx, ny = grid_size
            x = np.linspace(0, substrate["length"], nx)
            y = np.linspace(0, substrate["width"], ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Get stress at defect locations
            stress_at_defects = stress_calculator.compute_stress_at_defects(
                stress_field, defect_result.positions, X, Y
            )
            
            # Compute stress intensity factors
            K_I = self.compute_stress_intensity(
                stress_at_defects, defect_result.flaw_sizes
            )
            
            # Apply Griffith criterion
            griffith_ratios = self.griffith_criterion(K_I)
            
            # Find nucleated cracks
            nucleated_mask = griffith_ratios >= self.nucleation_threshold
            nucleated_positions = defect_result.positions[nucleated_mask]
            
            nucleation_events.append(nucleated_positions.tolist())
            nucleation_density = len(nucleated_positions) / substrate["length"] / substrate["width"] / substrate["thickness"]
            nucleation_densities.append(nucleation_density)
            
            # Estimate time to first crack (simplified)
            if len(nucleated_positions) > 0:
                # Use subcritical crack growth model for timing
                max_ratio = np.max(griffith_ratios)
                time_to_crack = 1.0 / max_ratio  # Simplified scaling
                first_crack_times.append(time_to_crack)
            else:
                first_crack_times.append(np.inf)
        
        return {
            "nucleation_probability": np.mean([len(events) > 0 for events in nucleation_events]),
            "mean_nucleation_density": np.mean(nucleation_densities),
            "std_nucleation_density": np.std(nucleation_densities),
            "nucleation_events": nucleation_events,
            "time_to_first_crack_stats": {
                "mean": np.mean([t for t in first_crack_times if np.isfinite(t)]),
                "std": np.std([t for t in first_crack_times if np.isfinite(t)]),
                "min": np.min([t for t in first_crack_times if np.isfinite(t)]) if any(np.isfinite(first_crack_times)) else np.inf
            }
        }
    
    def run_thermal_cycling(self, n_cycles: int, cycle_params: Dict,
                          defect_params: Dict, substrate: Dict) -> Dict:
        """
        Simulate nucleation under repeated thermal cycling.
        
        Args:
            n_cycles: Number of thermal cycles
            cycle_params: Parameters for each cycle (dose, delta_T, etc.)
            defect_params: Defect generation parameters
            substrate: Substrate geometry
            
        Returns:
            Dictionary with cumulative damage analysis
        """
        # Generate initial defect field
        defect_field = DefectField(substrate)
        domain_size = (substrate["length"], substrate["width"], substrate["thickness"])
        
        defect_result = defect_field.generate_poisson(
            density=defect_params["density"],
            domain_size=domain_size,
            seed=SIMULATION.get("random_seed")
        )
        
        if len(defect_result.positions) == 0:
            return {
                "nucleation_probability_vs_cycles": np.zeros(n_cycles),
                "cumulative_damage": np.zeros(n_cycles),
                "total_nucleations": 0
            }
        
        nucleation_probability_history = []
        cumulative_damage = np.zeros(len(defect_result.positions))
        total_nucleations = 0
        
        stress_calculator = ThermoelasticStress()
        cte_generator = CTEMapGenerator()
        
        for cycle in range(n_cycles):
            # Generate CTE map for this cycle
            grid_size = SIMULATION["grid_2d"] 
            cte_map = cte_generator.generate_gaussian_random_field(
                grid_size, ULE_GLASS["CTE_sigma"], 
                cycle_params.get("correlation_length", 1e-3), 
                seed=SIMULATION.get("random_seed", 42) + cycle
            )
            
            # Compute stress field for this cycle
            thermal_field = stress_calculator.compute_thermal_field(
                dose=cycle_params["dose"],
                delta_T=cycle_params["delta_T"],
                substrate_geometry=substrate
            )
            
            stress_field = stress_calculator.compute_stress_field(
                thermal_field, cte_map, ULE_GLASS["E_young"], ULE_GLASS["nu_poisson"]
            )
            
            # Create coordinate grids
            nx, ny = grid_size
            x = np.linspace(0, substrate["length"], nx)
            y = np.linspace(0, substrate["width"], ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Get stress at defect locations
            stress_at_defects = stress_calculator.compute_stress_at_defects(
                stress_field, defect_result.positions, X, Y
            )
            
            # Compute stress intensity factors
            K_I = self.compute_stress_intensity(
                stress_at_defects, defect_result.flaw_sizes
            )
            
            # Accumulate subcritical damage
            subcritical_mask = self.subcritical_check(K_I)
            damage_increment = (K_I / self.K_IC) ** ULE_GLASS.get("scg_n", 20.0)
            cumulative_damage += damage_increment
            
            # Check for new nucleations (damage threshold exceeded)
            damage_threshold = 1.0
            new_nucleations = (cumulative_damage >= damage_threshold) & (cumulative_damage < damage_threshold + damage_increment)
            total_nucleations += np.sum(new_nucleations)
            
            # Calculate nucleation probability for this cycle
            nucleation_prob = total_nucleations / len(defect_result.positions)
            nucleation_probability_history.append(nucleation_prob)
        
        return {
            "nucleation_probability_vs_cycles": np.array(nucleation_probability_history),
            "cumulative_damage": cumulative_damage,
            "total_nucleations": total_nucleations,
            "cycles_to_first_nucleation": np.argmax(np.array(nucleation_probability_history) > 0) + 1 if total_nucleations > 0 else n_cycles
        }