"""
Module M5: Process Attribution Engine

Quantitative decomposition of semiconductor process anomaly signals to determine
how much is due to substrate degradation (cracks) versus other sources.

Classes:
    OverlayDegradationModel: Model overlay degradation from crack state
    VarianceDecomposition: Decompose process variance components with Bayesian changepoint detection
    ReplacementOptimizer: Optimize substrate replacement timing based on cost model
    ProcessSimulator: Simulate various lithography processes with crack effects
    AttributionResult: Result container for attribution analysis

Author: Glass Crack Lifecycle Simulator Team
Date: 2026-02-25
"""

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats, special
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Import configuration parameters
from config import ATTRIBUTION


@dataclass
class AttributionResult:
    """Result container for process attribution analysis."""
    total_overlay: float                          # nm RMS
    scanner_contribution: float                   # %
    mask_pristine_contribution: float             # %
    mask_degradation_contribution: float          # %
    process_contribution: float                   # %
    changepoint_detected: bool
    changepoint_time: Optional[float]
    changepoint_confidence: float
    optimal_replacement_time: Optional[float]
    total_cost_optimal: float                     # USD
    total_cost_no_replacement: float              # USD
    cost_savings: float                           # USD


@dataclass
class ProcessMetrics:
    """Container for lithography process metrics."""
    overlay_x: float        # nm RMS
    overlay_y: float        # nm RMS
    overlay_total: float    # nm RMS
    cdu: float             # nm 3-sigma
    epe: float             # nm 3-sigma
    yield_loss: float      # fraction
    process_type: str


class OverlayDegradationModel:
    """
    Model overlay degradation from substrate crack state.
    
    Implements power-law relationship between crack density and overlay degradation
    through thermal deformation and optical phase errors.
    """
    
    def __init__(self):
        """Initialize with attribution parameters from config."""
        self.scanner_sigma = ATTRIBUTION["overlay_scanner"]
        self.mask_pristine_sigma = ATTRIBUTION["overlay_mask_pristine"]
        self.process_sigma = ATTRIBUTION["overlay_process"]
        self.degradation_exponent = ATTRIBUTION["degradation_exponent"]
        
    def compute_pristine_overlay(self, scanner_sigma: Optional[float] = None,
                                mask_pristine_sigma: Optional[float] = None,
                                process_sigma: Optional[float] = None) -> float:
        """
        Compute pristine (undegraded) overlay error budget.
        
        Args:
            scanner_sigma: Scanner contribution [nm RMS] (uses config default if None)
            mask_pristine_sigma: Pristine mask contribution [nm RMS] (uses config default if None)
            process_sigma: Process contribution [nm RMS] (uses config default if None)
            
        Returns:
            Total pristine overlay error [nm RMS]
            
        Raises:
            ValueError: If any sigma values are negative
        """
        # Use config defaults if not provided
        scanner = scanner_sigma if scanner_sigma is not None else self.scanner_sigma
        mask_pristine = mask_pristine_sigma if mask_pristine_sigma is not None else self.mask_pristine_sigma
        process = process_sigma if process_sigma is not None else self.process_sigma
        
        if scanner < 0 or mask_pristine < 0 or process < 0:
            raise ValueError("Overlay sigma values must be non-negative")
            
        # RSS combination: σ²_total = σ²_scanner + σ²_mask,pristine + σ²_process
        total_variance = scanner**2 + mask_pristine**2 + process**2
        return np.sqrt(total_variance)
    
    def compute_degraded_overlay(self, crack_state: Union[float, npt.NDArray[np.float64]], 
                                pristine_overlay: float) -> Union[float, npt.NDArray[np.float64]]:
        """
        Compute overlay error including crack-induced degradation.
        
        Args:
            crack_state: Crack density [m⁻²] or dimensionless damage parameter
            pristine_overlay: Pristine overlay error [nm RMS]
            
        Returns:
            Total overlay error with degradation [nm RMS]
            
        Raises:
            ValueError: If crack_state is negative or pristine_overlay is invalid
        """
        crack_array = np.asarray(crack_state)
        if np.any(crack_array < 0):
            raise ValueError("Crack state must be non-negative")
        if pristine_overlay < 0:
            raise ValueError("Pristine overlay must be non-negative")
            
        # Power-law degradation model: σ_deg ∝ (crack_density)^α
        # Physics rationale:
        # - crack → local CTE anomaly → thermal deformation → overlay change
        # - crack → local Δn anomaly → phase/registration error
        
        # Normalize crack density for stability (typical range: 1e6 - 1e12 m⁻²)
        # Use dimensionless form: (crack_density / reference_density)^α
        reference_density = 1e8  # m⁻² reference crack density
        normalized_crack_density = crack_array / reference_density
        
        # Degradation contribution
        degradation_sigma = self.mask_pristine_sigma * (normalized_crack_density**self.degradation_exponent)
        
        # Total mask variance: σ²_mask = σ²_mask,pristine + σ²_mask,degradation
        total_mask_variance = self.mask_pristine_sigma**2 + degradation_sigma**2
        
        # Total overlay variance
        total_variance = (self.scanner_sigma**2 + total_mask_variance + 
                         self.process_sigma**2)
        
        return np.sqrt(total_variance)
    
    def overlay_time_series(self, crack_density_vs_time: npt.NDArray[np.float64],
                           time_points: Optional[npt.NDArray[np.float64]] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute overlay degradation time series.
        
        Args:
            crack_density_vs_time: Crack density evolution [m⁻² or dimensionless]
            time_points: Time points [hours or lots] (creates default if None)
            
        Returns:
            Tuple of (time_points, overlay_time_series)
        """
        if len(crack_density_vs_time) == 0:
            raise ValueError("Crack density time series cannot be empty")
            
        if time_points is None:
            time_points = np.arange(len(crack_density_vs_time), dtype=float)
        
        if len(time_points) != len(crack_density_vs_time):
            raise ValueError("Time points and crack density arrays must have same length")
            
        pristine_overlay = self.compute_pristine_overlay()
        overlay_series = self.compute_degraded_overlay(crack_density_vs_time, pristine_overlay)
        
        return time_points, overlay_series


class VarianceDecomposition:
    """
    Decompose process variance into components with Bayesian changepoint detection.
    
    Identifies when substrate degradation becomes statistically significant.
    """
    
    def __init__(self):
        """Initialize decomposition engine."""
        pass
        
    def decompose(self, total_variance_series: npt.NDArray[np.float64],
                 known_components: Dict[str, float]) -> Dict[str, Union[float, npt.NDArray]]:
        """
        Decompose total variance into constituent components.
        
        Args:
            total_variance_series: Time series of total overlay variance [nm²]
            known_components: Known variance components {"scanner": σ², "process": σ²}
            
        Returns:
            Dictionary with component contributions and statistics
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(total_variance_series) == 0:
            raise ValueError("Variance series cannot be empty")
        if np.any(total_variance_series < 0):
            raise ValueError("Variance values must be non-negative")
            
        total_var = np.mean(total_variance_series)
        
        # Extract known components
        scanner_var = known_components.get("scanner", 0.0)**2
        process_var = known_components.get("process", 0.0)**2
        
        # Remaining variance attributed to mask (pristine + degradation)
        mask_total_var = max(0.0, total_var - scanner_var - process_var)
        
        # Estimate pristine mask contribution (use config default)
        mask_pristine_var = ATTRIBUTION["overlay_mask_pristine"]**2
        
        # Degradation component
        mask_degradation_var = max(0.0, mask_total_var - mask_pristine_var)
        
        # Convert to percentages (avoid division by zero)
        total_nonzero = max(total_var, 1e-12)
        
        result = {
            "scanner_contribution": 100.0 * scanner_var / total_nonzero,
            "mask_pristine_contribution": 100.0 * mask_pristine_var / total_nonzero,
            "mask_degradation_contribution": 100.0 * mask_degradation_var / total_nonzero,
            "process_contribution": 100.0 * process_var / total_nonzero,
            "total_variance": total_var,
            "variance_series": total_variance_series
        }
        
        return result
    
    def bayesian_changepoint(self, time_series: npt.NDArray[np.float64],
                            prior_params: Optional[Dict] = None) -> Tuple[Optional[float], float]:
        """
        Bayesian changepoint detection for degradation onset.
        
        Args:
            time_series: Overlay error time series [nm RMS]
            prior_params: Prior parameters for changepoint model
            
        Returns:
            Tuple of (changepoint_time, confidence)
        """
        if len(time_series) < 3:
            return None, 0.0
            
        # Default prior parameters
        if prior_params is None:
            prior_params = {
                "hazard_rate": 1.0 / len(time_series),  # Expect ~1 changepoint per series
                "sigma_ratio_threshold": 1.5  # Minimum signal increase to detect
            }
            
        n_points = len(time_series)
        hazard_rate = prior_params["hazard_rate"]
        
        # Simplified online changepoint detection using CUSUM-like approach
        # H0: mean = μ₀ (pristine state)
        # H1: mean = μ₁ > μ₀ (degraded state)
        
        # Estimate pristine mean (first quartile of data as baseline)
        baseline_length = max(3, n_points // 4)
        mu_0 = np.mean(time_series[:baseline_length])
        sigma_0 = max(np.std(time_series[:baseline_length]), 0.01)  # Avoid zero std
        
        # Track posterior probability of changepoint at each time
        log_posterior_odds = np.zeros(n_points)
        max_log_odds = -np.inf
        changepoint_idx = None
        
        for t in range(baseline_length, n_points):
            # Likelihood ratio for changepoint at time t
            # Simple approach: compare means before and after
            y_before = time_series[:t]
            y_after = time_series[t:]
            
            if len(y_after) < 2:
                continue
                
            mean_before = np.mean(y_before)
            mean_after = np.mean(y_after)
            
            # Only consider increases (degradation)
            if mean_after <= mean_before:
                continue
                
            # Log likelihood ratio (simplified Gaussian model)
            # Assume known variance for simplicity
            n_before = len(y_before)
            n_after = len(y_after)
            
            # Test statistic: standardized difference in means
            mean_diff = mean_after - mean_before
            pooled_std = sigma_0  # Simplified: use baseline std
            
            # t-test statistic
            t_stat = mean_diff / (pooled_std * np.sqrt(1/n_before + 1/n_after))
            
            # Convert to log odds (approximate)
            log_odds = t_stat**2 / 2 - np.log(2 * np.pi) / 2
            
            # Add prior (Poisson process assumption)
            log_odds += np.log(hazard_rate)
            
            log_posterior_odds[t] = log_odds
            
            if log_odds > max_log_odds:
                max_log_odds = log_odds
                changepoint_idx = t
        
        # Convert to confidence (probability)
        if changepoint_idx is not None and max_log_odds > 0:
            confidence = 1.0 / (1.0 + np.exp(-max_log_odds))  # Sigmoid
            confidence = min(confidence, 0.99)  # Cap at 99%
            return float(changepoint_idx), confidence
        else:
            return None, 0.0
    
    def attribute_excursion(self, process_data: Dict[str, npt.NDArray],
                           crack_state: float) -> Dict[str, float]:
        """
        Attribute process excursion to crack vs other sources.
        
        Args:
            process_data: Dictionary with process metrics time series
            crack_state: Current crack density state
            
        Returns:
            Attribution probability dictionary
        """
        overlay_series = process_data.get("overlay", np.array([]))
        if len(overlay_series) == 0:
            return {"crack_attribution": 0.0, "confidence": 0.0}
            
        # Simple Bayesian attribution using threshold model
        # P(crack caused excursion) = P(excursion | crack) × P(crack) / P(excursion)
        
        # Define excursion threshold (e.g., >2σ above baseline)
        baseline_mean = np.mean(overlay_series[:len(overlay_series)//2])
        baseline_std = np.std(overlay_series[:len(overlay_series)//2])
        excursion_threshold = baseline_mean + 2.0 * baseline_std
        
        # Current overlay level
        current_overlay = overlay_series[-1] if len(overlay_series) > 0 else baseline_mean
        
        # Is there an excursion?
        is_excursion = current_overlay > excursion_threshold
        
        if not is_excursion:
            return {"crack_attribution": 0.0, "confidence": 0.9}
            
        # Model P(excursion | crack_state)
        # Higher crack density → higher probability of excursion
        reference_density = 1e8  # m⁻²
        normalized_crack = crack_state / reference_density
        p_excursion_given_crack = min(0.95, normalized_crack**0.3)  # Saturating function
        
        # Prior: P(crack) - depends on substrate age/history
        # For simplicity, use crack density as proxy
        p_crack_prior = min(0.5, normalized_crack**0.5)
        
        # P(excursion) - base rate from historical data (assume 10% excursion rate)
        p_excursion = 0.1
        
        # Bayes rule
        if p_excursion > 0:
            p_crack_given_excursion = (p_excursion_given_crack * p_crack_prior) / p_excursion
            p_crack_given_excursion = min(p_crack_given_excursion, 0.99)
        else:
            p_crack_given_excursion = 0.0
            
        return {
            "crack_attribution": p_crack_given_excursion,
            "confidence": 0.7  # Moderate confidence in simplified model
        }


class ReplacementOptimizer:
    """
    Optimize substrate replacement timing based on economic cost model.
    
    Balances continued use costs (yield loss) against replacement costs (downtime, new substrate).
    """
    
    def __init__(self):
        """Initialize with cost parameters from config."""
        self.cost_new_substrate = ATTRIBUTION["cost_new_substrate"]
        self.cost_inspection = ATTRIBUTION["cost_inspection"]
        self.cost_yield_loss_per_nm = ATTRIBUTION["cost_yield_loss_per_nm"]
        self.cost_downtime_per_hour = ATTRIBUTION["cost_downtime_per_hour"]
        
    def compute_cost_continued_use(self, crack_state: float, n_lots: int,
                                  yield_model: Optional[Callable] = None) -> float:
        """
        Compute cost of continued use with degraded substrate.
        
        Args:
            crack_state: Crack density or damage parameter
            n_lots: Number of lots to process
            yield_model: Function mapping overlay error to yield loss (optional)
            
        Returns:
            Total cost of continued use [USD]
        """
        if crack_state < 0 or n_lots < 0:
            raise ValueError("Crack state and n_lots must be non-negative")
            
        # Compute overlay degradation
        overlay_model = OverlayDegradationModel()
        pristine_overlay = overlay_model.compute_pristine_overlay()
        degraded_overlay = overlay_model.compute_degraded_overlay(crack_state, pristine_overlay)
        
        # Overlay degradation above pristine level
        overlay_degradation = max(0.0, degraded_overlay - pristine_overlay)
        
        # Default yield model: linear relationship
        if yield_model is None:
            def default_yield_model(overlay_error: float) -> float:
                # Assume 1% yield loss per nm of overlay error above budget
                return min(0.5, overlay_error * 0.01)  # Cap at 50% loss
            yield_model = default_yield_model
            
        yield_loss_per_lot = yield_model(overlay_degradation)
        
        # Cost calculation
        # Assume: cost = yield_loss × cost_per_nm_per_lot × n_lots
        cost_per_lot = yield_loss_per_lot * self.cost_yield_loss_per_nm
        total_cost = cost_per_lot * n_lots
        
        return total_cost
    
    def compute_cost_replacement(self, downtime_hours: float,
                               new_substrate_cost: Optional[float] = None,
                               inspection_cost: Optional[float] = None) -> float:
        """
        Compute cost of substrate replacement.
        
        Args:
            downtime_hours: Tool downtime for replacement [hours]
            new_substrate_cost: Cost of new substrate [USD] (uses config default if None)
            inspection_cost: Inspection cost [USD] (uses config default if None)
            
        Returns:
            Total replacement cost [USD]
        """
        if downtime_hours < 0:
            raise ValueError("Downtime hours must be non-negative")
            
        substrate_cost = new_substrate_cost if new_substrate_cost is not None else self.cost_new_substrate
        inspect_cost = inspection_cost if inspection_cost is not None else self.cost_inspection
        downtime_cost = downtime_hours * self.cost_downtime_per_hour
        
        total_cost = substrate_cost + inspect_cost + downtime_cost
        return total_cost
    
    def optimal_replacement_time(self, crack_density_trajectory: npt.NDArray[np.float64],
                               cost_params: Optional[Dict] = None) -> Tuple[Optional[float], float]:
        """
        Find optimal replacement time minimizing total cost.
        
        Args:
            crack_density_trajectory: Crack density vs time/lots
            cost_params: Cost model parameters (downtime_hours, lots_per_day, etc.)
            
        Returns:
            Tuple of (optimal_time, minimum_total_cost)
        """
        if len(crack_density_trajectory) == 0:
            raise ValueError("Crack trajectory cannot be empty")
            
        # Default cost parameters
        if cost_params is None:
            cost_params = {
                "downtime_hours": 8.0,      # 8 hours for replacement
                "lots_per_day": 24.0,       # Typical lot rate
                "planning_horizon": 365.0   # days
            }
            
        n_time_points = len(crack_density_trajectory)
        time_points = np.arange(n_time_points, dtype=float)  # In lots or time units
        
        total_costs = np.zeros(n_time_points)
        
        for t_replace in range(n_time_points):
            # Cost of continued use until replacement
            remaining_lots = n_time_points - t_replace
            crack_at_replacement = crack_density_trajectory[t_replace]
            cost_continued = self.compute_cost_continued_use(crack_at_replacement, remaining_lots)
            
            # Cost of replacement
            cost_replacement = self.compute_cost_replacement(cost_params["downtime_hours"])
            
            total_costs[t_replace] = cost_continued + cost_replacement
        
        # Find minimum cost time
        optimal_idx = np.argmin(total_costs)
        optimal_time = time_points[optimal_idx]
        minimum_cost = total_costs[optimal_idx]
        
        return optimal_time, minimum_cost
    
    def sensitivity_analysis(self, params_ranges: Dict[str, Tuple[float, float]],
                           base_crack_trajectory: npt.NDArray[np.float64],
                           n_samples: int = 20) -> Dict[str, npt.NDArray]:
        """
        Analyze sensitivity of optimal replacement time to key parameters.
        
        Args:
            params_ranges: Parameter ranges to explore {"param_name": (min, max)}
            base_crack_trajectory: Baseline crack trajectory
            n_samples: Number of sample points per parameter
            
        Returns:
            Dictionary with sensitivity results
        """
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for sensitivity analysis")
            
        sensitivity_results = {}
        
        for param_name, (param_min, param_max) in params_ranges.items():
            param_values = np.linspace(param_min, param_max, n_samples)
            optimal_times = np.zeros(n_samples)
            optimal_costs = np.zeros(n_samples)
            
            for i, param_value in enumerate(param_values):
                # Modify cost parameters based on current parameter
                cost_params = {"downtime_hours": 8.0, "lots_per_day": 24.0}
                
                if param_name == "substrate_cost":
                    self.cost_new_substrate = param_value
                elif param_name == "downtime_rate":
                    self.cost_downtime_per_hour = param_value
                elif param_name == "yield_loss_rate":
                    self.cost_yield_loss_per_nm = param_value
                elif param_name == "downtime_hours":
                    cost_params["downtime_hours"] = param_value
                
                # Compute optimal replacement time
                opt_time, opt_cost = self.optimal_replacement_time(base_crack_trajectory, cost_params)
                optimal_times[i] = opt_time if opt_time is not None else len(base_crack_trajectory)
                optimal_costs[i] = opt_cost
            
            sensitivity_results[param_name] = {
                "param_values": param_values,
                "optimal_times": optimal_times,
                "optimal_costs": optimal_costs,
                "sensitivity": np.std(optimal_times) / np.mean(optimal_times) if np.mean(optimal_times) > 0 else 0.0
            }
        
        return sensitivity_results


class ProcessSimulator:
    """
    Simulate various lithography processes with crack-induced effects.
    
    Models different EUV technologies and their sensitivity to substrate degradation.
    """
    
    def __init__(self):
        """Initialize process simulator."""
        self.process_types = {
            "low_na_euv": {"NA": 0.33, "crack_sensitivity": 1.0},
            "high_na_euv": {"NA": 0.55, "crack_sensitivity": 1.8},  # More sensitive
            "duv_arf": {"NA": 1.35, "crack_sensitivity": 0.3}       # Less sensitive
        }
    
    def simulate_lithography_process(self, substrate_state: Dict[str, float],
                                   process_type: str, n_wafers: int) -> ProcessMetrics:
        """
        Simulate lithography process metrics with substrate effects.
        
        Args:
            substrate_state: Dictionary with crack density, temperature, etc.
            process_type: One of "low_na_euv", "high_na_euv", "duv_arf"
            n_wafers: Number of wafers to simulate
            
        Returns:
            ProcessMetrics with overlay, CDU, EPE, yield loss
        """
        if process_type not in self.process_types:
            raise ValueError(f"Unknown process type: {process_type}")
        if n_wafers <= 0:
            raise ValueError("Number of wafers must be positive")
            
        process_params = self.process_types[process_type]
        crack_density = substrate_state.get("crack_density", 0.0)
        
        # Baseline process performance (pristine substrate)
        if process_type == "low_na_euv":
            baseline_overlay = 1.2  # nm RMS
            baseline_cdu = 2.0     # nm 3σ
            baseline_epe = 3.0     # nm 3σ
        elif process_type == "high_na_euv":
            baseline_overlay = 0.8  # nm RMS (tighter requirements)
            baseline_cdu = 1.5     # nm 3σ
            baseline_epe = 2.2     # nm 3σ
        else:  # duv_arf
            baseline_overlay = 2.5  # nm RMS (relaxed)
            baseline_cdu = 3.5     # nm 3σ
            baseline_epe = 5.0     # nm 3σ
        
        # Apply crack-induced degradation
        overlay_model = OverlayDegradationModel()
        pristine_overlay = overlay_model.compute_pristine_overlay()
        degraded_overlay = overlay_model.compute_degraded_overlay(crack_density, pristine_overlay)
        
        # Scale by process sensitivity
        crack_sensitivity = process_params["crack_sensitivity"]
        overlay_degradation = (degraded_overlay - pristine_overlay) * crack_sensitivity
        
        # Final process metrics
        overlay_total = baseline_overlay + overlay_degradation
        overlay_x = overlay_total / np.sqrt(2)  # Assume equal X, Y components
        overlay_y = overlay_x
        
        # CDU and EPE also affected by substrate stability
        cdu_degradation = overlay_degradation * 0.5  # CDU less sensitive than overlay
        epe_degradation = overlay_degradation * 1.2  # EPE more sensitive
        
        cdu_total = baseline_cdu + cdu_degradation
        epe_total = baseline_epe + epe_degradation
        
        # Yield loss model (simplified)
        # Assume exponential relationship: yield_loss = 1 - exp(-overlay_degradation/characteristic_length)
        characteristic_overlay = 2.0  # nm - overlay error that causes ~63% yield impact
        yield_loss = 1.0 - np.exp(-overlay_degradation / characteristic_overlay)
        yield_loss = max(0.0, min(yield_loss, 0.8))  # Clamp to reasonable range
        
        return ProcessMetrics(
            overlay_x=overlay_x,
            overlay_y=overlay_y,
            overlay_total=overlay_total,
            cdu=cdu_total,
            epe=epe_total,
            yield_loss=yield_loss,
            process_type=process_type
        )
    
    def simulate_lot_sequence(self, n_lots: int, 
                             crack_evolution_func: Callable[[int], float]) -> List[ProcessMetrics]:
        """
        Simulate lot-by-lot processing with evolving crack state.
        
        Args:
            n_lots: Number of lots to simulate
            crack_evolution_func: Function mapping lot number to crack density
            
        Returns:
            List of ProcessMetrics for each lot
        """
        if n_lots <= 0:
            raise ValueError("Number of lots must be positive")
            
        lot_metrics = []
        
        for lot_num in range(n_lots):
            # Get crack density for this lot
            crack_density = crack_evolution_func(lot_num)
            
            # Simulate process (default to low_na_euv)
            substrate_state = {"crack_density": crack_density}
            metrics = self.simulate_lithography_process(substrate_state, "low_na_euv", 25)  # 25 wafers/lot
            
            lot_metrics.append(metrics)
        
        return lot_metrics


def run_attribution_analysis(crack_density_history: npt.NDArray[np.float64],
                           overlay_measurements: npt.NDArray[np.float64],
                           process_type: str = "low_na_euv",
                           cost_params: Optional[Dict] = None) -> AttributionResult:
    """
    Main function to run complete process attribution analysis.
    
    Args:
        crack_density_history: Time series of crack density evolution
        overlay_measurements: Measured overlay error time series [nm RMS]
        process_type: Lithography process type
        cost_params: Cost model parameters
        
    Returns:
        Complete AttributionResult with all analysis components
    """
    if len(crack_density_history) == 0 or len(overlay_measurements) == 0:
        raise ValueError("Input time series cannot be empty")
    
    # Initialize analysis components
    overlay_model = OverlayDegradationModel()
    decomposer = VarianceDecomposition()
    optimizer = ReplacementOptimizer()
    
    # 1. Overlay degradation analysis
    current_crack_density = crack_density_history[-1] if len(crack_density_history) > 0 else 0.0
    pristine_overlay = overlay_model.compute_pristine_overlay()
    degraded_overlay = overlay_model.compute_degraded_overlay(current_crack_density, pristine_overlay)
    
    # 2. Variance decomposition
    variance_series = overlay_measurements**2
    known_components = {
        "scanner": ATTRIBUTION["overlay_scanner"],
        "process": ATTRIBUTION["overlay_process"]
    }
    decomposition = decomposer.decompose(variance_series, known_components)
    
    # 3. Bayesian changepoint detection
    changepoint_time, changepoint_confidence = decomposer.bayesian_changepoint(overlay_measurements)
    
    # 4. Replacement optimization
    if cost_params is None:
        cost_params = {"downtime_hours": 8.0}
    
    optimal_time, optimal_cost = optimizer.optimal_replacement_time(crack_density_history, cost_params)
    
    # Cost of no replacement (continue to end of planning horizon)
    n_remaining_lots = max(1, len(crack_density_history) - (optimal_time or 0))
    cost_no_replacement = optimizer.compute_cost_continued_use(current_crack_density, int(n_remaining_lots))
    
    cost_savings = max(0.0, cost_no_replacement - optimal_cost)
    
    # 5. Compile results
    result = AttributionResult(
        total_overlay=degraded_overlay,
        scanner_contribution=decomposition["scanner_contribution"],
        mask_pristine_contribution=decomposition["mask_pristine_contribution"], 
        mask_degradation_contribution=decomposition["mask_degradation_contribution"],
        process_contribution=decomposition["process_contribution"],
        changepoint_detected=changepoint_time is not None,
        changepoint_time=changepoint_time,
        changepoint_confidence=changepoint_confidence,
        optimal_replacement_time=optimal_time,
        total_cost_optimal=optimal_cost,
        total_cost_no_replacement=cost_no_replacement,
        cost_savings=cost_savings
    )
    
    return result