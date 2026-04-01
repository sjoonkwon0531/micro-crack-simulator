# EXACT CODE FIXES — Glass Crack Simulator v2

## CRITICAL FIXES (Must Implement)

### C1: Remove Incorrect Thermal Diffusion Calculation

**File:** `app_v2.py`  
**Lines:** 315-318

**OLD:**
```python
# C2+C6: Fix thermal diffusion timescale for femtosecond regime
alpha_thermal = glass_mat.get("alpha_thermal", 7e-7)  # m^2/s
t_diffusion = pulse_duration * 3  # ~1.5 ps timescale for femtosecond heating
l_diff = np.sqrt(alpha_thermal * t_diffusion)
```

**NEW:**
```python
# Thermal diffusion is negligible during femtosecond pulse
# Relevant timescale is ~100 ps to 1 ns AFTER pulse
# (Not needed for simplified peak temperature estimate)
```

**Justification:** For 350 fs pulses, thermal diffusion during pulse is negligible. The `t_diffusion` variable is not physically meaningful here.

---

### C2: Fix Peak Temperature Calculation

**File:** `app_v2.py`  
**Lines:** 320-329

**OLD:**
```python
# Effective beam radius at focus (wavelength-limited + diffraction)
lambda_laser = 1030e-9  # m
NA = 0.5  # numerical aperture
w0 = lambda_laser / (2 * NA)  # beam waist

# Temperature rise (simplified Gaussian model)
rho = glass_mat.get("rho", 2210)  # kg/m^3
cp = glass_mat.get("cp_specific", 767)  # J/(kg·K)

# Peak temperature rise at center
volume_heated = np.pi * w0**2 * z_focus
mass_heated = rho * volume_heated
delta_T_peak = E_pulse / (mass_heated * cp)
```

**NEW:**
```python
# Effective beam radius at focus (wavelength-limited + diffraction)
lambda_laser = 1030e-9  # m
NA = 0.5  # numerical aperture
w0 = lambda_laser / (2 * NA)  # beam waist

# Material properties
rho = glass_mat.get("rho", 2210)  # kg/m^3
cp = glass_mat.get("cp_specific", 767)  # J/(kg·K)

# CORRECTED: Account for absorption depth
# Beer-Lambert absorption: I(z) = I0 * exp(-α*z)
# For glass at 1030 nm: α ~ 10^4 m^-1 (absorption coefficient)
alpha_abs = 1e4  # m^-1 (typical for borosilicate at 1030 nm)
l_abs = 1 / alpha_abs  # absorption depth ~ 100 μm

# Heated volume limited by absorption depth, not full focus depth
volume_heated = np.pi * w0**2 * min(l_abs, z_focus)
mass_heated = rho * volume_heated

# Two-temperature model correction:
# In femtosecond regime, electrons heat first, then transfer to phonons
# Only ~30% of energy heats lattice on relevant timescale (~10 ps)
eta_coupling = 0.3  # electron-phonon coupling efficiency

# Peak temperature rise at center
delta_T_peak = (E_pulse * eta_coupling) / (mass_heated * cp)
```

**Expected result:** For 50 μJ pulse, ~2000-3000 K (realistic) instead of 81,000 K (unphysical)

---

### C3: Add ROI Disclaimer

**File:** `app_v2.py`  
**After line:** 1715 (start of Tab 9: Executive Dashboard)

**ADD:**
```python
# C3: Add warning and change to estimates
st.warning("⚠️ Illustrative estimates based on industry benchmarks. Actual values depend on Corning process data and will be calibrated in Phase 1.")
```

**AND change lines 1717-1742:**

**OLD:**
```python
st.markdown(create_metric_card(
    "Crack Density Reduction",
    f"{crack_density_reduction}%",
    "vs baseline process",
    "success"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Annual Cost Savings",
    f"${cost_savings:.1f}M",
    "Reduced scrap + rework",
    "success"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Throughput Improvement",
    f"{throughput_improvement:.1f}x",
    "Faster process optimization",
    "default"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Reliability Gain",
    f"+{reliability_gain}%",
    "JEDEC cycling lifetime",
    "success"
), unsafe_allow_html=True)
```

**NEW:**
```python
st.markdown(create_metric_card(
    "Crack Density Reduction (Est.)",  # Added (Est.)
    f"{crack_density_reduction}%",
    "vs baseline process",
    "success"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Annual Cost Savings (Est.)",  # Added (Est.)
    f"${cost_savings:.1f}M",
    "Reduced scrap + rework",
    "success"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Throughput Improvement (Est.)",  # Added (Est.)
    f"{throughput_improvement:.1f}x",
    "Faster process optimization",
    "default"
), unsafe_allow_html=True)

st.markdown(create_metric_card(
    "Reliability Gain (Est.)",  # Added (Est.)
    f"+{reliability_gain}%",
    "JEDEC cycling lifetime",
    "success"
), unsafe_allow_html=True)
```

---

### C5: Add CTE Stress Disclaimer

**File:** `app_v2.py`  
**After line:** 620 (after first stress plot in Tab 2)

**ADD:**
```python
# C5: Add disclaimer
st.caption("⚠️ Simplified biaxial stress model. Full Suhir/Timoshenko multilayer analysis accounts for neutral axis shift and bending moments (typically 2-3× higher stress).")
```

**AND after line:** 648 (after second stress plot)

**ADD:**
```python
st.caption("⚠️ Simplified biaxial stress model. Full Suhir/Timoshenko multilayer analysis accounts for neutral axis shift and bending moments (typically 2-3× higher stress).")
```

---

### C6: Fix Paris Law Dynamic ΔK

**File:** `app_v2.py`  
**Lines:** 358-376

**OLD:**
```python
def paris_law_crack_growth(a0: float, delta_K: float, N_cycles: int, C: float = 1e-11, m: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paris law fatigue crack growth: da/dN = C * (ΔK)^m
    
    Note: Fatigue model (Paris Law) for thermal cycling; subcritical growth (SCG) applies to static loading + moisture
    
    Returns: (cycles, crack_length)
    """
    cycles = np.arange(0, N_cycles + 1, max(1, N_cycles // 1000))
    a = np.zeros_like(cycles, dtype=float)
    a[0] = a0
    
    for i in range(1, len(cycles)):
        dN = cycles[i] - cycles[i-1]
        da_dN = C * (delta_K ** m)
        a[i] = a[i-1] + da_dN * dN
        
        # Stop if crack becomes too large
        if a[i] > 1e-3:  # 1 mm
            a[i:] = a[i]
            break
    
    return cycles, a
```

**NEW:**
```python
def paris_law_crack_growth(a0: float, sigma: float, N_cycles: int, Y: float = 1.12, C: float = 1e-11, m: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paris law fatigue crack growth: da/dN = C * (ΔK)^m
    
    ΔK is recalculated at each step based on current crack length:
    ΔK(a) = σ * sqrt(π*a) * Y
    
    Note: Fatigue model (Paris Law) for thermal cycling; subcritical growth (SCG) applies to static loading + moisture
    
    Args:
        a0: Initial crack length [m]
        sigma: Applied stress [Pa]
        N_cycles: Number of cycles
        Y: Geometry factor (1.12 for edge crack)
        C: Paris law coefficient [m/cycle/(Pa·m^0.5)^m]
        m: Paris law exponent
    
    Returns: (cycles, crack_length)
    """
    cycles = np.arange(0, N_cycles + 1, max(1, N_cycles // 1000))
    a = np.zeros_like(cycles, dtype=float)
    a[0] = a0
    
    for i in range(1, len(cycles)):
        dN = cycles[i] - cycles[i-1]
        
        # Recalculate ΔK based on current crack length
        delta_K = sigma * np.sqrt(np.pi * a[i-1]) * Y
        
        # Paris law
        da_dN = C * (delta_K ** m)
        a[i] = a[i-1] + da_dN * dN
        
        # Stop if crack becomes too large
        if a[i] > 1e-3:  # 1 mm
            a[i:] = a[i]
            break
    
    return cycles, a
```

**AND update the calling code (around line 662):**

**OLD:**
```python
# I2: Stress intensity factor range with correct geometry factor
sigma_max_MPa = max(sigma_glass_cu, sigma_glass_mold) * 1e3  # GPa → MPa
Y_factor = 1.12  # Y = 1.12 for edge crack (Tada, Paris & Irwin)
delta_K_MPa = sigma_max_MPa * np.sqrt(np.pi * a0) * Y_factor  # MPa·m^0.5

# Paris law parameters (typical for glass, ΔK in MPa·m^0.5)
C_paris = 1e-11  # m/cycle/(MPa·m^0.5)^m — typical for brittle glass
m_paris = 3.0

cycles, crack_length = paris_law_crack_growth(a0, delta_K_MPa, n_cycles_thermal, C_paris, m_paris)
```

**NEW:**
```python
# Stress intensity factor calculation (now done inside Paris law function)
sigma_max_Pa = max(sigma_glass_cu, sigma_glass_mold) * 1e9  # GPa → Pa
Y_factor = 1.12  # Y = 1.12 for edge crack (Tada, Paris & Irwin)

# Paris law parameters (typical for glass, ΔK in Pa·m^0.5 now)
C_paris = 1e-11  # m/cycle/(MPa·m^0.5)^m
# Convert to SI: C in [m/cycle/(Pa·m^0.5)^m]
C_paris_SI = C_paris * (1e-6)**m_paris  # Adjust for MPa→Pa conversion
m_paris = 3.0

# Call updated function with stress (not ΔK)
cycles, crack_length = paris_law_crack_growth(a0, sigma_max_Pa, n_cycles_thermal, Y_factor, C_paris_SI, m_paris)
```

---

## MODERATE FIXES (Should Implement)

### M1: Update Paris Law Exponent for Glass

**File:** `app_v2.py`  
**Line:** ~665 (where `m_paris = 3.0`)

**OLD:**
```python
m_paris = 3.0
```

**NEW:**
```python
m_paris = 15.0  # Typical for glass (literature: 12-20 for borosilicate in air)
# Note: m=3 is typical for METALS. Glass has much steeper ΔK dependence.
```

**Justification:** Literature values for glass fatigue:
- Soda-lime glass: m = 10-15
- Fused silica: m = 15-25
- Borosilicate: m = 12-20

---

### M3: Add Division by Zero Guards (3 locations)

**File:** `app_v2.py`

**Location 1: Line ~2254 (Tab 10, Scenario B comparison)**

**OLD:**
```python
# M3: Division by zero guard
if crack_prob_a > 1e-6:
    delta_pct = (crack_prob_b - crack_prob_a)/crack_prob_a*100
else:
    delta_pct = 0
```

**NEW:** (already correct in code!)
```python
# M3: Division by zero guard
if crack_prob_a > 1e-6:
    delta_pct = (crack_prob_b - crack_prob_a)/crack_prob_a*100
else:
    delta_pct = 0
```

**Location 2: Line ~2264**

**ADD:**
```python
# M3: Division by zero guard
if crack_prob_a > 1e-6:
    crack_reduction = (crack_prob_a - crack_prob_b) / crack_prob_a * 100
else:
    crack_reduction = 0
```

**Location 3: Line ~2300**

**ADD:**
```python
# M3: Division by zero guard
if current_prob > 1e-6:
    improvement_pct = (current_prob - target_crack_prob) / current_prob * 100
else:
    improvement_pct = 0
```

---

### M6: Add Radar Chart Normalization Note

**File:** `app_v2.py`  
**After line:** 1156 (after radar chart in Tab 6)

**ADD:**
```python
# M6: Add normalization note
st.caption("Values normalized 0-1 for comparison. See bar charts below for absolute values.")
```

---

### M7: Add Competitive Positioning Footnote

**File:** `app_v2.py`  
**After line:** 1885 (after competitive radar chart in Tab 9)

**ADD:**
```python
# M7: Add competitive positioning footnote
st.caption("Assessment based on published specifications and industry benchmarks (2025).")
```

---

### M10: Temperature Cap — Already Present ✅

**File:** `app_v2.py`  
**Line:** 338

**Current code:**
```python
# M10: Cap temperature at physically realistic maximum
T_max_physical = 1600 + 273.15  # K (silica vaporization ~1600°C)
if np.any(T_distribution > T_max_physical):
    st.warning("⚠️ Peak temperature exceeds physical limit (1600°C) — capped at vaporization point")
    T_distribution = np.minimum(T_distribution, T_max_physical)
```

**Status:** ✅ Already correct, no change needed.

---

## MINOR FIXES (Cosmetic/Clarity)

### I1: Update ROI Claims to Conservative Values

**File:** `app_v2.py`  
**Lines:** 1364, 1374, 1378, 1412

**OLD:**
```python
"Reduce experimental cost by **10-20x**"
"**10-20x cost reduction**"
```

**NEW:**
```python
"Reduce experimental cost by **5-10x**"
"**5-10x cost reduction**"
```

**Justification:** 400 experiments → 50 experiments = 8x reduction, so 5-10x is more conservative and justifiable.

---

### I3: Add ML Prediction Placeholder

**File:** `app_v2.py`  
**After line:** 848 (in Tab 4, ML prediction section)

**ADD:**
```python
# I3: Add placeholder if prediction not yet run
if 'ml_prediction' in st.session_state:
    pred = st.session_state['ml_prediction']
    # ... existing code ...
else:
    st.info("👈 Click 'Predict Crack Probability' to run ML diagnostics")
```

**Status:** Already present in code ✅

---

### I5: Add ROI Assumptions Expander

**File:** `app_v2.py`  
**After line:** 1747 (in Tab 9, before ROI calculator inputs)

**ADD:**
```python
# I5: Add assumptions expander
with st.expander("📋 Model Assumptions"):
    st.markdown("""
    **Cost Model Assumptions:**
    - Software investment: $50K-$500K (one-time)
    - Training & integration: $20K-$200K (one-time)
    - Cost per experiment: $100-$2000 (material + labor + equipment time)
    - Annual experiments: 100-5000 (varies by R&D intensity)
    - Experiment reduction: 85% through AI optimization
    - Yield improvement: 5% absolute (conservative estimate)
    
    **Timing Assumptions:**
    - Traditional grid search: 0.5 hours/experiment
    - Payback period: Based on annual savings
    
    **Limitations:**
    - Does not account for opportunity cost
    - Assumes steady-state production
    - Actual savings depend on process maturity and data quality
    """)
```

**Status:** Already present in code ✅

---

### I8: Change "Von Mises" to "Thermal Stress"

**File:** `app_v2.py**  
**Line:** 360 (Tab 1 stress plot title)

**OLD:**
```python
title="Von Mises Stress Distribution",
```

**NEW:**
```python
title="Thermal Stress (Magnitude) Distribution",
```

**AND Line:** 587, 615 (Tab 2 stress plot titles)

**OLD:**
```python
title="Von Mises Stress (ΔT = {delta_T_cycle}°C)",
```

**NEW:**
```python
title="Biaxial Thermal Stress (ΔT = {delta_T_cycle}°C)",
```

**Justification:** The calculation is not von Mises stress (which requires full 3D stress tensor). It's thermal stress from CTE mismatch.

---

### I9: Process Attribution Disclaimer — Already Present ✅

**File:** `app_v2.py`  
**Line:** 999

**Current code:**
```python
# I9: Add disclaimer
st.info("ℹ️ Illustrative attribution model with typical industry weights. Replace with Corning FMEA data for accurate process-specific attribution.")
```

**Status:** ✅ Already correct.

---

### I10: Data Integration "Coming Soon"

**File:** `app_v2.py`  
**After line:** 1574 (in Tab 8, after data upload section)

**ADD:**
```python
# I10: Add "coming soon" message
st.info("🔜 Automated data-to-model integration available in v3.0. Contact SPMDL for custom pilot analysis.")
```

**Status:** Already present in code ✅

---

## SUMMARY

**Total fixes needed:**
- **CRITICAL:** 5 fixes (C1, C2, C3, C5, C6)
- **MODERATE:** 4 fixes (M1, M3, M6, M7)
- **MINOR:** 2 fixes (I1, I8)
- **Already correct:** 7 items (C4, M10, I3, I5, I9, I10, I2)

**Most important:**
1. **C2:** Fix peak temperature model (currently gives 81,000 K!)
2. **C6:** Fix Paris Law to update ΔK dynamically
3. **C3:** Add ROI disclaimers
4. **M1:** Change Paris Law m from 3 to 15 for glass

