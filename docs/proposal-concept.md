# Job 10: Glass Micro-Crack Lifecycle Simulator
## Project Concept & Proposal Draft

### For: Corning Incorporated (Industry-Academia Collaboration with SKKU SPMDL)

---

## Simulator Architecture (5 Modules)

### M1: Crack Nucleation Probability Engine
**Physics**: Given a glass substrate with known defect/impurity distribution, compute nucleation probability under EUV thermo-mechanical stress cycling.

- **Stress field**: Thermoelastic PDE (from Corning report Helmholtz eq.) + transient thermal profile from EUV dose absorption
- **Nucleation criterion**: Modified Griffith energy balance at defect sites
  - G_I(σ, a_0) ≥ G_IC → nucleation
  - Where a_0 = initial flaw size, σ = local stress (from thermal + residual)
- **Stochastic layer**: 
  - Defect spatial distribution: Poisson field or clustered (Neyman-Scott process)
  - Impurity concentration fluctuations: Gaussian random field with correlation length
  - Monte Carlo sampling: N_runs × defect realizations → P(nucleation | dose, ΔT, defect_density)
- **Subcritical threshold**: K_0/K_IC ratio for ULE glass (stress corrosion)
- **Output**: Nucleation probability map P(x,y), expected nucleation density, time-to-first-crack distribution

### M2: Crack Propagation & Morphology Evolution
**Physics**: Once nucleated, simulate crack growth path, branching, and arrest in thin glass substrate.

- **Deterministic core**: Phase-field fracture model (Bourdin-Francfort-Marigo variational approach)
  - Energy functional: E = ∫(ψ_elastic + G_c·γ(d,∇d))dΩ
  - d = phase-field damage variable (0=intact, 1=fully cracked)
  - Coupled with thermoelastic stress from M1
- **Subcritical crack growth**: Charles-Hillig model for glass
  - v = v_0·exp(-ΔH/RT)·(K_I/K_IC)^n (n~15-30 for silicate glasses)
  - Environment-dependent: humidity, temperature
- **Thin-film constraints**: 
  - Interface with Mo/Si multilayer → stress mismatch, delamination risk
  - Depth-dependent CTE (axial variation from Corning report)
- **Thermal cycling fatigue**: Cumulative damage under repeated EUV exposure
  - Paris-law analog: da/dN = C·(ΔK)^m
- **Percolation analysis**: Graph-theoretic connectivity of crack network
  - When does isolated micro-crack population → connected macro-crack?
- **Output**: Crack morphology evolution (2D/3D), crack density vs time/cycles, percolation threshold

### M3: Inspection Signal Forward Model
**Physics**: Simulate what NDT methods detect for cracked vs pristine substrates.

- **Acoustic methods**:
  - Elastic wave propagation (Lamb waves in thin plate): dispersion relation shifts with crack presence
  - Acoustic emission: stress wave from crack growth events → AE signal characteristics
  - Forward model: crack geometry → scattered wave field → detector signal
- **Optical/Laser methods**:
  - Laser scattering: Mie/Rayleigh scattering from sub-surface voids/cracks
  - Confocal Raman: stress-induced peak shift (Δω ∝ σ) mapping
  - 193nm interferometry: phase perturbation from crack-induced refractive index change
- **Electron-beam methods** (if needed):
  - EELS: local bonding environment change near crack tip (Si-O bond distortion)
  - KFM (Kelvin Force Microscopy): surface potential variation from charge trapping at crack sites
- **Reference comparison**: ΔSignal = Signal(cracked) - Signal(pristine)
  - SNR analysis: minimum detectable crack size per method
  - ROC curves: detection probability vs false alarm rate
- **Output**: Simulated inspection images/spectra, detectability maps, optimal inspection strategy

### M4: Inverse ML for Crack Diagnosis
**Physics-informed ML**: From experimental inspection data → crack distribution, density, root cause.

- **Feature engineering** (physics-based):
  - From acoustic: dispersion curve anomalies, AE event rate/energy
  - From optical: scattering intensity spatial statistics, Raman shift maps
  - From thermal history: cumulative dose, ΔT cycles, time-at-temperature
- **Architecture options**:
  - Physics-Informed Neural Network (PINN): embed Griffith + Charles-Hillig as loss constraints
  - Bayesian inference: P(crack_state | observations) with physics prior
  - Gaussian Process with mechanistic kernel
- **Training data**: 
  - Synthetic: M1+M2+M3 forward simulations (large ensemble)
  - Literature: published crack data in silicate/ULE glasses
  - Experimental: Corning/SKKU measurements (when available)
- **Output**: Crack density estimate, spatial distribution, probable nucleation cause, confidence interval

### M5: Process Impact Attribution Engine
**Physics**: Quantify how cracked substrate degrades semiconductor manufacturing metrics.

- **Overlay/EPE degradation model**:
  - σ²_mask = σ²_mask,pristine + σ²_mask,degradation(crack_state)
  - Crack → local CTE anomaly → thermal deformation change → overlay residual
  - Crack → local Δn anomaly → phase/registration error
- **Yield impact**: 
  - Crack-induced EPE tail → defect probability increase
  - Monte Carlo: sample crack states × process conditions → yield distribution
- **Attribution algorithm**:
  - Given: time series of overlay/CDU/EPE from fab
  - Decompose: scanner + mask(pristine) + mask(degradation) + process + noise
  - Bayesian change-point detection: when does mask(degradation) term become significant?
- **Replacement decision model**:
  - Cost(continued use with degraded substrate) vs Cost(replacement)
  - Optimal replacement time minimizing total cost (inspection + yield loss + downtime)
- **Output**: Attribution report, replacement recommendation, cost-benefit analysis

---

## Technology Stack
- **Language**: Python
- **Core libraries**: NumPy, SciPy, FEniCS/FEniCSx (phase-field FEM), scikit-learn, PyTorch
- **Visualization**: Streamlit + Plotly (consistent with DT5, EUV v4, Tandem PV)
- **Deployment**: Streamlit Cloud + GitHub

## Development Phases
- **Phase 1** (8 weeks): M1 + M2 core engines + basic Streamlit UI
- **Phase 2** (6 weeks): M3 forward models + M4 ML framework
- **Phase 3** (4 weeks): M5 attribution + integration + validation
- **Phase 4** (ongoing): Corning experimental data integration + refinement
