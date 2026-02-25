# Corning Technical Concept Proposal (English, A4 1-2pp, Arial 10pt, 1.15 spacing)

---

**Predictive Micro-Crack Lifecycle Management Framework for Sub-5nm EUV Lithography Glass Substrates**

S. Joon Kwon, Ph.D.
Associate Professor & Associate Dean, College of Engineering
Sungkyunkwan University (SKKU), Smart Process & Materials Design Lab (SPMDL)

---

**1. Problem Statement**

As EUV lithography transitions to High-NA (0.55 NA) configurations for sub-5nm patterning, photomask substrates face substantially elevated thermo-mechanical loading. Absorbed EUV dose generates transient temperature fields ΔT(x,y,t) on the order of 0.3–0.5 K with localized gradients, followed by rapid thermal cycling at production cadence. In ultra-low expansion (ULE) glass substrates, this repeated thermo-mechanical stress interacts with pre-existing microstructural heterogeneities—compositional fluctuations, dissolved gas inclusions, striae, surface/subsurface damage from polishing—that serve as potential nucleation sites for micro-cracks.

Once nucleated, micro-crack evolution in glass follows well-established subcritical crack growth kinetics (Charles-Hillig model: v = v₀·(K_I/K_IC)ⁿ, where n ~ 15–30 for silicate glasses), meaning that even stress intensities well below the critical fracture toughness K_IC can drive slow, persistent crack extension under sustained or cyclic loading. The spatial correlation among nucleation sites determines whether isolated micro-cracks remain benign or coalesce into networks that approach percolation thresholds—at which point degradation becomes systemic and potentially catastrophic.

The consequences for semiconductor manufacturing are multi-dimensional: (i) progressive degradation of CTE uniformity and optical homogeneity, directly impacting overlay and edge placement error (EPE) budgets; (ii) mechanical integrity reduction affecting reticle handling, cleaning, and pellicle adhesion reliability; (iii) unpredictable substrate lifetime, complicating inventory management and cost-of-ownership models; and (iv) increased process uncertainty propagating into yield loss whose root cause is difficult to attribute. Despite the criticality of this failure mode, no systematic, physics-based framework currently exists to predict micro-crack nucleation probability, simulate propagation dynamics, or quantify the downstream impact on fab-level process metrics.

**2. Proposed Research Framework**

We propose a five-module computational framework that spans the complete micro-crack lifecycle—from nucleation through propagation to process-level impact assessment—supported by physics-based simulation, forward inspection modeling, and machine-learning-based diagnostics.

**Module 1: Stochastic Crack Nucleation Engine.** Computes the probability of micro-crack nucleation under EUV-induced thermo-mechanical stress, given a substrate's defect/impurity landscape. The module couples a thermoelastic stress solver (building on the Helmholtz-equation framework established in our prior technical assessment) with a Griffith energy-balance criterion evaluated at stochastically distributed defect sites. Defect populations are modeled as spatial point processes (Poisson or clustered Neyman-Scott) with configurable density and correlation length. Monte Carlo sampling over defect realizations yields nucleation probability maps P(x,y) and time-to-first-crack distributions as functions of dose, thermal cycling parameters, and substrate grade specifications.

**Module 2: Crack Propagation & Morphology Evolution.** Simulates crack path, branching, and arrest using a variational phase-field fracture model coupled with subcritical crack growth kinetics. The phase-field approach naturally handles complex crack topologies (branching, merging, arrest at interfaces) without explicit crack tracking, making it well-suited for multi-defect substrates. The model incorporates thin-film mechanical constraints (Mo/Si multilayer interface stress mismatch), depth-dependent CTE variation, and environment-sensitive growth rates (humidity, temperature). A graph-theoretic percolation analysis quantifies the connectivity of the evolving crack network, identifying the transition from isolated defects to system-level degradation.

**Module 3: Inspection Signal Forward Model.** Generates synthetic inspection signatures for cracked versus pristine substrates across multiple modalities: acoustic (Lamb wave dispersion and scattering), optical (laser scattering, confocal Raman stress mapping, 193-nm interferometric phase perturbation), and electron-beam methods (EELS bonding-environment shifts, Kelvin probe force microscopy surface potential mapping) as needed. For each modality, the module computes signal-to-noise ratios as a function of crack size, orientation, and depth, producing receiver operating characteristic (ROC) curves and minimum detectable crack size estimates. This enables rational selection and optimization of inspection strategies before committing to expensive experimental campaigns.

**Module 4: Physics-Informed Inverse Diagnostics.** Given experimental inspection data, this module estimates crack spatial distribution, number density, and probable nucleation origin using physics-informed machine learning. Features are engineered from the governing physics (Griffith criterion parameters, stress intensity factors, thermal history metrics) rather than purely data-driven, ensuring physical consistency and interpretability. The architecture combines Bayesian inference with mechanistic priors derived from Modules 1–2, enabling reliable estimation even with limited initial experimental data. As Corning's measurement database grows, the model refines through transfer learning.

**Module 5: Process Impact Attribution.** Quantifies the contribution of substrate-level micro-crack degradation to fab-observable process metrics (overlay, CDU, EPE). The module decomposes total process variance as σ²_total = σ²_scanner + σ²_mask,pristine + σ²_mask,degradation + σ²_process, isolating the crack-driven degradation term through Bayesian change-point detection on time-series process data. This enables: (i) evidence-based substrate replacement timing that minimizes total cost (inspection + yield loss + downtime); (ii) clear attribution of process excursions to substrate degradation versus other root causes; and (iii) quantitative input to substrate lifetime specifications and warranty frameworks.

**3. Deliverables and Strategic Value**

The primary deliverable is an integrated simulation platform with validated physics engines, deployable as an interactive tool for Corning's R&D and quality engineering teams. Each module produces quantitative, auditable outputs traceable to governing equations and material parameters.

Strategic value to Corning: (a) a defensible, first-mover framework for substrate lifetime prediction—no competitor currently offers physics-based micro-crack lifecycle management; (b) direct input to customer qualification dossiers, strengthening Extreme-ULE positioning for High-NA supply contracts with Samsung and TSMC; (c) a rational basis for inspection protocol design and substrate grade differentiation; (d) a quantitative tool for warranty/liability discussions with IDM customers, translating substrate-level physics into fab-level cost impact.

**4. Team and Capabilities**

Prof. Kwon's laboratory combines expertise in EUV lithography physics, semiconductor process modeling, and computational materials science. Relevant prior work includes a comprehensive technical assessment of Corning's ULE substrate strategy for EUV/High-NA ecosystems (delivered 2025–2026), development of stochastic EUV photoresist simulators integrating aging kinetics with patterning physics, and digital twin platforms for energy systems employing Monte Carlo uncertainty quantification. The proposed framework leverages these established capabilities and modeling infrastructure directly.

---

*Contact: S. Joon Kwon, Ph.D. | sjoonkwon@skku.edu | sjoonkwon.com*
