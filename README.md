# Glass Micro-Crack Lifecycle Simulator (Gmcls)

**Job 10** — Corning × SKKU SPMDL Industry-Academia Collaboration

Predictive micro-crack lifecycle management framework for sub-5nm EUV lithography glass substrates.

## Architecture

| Module | Name | Physics |
|--------|------|---------|
| M1 | Crack Nucleation Engine | Griffith + stochastic defects + thermoelastic stress |
| M2 | Propagation & Morphology | Phase-field fracture + Charles-Hillig subcritical growth |
| M3 | Inspection Forward Model | Acoustic, laser, Raman, interferometry signal simulation |
| M4 | Inverse ML Diagnostics | Physics-informed Bayesian inference from inspection data |
| M5 | Process Attribution | Overlay/EPE degradation decomposition + replacement optimization |

## Project Structure

```
glass-crack-sim/
├── config.py              # Material properties, EUV conditions, parameters
├── modules/
│   ├── m01_nucleation.py  # M1: Crack nucleation probability
│   ├── m02_propagation.py # M2: Phase-field crack evolution
│   ├── m03_inspection.py  # M3: NDT signal forward model
│   ├── m04_inverse_ml.py  # M4: ML-based crack diagnosis
│   └── m05_attribution.py # M5: Process impact & replacement
├── app.py                 # Streamlit UI
├── tests/
├── data/
├── docs/
│   ├── proposal-corning.md  # Technical concept proposal (English)
│   └── proposal-concept.md  # Internal architecture document
└── scripts/
```

## Stack

Python, NumPy, SciPy, Streamlit, Plotly, scikit-learn, PyTorch

## Links

- EUV Simulator v4: https://github.com/sjoonkwon0531/EUV-Simulator-V4
- CEMS DT5: https://github.com/sjoonkwon0531/CESM_DT5
- Tandem PV: https://github.com/sjoonkwon0531/Tandem-PV-Simulator

## License

Proprietary — Corning × SKKU SPMDL
