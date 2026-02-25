"""
Glass Micro-Crack Lifecycle Simulator — Modules Package
Job 10: Corning × SKKU SPMDL Industry-Academia Project

This package contains the five-module architecture for glass crack simulation:

Modules:
- M1: Crack Nucleation Probability Engine  
- M2: Crack Propagation & Morphology Evolution  ✓ IMPLEMENTED
- M3: Inspection Signal Forward Model
- M4: Physics-Informed Inverse Diagnostics (ML)
- M5: Process Impact Attribution Engine

Author: Claude Code (OpenClaw Agent)
Date: 2026-02-25
"""

from .m02_propagation import (
    SubcriticalGrowth,
    PhaseFieldFracture,
    PercolationAnalysis,
    PropagationResult,
    propagate_from_nucleation_result
)

__all__ = [
    'SubcriticalGrowth',
    'PhaseFieldFracture', 
    'PercolationAnalysis',
    'PropagationResult',
    'propagate_from_nucleation_result'
]

__version__ = '0.1.0'