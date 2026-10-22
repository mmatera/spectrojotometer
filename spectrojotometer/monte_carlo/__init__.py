"""Monte Carlo simulation of the Ising model, for spectrojotometer.

The fit of spectrojotometer projects the Heisenberg Hamiltonian onto the
basis of collinear configurations and fits its diagonal part, which is
exactly an Ising Hamiltonian:

    H = E0 - sum_{<i,j>} J_ij sigma_i sigma_j

So the couplings of the model are already in the Ising convention, and
this module uses them without rescaling.
"""
from ._jit_kernels import HAS_NUMBA, warmup
from .analysis import (
    binder_crossing,
    estimate_tc,
    fit_curie_weiss,
    frustration_factor,
)
from .frustration import detect_frustration
from .ising_engine import IsingMonteCarlo
from .parallel_tempering import ParallelTempering
from .runner import run_ising_mc
from .wolff import WolffCluster

__all__ = [
    "HAS_NUMBA",
    "IsingMonteCarlo",
    "ParallelTempering",
    "WolffCluster",
    "binder_crossing",
    "detect_frustration",
    "estimate_tc",
    "fit_curie_weiss",
    "frustration_factor",
    "run_ising_mc",
    "warmup",
]
