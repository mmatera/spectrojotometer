"""Monte Carlo Ising module for Spectrojotometer.

El fit de Spectrojotometer proyecta el Hamiltoniano de Heisenberg sobre
la base de configuraciones colineales y ajusta la parte diagonal, que es
exactamente un Hamiltoniano de Ising:

    H = E0 - sum_{<i,j>} J_ij sigma_i sigma_j

Por lo tanto los J del modelo ya estan en convencion Ising. Este modulo
consume esos J sin reescalar.
"""
from ._jit_kernels import HAS_NUMBA, warmup
from .frustration import detect_frustration
from .ising_engine import IsingMonteCarlo
from .parallel_tempering import ParallelTempering
from .wolff import WolffCluster
from .analysis import (
    estimate_tc,
    fit_curie_weiss,
    binder_crossing,
    frustration_factor,
)
from .runner import run_ising_mc

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
