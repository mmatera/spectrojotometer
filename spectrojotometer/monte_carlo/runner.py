"""Orquestador de alto nivel para Monte Carlo Ising."""

import warnings
import numpy as np

from .frustration import detect_frustration
from .ising_engine import IsingMonteCarlo
from .parallel_tempering import ParallelTempering
from .wolff import WolffCluster
from .analysis import estimate_tc, fit_curie_weiss, frustration_factor


def run_ising_mc(spins, bonds, J_vals, E0,
                 T_min, T_max, n_temps,
                 n_equil=5000, n_sweeps=20000,
                 algorithm="auto",
                 n_restarts=1,
                 seed=None,
                 swap_every=10):
    """Ejecuta MC Ising sobre (spins, bonds, J_vals, E0).

    Los J ya estan en convencion Ising (vienen del fit de Spectrojotometer).
    No se aplica ningun factor S^2.
    """
    spins = np.asarray(spins, dtype=np.int8)
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.asarray(J_vals, dtype=np.float64)
    N = int(spins.size)

    frustrated, gauge, cycle = detect_frustration(bonds, J_vals, N)

    if algorithm == "auto":
        algorithm = "metropolis" if frustrated else "wolff"
        if frustrated:
            warnings.warn(
                "Modelo frustrado detectado "
                f"(ciclo impar: {cycle[:6]}{'...' if cycle and len(cycle) > 6 else ''}). "
                "Usando Metropolis. Para Tc preciso considerar algorithm='pt'.",
                UserWarning,
            )
    elif algorithm == "wolff" and frustrated:
        raise ValueError(
            "Wolff no es ergodico en sistemas frustrados. "
            "Usar 'metropolis' o 'pt'."
        )

    T_grid = np.geomspace(T_min, T_max, n_temps)

    if algorithm == "pt":
        sim = ParallelTempering(spins, bonds, J_vals, E0, T_grid, seed=seed)
        results = sim.run(n_equil, n_sweeps, swap_every=swap_every)
    elif algorithm == "wolff":
        sim = WolffCluster(spins, bonds, J_vals, E0, seed=seed, gauge=gauge)
        results = sim.sweep_temperatures(T_grid, n_equil, n_sweeps)
    elif algorithm == "metropolis":
        results = _run_metropolis_with_restarts(
            spins, bonds, J_vals, E0, T_grid,
            n_equil, n_sweeps, n_restarts, seed,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")

    results["frustrated"] = bool(frustrated)
    results["frustration_cycle"] = cycle
    results["algorithm"] = algorithm
    results["N"] = N

    if len(results["T"]) >= 3:
        results["Tc_peak"] = estimate_tc(results["T"], results["chi"])
        results["curie_weiss"] = fit_curie_weiss(
            results["T"], results["chi"], T_cut=1.2 * results["Tc_peak"],
        )
        results["frustration_factor"] = frustration_factor(
            results["Tc_peak"], J_vals, N,
        )
    else:
        results["Tc_peak"] = None
        results["curie_weiss"] = None
        results["frustration_factor"] = None

    return results


def _run_metropolis_with_restarts(spins, bonds, J_vals, E0, T_grid,
                                  n_equil, n_sweeps, n_restarts, seed):
    """Corre Metropolis desde varias condiciones iniciales y promedia."""
    keys = ["M", "M2", "M4", "E", "chi", "C", "U4"]
    rng = np.random.default_rng(seed)

    accum = {k: np.zeros(len(T_grid)) for k in keys}
    accum["T"] = list(T_grid)

    for _ in range(n_restarts):
        init = 2 * rng.integers(0, 2, size=spins.size).astype(np.int8) - 1
        sim = IsingMonteCarlo(init, bonds, J_vals, E0,
                              seed=int(rng.integers(0, 2**31 - 1)))
        for t_idx, T in enumerate(T_grid):
            sim.set_temperature(T)
            sim.equilibrate(n_equil)
            obs = sim.sample(n_sweeps)
            for k in keys:
                accum[k][t_idx] += obs[k]

    for k in keys:
        accum[k] = (accum[k] / n_restarts).tolist()
    accum["T"] = list(map(float, T_grid))

    if n_restarts > 1:
        E_vals = np.array(accum["E"])
        spread = (E_vals.max() - E_vals.min()) / max(abs(E_vals.mean()), 1e-12)
        if spread > 0.05:
            warnings.warn(
                f"Dispersion entre replicas ({spread:.1%}) sugiere "
                "no-ergodicidad. Aumentar n_equil/n_sweeps o usar 'pt'.",
                UserWarning,
            )
    return accum
