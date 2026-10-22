"""High-level driver of the Monte Carlo simulations of the Ising model."""

import warnings

import numpy as np

from .analysis import estimate_tc, fit_curie_weiss, frustration_factor
from .frustration import detect_frustration
from .ising_engine import IsingMonteCarlo
from .parallel_tempering import ParallelTempering
from .wolff import WolffCluster


def run_ising_mc(spins, bonds, J_vals, E0,
                 T_min, T_max, n_temps,
                 n_equil=5000, n_sweeps=20000,
                 algorithm="auto",
                 n_restarts=1,
                 seed=None,
                 swap_every=10,
                 progress_callback=None):
    """
    Simulate the Ising model H = E0 - sum_b J_b s_i s_j (s = +-1) at
    n_temps temperatures, geometrically spaced between T_min and T_max
    (in the units of J; Boltzmann constant 1).

    spins, bonds, J_vals, E0: as returned by
        monte_carlo.adapter.magnetic_model_to_ising (initial spins, the
        pairs (i, j) of each bond, its coupling, and the base energy).
    n_equil, n_sweeps: sweeps discarded at each temperature, and sweeps
        averaged.
    algorithm: "wolff" (cluster updates; only for unfrustrated models,
        which are mapped onto ferromagnetic ones by a gauge transform),
        "metropolis", "pt" (parallel tempering), or "auto": Wolff if the
        model is not frustrated, Metropolis otherwise (with a warning).
    n_restarts: Metropolis only, independent runs from random states;
        a warning is issued if, at some temperature, their energies
        differ by more than 5% of the energy scale of the model
        (sum |J| / N).
    seed: for reproducible results.
    swap_every: parallel tempering only, sweeps between swaps.
    progress_callback: called as progress_callback(done, total, message).

    Returns a dict with lists, one value per temperature: "T", "E"
    (energy per site), "M", "M2", "M4" (magnetization per site and its
    moments), "chi", "chi_connected" (susceptibility), "C" (specific
    heat), "U4" (Binder cumulant); and "algorithm", "N" (number of
    sites), "frustrated", "frustration_cycle" (an odd cycle of frustrated
    bonds, if any), "Tc_peak" (temperature of the peak of the connected
    susceptibility), "curie_weiss" and "frustration_factor" (None with
    fewer than three temperatures).
    """
    spins = np.asarray(spins, dtype=np.int8)
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.asarray(J_vals, dtype=np.float64)
    N = int(spins.size)

    def notify(done, total, msg):
        if progress_callback is not None:
            try:
                progress_callback(done, total, msg)
            except Exception:
                pass

    frustrated, gauge, cycle = detect_frustration(bonds, J_vals, N)

    if algorithm == "auto":
        algorithm = "metropolis" if frustrated else "wolff"
        if frustrated:
            warnings.warn(
                "Frustrated model detected "
                f"(odd cycle: {cycle[:6]}{'...' if cycle and len(cycle) > 6 else ''}). "
                "Using Metropolis. For an accurate Tc, consider algorithm='pt'.",
                UserWarning,
            )
    elif algorithm == "wolff" and frustrated:
        raise ValueError(
            "Wolff is not ergodic for frustrated models. "
            "Use 'metropolis' or 'pt'."
        )

    T_grid = np.geomspace(T_min, T_max, n_temps)
    notify(0, n_temps, f"Algoritmo: {algorithm}")

    if algorithm == "pt":
        sim = ParallelTempering(spins, bonds, J_vals, E0, T_grid, seed=seed)
        results = sim.run(n_equil, n_sweeps, swap_every=swap_every)
        notify(n_temps, n_temps, "PT completado")
    elif algorithm == "wolff":
        sim = WolffCluster(spins, bonds, J_vals, E0, seed=seed, gauge=gauge)
        results = _sweep_with_progress(sim, T_grid, n_equil, n_sweeps, notify)
    elif algorithm == "metropolis":
        results = _run_metropolis_with_restarts(
            spins, bonds, J_vals, E0, T_grid,
            n_equil, n_sweeps, n_restarts, seed,
            progress_callback=notify,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")

    results["frustrated"] = bool(frustrated)
    results["frustration_cycle"] = cycle
    results["algorithm"] = algorithm
    results["N"] = N

    if len(results["T"]) >= 3:
        results["Tc_peak"] = estimate_tc(results["T"], results["chi_connected"])
        results["curie_weiss"] = fit_curie_weiss(
            results["T"], results["chi_connected"],
            T_cut=1.2 * results["Tc_peak"],
        )
        results["frustration_factor"] = frustration_factor(
            results["Tc_peak"], J_vals, N,
        )
    else:
        results["Tc_peak"] = None
        results["curie_weiss"] = None
        results["frustration_factor"] = None

    return results


def _sweep_with_progress(sim, T_grid, n_equil, n_sweeps, notify):
    """Like sim.sweep_temperatures, but reporting the progress."""
    keys = ["T", "M", "M2", "M4", "E", "chi", "chi_connected", "C", "U4"]
    results = {k: [] for k in keys}
    total = len(T_grid)
    for i, T in enumerate(T_grid):
        sim.set_temperature(T)
        sim.equilibrate(n_equil)
        obs = sim.sample(n_sweeps)
        results["T"].append(float(T))
        for k in keys[1:]:
            if k in obs:
                results[k].append(obs[k])
        notify(i + 1, total, f"T = {T:.3f}")
    return results


def _run_metropolis_with_restarts(spins, bonds, J_vals, E0, T_grid,
                                  n_equil, n_sweeps, n_restarts, seed,
                                  progress_callback=None):
    keys = ["M", "M2", "M4", "E", "chi", "chi_connected", "C", "U4"]
    rng = np.random.default_rng(seed)

    accum = {k: np.zeros(len(T_grid)) for k in keys}
    accum["T"] = list(T_grid)
    energies = np.zeros((n_restarts, len(T_grid)))  # per replica

    total_steps = n_restarts * len(T_grid)
    step = 0

    for replica in range(n_restarts):
        init = 2 * rng.integers(0, 2, size=spins.size).astype(np.int8) - 1
        sim = IsingMonteCarlo(init, bonds, J_vals, E0,
                              seed=int(rng.integers(0, 2**31 - 1)))
        for t_idx, T in enumerate(T_grid):
            sim.set_temperature(T)
            sim.equilibrate(n_equil)
            obs = sim.sample(n_sweeps)
            for k in keys:
                accum[k][t_idx] += obs[k]
            energies[replica, t_idx] = obs["E"]
            step += 1
            if progress_callback is not None:
                progress_callback(step, total_steps, f"T = {T:.3f}")

    for k in keys:
        accum[k] = (accum[k] / n_restarts).tolist()
    accum["T"] = list(map(float, T_grid))

    if n_restarts > 1:
        # Spread of the energies of the replicas at the same temperature
        # (it compared the averaged energies of different temperatures),
        # relative to the energy scale of the model: the largest possible
        # energy per site, sum |J| / N. (Relative to the energy itself, any
        # difference is large where the energy is close to 0.)
        scale = max(float(np.sum(np.abs(J_vals))) / spins.size, 1e-12)
        spread = np.max(energies.max(axis=0) - energies.min(axis=0)) / scale
        if spread > 0.05:
            warnings.warn(
                f"The energies of the replicas differ by {spread:.1%} of the "
                "energy scale of the model: the sampling may not be ergodic. "
                "Increase n_equil / n_sweeps, or use algorithm='pt'.",
                UserWarning,
            )
    return accum

