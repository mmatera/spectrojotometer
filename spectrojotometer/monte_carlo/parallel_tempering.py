"""Parallel tempering para Ising clasico."""

import numpy as np

from .ising_engine import IsingMonteCarlo


class ParallelTempering:
    """Replica exchange Monte Carlo.

    Correcto para sistemas frustrados y no frustrados. Es el algoritmo
    recomendado cuando se necesita Tc preciso en presencia de frustration.
    """

    def __init__(self, spins, bonds, J_vals, E0, T_grid, seed=None):
        self.T_grid = np.asarray(T_grid, dtype=np.float64)
        if np.any(self.T_grid <= 0):
            raise ValueError("Todas las temperaturas deben ser positivas.")
        self.R = int(self.T_grid.size)

        rng_master = np.random.default_rng(seed)
        self.replicas = []
        for r in range(self.R):
            init = 2 * rng_master.integers(0, 2, size=len(spins)).astype(np.int8) - 1
            rep = IsingMonteCarlo(
                init, bonds, J_vals, E0,
                seed=int(rng_master.integers(0, 2**31 - 1)),
            )
            rep.set_temperature(self.T_grid[r])
            self.replicas.append(rep)

        self.rng = np.random.default_rng(seed)
        self.n_swap_attempts = 0
        self.n_swap_accepted = 0

    # ---------- swaps ----------

    def _swap(self, i, j):
        ri, rj = self.replicas[i], self.replicas[j]
        beta_i = 1.0 / ri.T
        beta_j = 1.0 / rj.T
        arg = (beta_i - beta_j) * (ri.energy - rj.energy)
        self.n_swap_attempts += 1
        if arg >= 0.0 or self.rng.random() < np.exp(arg):
            ri.spins, rj.spins = rj.spins.copy(), ri.spins.copy()
            ri._energy_dirty = True
            rj._energy_dirty = True
            self.n_swap_accepted += 1

    def _attempt_swaps(self):
        for i in range(self.R - 1):
            self._swap(i, i + 1)

    # ---------- runs ----------

    def run(self, n_equil, n_sweeps, swap_every=10, measure_every=1):
        for _ in range(n_equil):
            for rep in self.replicas:
                rep.metropolis_sweep()
            if self.rng.random() < 1.0 / swap_every:
                self._attempt_swaps()

        keys = ["T", "M", "M2", "M4", "E", "chi", "chi_connected",
                "C", "U4"]
        results = {k: [] for k in keys}

        acc = [
            dict(M=0.0, M_abs=0.0, M2=0.0, M4=0.0,
                 E=0.0, E2=0.0, n=0)
            for _ in range(self.R)
        ]

        for sweep in range(n_sweeps):
            for rep in self.replicas:
                rep.metropolis_sweep()
            if self.rng.random() < 1.0 / swap_every:
                self._attempt_swaps()

            if (sweep + 1) % measure_every == 0:
                for k, rep in enumerate(self.replicas):
                    M = int(rep.spins.sum(dtype=np.int64))
                    E = rep.energy
                    a = acc[k]
                    a["M"] += M
                    a["M_abs"] += abs(M)
                    a["M2"] += M * M
                    a["M4"] += M * M * M * M
                    a["E"] += E
                    a["E2"] += E * E
                    a["n"] += 1

        N = self.replicas[0].N
        for k in range(self.R):
            a = acc[k]
            n = a["n"]
            beta = 1.0 / self.T_grid[k]
            M_avg = a["M"] / n
            M_abs_avg = a["M_abs"] / n
            M2_avg = a["M2"] / n
            M4_avg = a["M4"] / n
            E_avg = a["E"] / n
            E2_avg = a["E2"] / n

            chi = beta / N * (M2_avg - M_avg * M_avg)
            chi_connected = beta / N * (M2_avg - M_abs_avg * M_abs_avg)

            results["T"].append(float(self.T_grid[k]))
            results["M"].append(abs(M_abs_avg) / N)
            results["M2"].append(M2_avg / N ** 2)
            results["M4"].append(M4_avg / N ** 4)
            results["E"].append(E_avg / N)
            results["chi"].append(chi)
            results["chi_connected"].append(chi_connected)
            results["C"].append(beta ** 2 / N * (E2_avg - E_avg * E_avg))
            U4 = 1.0 - M4_avg / (3.0 * M2_avg ** 2) if M2_avg > 0 else 0.0
            results["U4"].append(U4)

        results["swap_acceptance"] = (
            self.n_swap_accepted / self.n_swap_attempts
            if self.n_swap_attempts > 0 else 0.0
        )
        return results
