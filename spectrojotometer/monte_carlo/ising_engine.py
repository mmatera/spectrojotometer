"""Motor Metropolis para Ising clasico con kernels Numba."""

import numpy as np

from ._jit_kernels import (
    _interaction_energy_kernel,
    _metropolis_sweep_kernel,
)


class IsingMonteCarlo:
    """Simulacion Metropolis single-flip para Ising.

    Convencion: sigma_i in {-1, +1}, H = E0 - sum J_ij sigma_i sigma_j.
    """

    def __init__(self, spins, bonds, J_vals, E0=0.0, seed=None):
        self.spins = np.ascontiguousarray(spins, dtype=np.int8).copy()
        self.bonds = np.ascontiguousarray(bonds, dtype=np.int64)
        self.J_vals = np.ascontiguousarray(J_vals, dtype=np.float64)
        self.E0 = float(E0)
        self.N = int(self.spins.size)

        # RNG Python (para inicializacion) y RNG Numba (para la dinamica)
        self.rng = np.random.default_rng(seed)
        seed_jit = int(self.rng.integers(1, 2**63 - 1))
        self._jit_state = np.array([np.uint64(seed_jit)], dtype=np.uint64)

        if self.bonds.size == 0:
            self.bonds = np.zeros((0, 2), dtype=np.int64)
            self.J_vals = np.zeros((0,), dtype=np.float64)

        self._build_neighbor_lists()

        self.T = 1.0
        self.beta = 1.0
        self._energy = None
        self._energy_dirty = True

    # ---------- construccion ----------

    def _build_neighbor_lists(self):
        """Construye la representacion CSR plana de vecinos."""
        nbrs = [[] for _ in range(self.N)]
        Js = [[] for _ in range(self.N)]
        for (i, j), J in zip(self.bonds, self.J_vals):
            nbrs[int(i)].append(int(j))
            Js[int(i)].append(float(J))
            nbrs[int(j)].append(int(i))
            Js[int(j)].append(float(J))

        starts = np.zeros(self.N + 1, dtype=np.int64)
        for i in range(self.N):
            starts[i + 1] = starts[i] + len(nbrs[i])
        total = int(starts[-1])

        nbr_idx_flat = np.empty(total, dtype=np.int64)
        nbr_J_flat = np.empty(total, dtype=np.float64)
        for i in range(self.N):
            a, b = int(starts[i]), int(starts[i + 1])
            nbr_idx_flat[a:b] = nbrs[i]
            nbr_J_flat[a:b] = Js[i]

        self._nbr_idx_flat = nbr_idx_flat
        self._nbr_J_flat = nbr_J_flat
        self._nbr_start = starts

    # ---------- energia ----------

    @property
    def energy(self):
        """Energia total, recalculada si es necesario."""
        if self._energy_dirty:
            self._energy = self._compute_energy()
            self._energy_dirty = False
        return self._energy

    def _compute_energy(self):
        if self.bonds.shape[0] == 0:
            return self.E0
        return self.E0 - _interaction_energy_kernel(
            self.spins, self.bonds, self.J_vals
        )

    # ---------- dinamica ----------

    def set_temperature(self, T):
        if T <= 0:
            raise ValueError("La temperatura debe ser positiva.")
        self.T = float(T)
        self.beta = 1.0 / self.T

    def metropolis_sweep(self):
        _metropolis_sweep_kernel(
            self.spins,
            self._nbr_idx_flat,
            self._nbr_J_flat,
            self._nbr_start,
            self.beta,
            self._jit_state,
        )
        self._energy_dirty = True

    def equilibrate(self, n_sweeps):
        for _ in range(n_sweeps):
            self.metropolis_sweep()

    # ---------- medicion ----------

    def ___old_sample(self, n_sweeps, measure_every=1):
        M_sum = 0.0
        M_abs_sum = 0.0
        M2_sum = 0.0
        M4_sum = 0.0
        E_sum = 0.0
        E2_sum = 0.0
        n = 0
        for sweep in range(n_sweeps):
            self.metropolis_sweep()
            if (sweep + 1) % measure_every == 0:
                M = int(self.spins.sum(dtype=np.int64))
                E = self.energy
                M_sum += M
                M_abs_sum += abs(M)
                M2_sum += M * M
                M4_sum += M * M * M * M
                E_sum += E
                E2_sum += E * E
                n += 1

        if n == 0:
            raise RuntimeError("No se tomaron muestras.")

        M_avg = M_sum / n
        M_abs_avg = M_abs_sum / n
        M2_avg = M2_sum / n
        M4_avg = M4_sum / n
        E_avg = E_sum / n
        E2_avg = E2_sum / n

        chi = self.beta / self.N * (M2_avg - M_avg * M_avg)
        chi_connected = self.beta / self.N * (M2_avg - M_abs_avg * M_abs_avg)
        C = self.beta ** 2 / self.N * (E2_avg - E_avg * E_avg)
        U4 = 1.0 - M4_avg / (3.0 * M2_avg ** 2) if M2_avg > 0 else 0.0


    def sample(self, n_sweeps, measure_every=1):
        M_sum = 0.0
        M_abs_sum = 0.0
        M2_sum = 0.0
        M4_sum = 0.0
        E_sum = 0.0
        E2_sum = 0.0
        n = 0
        for sweep in range(n_sweeps):
            self.metropolis_sweep()
            if (sweep + 1) % measure_every == 0:
                M = int(self.spins.sum(dtype=np.int64))
                E = self.energy
                M_sum += M
                M_abs_sum += abs(M)
                M2_sum += M * M
                M4_sum += M * M * M * M
                E_sum += E
                E2_sum += E * E
                n += 1

        if n == 0:
            raise RuntimeError("No se tomaron muestras.")

        M_avg = M_sum / n
        M_abs_avg = M_abs_sum / n
        M2_avg = M2_sum / n
        M4_avg = M4_sum / n
        E_avg = E_sum / n
        E2_avg = E2_sum / n

        chi = self.beta / self.N * (M2_avg - M_avg * M_avg)
        chi_connected = self.beta / self.N * (M2_avg - M_abs_avg * M_abs_avg)
        C = self.beta ** 2 / self.N * (E2_avg - E_avg * E_avg)
        U4 = 1.0 - M4_avg / (3.0 * M2_avg ** 2) if M2_avg > 0 else 0.0

        return {
            "M": M_abs_avg / self.N,
            "M_signed": M_avg / self.N,
            "M2": M2_avg / self.N ** 2,
            "M4": M4_avg / self.N ** 4,
            "E": E_avg / self.N,
            "E_total": E_avg,
            "chi": chi,
            "chi_connected": chi_connected,
            "C": C,
            "U4": U4,
            "n_samples": n,
        }

    def sweep_temperatures(self, T_grid, n_equil, n_sweeps):
        keys = ["T", "M", "M2", "M4", "E", "chi", "chi_connected", "C", "U4"]
        results = {k: [] for k in keys}
        for T in T_grid:
            self.set_temperature(T)
            self.equilibrate(n_equil)
            obs = self.sample(n_sweeps)
            results["T"].append(float(T))
            for k in keys[1:]:
                results[k].append(obs[k])
        return results
