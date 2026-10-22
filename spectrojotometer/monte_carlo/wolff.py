"""Wolff cluster algorithm for the unfrustrated Ising model, with numba kernels."""

import numpy as np

from ._jit_kernels import _wolff_step_kernel
from .ising_engine import IsingMonteCarlo


class WolffCluster(IsingMonteCarlo):
    """Wolff cluster algorithm, with a gauge transformation.

    The gauge (tau) must come from detect_frustration(): for a frustrated
    model this algorithm is NOT ergodic and must not be used.

    The initial state
    -----------------
    At low temperatures (T << Tc) the Wolff cluster covers almost the
    whole lattice, so each step flips the whole system. Starting from a
    high-energy state in the gauge basis (e.g. a Neel state of a
    bipartite antiferromagnet), the algorithm gets stuck. By default, the
    constructor starts from the "ordered" state in the gauge basis (every
    spin = +1), which is the physical Neel state of a bipartite
    antiferromagnet.
    """

    def __init__(self, spins, bonds, J_vals, E0=0.0, seed=None,
                 gauge=None, initial_state="ordered"):
        if gauge is None:
            raise ValueError(
                "WolffCluster needs the gauge computed by "
                "detect_frustration(). If the model is frustrated, "
                "use Metropolis or ParallelTempering."
            )
        super().__init__(spins, bonds, J_vals, E0, seed)

        self.gauge = np.asarray(gauge, dtype=np.int8)
        if self.gauge.shape != (self.N,):
            raise ValueError("gauge must have shape (N,)")

        # To the gauge basis
        self.spins = (self.spins * self.gauge).astype(np.int8)

        if initial_state == "ordered":
            self.spins = np.ones(self.N, dtype=np.int8)
        elif initial_state == "random":
            self.spins = (2 * self.rng.integers(0, 2, size=self.N)
                          .astype(np.int8) - 1)
        elif initial_state == "user":
            pass
        else:
            raise ValueError(
                f"unknown initial_state: {initial_state!r}. "
                "Use 'ordered', 'random' or 'user'."
            )

        # After the gauge transformation, every effective J is |J|
        self.J_vals = np.abs(self.J_vals)
        self._build_neighbor_lists()
        self._energy_dirty = True

        # Preallocated buffers for the Wolff kernel
        self._cluster_mask = np.zeros(self.N, dtype=np.bool_)
        self._cluster_stack = np.zeros(self.N, dtype=np.int64)

    # ---------- dynamics ----------

    def wolff_step(self):
        size = _wolff_step_kernel(
            self.spins,
            self._nbr_idx_flat,
            self._nbr_J_flat,
            self._nbr_start,
            self.beta,
            self._jit_state,
            self._cluster_mask,
            self._cluster_stack,
        )
        self._energy_dirty = True
        return size

    def metropolis_sweep(self, n_clusters=None):
        """A Wolff sweep: N cluster attempts."""
        if n_clusters is None:
            n_clusters = self.N
        for _ in range(n_clusters):
            self.wolff_step()

    # ---------- measurement ----------
    def sample(self, n_sweeps, measure_every=1):
        M_sum = 0.0
        M_abs_sum = 0.0
        M2_sum = 0.0
        M4_sum = 0.0
        Mu_sum = 0.0
        E_sum = 0.0
        E2_sum = 0.0
        n = 0
        gauge_int = self.gauge.astype(np.int64)
        for sweep in range(n_sweeps):
            self.metropolis_sweep()
            if (sweep + 1) % measure_every == 0:
                M = int(self.spins.sum(dtype=np.int64))
                Mu = int((gauge_int * self.spins).sum())
                E = self.energy
                M_sum += M
                M_abs_sum += abs(M)
                M2_sum += M * M
                M4_sum += M * M * M * M
                Mu_sum += Mu
                E_sum += E
                E2_sum += E * E
                n += 1

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
            "M_uniform": abs(Mu_sum / n) / self.N,
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
