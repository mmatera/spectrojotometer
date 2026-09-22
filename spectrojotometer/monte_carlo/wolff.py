"""Algoritmo de cluster de Wolff para Ising unfrustrated, con kernels Numba."""

import numpy as np

from .ising_engine import IsingMonteCarlo
from ._jit_kernels import _wolff_step_kernel


class WolffCluster(IsingMonteCarlo):
    """Wolff cluster con gauge transform.

    El gauge (tau) debe provenir de detect_frustration(); si el modelo
    esta frustrado este algoritmo NO es ergodico y no debe usarse.

    Notas sobre el estado inicial
    -----------------------------
    A temperaturas bajas (T << Tc) el cluster de Wolff cubre
    practicamente todo el reticulo, por lo que cada paso invierte el
    sistema completo. Si se parte de un estado de alta energia en la
    base gauge (p.ej. un estado Neel para un AFM bipartito), el
    algoritmo queda atascado. El constructor arranca por defecto desde
    el estado "ordered" en base gauge (todos los spines = +1), que
    corresponde al Neel fisico para AFM bipartito.
    """

    def __init__(self, spins, bonds, J_vals, E0=0.0, seed=None,
                 gauge=None, initial_state="ordered"):
        if gauge is None:
            raise ValueError(
                "WolffCluster requiere el gauge calculado por "
                "detect_frustration(). Si el modelo esta frustrado, "
                "usar Metropolis o ParallelTempering."
            )
        super().__init__(spins, bonds, J_vals, E0, seed)

        self.gauge = np.asarray(gauge, dtype=np.int8)
        if self.gauge.shape != (self.N,):
            raise ValueError("gauge debe tener shape (N,)")

        # Pasar a base gauge
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
                f"initial_state desconocido: {initial_state!r}. "
                "Usar 'ordered', 'random' o 'user'."
            )

        # Todos los J efectivos son |J| tras el gauge
        self.J_vals = np.abs(self.J_vals)
        self._build_neighbor_lists()
        self._energy_dirty = True

        # Buffers preallocados para el kernel de Wolff
        self._cluster_mask = np.zeros(self.N, dtype=np.bool_)
        self._cluster_stack = np.zeros(self.N, dtype=np.int64)

    # ---------- dinamica ----------

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
        """Un sweep de Wolff = N intentos de cluster."""
        if n_clusters is None:
            n_clusters = self.N
        for _ in range(n_clusters):
            self.wolff_step()

    # ---------- medicion ----------
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
