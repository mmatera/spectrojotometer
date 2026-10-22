"""Smoke tests del panel Monte Carlo y del adaptador de tiling."""

import queue
import time

import numpy as np
import pytest

from spectrojotometer.monte_carlo.adapter import _tile_bonds
from spectrojotometer.monte_carlo.lattices import build_square_lattice_2d

class TestAdapter:
    @staticmethod
    def _mock_model():
        """Mock con la estructura real de MagneticModel.bonds."""
        class Mock:
            n_sites = 4
            bonds = {
                "nn1": {
                    "distance": 2.9386,
                    "value": -5.42,
                    "bonds": [
                        (0, 1, "."), (1, 2, "."), (2, 3, "."), (3, 0, "."),
                    ],
                },
                "nn2": {
                    "distance": 5.1234,
                    "value": 1.23,
                    "bonds": [
                        (0, 2, "."), (1, 3, "."),
                        (0, 3, "1_100"), (2, 1, "1_900"),
                    ],
                },
            }
        return Mock()

    def test_parse_bond_dict(self):
        from spectrojotometer.monte_carlo.adapter import _extract_bond_arrays
        bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(
            self._mock_model()
        )
        assert n_sites == 4
        assert len(bonds_cell) == 8
        offsets = [tuple(o) for o in offset_cell]
        assert offsets.count((0, 0, 0)) == 6
        assert offsets.count((1, 0, 0)) == 1
        assert offsets.count((-1, 0, 0)) == 1

    def test_adapter_full_pipeline(self):
        from spectrojotometer.monte_carlo.adapter import magnetic_model_to_ising
        spins, bonds, J, E0 = magnetic_model_to_ising(
            self._mock_model(), supercell=(1, 1, 1)
        )
        assert len(spins) == 4
        assert bonds.shape[1] == 2
        assert len(J) == len(bonds)
        assert np.all(bonds[:, 0] != bonds[:, 1])

    def test_adapter_supercell_2x1x1(self):
        from spectrojotometer.monte_carlo.adapter import magnetic_model_to_ising
        spins, bonds, J, E0 = magnetic_model_to_ising(
            self._mock_model(), supercell=(2, 1, 1)
        )
        assert len(spins) == 8
        assert len(bonds) > 0
        assert np.all(bonds[:, 0] != bonds[:, 1])

    def test_adapter_missing_value_raises(self):
        from spectrojotometer.monte_carlo.adapter import _extract_bond_arrays
        class BadMock:
            n_sites = 2
            bonds = {"nn": {"distance": 1.0, "bonds": [(0, 1, ".")]}}
        with pytest.raises(KeyError, match="value"):
            _extract_bond_arrays(BadMock())

    def test_adapter_bad_bond_format_raises(self):
        from spectrojotometer.monte_carlo.adapter import _extract_bond_arrays
        class BadMock:
            n_sites = 2
            bonds = {"nn": {"distance": 1.0, "value": 1.0,
                            "bonds": [(0, 1)]}}  # falta el tag
        with pytest.raises(ValueError, match="formato inesperado"):
            _extract_bond_arrays(BadMock())

    def test_adapter_infer_n_sites(self):
        """Si el modelo no tiene n_sites ni natoms, se infiere del max indice."""
        from spectrojotometer.monte_carlo.adapter import _extract_bond_arrays
        class NoNSitesMock:
            bonds = {"nn": {"distance": 1.0, "value": 1.0,
                            "bonds": [(0, 5, ".")]}}
        bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(
            NoNSitesMock()
        )
        assert n_sites == 6  # max(0,5) + 1


class TestTiling:

    def test_bond_intracell_no_offset(self):
        """Bond entre dos atomos distintos de la misma celda."""
        bonds_cell = np.array([[0, 1]], dtype=np.int64)
        offset_cell = np.array([[0, 0, 0]], dtype=np.int64)
        J_cell = np.array([1.0])
        N, bonds, J = _tile_bonds(bonds_cell, offset_cell, J_cell,
                                  N_cell=2, supercell=(1, 1, 1))
        assert N == 2
        assert len(bonds) == 1
        assert np.array_equal(bonds[0], [0, 1])
        assert J[0] == 1.0

    def test_bond_intercell_wraps_to_self_in_1x1x1(self):
        """Bond entre celdas en 1x1x1: la periodicidad lo convierte en
        self-loop y debe descartarse."""
        bonds_cell = np.array([[0, 0]], dtype=np.int64)
        offset_cell = np.array([[1, 0, 0]], dtype=np.int64)
        J_cell = np.array([1.0])
        N, bonds, J = _tile_bonds(bonds_cell, offset_cell, J_cell,
                                  N_cell=1, supercell=(1, 1, 1))
        assert N == 1
        # El bond se descarta porque i_glob == j_glob
        assert len(bonds) == 0
        assert len(J) == 0

    def test_single_cell_with_2_atoms_2_bonds(self):
        """Celda con 2 atomos, 1 bond intra-celda + 1 bond inter-celda
        en direccion x. Con supercell 2x1x1 se materializan ambos."""
        bonds_cell = np.array([[0, 0], [0, 1]], dtype=np.int64)
        offset_cell = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int64)
        J_cell = np.array([1.0, 1.0])
        N, bonds, J = _tile_bonds(bonds_cell, offset_cell, J_cell,
                                  N_cell=2, supercell=(2, 1, 1))
        assert N == 4
        # 2 celdas × 2 bonds = 4 bonds, ninguno self-loop
        assert len(bonds) == 4
        assert np.all(bonds[:, 0] != bonds[:, 1])
        # El bond intra-celda (0,1) debe aparecer 2 veces
        intra = np.sum(np.all(bonds == [0, 1], axis=1))
        assert intra == 1
        intra = np.sum(np.all(bonds == [2, 3], axis=1))
        assert intra == 1

    def test_supercell_2x2x1(self):
        """Red 2x2x1 con bond +x: 4 bonds, sin self-loops."""
        bonds_cell = np.array([[0, 0]], dtype=np.int64)
        offset_cell = np.array([[1, 0, 0]], dtype=np.int64)
        J_cell = np.array([1.0])
        N, bonds, J = _tile_bonds(bonds_cell, offset_cell, J_cell,
                                  N_cell=1, supercell=(2, 2, 1))
        assert N == 4
        assert len(bonds) == 4
        assert np.all(bonds[:, 0] != bonds[:, 1])

    def test_tiling_conserves_J_values(self):
        """Los J deben replicarse uno a uno, sin modificaciones."""
        bonds_cell = np.array([[0, 1], [0, 0]], dtype=np.int64)
        offset_cell = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        J_cell = np.array([-1.0, 0.5])
        N, bonds, J = _tile_bonds(bonds_cell, offset_cell, J_cell,
                                  N_cell=2, supercell=(2, 1, 1))
        # 2 celdas × 2 bonds = 4 (todos con offset no nulo en x, todos
        # entre sitios distintos)
        assert len(J) == 4
        # Los J deben ser {-1.0, 0.5} cada uno dos veces
        assert np.sum(J == -1.0) == 2
        assert np.sum(J == 0.5) == 2


class TestWorkerSmoke:

    def test_worker_runs_to_completion(self):
        """Corre el worker con un sistema chico y verifica que termina."""
        from spectrojotometer.gui.mc_worker import MCWorker

        spins, bonds, J, E0 = build_square_lattice_2d(L=4, J=1.0)
        params = dict(
            spins=spins, bonds=bonds, J_vals=J, E0=E0,
            T_min=1.5, T_max=3.5, n_temps=5,
            n_equil=20, n_sweeps=50,
            n_restarts=1, algorithm="wolff", seed=0,
        )
        q = queue.Queue()
        w = MCWorker(params, q)
        w.start()
        w.join(timeout=30)
        assert not w.is_alive()

        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        kinds = [m[0] for m in msgs]
        assert "done" in kinds, f"kinds={kinds}"

        done_msg = next(m for m in msgs if m[0] == "done")
        results = done_msg[1]
        assert "T" in results
        assert len(results["T"]) == 5
        assert "chi_connected" in results

    def test_worker_cancellation(self):
        """El worker debe responder a cancel()."""
        from spectrojotometer.gui.mc_worker import MCWorker

        spins, bonds, J, E0 = build_square_lattice_2d(L=8, J=1.0)
        params = dict(
            spins=spins, bonds=bonds, J_vals=J, E0=E0,
            T_min=1.5, T_max=3.5, n_temps=20,
            n_equil=500, n_sweeps=2000,
            n_restarts=1, algorithm="wolff", seed=0,
        )
        q = queue.Queue()
        w = MCWorker(params, q)
        w.start()
        time.sleep(0.5)
        w.cancel()
        w.join(timeout=10)
        assert not w.is_alive()

        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        kinds = [m[0] for m in msgs]
        assert "cancelled" in kinds or "done" in kinds

    def test_worker_reports_error_on_bad_algorithm(self):
        """Un algoritmo invalido debe producir ('error', traceback)."""
        from spectrojotometer.gui.mc_worker import MCWorker

        spins, bonds, J, E0 = build_square_lattice_2d(L=4, J=1.0)
        params = dict(
            spins=spins, bonds=bonds, J_vals=J, E0=E0,
            T_min=1.5, T_max=3.5, n_temps=5,
            n_equil=20, n_sweeps=50,
            n_restarts=1, algorithm="no_existe", seed=0,
        )
        q = queue.Queue()
        w = MCWorker(params, q)
        w.start()
        w.join(timeout=10)
        kinds = []
        while not q.empty():
            kinds.append(q.get_nowait()[0])
        assert "error" in kinds, f"kinds={kinds}"
