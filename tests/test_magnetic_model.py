import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ring_model(n_unknown_bonds=2):
    """4-site cell, J1 = nearest neighbours, J2 = next-nearest neighbours."""
    pos = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    lat = [np.array([4.0, 0.0, 0.0])]
    j1 = [(0, 1, (0,)), (1, 2, (0,)), (2, 3, (0,)), (3, 0, (0,))]
    j2 = [(0, 2, (0,)), (1, 3, (0,))]
    lists = [j1, j2][:n_unknown_bonds]
    return MagneticModel(
        pos, lat, bond_lists=lists, bond_distances=[1.0, 2.0][:n_unknown_bonds]
    )


def all_confs(size=4):
    return [[0] + list(b) for b in itertools.product([0, 1], repeat=size - 1)]


def rank(model, confs):
    return np.linalg.matrix_rank(model.coefficient_matrix(confs, False), tol=1e-6)


def test_forced_nonempty_length_none():
    m = ring_model()
    confs = all_confs()
    forced = [confs[1]]
    res, cost = m.optimize_independent_set(confs, forced=forced)
    assert np.isfinite(cost)
    assert forced[0] not in res
    assert rank(m, forced + res) == 3  # 2 bonds + base energy


def test_small_length_does_not_crash():
    m = ring_model()
    res, cost = m.optimize_independent_set(all_confs(), length=1)
    # The rank gets recovered by enlarging the set instead of raising.
    assert rank(m, res) == 3
    assert np.isfinite(cost)


def test_length_with_forced():
    m = ring_model()
    confs = all_confs()
    res, cost = m.optimize_independent_set(confs, length=1, forced=[confs[1]])
    assert rank(m, [confs[1]] + res) == 3
    assert np.isfinite(cost)


def test_underdetermined_returns_inf():
    m = ring_model()
    res, cost = m.optimize_independent_set([all_confs()[1]])
    assert cost == np.inf


def test_no_information_returns_inf():
    m = ring_model()
    # All-equal spins: every bond sum is constant -> only the base energy
    # column is non-zero, so the information is partial.
    res, cost = m.optimize_independent_set([[0, 0, 0, 0]])
    assert cost == np.inf


def test_empty_input():
    m = ring_model()
    assert m.optimize_independent_set([]) == ([], np.inf)


def test_add_independent_confs_typo_fixed():
    m = ring_model()
    confs = [[0, 0, 1, 1]]
    m.add_independent_confs(confs, [[0, 1, 0, 1], [0, 1, 0, 1], []])
    assert confs == [[0, 0, 1, 1], [0, 1, 0, 1]]


def test_random_configurations_shape():
    m = ring_model()
    confs = list(m.generate_random_configurations(20))
    assert len(confs) == 20
    assert all(len(c) == 4 and c[0] == 0 and set(c) <= {0, 1} for c in confs)


def test_find_optimal_configurations_runs():
    m = ring_model()
    # The search is randomized. Starting from the full set of
    # configurations makes the outcome deterministic as far as the rank
    # goes: the information is complete, so the cost must be finite.
    res, cost = m.find_optimal_configurations(start=all_confs(), its=5)
    assert np.isfinite(cost)
    assert rank(m, res) == 3


def test_find_optimal_configurations_partial_info_is_reported():
    m = ring_model()
    # Few random draws may not reach full rank: that is legitimate, but it
    # must then be reported through an infinite cost, never as a crash.
    for _ in range(20):
        res, cost = m.find_optimal_configurations(its=2)
        assert rank(m, res) == 3 or cost == np.inf


def test_compute_couplings_mcsteps_none():
    m = ring_model()
    confs = all_confs()
    coeffs = m.coefficient_matrix(confs, centered=True)
    energs = coeffs.dot(np.array([1.0, -0.5, 3.0]))
    js, dj, chi, rate = m.compute_couplings(confs, energs, err_energs=0.01)
    assert np.allclose(js[:2], [1.0, -0.5], atol=0.05)


def test_save_cif_all_dimensions(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0]])
    for lat in (
        [[3.0, 0, 0]],
        [[3.0, 0, 0], [0, 4.0, 0]],
        [[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]],
    ):
        m = MagneticModel(pos, [np.array(v) for v in lat])
        out = tmp_path / f"m{len(lat)}.cif"
        assert m.save_cif(str(out))
        assert out.read_text()


def test_cif_angles_are_right_way_round(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0]])
    # b ⟂ c, a·c = cos(60°)·|a||c|, a ⟂ b  -> alpha=90, beta=60, gamma=90
    a = np.array([1.0, 0, 0])
    b = np.array([0, 1.0, 0])
    c = np.array([0.5, 0, np.sqrt(3) / 2])
    m = MagneticModel(pos, [a, b, c])
    out = tmp_path / "m.cif"
    m.save_cif(str(out))
    text = out.read_text()
    assert "_cell_angle_alpha \t\t\t90.0" in text
    assert "_cell_angle_beta \t\t\t60.0" in text
    assert "_cell_angle_gamma \t\t\t90.0" in text


def test_bond_keys_independent_of_hash_seed():
    code = (
        "import numpy as np;"
        "from spectrojotometer.magnetic_model import MagneticModel;"
        "m=MagneticModel(np.array([[0.,0,0],[1.,0,0]]),[np.array([2.,0,0])],"
        "magnetic_species=['A','B'],supercell_size=1);"
        "m.generate_bonds(0.02,[[0,3.5]]);"
        "print(sorted((k,round(v['distance'],2),len(v['bonds'])) for k,v in m.bonds.items()))"
    )
    outs = set()
    for seed in ("0", "1", "2", "3", "4"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=ROOT)
        r = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, check=True,
        )
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1
