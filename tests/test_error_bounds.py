"""
error_bounds / cost: the largest error of each coupling per unit error in
the energies, and the selection of configurations based on it.
"""

import itertools
import logging

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.tools import compatibility_box


def ring(n_sites, n_bonds):
    pos = [np.array([float(i), 0.0, 0.0]) for i in range(n_sites)]
    lists = [
        [(i, (i + d) % n_sites, (0,)) for i in range(n_sites)]
        for d in range(1, n_bonds + 1)
    ]
    return MagneticModel(
        pos,
        [np.array([float(n_sites), 0.0, 0.0])],
        bond_lists=lists,
        bond_distances=list(range(1, n_bonds + 1)),
    )


def all_confs(n_sites):
    return [[0] + list(b) for b in itertools.product([0, 1], repeat=n_sites - 1)]


def random_confs(n_sites, count, rng):
    return [[0] + list(rng.integers(0, 2, n_sites - 1)) for _ in range(count)]


def determining_sets(model, n_sites, size, count, rng):
    sets = []
    while len(sets) < count:
        confs = random_confs(n_sites, size, rng)
        if np.all(np.isfinite(model.error_bounds(confs, "ellipse"))):
            sets.append(confs)
    return sets


@pytest.mark.parametrize("seed", range(5))
def test_exact_bounds_are_the_box_for_zero_residual(seed):
    """With energies reproduced exactly, compute_couplings' error bars are
    tolerance * error_bounds."""
    rng = np.random.default_rng(seed)
    model = ring(8, 3)
    (confs,) = determining_sets(model, 8, 8, 1, rng)
    couplings = np.r_[rng.normal(0, 0.01, 3), -100.0]
    energies = model.coefficient_matrix(confs).dot(couplings)
    tol = 1e-3
    _, d_js, _, _ = model.compute_couplings(confs, energies, tol, method="exact")
    np.testing.assert_allclose(d_js[:-1], tol * model.error_bounds(confs), rtol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_ellipse_bounds_the_exact_ones(seed):
    rng = np.random.default_rng(seed)
    model = ring(10, 4)
    for confs in determining_sets(model, 10, 8, 5, rng):
        exact = model.error_bounds(confs, "exact")
        ellipse = model.error_bounds(confs, "ellipse")
        assert np.all(exact <= ellipse * (1 + 1e-9))


@pytest.mark.parametrize("seed", range(5))
def test_exact_bounds_never_grow_with_more_configurations(seed):
    rng = np.random.default_rng(seed)
    model = ring(10, 4)
    for confs in determining_sets(model, 10, 6, 5, rng):
        more = confs + random_confs(10, 1, rng)
        assert np.all(
            model.error_bounds(more) <= model.error_bounds(confs) * (1 + 1e-9)
        )


def test_bounds_do_not_depend_on_scaling_a_coupling():
    """Each bound is a property of its own coupling: rescaling a column of
    the problem (a change of units of that coupling) only rescales its own
    bound."""
    rng = np.random.default_rng(0)
    model = ring(10, 4)
    (confs,) = determining_sets(model, 10, 8, 1, rng)
    coeffs = model.coefficient_matrix(confs)
    lower, upper = compatibility_box(coeffs, np.zeros(len(confs)), 1.0)
    scaled = coeffs.copy()
    scaled[:, 0] *= 10.0
    lower2, upper2 = compatibility_box(scaled, np.zeros(len(confs)), 1.0)
    np.testing.assert_allclose((upper2 - lower2)[0] * 10.0, (upper - lower)[0], rtol=1e-6)
    np.testing.assert_allclose((upper2 - lower2)[1:], (upper - lower)[1:], rtol=1e-6)


def test_undetermined_couplings_and_empty_sets():
    model = ring(8, 3)
    assert np.all(model.error_bounds([]) == np.inf)
    assert model.cost([]) == np.inf
    ferro = [[0] * 8, [0] * 8]
    for method in ("exact", "ellipse"):
        assert np.all(model.error_bounds(ferro, method) == np.inf)
    with pytest.raises(ValueError):
        model.error_bounds(ferro, "bogus")


def test_cost_is_the_largest_exact_bound():
    rng = np.random.default_rng(1)
    model = ring(10, 4)
    (confs,) = determining_sets(model, 10, 7, 1, rng)
    assert model.cost(confs) == pytest.approx(np.max(model.error_bounds(confs)))
    assert model.cost(confs, "ellipse") >= model.cost(confs)


def test_selection_finds_the_optimum_of_a_small_model():
    """Compared with the exhaustive search over every subset of the minimal
    size, the selection reaches the best ellipse score."""
    model = ring(6, 2)
    candidates = model.get_independent_confs(all_confs(6))
    size = 3  # two couplings and the base energy
    best = min(
        model.cost(list(subset), "ellipse")
        for subset in itertools.combinations(candidates, size)
    )
    chosen, cost = model.optimize_independent_set(candidates, length=size)
    assert len(chosen) == size
    assert model.cost(chosen, "ellipse") == pytest.approx(best)
    assert cost == pytest.approx(model.cost(chosen))


def test_selection_beats_random_subsets():
    rng = np.random.default_rng(2)
    model = ring(12, 5)
    pool = random_confs(12, 30, rng)
    chosen, cost = model.optimize_independent_set(pool, length=6)
    assert len(chosen) == 6
    candidates = model.get_independent_confs(pool)
    random_costs = []
    for _ in range(200):
        subset = [candidates[i] for i in rng.choice(len(candidates), 6, replace=False)]
        random_costs.append(model.cost(subset, "ellipse"))
    assert model.cost(chosen, "ellipse") <= np.min(random_costs) * (1 + 1e-9)


def test_automatic_size_follows_the_improvement_threshold():
    """A larger threshold never gives more configurations; with 100% the
    set is the minimal one."""
    rng = np.random.default_rng(3)
    model = ring(10, 4)
    pool = random_confs(10, 25, rng)
    sizes = [
        len(model.optimize_independent_set(pool, min_improvement=m)[0])
        for m in (0.0, 0.01, 0.05, 0.2, 1.0)
    ]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == 5  # four couplings and the base energy
    assert sizes[0] > sizes[-1]


def test_nothing_new_to_add_keeps_the_cost_of_the_known_set():
    model = ring(6, 2)
    known = model.get_independent_confs(all_confs(6))
    chosen, cost = model.optimize_independent_set(known, forced=known)
    assert chosen == []
    assert cost == pytest.approx(model.cost(known))
    chosen, cost = model.find_optimal_configurations(known=known, its=5)
    assert chosen == []
    assert cost == pytest.approx(model.cost(known))


def test_printeqs_reports_the_bounds(capsys):
    rng = np.random.default_rng(4)
    model = ring(8, 3)
    (confs,) = determining_sets(model, 8, 6, 1, rng)
    energies = model.coefficient_matrix(confs).dot(np.r_[0.01, -0.005, 0.002, -10.0])
    logging.disable(logging.CRITICAL)
    try:
        model.compute_couplings(confs, energies, 1e-3, printeqs=True)
    finally:
        logging.disable(logging.NOTSET)
    out = capsys.readouterr().out
    assert "Largest error of each coupling per unit error in the energies" in out
    for name in model.bonds:
        assert f"{name}:" in out
