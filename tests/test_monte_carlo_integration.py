"""
The Monte Carlo simulation and the fitted model must describe the same
system: same lattice (bonds between the right sites, also across cells),
same sign convention, and the couplings the user chose.
"""

import itertools
import pathlib

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.monte_carlo.adapter import _offset_of, magnetic_model_to_ising
from spectrojotometer.monte_carlo.ising_engine import IsingMonteCarlo
from spectrojotometer.tools import pack_offset

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def ring(n_sites, distances, tuple_offsets=False):
    """Ring of n_sites along x; bonds that cross the cell carry an offset."""
    def tag(offset):
        return offset if tuple_offsets else pack_offset(offset)

    pos = [np.array([float(i), 0.0, 0.0]) for i in range(n_sites)]
    lists = [
        [
            (i, (i + d) % n_sites, tag((1, 0, 0) if i + d >= n_sites else (0, 0, 0)))
            for i in range(n_sites)
        ]
        for d in distances
    ]
    return MagneticModel(
        pos,
        [np.array([float(n_sites), 0, 0]), np.array([0, 9.0, 0]), np.array([0, 0, 9.0])],
        bond_lists=lists,
        bond_distances=[float(d) for d in distances],
    )


@pytest.mark.parametrize(
    "tag, offset",
    [(".", (0, 0, 0)), ("1_555", (-5, -5, -5)), (pack_offset((1, 0, 0)), (1, 0, 0)),
     (pack_offset((-1, 0, 1)), (-1, 0, 1)), ((0, 1, -1), (0, 1, -1)), (None, (0, 0, 0))],
)
def test_offsets_are_decoded(tag, offset):
    assert tuple(_offset_of(tag)) == offset


def supercell_bond_lengths(model, supercell, overrides):
    """Distance (minimum image in the supercell) of every bond of the
    lattice built by the adapter, grouped by coupling."""
    names = list(model.bonds)
    index = {name: float(k + 1) for k, name in enumerate(names)}
    _, bonds, j_vals, _ = magnetic_model_to_ising(
        model, supercell=supercell, J_overrides={n: index[n] for n in names}
    )
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lattice = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    n = len(pos)
    cells = list(itertools.product(*(range(s) for s in supercell)))
    big = np.array(supercell)[:, None] * lattice

    def site(idx):
        cell, atom = divmod(int(idx), n)
        return pos[atom] + np.array(cells[cell]) @ lattice

    lengths = {}
    for (a, b), j in zip(bonds, j_vals):
        frac = np.linalg.solve(big.T, site(b) - site(a))
        frac -= np.round(frac)
        lengths.setdefault(names[int(j) - 1], []).append(np.linalg.norm(big.T @ frac))
    return lengths


@pytest.mark.parametrize("supercell", [(2, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("tuple_offsets", [False, True])
def test_lattice_of_a_ring(supercell, tuple_offsets):
    model = ring(6, (1, 2), tuple_offsets)
    for name, lengths in supercell_bond_lengths(model, supercell, None).items():
        np.testing.assert_allclose(lengths, model.bonds[name]["distance"], atol=1e-9)


def test_lattice_of_a_real_model():
    """Before the fix, 144 of 384 bonds of this lattice joined sites 6.6 A
    apart instead of 2.9 A: bonds to neighbouring cells ended up inside
    the same cell."""
    model = magnetic_model_from_file(str(EXAMPLES / "cromita_ortogonal.cif"))
    model.generate_bonds(0.02, [[0, 3.5]])
    for name, lengths in supercell_bond_lengths(model, (2, 2, 2), None).items():
        np.testing.assert_allclose(lengths, model.bonds[name]["distance"], atol=0.02)


@pytest.mark.parametrize("supercell", [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def test_energy_convention_matches_the_fit(supercell):
    """E = E0 - sum J s_i s_j, in the fit and in the simulation."""
    model = ring(6, (1, 2, 3))
    couplings = np.array([-0.4, 0.25, 0.1])
    names = list(model.bonds)
    _, bonds, j_vals, _ = magnetic_model_to_ising(
        model, supercell=supercell, J_overrides=dict(zip(names, couplings))
    )
    cells = int(np.prod(supercell))
    for conf in itertools.product([0, 1], repeat=6):
        fitted = model.coefficient_matrix([list(conf)])[0][:-1].dot(couplings)
        spins = np.tile(1 - 2 * np.array(conf), cells).astype(np.int8)
        engine = IsingMonteCarlo(spins, bonds, j_vals, E0=0.0, seed=1)
        assert engine.energy == pytest.approx(cells * fitted, abs=1e-12)


def test_isolated_last_site_is_kept():
    """The number of sites comes from the model, not from the bonds."""
    model = MagneticModel(
        [np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0])],
        [np.array([2.0, 0, 0]), np.array([0, 9.0, 0]), np.array([0, 0, 9.0])],
        bond_lists=[[(0, 1, ".")]],
        bond_distances=[1.0],
    )
    spins, _, _, _ = magnetic_model_to_ising(model, (2, 1, 1), J_overrides={"J0": 1.0})
    assert len(spins) == 6


def test_couplings_without_value_are_an_error():
    model = ring(4, (1,))
    with pytest.raises(ValueError, match="no value"):
        magnetic_model_to_ising(model, (1, 1, 1))
    spins, _, _, _ = magnetic_model_to_ising(model, (1, 1, 1), J_overrides={"J0": 0.5})
    assert len(spins) == 4


@pytest.mark.parametrize("method", ["exact", "montecarlo", "ellipse"])
def test_compute_couplings_stores_the_values(method):
    model = ring(6, (1, 2))
    couplings = np.array([0.3, -0.1])
    confs = [[0] + list(c) for c in itertools.product([0, 1], repeat=5)]
    energies = model.coefficient_matrix(confs).dot(np.r_[couplings, -7.0])
    js, d_js, _, _ = model.compute_couplings(confs, energies, 1e-3, method=method, rng=0)
    for name, j in zip(model.bonds, js):
        assert model.bonds[name]["value"] == pytest.approx(j)
    np.testing.assert_allclose(
        [model.bonds[n]["value"] for n in model.bonds], couplings, atol=1e-3
    )
    # The stored values can be simulated directly.
    magnetic_model_to_ising(model, (2, 1, 1))


def test_undetermined_couplings_are_stored_as_nan():
    model = ring(6, (1, 2))
    model.compute_couplings([[0] * 6, [0] * 6], [1.0, 1.0], 0.01)
    assert all(np.isnan(b["value"]) for b in model.bonds.values())


@pytest.fixture
def tk_root():
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


def test_couplings_editor(tk_root):
    from spectrojotometer.gui.couplings_editor import CouplingsEditor

    changes = []
    editor = CouplingsEditor(tk_root, on_change=lambda n, v: changes.append((n, v)))
    assert editor.get_values() == {}
    bonds = {
        "J0": {"distance": 1.0, "bonds": [(0, 1, ".")], "value": 0.5},
        "J1": {"distance": 2.0, "bonds": [(0, 2, ".")], "value": np.nan},
    }
    editor.set_couplings(bonds, reset_baseline=True)
    assert editor.get_values() is None  # J1 not estimated: no value to simulate
    editor._rows["J1"]["var"].set("-0.25")
    editor._confirm("J1")
    assert changes == [("J1", -0.25)]
    assert editor.get_values() == {"J0": 0.5, "J1": -0.25}
    assert editor.is_modified()
    editor._rows["J0"]["var"].set("abc")
    assert editor.get_values() is None
    editor.reset()
    assert editor._rows["J0"]["var"].get() == "0.5"
    assert editor._rows["J1"]["var"].get() == ""
    assert not editor.is_modified()
    # Without reset_baseline, the values typed by the user are kept.
    editor._rows["J0"]["var"].set("1.5")
    editor.set_couplings(bonds)
    assert editor._rows["J0"]["var"].get() == "1.5"
    editor.set_couplings({})
    assert editor._rows == {}


def test_panel_simulates_with_the_values_of_the_editor(tk_root, monkeypatch):
    """The values of the editor used to be read and then ignored."""
    from spectrojotometer.gui import monte_carlo_panel

    model = ring(4, (1,))
    model.bonds["J0"]["value"] = 0.5
    calls = []

    def adapter(model, supercell, J_overrides=None):
        calls.append(J_overrides)
        raise RuntimeError("stop here")

    monkeypatch.setattr(monte_carlo_panel.messagebox, "showerror", lambda *a, **k: None)
    panel = monte_carlo_panel.MonteCarloPanel(tk_root, get_model=lambda: model, adapter=adapter)
    panel.refresh_couplings(model)
    panel.editor._rows["J0"]["var"].set("-1.25")
    panel._on_run()
    assert calls == [{"J0": -1.25}]


@pytest.mark.parametrize(
    "equilibrated, temperatures, kwargs",
    [(True, (0.5, 4.0), dict(n_equil=2000, n_sweeps=5000)),
     (False, (0.05, 0.06), dict(n_equil=0, n_sweeps=1))],
)
def test_replica_warning(equilibrated, temperatures, kwargs):
    """Metropolis with several replicas warns when they disagree at the same
    temperature (it used to warn always: it compared the energies of
    different temperatures)."""
    import warnings

    from spectrojotometer.monte_carlo import run_ising_mc

    model = ring(4, (1,))
    spins, bonds, j_vals, _ = magnetic_model_to_ising(model, (3, 1, 1), J_overrides={"J0": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_ising_mc(spins, bonds, j_vals, 0.0, *temperatures, 3,
                     algorithm="metropolis", n_restarts=4, seed=3, **kwargs)
    warned = any("replicas" in str(w.message) for w in caught)
    assert warned is not equilibrated


def test_algorithms_agree_with_the_exact_energy():
    """Wolff (with the gauge transform) and Metropolis, on an unfrustrated
    antiferromagnetic ring of 12 sites, against the exact mean energy."""
    from spectrojotometer.monte_carlo import run_ising_mc

    model = ring(4, (1,))
    spins, bonds, j_vals, _ = magnetic_model_to_ising(model, (3, 1, 1), J_overrides={"J0": -1.0})
    n = len(spins)
    temps = np.geomspace(0.5, 4.0, 3)
    confs = np.array(list(itertools.product([1, -1], repeat=n)))
    energies = -(confs[:, bonds[:, 0]] * confs[:, bonds[:, 1]] * j_vals).sum(axis=1)
    exact = [
        np.sum(energies * np.exp(-(energies - energies.min()) / t))
        / np.sum(np.exp(-(energies - energies.min()) / t)) / n
        for t in temps
    ]
    for algorithm in ("wolff", "metropolis"):
        res = run_ising_mc(spins.copy(), bonds, j_vals, 0.0, 0.5, 4.0, 3,
                           n_equil=1000, n_sweeps=10000, algorithm=algorithm, seed=1)
        np.testing.assert_allclose(res["E"], exact, atol=0.02, err_msg=algorithm)


def visible_in(widget, container):
    """Whether widget is entirely within the visible area of container."""
    top = widget.winfo_rooty() - container.winfo_rooty()
    return 0 <= top and top + widget.winfo_height() <= container.winfo_height()


def test_panel_controls_scroll_in_a_short_window(tk_root):
    """In a short window, the last controls (Run, Save) are reachable by
    scrolling; in a tall one, there is no scrollbar."""
    from spectrojotometer.gui.monte_carlo_panel import MonteCarloPanel

    tk_root.deiconify()
    panel = MonteCarloPanel(tk_root)
    panel.pack(fill="both", expand=True)
    panel.refresh_couplings(ring(6, (1, 2, 3)))  # a few rows in the editor
    tk_root.geometry("1000x300")
    tk_root.update()
    controls = panel.controls
    assert controls.needs_scrolling()
    assert controls.scrollbar.winfo_ismapped()
    assert not visible_in(panel.btn_run, controls.canvas)

    controls.canvas.yview_moveto(1.0)
    tk_root.update()
    assert visible_in(panel.btn_run, controls.canvas)
    assert visible_in(panel.btn_save, controls.canvas)

    # The mouse wheel scrolls up and down (X11 and Windows/macOS events).
    class Wheel:
        def __init__(self, num=None, delta=0):
            self.num, self.delta = num, delta

    top_before = controls.canvas.yview()[0]
    controls._on_wheel(Wheel(num=4))
    controls._on_wheel(Wheel(delta=120))
    tk_root.update()
    assert controls.canvas.yview()[0] < top_before

    tk_root.geometry("1000x1400")
    tk_root.update()
    assert not controls.needs_scrolling()
    assert not controls.scrollbar.winfo_ismapped()
    assert visible_in(panel.btn_run, controls.canvas)
