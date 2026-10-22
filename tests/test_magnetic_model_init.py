"""
Behaviour of the MagneticModel constructor: how it normalizes its
arguments and fills in the defaults.
"""

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel

POSITIONS = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]
BRAVAIS = [np.array([3.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 3.0])]


def model(**kwargs):
    return MagneticModel(POSITIONS, BRAVAIS, **kwargs)


@pytest.mark.parametrize(
    "given, expected",
    [(1, 1), (3, 3), (0, 1), (7, 4), (2.9, 2), ([1, 2, 9], [1, 2, 4]), ((0, 1, 3), [1, 1, 3])],
)
def test_supercell_size_is_clamped(given, expected):
    assert model(supercell_size=given).lattice_properties["supercell_size"] == expected


def test_site_defaults():
    m = model()
    assert m.site_properties["magnetic_species"] == ["X", "X", "X"]
    assert m.site_properties["g_lande_factors"] == [2.0, 2.0, 2.0]
    assert m.site_properties["spin_repr"] == [0.5, 0.5, 0.5]
    assert m.lattice_properties["cell_size"] == 3


def test_site_values_from_cif_strings():
    """CIF files give the values as strings, with '.' for 'not given'."""
    m = model(g_lande_factors=["2.1", ".", 1.9], spin_repr=[".", "1.5", 2])
    assert m.site_properties["g_lande_factors"] == [2.1, 2.0, 1.9]
    assert m.site_properties["spin_repr"] == [0.5, 1.5, 2.0]


def test_arguments_are_not_modified():
    g_factors = ["2.1", ".", "1.9"]
    spins = [".", "1.5", "2"]
    model(g_lande_factors=g_factors, spin_repr=spins)
    assert g_factors == ["2.1", ".", "1.9"]
    assert spins == [".", "1.5", "2"]


@pytest.mark.parametrize("kwargs", [{"ranges": 3.5}, {"discretization": 0.1}])
def test_bond_generation_arguments_are_deprecated(kwargs):
    """They never had an effect in the constructor: use generate_bonds."""
    with pytest.warns(DeprecationWarning, match="generate_bonds"):
        m = model(**kwargs)
    assert m.bonds == {}


def test_bonds_from_lists():
    lists = [[(0, 1, (0,))], [(0, 2, (0,))]]
    m = model(bond_lists=lists, bond_distances=[1.0, 2.0])
    assert list(m.bonds) == ["J0", "J1"]
    assert m.bonds["J0"] == {"distance": 1.0, "bonds": lists[0]}
    assert m.bonds["J1"]["distance"] == 2.0


def test_bond_names_and_default_distances():
    lists = [[(0, 1, (0,))], [(0, 2, (0,))]]
    m = model(bond_lists=lists, bond_names=["Ja", "Jb"])
    assert list(m.bonds) == ["Ja", "Jb"]
    assert [b["distance"] for b in m.bonds.values()] == [0, 0]


def test_bond_distances_must_match_bond_lists():
    with pytest.raises(ValueError, match="same length"):
        model(bond_lists=[[(0, 1, (0,))]], bond_distances=[1.0, 2.0])


def test_no_bonds():
    assert model().bonds == {}


def test_symmetries_and_space_group():
    m = model(symmetries=({"op": 1},), space_group_symbol="P 1")
    assert m.lattice_properties["symmetries"] == [{"op": 1}]
    assert m.lattice_properties["space_group_symbol"] == "P 1"
    assert model().lattice_properties["symmetries"] == []


def test_coefficient_matrix_centered():
    m = model(bond_lists=[[(0, 1, (0,))], [(0, 2, (0,))]], bond_distances=[1.0, 2.0])
    confs = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    raw = m.coefficient_matrix(confs)
    centered = m.coefficient_matrix(confs, centered=True)
    np.testing.assert_allclose(centered[:, :-1], raw[:, :-1] - raw[:, :-1].mean(axis=0))
    np.testing.assert_allclose(centered[:, -1], 1.0)
    np.testing.assert_allclose(m.coefficient_matrix(confs, True), centered)


def test_coefficient_matrix_normalizar_is_deprecated():
    m = model(bond_lists=[[(0, 1, (0,))]], bond_distances=[1.0])
    confs = [[0, 0, 0], [0, 1, 0]]
    with pytest.warns(DeprecationWarning, match="centered"):
        old = m.coefficient_matrix(confs, normalizar=True)
    np.testing.assert_allclose(old, m.coefficient_matrix(confs, centered=True))


def test_remove_bond_by_name():
    m = model(bond_lists=[[(0, 1, (0,))], [(0, 2, (0,))]])
    m.remove_bond_by_name("J0")
    assert list(m.bonds) == ["J1"]
    with pytest.warns(DeprecationWarning, match="remove_bond_by_name"):
        m.remover_bond_by_name("J1")
    assert m.bonds == {}
