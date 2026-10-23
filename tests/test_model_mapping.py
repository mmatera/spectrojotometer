"""
Mapping configurations between models: site_map and map_configurations.
"""

import pathlib

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel, map_configurations, site_map
from spectrojotometer.model_io import magnetic_model_from_file

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def chain(xs, period):
    """Sites at the positions xs along x, with lattice period `period`."""
    return MagneticModel(
        [np.array([float(x), 0.0, 0.0]) for x in xs],
        [np.array([float(period), 0.0, 0.0]), np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 5.0])],
    )


def test_double_cell():
    source = chain([0, 1], 2)
    target = chain([0, 1, 2, 3], 4)
    assert site_map(source, target) == [0, 1, 0, 1]
    confs = [[0, 1], [0, 0]]
    assert map_configurations(source, confs, target) == [[0, 1, 0, 1], [0, 0, 0, 0]]


def test_translation_beyond_the_supercell():
    """Matching is modulo the source lattice, however far the images are."""
    source = chain([0, 1], 2)
    target = chain([20, 21, 22, 23], 4)
    assert site_map(source, target) == [0, 1, 0, 1]


def test_tolerance():
    source = chain([0, 1], 2)
    target = chain([0.05, 1.0], 2)
    assert site_map(source, target, tol=0.1, method="position") == [0, 1]
    assert site_map(source, target, tol=0.01, method="position") == [-1, 1]


def test_sites_without_counterpart():
    source = chain([0, 1], 2)
    target = chain([0, 0.5, 1], 2)
    assert site_map(source, target) == [0, -1, 1]
    with pytest.raises(ValueError, match=r"sites \[1\]"):
        map_configurations(source, [[0, 1]], target)
    assert map_configurations(source, [[1, 1]], target, fill=0) == [[1, 0, 1]]


def test_configuration_length_is_checked():
    source = chain([0, 1], 2)
    with pytest.raises(ValueError, match="has 3 values"):
        map_configurations(source, [[0, 1, 0]], chain([0, 1], 2))


def test_model_onto_itself_is_the_identity():
    model = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    size = len(model.site_properties["coord_atomos"])
    assert site_map(model, model) == list(range(size))
    confs = [[0] * size, [0] + [1] * (size - 1)]
    assert map_configurations(model, confs, model) == confs


def test_parse_spin_configurations_comments():
    """The comment starts at the first '#', with or without a space: its
    digits are not part of the configuration."""
    from spectrojotometer.model_io import parse_spin_configurations

    text = "# header\n1.5\t[0, 1, 0]\t# a\nNAN [0, 1, 1] #10\n2.5 [1, 0, 0]#7\n\n"
    energies, confs, comments = parse_spin_configurations(text)
    assert confs == [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
    assert comments == [" a", "10", "7"]
    assert energies[0] == 1.5 and np.isnan(energies[1])


def test_supercell_onto_cell_needs_periodic_configurations():
    from spectrojotometer.magnetic_model import compatible_configurations

    cell = chain([0, 1], 2)
    double = chain([0, 1, 2, 3], 4)
    confs = [[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
    assert compatible_configurations(double, confs, cell) == [True, False, True]
    with pytest.raises(ValueError, match=r"configurations \[1\] are not periodic"):
        map_configurations(double, confs, cell)
    assert map_configurations(double, [confs[0], confs[2]], cell) == [[0, 1], [1, 1]]
    # From the cell onto its supercell every configuration is representable.
    assert compatible_configurations(cell, [[0, 1], [1, 1]], double) == [True, True]


def test_explicit_site_map():
    source = chain([0, 1], 2)
    target = chain([0, 1], 2)
    assert map_configurations(source, [[0, 1]], target, mapping=[1, 0]) == [[1, 0]]
    with pytest.raises(ValueError, match="2 entries"):
        map_configurations(source, [[0, 1]], chain([0, 1, 2, 3], 4), mapping=[1, 0])
    with pytest.raises(ValueError, match="beyond"):
        map_configurations(source, [[0, 1]], target, mapping=[0, 5])


def moved_copy(model, perm=(1, 2, 0), shift=(0.13, 0.31, 0.07)):
    """The same structure, with the axes permuted and the origin shifted:
    the positions no longer match directly."""
    lattice = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    frac = np.linalg.solve(lattice.T, np.array(model.site_properties["coord_atomos"], dtype=float).T).T
    new_lattice = lattice[list(perm)][:, list(perm)]
    new_frac = (frac[:, list(perm)] + np.array(shift)) % 1
    return MagneticModel(
        list(new_frac @ new_lattice),
        list(new_lattice),
        magnetic_species=list(model.site_properties["magnetic_species"]),
    )


@pytest.fixture(scope="module")
def spinel():
    model = magnetic_model_from_file(str(EXAMPLES / "cromita_ortogonal.cif"))
    model.generate_bonds(0.02, [[0, 3.5]])
    return model


def test_structure_matcher_finds_a_moved_copy(spinel):
    from spectrojotometer.magnetic_model import site_maps

    moved = moved_copy(spinel)
    assert site_map(moved, spinel, method="position") == [-1] * 16
    forward, reverse, used = site_maps(moved, spinel)  # "auto"
    assert used == "structure"
    assert sorted(forward) == list(range(16))
    # A valid alignment: every configuration keeps its pairs of each
    # coupling, so its energy in the target model.
    rng = np.random.default_rng(0)
    confs = [list(rng.integers(0, 2, 16)) for _ in range(300)]
    via_moved = map_configurations(moved, confs, spinel)
    direct = map_configurations(spinel, confs, spinel)
    np.testing.assert_array_equal(
        spinel.coefficient_matrix(via_moved), spinel.coefficient_matrix(direct)
    )


def test_structure_matcher_between_cell_and_supercell():
    """With a different origin, from a cell onto its double and back; the
    check of periodicity uses a single alignment."""
    cell = chain([0, 1], 2)
    double = chain([0.3, 1.3, 2.3, 3.3], 4)  # shifted: no direct match
    assert site_map(cell, double, method="position") == [-1] * 4
    forward = site_map(cell, double)
    assert forward in ([0, 1, 0, 1], [1, 0, 1, 0])
    assert map_configurations(cell, [[0, 1]], double) == [[[0, 1][j] for j in forward]]
    from spectrojotometer.magnetic_model import compatible_configurations

    assert compatible_configurations(double, [[0, 1, 0, 1], [0, 1, 1, 1]], cell) == [True, False]


def test_auto_keeps_the_partial_map_by_positions():
    source = chain([0, 1], 2)
    target = chain([0, 0.5, 1], 2)  # different structures: no match at all
    assert site_map(source, target) == [0, -1, 1]


def distorted_reordered_copy(model, strain=(0.997, 1.0, 1.003), seed=0):
    """The same structure with a small orthorhombic strain and the sites in
    another order, in the same frame: like a slightly distorted phase."""
    lattice = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    frac = np.linalg.solve(lattice.T, np.array(model.site_properties["coord_atomos"], dtype=float).T).T
    order = np.random.default_rng(seed).permutation(len(frac))
    new_lattice = lattice * np.array(strain)[:, None]
    copy = MagneticModel(
        list(frac[order] @ new_lattice),
        list(new_lattice),
        magnetic_species=[model.site_properties["magnetic_species"][i] for i in order],
    )
    copy.generate_bonds(0.02, [[0, 3.5]])
    return copy


def check_equivalent(eq, source, target, confs):
    """Each configuration of the target, in the equivalent model, has the
    coefficients of the corresponding configuration of the source."""
    from spectrojotometer.magnetic_model import site_maps

    forward = site_maps(source, target)[0]
    assert len(eq.site_properties["coord_atomos"]) == len(forward)
    assert list(eq.bonds) == list(source.bonds)
    inverse = {s: t for t, s in enumerate(forward)}
    for conf in confs:
        source_conf = [conf[inverse[s]] for s in range(len(source.site_properties["coord_atomos"]))]
        np.testing.assert_array_equal(
            eq.coefficient_matrix([conf]), source.coefficient_matrix([source_conf])
        )


def test_equivalent_model_of_a_distorted_phase(spinel, tmp_path):
    from spectrojotometer.magnetic_model import equivalent_model

    source = distorted_reordered_copy(spinel)
    eq = equivalent_model(source, spinel)
    rng = np.random.default_rng(0)
    confs = [list(rng.integers(0, 2, 16)) for _ in range(200)]
    check_equivalent(eq, source, spinel, confs)
    # The cell and the positions are the ones of the source (its strain).
    np.testing.assert_allclose(
        np.array(eq.lattice_properties["bravais_vectors"]),
        np.array(source.lattice_properties["bravais_vectors"]),
    )
    # Saved and read back, it is the same model.
    eq.save_cif(str(tmp_path / "eq.cif"))
    back = magnetic_model_from_file(str(tmp_path / "eq.cif"))
    np.testing.assert_allclose(
        np.array(back.site_properties["coord_atomos"]),
        np.array(eq.site_properties["coord_atomos"]),
        atol=1e-4,
    )
    for conf in confs[:50]:
        np.testing.assert_array_equal(back.coefficient_matrix([conf]), eq.coefficient_matrix([conf]))


def test_equivalent_model_in_another_frame(spinel):
    from spectrojotometer.magnetic_model import equivalent_model, site_maps

    source = moved_copy(spinel)
    source.generate_bonds(0.02, [[0, 3.5]])
    assert site_maps(source, spinel)[2] == "structure"
    rng = np.random.default_rng(1)
    check_equivalent(
        equivalent_model(source, spinel), source, spinel,
        [list(rng.integers(0, 2, 16)) for _ in range(200)],
    )


def test_equivalent_model_of_a_supercell():
    from spectrojotometer.magnetic_model import equivalent_model

    cell = chain([0, 1], 2)
    cell.generate_bonds(0.02, [[0, 1.5]])
    double = chain([0, 1, 2, 3], 4)
    eq = equivalent_model(cell, double)
    assert np.allclose(np.array(eq.lattice_properties["bravais_vectors"])[0], [4, 0, 0])
    assert {k: len(v["bonds"]) for k, v in eq.bonds.items()} == {
        k: 2 * len(v["bonds"]) for k, v in cell.bonds.items()
    }
    for conf in ([0, 1], [1, 1], [0, 0]):
        np.testing.assert_array_equal(
            eq.coefficient_matrix([conf * 2])[0][:-1],
            2 * cell.coefficient_matrix([conf])[0][:-1],
        )
    with pytest.raises(ValueError, match="smaller"):
        equivalent_model(double, cell)
