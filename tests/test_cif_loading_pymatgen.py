"""
Tests para spectrojotometer.model_io_pymatgen.magnetic_model_from_cif_pymatgen.

Son deliberadamente independientes de `test_cif_loading.py` (no
comparan contra la implementación actual, la ejercitan sola) para no
quedar ciegos ante un bug que las dos implementaciones compartieran
por reusar la misma función de más bajo nivel -- que es exactamente lo
que pasó con `test_bond_via_non_identity_symmetry_is_currently_dropped`
en `test_cif_parser_equivalence.py`. La comparación lado a lado vive
en ese otro archivo.

Requiere pymatgen instalado; si no está, estos tests se saltean.
"""
import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen")

from spectrojotometer.model_io.model_io_pymatgen import magnetic_model_from_cif_pymatgen

from .conftest import all_bond_lengths

CR_INCLUSIVE_ATOMS = (
    "Mn", "Fe", "Co", "Ni", "Dy", "Tb", "Eu", "Cu", "V", "Cr",
)


# ---------------------------------------------------------------------
# Fixture sintética: geometría mínima y controlada
# ---------------------------------------------------------------------

def test_synthetic_two_atom_positions(fixtures_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(fixtures_dir / "synthetic_two_atom.cif")
    )

    positions = model.site_properties["coord_atomos"]
    species = model.site_properties["magnetic_species"]

    assert species == ["Cu", "Cu"]
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(positions[1], [1.0, 0.0, 0.0], atol=1e-6)


def test_synthetic_two_atom_bravais_vectors(fixtures_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(fixtures_dir / "synthetic_two_atom.cif")
    )
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])
    np.testing.assert_allclose(
        bravais_vectors, np.diag([10.0, 10.0, 10.0]), atol=1e-6
    )


def test_synthetic_two_atom_bond_matches_declared_distance(fixtures_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(fixtures_dir / "synthetic_two_atom.cif")
    )

    assert set(model.bonds) == {"J0"}
    assert model.bonds["J0"]["distance"] == pytest.approx(1.0)
    lengths = all_bond_lengths(model, "J0")
    assert len(lengths) == 1
    assert lengths[0] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------
# examples/example1.cif
# ---------------------------------------------------------------------

def test_example1_atom_count_and_species(examples_dir):
    model = magnetic_model_from_cif_pymatgen(str(examples_dir / "example1.cif"))
    assert len(model.site_properties["coord_atomos"]) == 3
    assert model.site_properties["magnetic_species"] == ["Cu", "Cu", "Cu"]


def test_example1_bonds_are_read(examples_dir):
    model = magnetic_model_from_cif_pymatgen(str(examples_dir / "example1.cif"))
    assert set(model.bonds) == {"J0"}
    assert len(model.bonds["J0"]["bonds"]) == 2
    assert model.bonds["J0"]["distance"] == pytest.approx(2.9386)


# ---------------------------------------------------------------------
# examples/cromita_ortogonal.cif: centrado FCC, resuelto por pymatgen
# a partir del símbolo de grupo espacial (no de la heurística de
# `centering_letter_from_symbol`, que acá ni se usa).
# ---------------------------------------------------------------------

def test_cromita_expands_atoms_via_pymatgen_symops(examples_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    positions = model.site_properties["coord_atomos"]

    assert len(positions) == 16
    assert model.site_properties["magnetic_species"] == ["Cr"] * 16
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert np.linalg.norm(positions[i] - positions[j]) > 1e-4


def test_cromita_primitive_cell_volume_matches_centering_order(examples_dir):
    conventional = magnetic_model_from_cif_pymatgen(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    primitive = magnetic_model_from_cif_pymatgen(
        str(examples_dir / "cromita_ortogonal.cif"),
        magnetic_atoms=CR_INCLUSIVE_ATOMS,
        primitive_cell=True,
    )

    assert len(primitive.site_properties["coord_atomos"]) == 4
    vol_conventional = abs(
        np.linalg.det(np.array(conventional.lattice_properties["bravais_vectors"]))
    )
    vol_primitive = abs(
        np.linalg.det(np.array(primitive.lattice_properties["bravais_vectors"]))
    )
    assert vol_conventional / vol_primitive == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------
# Los dos archivos que rompían la implementación línea-por-línea
# ---------------------------------------------------------------------

def test_h2o_loads(examples_dir):
    """
    `examples/h2o.cif` es justamente el caso que motivó todo esto:
    columna de índice antepuesta al operador de simetría entre
    comillas, y loops consecutivos sin línea en blanco. A pymatgen no
    le afecta ninguna de las dos cosas.
    """
    model = magnetic_model_from_cif_pymatgen(
        str(examples_dir / "h2o.cif"), magnetic_atoms=("O",)
    )
    assert len(model.site_properties["coord_atomos"]) == 12
    assert model.site_properties["magnetic_species"] == ["O"] * 12


def test_fe2o_fixture_loads(fixtures_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(fixtures_dir / "fe2o_synthetic.cif")
    )
    assert len(model.site_properties["coord_atomos"]) == 24
    assert model.site_properties["magnetic_species"] == ["Fe"] * 24


# ---------------------------------------------------------------------
# Columnas custom no estándar (_atom_site_g_factor, _atom_site_spin)
# ---------------------------------------------------------------------

def test_g_factor_and_spin_columns_are_preserved(fixtures_dir):
    """
    Estas columnas no son parte del estándar CIF; pymatgen no las
    conoce, pero como el prototipo lee el loop de átomos con la misma
    `cif_read_loop_atoms` de siempre (sólo cambia de dónde vienen
    `labels`/`entries`), se siguen leyendo igual.
    """
    model = magnetic_model_from_cif_pymatgen(
        str(fixtures_dir / "synthetic_g_factor_spin.cif")
    )
    assert model.site_properties["g_lande_factors"] == [2.1, 2.3]


# ---------------------------------------------------------------------
# Filtro de magnetic_atoms sin coincidencias -> modelo vacío
# ---------------------------------------------------------------------

def test_magnetic_atoms_filter_can_exclude_everything(examples_dir):
    model = magnetic_model_from_cif_pymatgen(
        str(examples_dir / "h2o.cif")  # default: sin H ni O
    )
    assert model.site_properties["coord_atomos"] == []
    assert model.site_properties["magnetic_species"] == []
