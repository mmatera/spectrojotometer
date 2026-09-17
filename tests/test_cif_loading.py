"""
Tests para spectrojotometer.model_io.magnetic_model_from_cif.
"""
import numpy as np
import pytest

from spectrojotometer.model_io import magnetic_model_from_cif

from .conftest import all_bond_lengths

# Especies magnéticas usadas en cromita_ortogonal.cif (Cr). Nótese que
# NO forma parte del default de magnetic_model_from_cif -- ver
# test_default_magnetic_atoms_inconsistencies.py para el detalle.
CR_INCLUSIVE_ATOMS = (
    "Mn", "Fe", "Co", "Ni", "Dy", "Tb", "Eu", "Cu", "V", "Cr",
)


# ---------------------------------------------------------------------
# Fixture sintética: geometría mínima y controlada
# ---------------------------------------------------------------------

def test_synthetic_two_atom_positions(fixtures_dir):
    """Las posiciones cartesianas deben ser (frac . bravais_vectors)."""
    model = magnetic_model_from_cif(str(fixtures_dir / "synthetic_two_atom.cif"))

    positions = model.site_properties["coord_atomos"]
    species = model.site_properties["magnetic_species"]

    assert species == ["Cu", "Cu"]
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(positions[1], [1.0, 0.0, 0.0], atol=1e-8)


def test_synthetic_two_atom_bravais_vectors(fixtures_dir):
    model = magnetic_model_from_cif(str(fixtures_dir / "synthetic_two_atom.cif"))
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])

    np.testing.assert_allclose(
        bravais_vectors, np.diag([10.0, 10.0, 10.0]), atol=1e-6
    )


def test_synthetic_two_atom_bond_matches_declared_distance(fixtures_dir):
    """
    La fixture fue construida a mano para que la distancia real entre
    Cu1 y Cu2 sea exactamente 1.0, igual a la declarada en el CIF.
    """
    model = magnetic_model_from_cif(str(fixtures_dir / "synthetic_two_atom.cif"))

    assert set(model.bonds) == {"J0"}
    assert model.bonds["J0"]["distance"] == pytest.approx(1.0)

    lengths = all_bond_lengths(model, "J0")
    assert len(lengths) == 1
    assert lengths[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# examples/example1.cif: caso real sin simetrías no triviales
# ---------------------------------------------------------------------

def test_example1_atom_count_and_species(examples_dir):
    model = magnetic_model_from_cif(str(examples_dir / "example1.cif"))

    assert len(model.site_properties["coord_atomos"]) == 3
    assert model.site_properties["magnetic_species"] == ["Cu", "Cu", "Cu"]


def test_example1_bonds_are_read(examples_dir):
    model = magnetic_model_from_cif(str(examples_dir / "example1.cif"))

    # El archivo declara dos enlaces Cu-Cu, ambos etiquetados "J0".
    assert set(model.bonds) == {"J0"}
    assert len(model.bonds["J0"]["bonds"]) == 2
    # El valor de "distance" es el que aparece literalmente en el CIF;
    # no se recalcula a partir de las posiciones (ver
    # test_geometry_consistency.py para por qué no debe asumirse que
    # coincide con la distancia real en este archivo de ejemplo).
    assert model.bonds["J0"]["distance"] == pytest.approx(2.9386)


def test_example1_magnetic_atoms_filter_excludes_non_magnetic(examples_dir):
    """
    If the structure does not contain magnetic atoms, no magnetic site
    must be loaded.
    """
    
    model =  magnetic_model_from_cif(
            str(examples_dir / "example1.cif"), magnetic_atoms=("Fe",)
        )
    assert len(model.site_properties["coord_atomos"])==0


# ---------------------------------------------------------------------
# examples/cromita_ortogonal.cif: simetrías de centrado FCC
# ---------------------------------------------------------------------

def test_cromita_expands_atoms_with_symmetries(examples_dir):
    """
    4 átomos en la unidad asimétrica x 4 operadores de centrado FCC
    deben producir 16 posiciones magnéticas, todas distintas.
    """
    model = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    positions = model.site_properties["coord_atomos"]

    assert len(positions) == 16
    assert model.site_properties["magnetic_species"] == ["Cr"] * 16

    # Ninguna posición debería repetirse (dentro de tolerancia numérica).
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert np.linalg.norm(positions[i] - positions[j]) > 1e-4


def test_cromita_primitive_cell_keeps_asymmetric_unit(examples_dir):
    """
    Con primitive_cell=True los átomos NO se expanden: deben quedar
    exactamente los 4 de `_atom_site`, en el mismo orden y con las
    mismas coordenadas cartesianas que en el modelo expandido.
    """
    expanded = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    primitive = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"),
        magnetic_atoms=CR_INCLUSIVE_ATOMS,
        primitive_cell=True,
    )

    assert len(primitive.site_properties["coord_atomos"]) == 4
    np.testing.assert_allclose(
        primitive.site_properties["coord_atomos"],
        expanded.site_properties["coord_atomos"][:4],
        atol=1e-6,
    )


def test_cromita_primitive_cell_volume_matches_centering_order(examples_dir):
    """
    El volumen de la celda primitiva de una red centrada F debe ser
    1/4 del volumen de la celda convencional (4 traslaciones de
    centrado => 4 átomos por celda convencional por cada átomo de la
    celda primitiva).
    """
    conventional = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    primitive = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"),
        magnetic_atoms=CR_INCLUSIVE_ATOMS,
        primitive_cell=True,
    )

    vol_conventional = abs(
        np.linalg.det(np.array(conventional.lattice_properties["bravais_vectors"]))
    )
    vol_primitive = abs(
        np.linalg.det(np.array(primitive.lattice_properties["bravais_vectors"]))
    )

    assert vol_conventional / vol_primitive == pytest.approx(4.0, rel=1e-6)


def test_cromita_without_cr_in_magnetic_atoms_returns_empty_model(examples_dir):
    """
    Con el fix "handle files without magnetic atoms", si
    `magnetic_atoms` no incluye ninguna especie presente en el
    archivo, el modelo resultante queda vacío en lugar de fallar.
    `h2o.cif` sólo tiene H y O, ninguno magnético por default.
    """
    model = magnetic_model_from_cif(str(examples_dir / "h2o.cif"))
    assert model.site_properties["coord_atomos"] == []
    assert model.site_properties["magnetic_species"] == []


# ---------------------------------------------------------------------
# CIFs "estilo pymatgen": columna de índice antepuesta a la simetría
# entre comillas, y bloques `loop_` consecutivos sin línea en blanco
# entre ellos. Antes rompía cif_read_loop_symmetries con un
# ValueError al intentar convertir "'x," a float, porque la fila
# `1  'x, y, z'` se tokenizaba con un simple `.split()` que no
# respeta las comillas.
# ---------------------------------------------------------------------

def test_h2o_loads_despite_quoted_symmetry_with_leading_id_column(examples_dir):
    """
    `examples/h2o.cif` declara la simetría con
    `_symmetry_equiv_pos_site_id` + `_symmetry_equiv_pos_as_xyz`
    (en vez de `_space_group_symop_operation_xyz` sola), es decir con
    una columna numérica antes del operador entre comillas:

        loop_
         _symmetry_equiv_pos_site_id
         _symmetry_equiv_pos_as_xyz
          1  'x, y, z'
        loop_
         _atom_site_type_symbol
         ...

    Además, no hay una línea en blanco entre el `loop_` de simetrías
    y el siguiente `loop_` de átomos.
    """
    model = magnetic_model_from_cif(str(examples_dir / "h2o.cif"), magnetic_atoms=("O",))

    assert len(model.site_properties["coord_atomos"]) == 12
    assert model.site_properties["magnetic_species"] == ["O"] * 12
    np.testing.assert_allclose(
        model.lattice_properties["bravais_vectors"][2], [0.0, 0.0, 7.142962], atol=1e-4
    )


def test_h2o_atom_loop_labels_are_not_corrupted_by_previous_loop(examples_dir):
    """
    Regresión específica para el bug de "loops consecutivos sin línea
    en blanco": si el `loop_` de átomos quedara mal delimitado, sus
    posiciones fraccionarias (columnas 4, 5 y 6) quedarían corridas o
    directamente ausentes. Comprobamos la posición cartesiana del
    primer átomo de oxígeno contra el valor fraccionario declarado en
    el archivo (0.32664200, 0.0, 0.05565800) multiplicado por los
    vectores de Bravais.
    """
    model = magnetic_model_from_cif(str(examples_dir / "h2o.cif"), magnetic_atoms=("O",))
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])
    expected = np.array([0.326642, 0.0, 0.055658]).dot(bravais_vectors)

    np.testing.assert_allclose(
        model.site_properties["coord_atomos"][0], expected, atol=1e-4
    )


def test_fe2o_fixture_loads_correctly(fixtures_dir):
    """
    Caso concreto que disparó el bug: un CIF generado con pymatgen
    para una estructura "Fe2O" ficticia (H2O con el H reemplazado por
    Fe), en el mismo formato que `h2o.cif`.
    """
    model = magnetic_model_from_cif(str(fixtures_dir / "fe2o_synthetic.cif"))

    assert len(model.site_properties["coord_atomos"]) == 24
    assert model.site_properties["magnetic_species"] == ["Fe"] * 24
    # Celda hexagonal: a == b != c, gamma == 120°.
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])
    np.testing.assert_allclose(np.linalg.norm(bravais_vectors[0]), 7.6035663, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(bravais_vectors[1]), 7.6035663, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(bravais_vectors[2]), 7.142962, atol=1e-4)
