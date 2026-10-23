"""
Tests para spectrojotometer.model_io.magnetic_model_from_wk2_struct.
"""
import numpy as np

import pytest

from spectrojotometer.model_io import magnetic_model_from_wk2_struct
from spectrojotometer.model_io.struct import BOHR_IN_ANGSTROM as BOHR

# The fixtures do not say unit=ang: their lengths are in bohr, and the
# reader converts them to angstrom.


def test_single_atom_position_and_species(fixtures_dir):
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_single_atom.struct"), magnetic_atoms=("Cu",)
    )

    assert model.site_properties["magnetic_species"] == ["Cu"]
    # Fraccional (0.1, 0.2, 0.3) en una celda cúbica de lado 10.
    np.testing.assert_allclose(
        model.site_properties["coord_atomos"][0], np.array([1.0, 2.0, 3.0]) * BOHR, atol=1e-5
    )


def test_single_atom_bravais_vectors(fixtures_dir):
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_single_atom.struct"), magnetic_atoms=("Cu",)
    )
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])

    np.testing.assert_allclose(
        bravais_vectors, np.diag([10.0, 10.0, 10.0]) * BOHR, atol=1e-4
    )


def test_species_filtering_keeps_only_magnetic_atoms(fixtures_dir):
    """
    El archivo tiene un Mn (magnético, por default) y un O (no
    magnético): sólo el Mn debe sobrevivir al filtro.
    """
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_two_species.struct")
    )

    assert model.site_properties["magnetic_species"] == ["Mn"]
    np.testing.assert_allclose(
        model.site_properties["coord_atomos"][0], np.array([2.0, 0.0, 0.0]) * BOHR, atol=1e-5
    )


def test_magnetic_atoms_filter_can_exclude_everything(fixtures_dir):
    
    model = magnetic_model_from_wk2_struct(
            str(fixtures_dir / "synthetic_two_species.struct"), magnetic_atoms=("Fe",)
        )
    assert len(model.site_properties["magnetic_species"])==0


def test_no_bonds_are_generated_from_struct(fixtures_dir):
    """
    A diferencia de `magnetic_model_from_cif`, el lector de `.struct`
    no construye enlaces: el modelo se crea sin `bond_lists`.
    """
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_single_atom.struct"), magnetic_atoms=("Cu",)
    )
    assert model.bonds == {}


def test_multiplicity_replica_uses_its_own_z_coordinate(fixtures_dir):
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_mult2.struct"), magnetic_atoms=("Cu",)
    )
    positions = model.site_properties["coord_atomos"]

    assert len(positions) == 2
    # Fraccional (0.4, 0.5, 0.6) en una celda cúbica de lado 10.
    np.testing.assert_allclose(positions[1], np.array([4.0, 5.0, 6.0]) * BOHR, atol=1e-4)


def write_struct(path, lattice_type, atoms, cell="  4.000000  5.000000  6.000000 90.000000 90.000000 90.000000",
                 mode="MODE OF CALC=RELA unit=bohr", index_style="positive"):
    """A WIEN2k .struct file; atoms: (name, [(x, y, z), ...])."""
    lines = ["synthetic", f"{lattice_type:<4}LATTICE,NONEQUIV. ATOMS{len(atoms):3d}", mode, cell]
    for k, (name, positions) in enumerate(atoms, start=1):
        index = f"{k:4d}" if index_style == "positive" else f"{-k:4d}"
        x, y, z = positions[0]
        lines.append(f"ATOM{index}: X={x:10.8f} Y={y:10.8f} Z={z:10.8f}")
        lines.append(f"          MULT={len(positions):2d}          ISPLIT= 8")
        for x, y, z in positions[1:]:
            lines.append(f"    {index}: X={x:10.8f} Y={y:10.8f} Z={z:10.8f}")
        lines.append(f"{name:<10} NPT=  781  R0=0.00005000 RMT=    2.0000   Z: 24.0")
        lines.append("LOCAL ROT MATRIX:    1.0000000 0.0000000 0.0000000")
        lines.append("                     0.0000000 1.0000000 0.0000000")
        lines.append("                     0.0000000 0.0000000 1.0000000")
    lines.append("   0      NUMBER OF SYMMETRY OPERATIONS")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def fractional(model):
    lattice = np.array(model.lattice_properties["bravais_vectors"])
    return np.linalg.solve(lattice.T, np.array(model.site_properties["coord_atomos"]).T).T


@pytest.mark.parametrize("index_style", ["positive", "negative"])
def test_both_styles_of_atom_index(tmp_path, index_style):
    """'ATOM   1:' and 'ATOM  -1:' (the first one used to fail)."""
    path = write_struct(tmp_path / "a.struct", "P",
                        [("Cr1", [(0.1, 0.2, 0.3), (0.6, 0.7, 0.8)])], index_style=index_style)
    model = magnetic_model_from_wk2_struct(path)
    np.testing.assert_allclose(fractional(model), [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]], atol=1e-6)


def test_units(tmp_path):
    bohr = magnetic_model_from_wk2_struct(write_struct(tmp_path / "b.struct", "P", [("Cr", [(0, 0, 0)])]))
    ang = magnetic_model_from_wk2_struct(write_struct(
        tmp_path / "a.struct", "P", [("Cr", [(0, 0, 0)])], mode="MODE OF CALC=RELA unit=ang"))
    old = magnetic_model_from_wk2_struct(write_struct(
        tmp_path / "o.struct", "P", [("Cr", [(0, 0, 0)])], mode="             RELA"))
    np.testing.assert_allclose(np.diag(bohr.lattice_properties["bravais_vectors"]), np.array([4, 5, 6]) * BOHR)
    np.testing.assert_allclose(np.diag(ang.lattice_properties["bravais_vectors"]), [4, 5, 6])
    np.testing.assert_allclose(np.diag(old.lattice_properties["bravais_vectors"]), np.array([4, 5, 6]) * BOHR)


def test_lattice_parameters_without_spaces(tmp_path):
    """6F10.6: two wide values are not separated by a space."""
    cell = "100.000000100.000000100.000000 90.000000 90.000000 90.000000"
    model = magnetic_model_from_wk2_struct(write_struct(tmp_path / "w.struct", "P", [("Cr", [(0, 0, 0)])], cell=cell))
    np.testing.assert_allclose(np.diag(model.lattice_properties["bravais_vectors"]), np.full(3, 100 * BOHR))


@pytest.mark.parametrize("lattice_type, sites", [("F", 4), ("B", 2), ("CXY", 2), ("CYZ", 2), ("CXZ", 2), ("P", 1), ("H", 1)])
def test_centred_lattices(tmp_path, lattice_type, sites):
    """Centred lattices list one of the atoms related by the centring."""
    model = magnetic_model_from_wk2_struct(
        write_struct(tmp_path / "c.struct", lattice_type, [("Cr", [(0.1, 0.1, 0.1)])]))
    assert len(model.site_properties["coord_atomos"]) == sites


def test_centring_does_not_duplicate_listed_atoms(tmp_path):
    """If the file lists the centred atoms too, they are not repeated."""
    model = magnetic_model_from_wk2_struct(write_struct(
        tmp_path / "d.struct", "B", [("Cr", [(0.1, 0.1, 0.1), (0.6, 0.6, 0.6)])]))
    assert len(model.site_properties["coord_atomos"]) == 2


@pytest.mark.parametrize("name, species", [("Cr1", "Cr"), ("CR1", "Cr"), ("Cr 2", "Cr"), ("Mn", "Mn"), ("O1", "O"), ("Zn1", "Zn")])
def test_species_from_the_name(tmp_path, name, species):
    model = magnetic_model_from_wk2_struct(
        write_struct(tmp_path / "s.struct", "P", [(name, [(0, 0, 0)])]), magnetic_atoms=(species,))
    assert model.site_properties["magnetic_species"] == [species]


def test_rhombohedral_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="rhombohedral"):
        magnetic_model_from_wk2_struct(write_struct(tmp_path / "r.struct", "R", [("Cr", [(0, 0, 0)])]))
