"""
Tests para spectrojotometer.model_io.magnetic_model_from_wk2_struct.
"""
import numpy as np

from spectrojotometer.model_io import magnetic_model_from_wk2_struct


def test_single_atom_position_and_species(fixtures_dir):
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_single_atom.struct"), magnetic_atoms=("Cu",)
    )

    assert model.site_properties["magnetic_species"] == ["Cu"]
    # Fraccional (0.1, 0.2, 0.3) en una celda cúbica de lado 10.
    np.testing.assert_allclose(
        model.site_properties["coord_atomos"][0], [1.0, 2.0, 3.0], atol=1e-5
    )


def test_single_atom_bravais_vectors(fixtures_dir):
    model = magnetic_model_from_wk2_struct(
        str(fixtures_dir / "synthetic_single_atom.struct"), magnetic_atoms=("Cu",)
    )
    bravais_vectors = np.array(model.lattice_properties["bravais_vectors"])

    np.testing.assert_allclose(
        bravais_vectors, np.diag([10.0, 10.0, 10.0]), atol=1e-4
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
        model.site_properties["coord_atomos"][0], [2.0, 0.0, 0.0], atol=1e-5
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
    print("positions[1]:", positions[1])

    assert len(positions) == 2
    # Fraccional (0.4, 0.5, 0.6) en una celda cúbica de lado 10.
    np.testing.assert_allclose(positions[1], [4.0, 5.0, 6.0], atol=1e-4)
    print("OK")
