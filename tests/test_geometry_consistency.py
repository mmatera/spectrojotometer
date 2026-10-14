"""
Tests que verifican que los modelos generados a partir de los
distintos formatos de entrada son geométricamente consistentes:

- Un modelo guardado con `save_cif` y vuelto a cargar debe reproducir
  exactamente las posiciones, la red de Bravais y los enlaces.
- Las distancias reales entre átomos (calculadas a partir de las
  posiciones cartesianas) deben ser invariantes ante la forma en la
  que se calculó/declaró el enlace.
- El número de átomos y el volumen de la celda escalan de forma
  consistente entre la descripción convencional y la primitiva de una
  misma estructura.
"""
import numpy as np
import pytest

from spectrojotometer.model_io import magnetic_model_from_cif

from .conftest import all_bond_lengths

CR_INCLUSIVE_ATOMS = (
    "Mn", "Fe", "Co", "Ni", "Dy", "Tb", "Eu", "Cu", "V", "Cr",
)


@pytest.mark.parametrize(
    "filename,magnetic_atoms",
    [
        ("synthetic_two_atom.cif", None),
    ],
)
def test_save_and_reload_cif_preserves_geometry(fixtures_dir, filename, magnetic_atoms):
    kwargs = {} if magnetic_atoms is None else {"magnetic_atoms": magnetic_atoms}
    original = magnetic_model_from_cif(str(fixtures_dir / filename), **kwargs)

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "roundtrip.cif")
        original.save_cif(out_path)
        reloaded = magnetic_model_from_cif(out_path, **kwargs)

    np.testing.assert_allclose(
        reloaded.site_properties["coord_atomos"],
        original.site_properties["coord_atomos"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        reloaded.lattice_properties["bravais_vectors"],
        original.lattice_properties["bravais_vectors"],
        atol=1e-4,
    )
    assert set(reloaded.bonds) == set(original.bonds)
    for name in original.bonds:
        assert reloaded.bonds[name]["distance"] == pytest.approx(
            original.bonds[name]["distance"], abs=1e-4
        )
        assert sorted(reloaded.bonds[name]["bonds"]) == sorted(
            original.bonds[name]["bonds"]
        )


def test_example1_roundtrip_preserves_real_bond_lengths(examples_dir):
    """
    Igual que el test anterior pero sobre un archivo de ejemplo real:
    lo que debe conservarse tras guardar y releer no es sólo la
    "distancia declarada" (`bonds[name]["distance"]`, que es un dato
    tomado literalmente del archivo) sino también la distancia
    *real*, calculada a partir de las posiciones cartesianas.
    """
    import os
    import tempfile

    original = magnetic_model_from_cif(str(examples_dir / "example1.cif"))
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "roundtrip.cif")
        original.save_cif(out_path)
        reloaded = magnetic_model_from_cif(out_path)

    for name in original.bonds:
        original_lengths = sorted(all_bond_lengths(original, name))
        reloaded_lengths = sorted(all_bond_lengths(reloaded, name))
        np.testing.assert_allclose(reloaded_lengths, original_lengths, atol=1e-3)


def test_expanded_and_primitive_models_describe_the_same_lattice(examples_dir):
    """
    Invariante estructural entre la descripción convencional (átomos
    expandidos por simetría) y la primitiva (unidad asimétrica +
    vectores primitivos) de la MISMA estructura:

    (nro. átomos convencional) / (nro. átomos primitiva)
        ==
    (volumen convencional) / (volumen primitiva)

    porque ambas relaciones miden lo mismo: el orden del grupo de
    centrado (4 para F).
    """
    conventional = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    primitive = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"),
        magnetic_atoms=CR_INCLUSIVE_ATOMS,
        primitive_cell=True,
    )

    n_conventional = len(conventional.site_properties["coord_atomos"])
    n_primitive = len(primitive.site_properties["coord_atomos"])

    vol_conventional = abs(
        np.linalg.det(np.array(conventional.lattice_properties["bravais_vectors"]))
    )
    vol_primitive = abs(
        np.linalg.det(np.array(primitive.lattice_properties["bravais_vectors"]))
    )

    assert n_conventional / n_primitive == pytest.approx(
        vol_conventional / vol_primitive, rel=1e-6
    )


def test_atom_density_is_conserved_between_conventional_and_primitive(examples_dir):
    """
    Otra forma de mirar el mismo invariante: la densidad de átomos
    magnéticos por unidad de volumen debe ser igual en ambas
    descripciones (es la misma estructura física).
    """
    conventional = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"), magnetic_atoms=CR_INCLUSIVE_ATOMS
    )
    primitive = magnetic_model_from_cif(
        str(examples_dir / "cromita_ortogonal.cif"),
        magnetic_atoms=CR_INCLUSIVE_ATOMS,
        primitive_cell=True,
    )

    def density(model):
        n_atoms = len(model.site_properties["coord_atomos"])
        volume = abs(
            np.linalg.det(np.array(model.lattice_properties["bravais_vectors"]))
        )
        return n_atoms / volume

    assert density(conventional) == pytest.approx(density(primitive), rel=1e-6)
