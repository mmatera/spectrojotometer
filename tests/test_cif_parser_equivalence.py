"""
Compara, archivo por archivo, la implementación actual de
`magnetic_model_from_cif` contra el prototipo
`magnetic_model_from_cif_pymatgen`: para cada CIF que ya sabemos leer
correctamente, las dos deben describir la misma geometría física
(mismos átomos, misma celda, mismos enlaces), aunque el orden interno
de los átomos pueda diferir.

Requiere pymatgen instalado; si no está, estos tests se saltean.
"""
import pytest

pymatgen = pytest.importorskip("pymatgen")

from spectrojotometer.model_io import magnetic_model_from_cif
from spectrojotometer.model_io.model_io_pymatgen import magnetic_model_from_cif_pymatgen

from .conftest import assert_models_equivalent

CR_INCLUSIVE_ATOMS = (
    "Mn", "Fe", "Co", "Ni", "Dy", "Tb", "Eu", "Cu", "V", "Cr",
)


@pytest.mark.parametrize(
    "path_fn,kwargs",
    [
        pytest.param(
            lambda ex, fx: ex / "example1.cif", {}, id="example1"
        ),
        pytest.param(
            lambda ex, fx: ex / "cromita_ortogonal.cif",
            {"magnetic_atoms": CR_INCLUSIVE_ATOMS},
            id="cromita-conventional",
        ),
        pytest.param(
            lambda ex, fx: ex / "cromita_ortogonal.cif",
            {"magnetic_atoms": CR_INCLUSIVE_ATOMS, "primitive_cell": True},
            id="cromita-primitive",
        ),
        pytest.param(
            lambda ex, fx: ex / "h2o.cif", {"magnetic_atoms": ("O",)}, id="h2o"
        ),
        pytest.param(
            lambda ex, fx: fx / "fe2o_synthetic.cif", {}, id="fe2o"
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_two_atom.cif", {}, id="synthetic-two-atom"
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_no_blank_between_loops.cif",
            {},
            id="no-blank-between-loops",
        ),
        pytest.param(
            lambda ex, fx: fx / "synthetic_g_factor_spin.cif",
            {},
            id="g-factor-spin",
        ),
    ],
)
def test_pymatgen_prototype_matches_current_implementation(
    examples_dir, fixtures_dir, path_fn, kwargs
):
    path = str(path_fn(examples_dir, fixtures_dir))

    current = magnetic_model_from_cif(path, **kwargs)
    prototype = magnetic_model_from_cif_pymatgen(path, **kwargs)

    assert_models_equivalent(current, prototype)


def test_g_factor_and_spin_survive_in_both_implementations(fixtures_dir):
    """
    `assert_models_equivalent` no mira `g_lande_factors`/`spin_repr`
    (no forman parte de la geometría), así que se chequean aparte acá.
    """
    path = str(fixtures_dir / "synthetic_g_factor_spin.cif")
    current = magnetic_model_from_cif(path)
    prototype = magnetic_model_from_cif_pymatgen(path)

    assert (
        current.site_properties["g_lande_factors"]
        == prototype.site_properties["g_lande_factors"]
        == [2.1, 2.3]
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug preexistente y compartido por las dos implementaciones (el "
        "prototipo reusa cif_read_loop_bonds/generate_atoms_by_symmetries "
        "sin modificar): cuando un enlace se declara con "
        "`_geom_bond_site_symmetry_2` apuntando a un operador de simetría "
        "no-identidad (p.ej. '2_555'), `cif_read_loop_bonds` arma la "
        "etiqueta de la réplica como `<label>_<sym>` con `sym = índice_CIF "
        "- 1`, pero `generate_atoms_by_symmetries` había etiquetado esa "
        "misma réplica como `<label>_<sym + 1>`. La etiqueta no coincide "
        "con ninguna clave de `atomlabels`, así que el enlace se descarta "
        "en silencio (0 bonds en vez de 1). Este test documenta el "
        "comportamiento correcto esperado; falla mientras el bug siga "
        "presente en `model_io.py` (no es algo que el prototipo pueda "
        "arreglar por su cuenta, ya que reusa esa función tal cual)."
    ),
)
@pytest.mark.parametrize(
    "loader",
    [magnetic_model_from_cif, magnetic_model_from_cif_pymatgen],
    ids=["current", "pymatgen-prototype"],
)
def test_bond_via_non_identity_symmetry_is_currently_dropped(fixtures_dir, loader):
    model = loader(str(fixtures_dir / "synthetic_bond_via_symmetry.cif"))

    assert len(model.site_properties["coord_atomos"]) == 2
    assert set(model.bonds) == {"J0"}
    assert len(model.bonds["J0"]["bonds"]) == 1
