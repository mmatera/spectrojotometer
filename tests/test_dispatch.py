"""
Tests para spectrojotometer.model_io.magnetic_model_from_file: el
despachador que decide, según la extensión del archivo, si delegar en
magnetic_model_from_cif o en magnetic_model_from_wk2_struct.
"""
import shutil

import numpy as np
import pytest

from spectrojotometer.model_io import (
    magnetic_model_from_cif,
    magnetic_model_from_file,
    magnetic_model_from_wk2_struct,
)


def test_dispatch_to_cif_matches_direct_call(fixtures_dir):
    path = str(fixtures_dir / "synthetic_two_atom.cif")

    via_dispatch = magnetic_model_from_file(path)
    direct = magnetic_model_from_cif(path)

    np.testing.assert_allclose(
        via_dispatch.site_properties["coord_atomos"],
        direct.site_properties["coord_atomos"],
    )
    assert via_dispatch.bonds.keys() == direct.bonds.keys()


def test_dispatch_to_struct_matches_direct_call(fixtures_dir):
    path = str(fixtures_dir / "synthetic_single_atom.struct")

    via_dispatch = magnetic_model_from_file(path, magnetic_atoms=("Cu",))
    direct = magnetic_model_from_wk2_struct(path, magnetic_atoms=("Cu",))

    np.testing.assert_allclose(
        via_dispatch.site_properties["coord_atomos"],
        direct.site_properties["coord_atomos"],
    )


@pytest.mark.parametrize(
    "suffix", [".cif", ".CIF"], ids=["lowercase-ext", "uppercase-ext"]
)
def test_cif_extension_is_case_insensitive(fixtures_dir, tmp_path, suffix):
    src = fixtures_dir / "synthetic_two_atom.cif"
    dst = tmp_path / ("copy" + suffix)
    shutil.copyfile(src, dst)

    model = magnetic_model_from_file(str(dst))
    assert model.site_properties["magnetic_species"] == ["Cu", "Cu"]


@pytest.mark.parametrize(
    "suffix", [".struct", ".STRUCT"], ids=["lowercase-ext", "uppercase-ext"]
)
def test_struct_extension_is_case_insensitive(fixtures_dir, tmp_path, suffix):
    src = fixtures_dir / "synthetic_single_atom.struct"
    dst = tmp_path / ("copy" + suffix)
    shutil.copyfile(src, dst)

    model = magnetic_model_from_file(str(dst), magnetic_atoms=("Cu",))
    assert model.site_properties["magnetic_species"] == ["Cu"]


def test_unknown_extension_returns_sentinel_without_raising(tmp_path):
    """
    Comportamiento actual documentado: una extensión desconocida no
    levanta una excepción, sólo loguea un error y devuelve -1.
    """
    dst = tmp_path / "model.xyz"
    dst.write_text("not a real structure file")

    result = magnetic_model_from_file(str(dst))
    assert result == -1


def test_primitive_cell_true_is_ignored_with_warning_for_struct(
    fixtures_dir, caplog
):
    with caplog.at_level("WARNING"):
        model = magnetic_model_from_file(
            str(fixtures_dir / "synthetic_single_atom.struct"),
            magnetic_atoms=("Cu",),
            primitive_cell=True,
        )
    assert any(
        "primitive_cell" in record.message for record in caplog.records
    )
    # A pesar de pedir primitive_cell=True, el modelo se construye
    # igual (el parámetro simplemente se ignora para .struct).
    assert model.site_properties["magnetic_species"] == ["Cu"]


class TestDefaultMagneticAtomsInconsistencies:
    """
    Las tres funciones de este módulo definen, cada una por su lado,
    un default distinto para `magnetic_atoms`:

    - magnetic_model_from_cif:         ... Cu, V            (sin Cr, sin Ti)
    - magnetic_model_from_wk2_struct:  ... V                (sin Cu, sin Cr, sin Ti)
    - magnetic_model_from_file:        ... Cu, V, Ti, Cr     (superconjunto de ambas)

    Esto significa que un CIF con, por ejemplo, átomos de Cr sólo se
    puede cargar con sus valores por default a través del
    despachador `magnetic_model_from_file`, pero falla si se llama
    directamente a `magnetic_model_from_cif` sin pasar
    `magnetic_atoms` explícitamente. Estos tests documentan ese
    comportamiento para que un cambio futuro en los defaults sea una
    decisión consciente y no una regresión silenciosa.
    """

    def test_cif_direct_call_default_excludes_cr(self, examples_dir):
        with pytest.raises(ValueError):
            magnetic_model_from_cif(str(examples_dir / "h2o.cif"))

    def test_dispatch_default_includes_cr(self, examples_dir):
        model = magnetic_model_from_file(str(examples_dir / "cromita_ortogonal.cif"))
        assert "Cr" in model.site_properties["magnetic_species"]

    def test_struct_direct_call_default_excludes_zn(self, fixtures_dir):
        with pytest.raises(ValueError):
            magnetic_model_from_wk2_struct(
                str(fixtures_dir / "synthetic_nm_single_atom.struct")
            )

    def test_dispatch_default_includes_cu_for_struct(self, fixtures_dir):
        model = magnetic_model_from_file(
            str(fixtures_dir / "synthetic_single_atom.struct")
        )
        assert model.site_properties["magnetic_species"] == ["Cu"]
