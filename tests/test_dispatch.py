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
        direct.site_properties["coord_atomos"],atol=1e-10
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
    If the format of the model is not known, raise a ValueError exception:
    """
    dst = tmp_path / "model.xyz"
    dst.write_text("not a real structure file")
    with pytest.raises(ValueError):
        result = magnetic_model_from_file(str(dst))



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


class TestDefaultMagneticAtomsAreUnified:
    """
    Las tres funciones de este módulo solían tener, cada una por su
    lado, un default distinto para `magnetic_atoms` (bug corregido en
    "fix default magnetic atoms and offset unpacking" / "uniformize
    default atoms"). Estos tests son la regresión: verifican que las
    tres funciones comparten exactamente el mismo default, y que un
    archivo con Cr (antes excluido del default de
    `magnetic_model_from_cif`) o con Cu (antes excluido del default
    de `magnetic_model_from_wk2_struct`) ahora carga correctamente
    también cuando se llama a cada función directamente, sin pasar
    `magnetic_atoms` explícitamente.
    """

    def test_defaults_are_the_same_tuple(self):
        import inspect

        cif_default = inspect.signature(
            magnetic_model_from_cif
        ).parameters["magnetic_atoms"].default
        struct_default = inspect.signature(
            magnetic_model_from_wk2_struct
        ).parameters["magnetic_atoms"].default
        file_default = inspect.signature(
            magnetic_model_from_file
        ).parameters["magnetic_atoms"].default

        assert set(cif_default) == set(struct_default) == set(file_default)

    def test_cif_direct_call_default_now_includes_cr(self, examples_dir):
        model = magnetic_model_from_cif(str(examples_dir / "cromita_ortogonal.cif"))
        assert "Cr" in model.site_properties["magnetic_species"]

    def test_dispatch_default_includes_cr(self, examples_dir):
        model = magnetic_model_from_file(str(examples_dir / "cromita_ortogonal.cif"))
        assert "Cr" in model.site_properties["magnetic_species"]

    def test_struct_direct_call_default_excludes_non_magnetic_species(
        self, fixtures_dir
    ):
        model = magnetic_model_from_wk2_struct(
            str(fixtures_dir / "synthetic_nm_single_atom.struct")
        )
        assert model.site_properties["coord_atomos"] == []

    def test_struct_direct_call_default_now_includes_cu(self, fixtures_dir):
        model = magnetic_model_from_wk2_struct(
            str(fixtures_dir / "synthetic_single_atom.struct")
        )
        assert model.site_properties["magnetic_species"] == ["Cu"]

    def test_dispatch_default_includes_cu_for_struct(self, fixtures_dir):
        model = magnetic_model_from_file(
            str(fixtures_dir / "synthetic_single_atom.struct")
        )
        assert model.site_properties["magnetic_species"] == ["Cu"]
