"""
The commands declared in pyproject.toml point to existing functions, and
the console commands answer to --help.
"""

import importlib
import importlib.metadata
import pathlib
import subprocess
import sys
import tomllib

import pytest

from ._helpers import repo_env

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
CONSOLE = PROJECT["scripts"]
GUI = PROJECT["gui-scripts"]


@pytest.mark.parametrize("command", sorted(CONSOLE) + sorted(GUI))
def test_entry_point_is_callable(command):
    target = CONSOLE.get(command) or GUI[command]
    module_name, function_name = target.split(":")
    if command in GUI:
        pytest.importorskip("tkinter")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, function_name))


@pytest.mark.parametrize("command", sorted(CONSOLE))
def test_console_command_help(command):
    module_name = CONSOLE[command].split(":")[0]
    res = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        env=repo_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "usage:" in res.stdout


def test_version_is_consistent():
    import spectrojotometer

    metadata_version = importlib.metadata.version("spectrojotometer")
    assert spectrojotometer.__version__ == metadata_version


EXAMPLES = ROOT / "examples"


@pytest.mark.parametrize("extra", [[], ["--known", str(EXAMPLES / "example1.spin")]])
def test_optimize_configurations_runs(extra, tmp_path):
    # The command writes a resume file next to the model: use a copy.
    model_file = tmp_path / "example1.cif"
    model_file.write_text((EXAMPLES / "example1.cif").read_text())
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "spectrojotometer.cli.optimize_configurations",
            str(model_file),
            "-",
            "--it",
            "5",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=repo_env(),
    )
    assert res.returncode == 0, res.stderr
    assert "Largest error of a coupling per unit error in the energies" in res.stdout
    assert "inf" not in res.stdout.splitlines()[0]


def run_command(module, *args, cwd):
    return subprocess.run(
        [sys.executable, "-m", f"spectrojotometer.cli.{module}", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd,
        env=repo_env(),
    )


def test_bond_generator_runs(tmp_path):
    out = tmp_path / "bonds.cif"
    res = run_command(
        "bond_generator", EXAMPLES / "example1.cif", out, "--rmin", 2, "--rmax", 4,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    from spectrojotometer.model_io import magnetic_model_from_file

    model = magnetic_model_from_file(str(out))
    assert len(model.bonds) >= 2
    for name in model.bonds:
        assert f"{name}: distance" in res.stdout


@pytest.mark.parametrize("fmt", ["plain", "latex", "wolfram"])
def test_print_equations_runs(fmt, tmp_path):
    res = run_command(
        "print_equations", EXAMPLES / "example1.cif", EXAMPLES / "example1.spin",
        "--format", fmt, cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "Equations" in res.stdout and "J0" in res.stdout


def test_map_configs_runs(tmp_path):
    out = tmp_path / "mapped.spin"
    res = run_command(
        "map_configs", EXAMPLES / "example1.cif", EXAMPLES / "example1.cif",
        EXAMPLES / "example1.spin", out, cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    from spectrojotometer.model_io import (
        magnetic_model_from_file,
        read_spin_configurations_file,
    )

    model = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    original = read_spin_configurations_file(str(EXAMPLES / "example1.spin"), model)
    mapped = read_spin_configurations_file(str(out), model)
    assert mapped[0] == original[0] and mapped[1] == original[1]


def test_map_configs_rejects_incompatible_models(tmp_path):
    res = run_command(
        "map_configs", EXAMPLES / "example1.cif", EXAMPLES / "cromita_ortogonal.cif",
        EXAMPLES / "example1.spin", "-", cwd=tmp_path,
    )
    assert res.returncode == 1
    assert "no counterpart" in res.stderr


def test_documented_workflow(tmp_path):
    """The workflow of the README: bond_generator, then optimize-configurations
    alone, with --start and with --known."""
    (tmp_path / "example1.cif").write_text((EXAMPLES / "example1.cif").read_text())
    steps = [
        ("bond_generator", "--rmin", 0, "--rmax", 3.1, "--discretization", ".02",
         "example1.cif", "example1-nnn.cif"),
        ("optimize_configurations", "example1-nnn.cif", "optconfs1.out"),
        ("optimize_configurations", "--start", "example1-nnn-resume.tmp",
         "example1-nnn.cif", "optconfs1.out"),
        ("optimize_configurations", "--known", "optconfs1.out",
         "example1-nnn.cif", "optconfs2.out"),
    ]
    for module, *args in steps:
        res = run_command(module, *args, cwd=tmp_path)
        assert res.returncode == 0, (module, args, res.stderr)
    from spectrojotometer.model_io import magnetic_model_from_file

    model = magnetic_model_from_file(str(tmp_path / "example1-nnn.cif"))
    assert list(model.bonds) == ["J0", "J1"]
    first_line = (tmp_path / "optconfs1.out").read_text().splitlines()[0]
    assert first_line.endswith(": 1.0")
    assert "Together with the known configurations" in (tmp_path / "optconfs2.out").read_text()


def test_map_configs_skips_non_periodic_configurations(tmp_path):
    """From a supercell onto its cell, only the periodic configurations are
    mapped; the others are written as comments, with a warning."""
    import numpy as np

    from spectrojotometer.magnetic_model import MagneticModel
    from spectrojotometer.model_io import magnetic_model_from_file

    base = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    pos = np.array(base.site_properties["coord_atomos"], dtype=float)
    lattice = np.array(base.lattice_properties["bravais_vectors"], dtype=float)
    MagneticModel(
        list(pos) + list(pos + lattice[0]),
        [2 * lattice[0], lattice[1], lattice[2]],
        magnetic_species=base.site_properties["magnetic_species"] * 2,
    ).save_cif(str(tmp_path / "double.cif"))
    (tmp_path / "double.spin").write_text(
        "10.0 [1, 0, 0, 1, 0, 0] # periodic\n30.0 [0, 1, 0, 1, 1, 0] # not\n"
    )
    res = run_command(
        "map_configs", tmp_path / "double.cif", EXAMPLES / "example1.cif",
        tmp_path / "double.spin", "-", cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "5.0\t[1, 0, 0]\t# periodic" in res.stdout
    assert "# not periodic with the target cell: 30.0" in res.stdout
    assert "not periodic" in res.stderr
