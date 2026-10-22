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
