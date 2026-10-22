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
    )
    assert res.returncode == 0, res.stderr
    assert "Largest error of a coupling per unit error in the energies" in res.stdout
    assert "inf" not in res.stdout.splitlines()[0]
