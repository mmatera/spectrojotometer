"""
Every module of the library must be importable in a fresh interpreter.

Running Python with -S skips the site machinery: nothing (pytest, an
editable install, another module of the package) has imported
importlib.util or other modules beforehand, as in a plain installation.
Each module is imported in its own process, so that importing one of them
cannot hide a missing import in another.
"""

import pathlib
import subprocess
import sys
import sysconfig

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "spectrojotometer"


def library_modules():
    """Non-GUI modules (the GUI needs a display to be imported)."""
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if "gui" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)


CODE = """
import sys
assert "importlib.util" not in sys.modules
sys.path[:0] = [{root!r}, {purelib!r}, {platlib!r}]
import {module}
"""


@pytest.mark.parametrize("module", list(library_modules()))
def test_module_imports_in_fresh_interpreter(module):
    code = CODE.format(
        root=str(ROOT),
        purelib=sysconfig.get_paths()["purelib"],
        platlib=sysconfig.get_paths()["platlib"],
        module=module,
    )
    res = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stderr
