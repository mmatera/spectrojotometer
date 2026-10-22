"""
The library reports through ``logging`` and never writes to stdout/stderr
or configures logging by itself; applications attach their own handlers.
"""

import logging
import subprocess
import sys
import threading

import numpy as np
import pytest

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.tools import setup_cli_logging


def two_site_model():
    return MagneticModel(
        [np.zeros(3), np.array([1.0, 0.0, 0.0])],
        [np.array([2.0, 0.0, 0.0])],
        bond_lists=[[(0, 1, (0,))]],
        bond_distances=[1.0],
    )


def test_library_is_silent_and_does_not_configure_logging():
    """Loading a model and fitting it prints nothing, in a fresh process."""
    code = """
import logging
from spectrojotometer.model_io import magnetic_model_from_file, read_spin_configurations_file
model = magnetic_model_from_file("examples/example1.cif")
energs, confs, _ = read_spin_configurations_file("examples/example1.spin", model)
model.compute_couplings(confs, energs, 1e-3)
model.compute_couplings(confs, energs, 1e-3, method="montecarlo", rng=0)
assert logging.getLogger().handlers == [], logging.getLogger().handlers
"""
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""
    assert res.stderr == ""


def test_warning_for_incompatible_model(caplog):
    model = two_site_model()
    with caplog.at_level(logging.WARNING, logger="spectrojotometer"):
        model.compute_couplings([[0, 0], [0, 1], [0, 1]], [0.0, 1.0, -1.0], 0.01)
    assert any(
        r.levelno == logging.WARNING
        and r.name == "spectrojotometer.magnetic_model"
        and "incompatible" in r.getMessage()
        for r in caplog.records
    )


def test_diagnostics_only_at_debug(caplog):
    model = two_site_model()
    with caplog.at_level(logging.INFO, logger="spectrojotometer"):
        model.coefficient_matrix([[0, 0], [0, 1]])
    assert not any("raw coefficient matrix" in r.getMessage() for r in caplog.records)
    with caplog.at_level(logging.DEBUG, logger="spectrojotometer"):
        model.coefficient_matrix([[0, 0], [0, 1]])
    assert any("raw coefficient matrix" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "verbosity, level",
    [(0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG), (5, logging.DEBUG)],
)
def test_setup_cli_logging(verbosity, level):
    package_logger = logging.getLogger("spectrojotometer")
    old_level = package_logger.level
    try:
        setup_cli_logging(verbosity)
        assert package_logger.level == level
    finally:
        package_logger.setLevel(old_level)


def test_cli_output_is_clean():
    """Results on stdout; diagnostics only with -vv, on stderr."""
    cmd = [sys.executable, "bin/evaluate_cc", "examples/example1.cif", "examples/example1.spin"]
    quiet = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stderr == ""
    assert "J0" in quiet.stdout and "raw coefficient" not in quiet.stdout
    verbose = subprocess.run(cmd + ["-vv"], capture_output=True, text=True, timeout=120)
    assert "DEBUG:" in verbose.stderr
    assert verbose.stdout == quiet.stdout


@pytest.fixture
def tk_root():
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


def test_log_panel_shows_messages_from_any_thread(tk_root):
    from tkinter import END, Text

    from spectrojotometer.gui.log_panel import LogPanel

    text = Text(tk_root)
    panel = LogPanel(tk_root, text)
    try:
        log = logging.getLogger("spectrojotometer.test")
        log.info("from the main thread")
        worker = threading.Thread(target=lambda: log.warning("from a worker thread"))
        worker.start()
        worker.join()
        log.debug("hidden detail")
        panel.drain()
        content = text.get("1.0", END)
        assert "from the main thread" in content
        assert "WARNING: from a worker thread" in content
        assert "hidden detail" not in content
        start = text.search("from a worker thread", "1.0")
        assert "warning" in text.tag_names(start)

        panel.set_level(logging.DEBUG)
        log.debug("shown detail")
        panel.drain()
        assert "DEBUG: shown detail" in text.get("1.0", END)
    finally:
        panel.close()
    assert panel.handler not in logging.getLogger("spectrojotometer").handlers
