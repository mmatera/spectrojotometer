"""
Tests of visualbond's main window. Skipped when there is no display.
"""

import logging
import pathlib

import numpy as np
import pytest

tkinter = pytest.importorskip("tkinter")

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def app(monkeypatch):
    from spectrojotometer.gui import main as gui_main

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display available")
    root.withdraw()
    errors = []
    monkeypatch.setattr(gui_main.messagebox, "showerror", lambda *a, **k: errors.append(a))
    application = gui_main.ApplicationGUI(root)
    application.errors = errors
    yield application
    application.log_panel.close()
    root.destroy()


def load_example(app, monkeypatch):
    from spectrojotometer.gui import main as gui_main

    monkeypatch.setattr(
        gui_main.fdlg, "askopenfilename", lambda **k: str(EXAMPLES / "example1.cif")
    )
    app.open_model()
    assert app.model is not None


def set_energies(app, text):
    app.spinconfigsenerg.delete("1.0", tkinter.END)
    app.spinconfigsenerg.insert("1.0", text)


def results(app):
    return app.resparam.get("1.0", tkinter.END)


def test_constructor_does_not_block(app):
    """The window is built without entering the event loop."""
    assert app.model is None
    assert callable(app.run)


def test_evaluate_without_model_shows_error(app):
    app.evaluate_couplings()
    assert app.errors and "Model was not loaded" in app.errors[-1][1]


@pytest.mark.parametrize("fmt", ["plain", "LaTeX", "Wolfram"])
@pytest.mark.parametrize("method", ["exact", "montecarlo", "ellipse"])
def test_evaluate_couplings(app, monkeypatch, method, fmt):
    load_example(app, monkeypatch)
    set_energies(app, (EXAMPLES / "example1.spin").read_text())
    app.outputformat.set(fmt)
    app.parameters["page3"]["method"].set(method)
    app.print_full_equations()
    app.evaluate_couplings()
    assert "J0" in results(app)
    assert not app.errors


def test_not_enough_energies(app, monkeypatch):
    load_example(app, monkeypatch)
    first_line = [
        line
        for line in (EXAMPLES / "example1.spin").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ][0]
    set_energies(app, first_line)
    app.evaluate_couplings()
    assert app.errors and "not enough" in app.errors[-1][1]
    assert "J0" not in results(app)


def test_incompatible_model_warns_in_the_panel(app, monkeypatch):
    load_example(app, monkeypatch)
    lines = [
        line
        for line in (EXAMPLES / "example1.spin").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    # Shift the energies far beyond the tolerance, alternating the sign.
    shifted = []
    for k, line in enumerate(lines):
        energy, rest = line.split(None, 1)
        shifted.append(f"{float(energy) + (1 if k % 2 else -1)}\t{rest}")
    set_energies(app, "\n".join(shifted))
    app.evaluate_couplings()
    app.log_panel.drain()
    panel = app.status.get("1.0", tkinter.END)
    assert "WARNING" in panel and "incompatible" in panel


def test_verbose_log_toggle(app):
    app.verbose_log.set(True)
    app.toggle_verbose_log()
    assert logging.getLogger("spectrojotometer").level == logging.DEBUG
    app.verbose_log.set(False)
    app.toggle_verbose_log()
    assert logging.getLogger("spectrojotometer").level == logging.INFO


def test_example_energies_are_consistent():
    """Guard for the tests above: the example has enough configurations."""
    lines = [
        line
        for line in (EXAMPLES / "example1.spin").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) >= 2
    assert all(np.isfinite(float(line.split()[0])) for line in lines)


def test_import_configurations_from_another_model(app, monkeypatch):
    """The import window: choose a model, compute the site map, map the
    configurations and send them to the main window."""
    from spectrojotometer.gui import importconfig

    load_example(app, monkeypatch)
    files = iter([str(EXAMPLES / "example1.cif"), str(EXAMPLES / "example1.spin")])
    monkeypatch.setattr(importconfig.fdlg, "askopenfilename", lambda **k: next(files))
    # The window is modal: do not wait for it to be closed.
    monkeypatch.setattr(app.root, "wait_window", lambda *a: None)
    window = importconfig.ImportConfigWindow(app)
    try:
        window.selected_model.set("[other model]")
        window.onmodelselect()
        size = len(app.model.site_properties["coord_atomos"])
        assert window.dictatoms == list(range(size))
        assert window.sitemap.get() == ", ".join(str(i) for i in range(size))

        window.configs_from_file()
        window.map_confs()
        mapped = window.outputconfs.get("1.0", tkinter.END)
        source = (EXAMPLES / "example1.spin").read_text()
        for line in source.splitlines():
            if line.strip() and not line.startswith("#"):
                energy, rest = line.split(None, 1)
                config = rest.split("#")[0].strip()
                assert f"{float(energy)}\t{config}" in mapped

        window.send_to_application()
        assert "From " in app.spinconfigs.get("1.0", tkinter.END)
    finally:
        window.close_window()
