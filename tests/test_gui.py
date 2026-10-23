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


def set_page2(app, **values):
    for field, value in values.items():
        entry = app.parameters["page2"][field]
        entry.delete(0, tkinter.END)
        entry.insert(0, str(value))


@pytest.mark.parametrize("fmt", ["plain", "LaTeX", "Wolfram"])
def test_optimize_configs_reports_the_error_bound(app, monkeypatch, fmt):
    load_example(app, monkeypatch)
    app.outputformat.set(fmt)
    set_page2(app, **{"Number of configurations": 2, "Iterations": 5, "Bunch size": 2})
    app.optimize_configs()
    text = app.spinconfigs.get("1.0", tkinter.END)
    assert "J|/|" in text and "E| <" in text
    assert not app.errors


def test_optimal_independent_set_reports_the_error_bound(app, monkeypatch):
    load_example(app, monkeypatch)
    app.spinconfigs.insert(tkinter.END, (EXAMPLES / "example1.spin").read_text())
    set_page2(app, **{"Number of configurations": 2})
    app.optimal_independent_set()
    text = app.spinconfigs.get("1.0", tkinter.END)
    assert "Subset of optimal configurations" in text
    assert "J|/|" in text and "sqrt(l)" not in text
    assert not app.errors


def test_monte_carlo_tab_uses_the_estimated_couplings(app, monkeypatch):
    """After estimating the couplings in the main window, the table of the
    Monte Carlo tab shows them."""
    load_example(app, monkeypatch)
    set_energies(app, (EXAMPLES / "example1.spin").read_text())
    app.evaluate_couplings()
    panel = app.pages["page4"]
    panel.refresh_couplings()
    values = panel.editor.get_values()
    for name, bond in app.model.bonds.items():
        assert values[name] == pytest.approx(bond["value"], rel=1e-5)


def test_entering_the_monte_carlo_tab_syncs_the_couplings(app, monkeypatch):
    load_example(app, monkeypatch)
    panel = app.pages["page4"]
    for bond in app.model.bonds.values():
        bond["value"] = 0.75
    app.nb.tab(panel, state="normal")
    app.nb.select(panel)
    app._on_tab_changed()
    assert set(panel.editor.get_values().values()) == {0.75}
    panel._auto_sync.set(False)
    for bond in app.model.bonds.values():
        bond["value"] = -2.0
    app._on_tab_changed()
    assert set(panel.editor.get_values().values()) == {0.75}


def test_import_from_a_supercell(app, monkeypatch, tmp_path):
    """Driven like a user: the model is chosen in the menu, and the buttons
    are pressed. From a 2x1x1 supercell of the current model, periodic
    configurations are imported with half their energy, the others are
    kept as comments, and comments without a space after '#' survive."""
    from spectrojotometer.gui import importconfig
    from spectrojotometer.magnetic_model import MagneticModel
    from spectrojotometer.model_io import magnetic_model_from_file

    base = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    pos = np.array(base.site_properties["coord_atomos"], dtype=float)
    lattice = np.array(base.lattice_properties["bravais_vectors"], dtype=float)
    double = MagneticModel(
        list(pos) + list(pos + lattice[0]),
        [2 * lattice[0], lattice[1], lattice[2]],
        magnetic_species=base.site_properties["magnetic_species"] * 2,
    )
    double.save_cif(str(tmp_path / "double.cif"))
    (tmp_path / "double.spin").write_text(
        "# supercell\n"
        "10.0\t[0, 0, 0, 0, 0, 0]\t#first\n"
        "20.0\t[1, 0, 0, 1, 0, 0]\t# second\n"
        "30.0\t[0, 1, 0, 1, 1, 0]\t# not periodic\n"
    )

    load_example(app, monkeypatch)
    files = iter([str(tmp_path / "double.cif"), str(tmp_path / "double.spin")])
    monkeypatch.setattr(importconfig.fdlg, "askopenfilename", lambda **k: next(files))
    monkeypatch.setattr(app.root, "wait_window", lambda *a: None)
    window = importconfig.ImportConfigWindow(app)

    def button(text):
        stack = [window]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            if isinstance(widget, tkinter.Button) and widget.cget("text") == text:
                return widget
        raise LookupError(text)

    try:
        window.optmodels["menu"].invoke(0)  # "[other model]"
        assert window.sitemap.get() == "0, 1, 2"
        assert window.energy_scale == 0.5
        button("Load Configuration from File").invoke()
        button("Import").invoke()
        button("Import").invoke()  # importing again must not duplicate
        out = window.outputconfs.get("1.0", tkinter.END)
        assert out.count("5.0\t[0, 0, 0]\t# first") == 1
        assert out.count("10.0\t[1, 0, 0]\t# second") == 1
        assert "# skipped, not periodic with the current cell: 30.0" in out
        assert "15.0" not in out  # the non-periodic one was not imported
        button("Send to main application").invoke()
        sent = app.spinconfigs.get("1.0", tkinter.END)
        assert "5.0\t[0, 0, 0]" in sent and "15.0" not in sent
    finally:
        window.close_window()


def test_import_reports_sites_without_counterpart(app, monkeypatch):
    from spectrojotometer.gui import importconfig

    load_example(app, monkeypatch)
    files = iter([str(EXAMPLES / "example1.cif")])
    monkeypatch.setattr(importconfig.fdlg, "askopenfilename", lambda **k: next(files))
    monkeypatch.setattr(app.root, "wait_window", lambda *a: None)
    errors = []
    monkeypatch.setattr(importconfig.messagebox, "showerror", lambda *a: errors.append(a))
    window = importconfig.ImportConfigWindow(app)
    try:
        window.optmodels["menu"].invoke(0)
        window.sitemap.delete(0, tkinter.END)
        window.sitemap.insert(0, "0, -1, 2")  # a site without counterpart
        window.inputconfs.insert("1.0", "1.0 [0, 1, 0]\n")
        window.map_confs()
        assert errors and "no counterpart" in errors[-1][1]
    finally:
        window.close_window()


def drive_import(app, monkeypatch, other_model, spin_text, tmp_path, choose_first):
    """Open the import window, load the configurations (choosing the model
    in the menu first, or not) and import them. Returns the window."""
    from spectrojotometer.gui import importconfig

    spin = tmp_path / "configs.spin"
    spin.write_text(spin_text)
    answers = {"Select file to open": str(other_model), "Select file": str(spin)}
    monkeypatch.setattr(importconfig.fdlg, "askopenfilename", lambda **k: answers[k["title"]])
    monkeypatch.setattr(app.root, "wait_window", lambda *a: None)
    window = importconfig.ImportConfigWindow(app)
    if choose_first:
        window.optmodels["menu"].invoke(0)
    window.configs_from_file()
    window.map_confs()
    return window


def test_import_without_choosing_the_model_first(app, monkeypatch, tmp_path):
    """Loading the configurations before choosing the model asks for it
    (it used to fail on Import, with an empty site map)."""
    load_example(app, monkeypatch)
    window = drive_import(
        app, monkeypatch, EXAMPLES / "example1.cif",
        (EXAMPLES / "example1.spin").read_text(), tmp_path, choose_first=False,
    )
    try:
        assert window.sitemap.get() == "0, 1, 2"
        assert "49022.0027\t[0, 0, 0]" in window.outputconfs.get("1.0", tkinter.END)
    finally:
        window.close_window()


def test_import_from_a_model_in_another_frame(app, monkeypatch, tmp_path):
    """The other model has a different origin: the site map comes from the
    alignment of the structures, and the configurations are imported."""
    from spectrojotometer.magnetic_model import MagneticModel
    from spectrojotometer.model_io import magnetic_model_from_file

    base = magnetic_model_from_file(str(EXAMPLES / "example1.cif"))
    lattice = np.array(base.lattice_properties["bravais_vectors"], dtype=float)
    shift = np.array([0.11, 0.23, 0.07]) @ lattice
    MagneticModel(
        list(np.array(base.site_properties["coord_atomos"], dtype=float) + shift),
        list(lattice),
        magnetic_species=list(base.site_properties["magnetic_species"]),
    ).save_cif(str(tmp_path / "shifted.cif"))
    load_example(app, monkeypatch)
    window = drive_import(
        app, monkeypatch, tmp_path / "shifted.cif",
        "1.0 [0, 1, 0] # a\n2.0 [1, 1, 0] # b\n", tmp_path, choose_first=True,
    )
    try:
        mapping = [int(x) for x in window.sitemap.get().split(",")]
        assert sorted(mapping) == [0, 1, 2]
        out = window.outputconfs.get("1.0", tkinter.END)
        assert "1.0\t" in out and "2.0\t" in out and "skipped" not in out
    finally:
        window.close_window()
