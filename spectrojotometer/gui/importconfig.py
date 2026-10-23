#!/usr/bin/env python3
#  from tkmessagebox import *
"""
Import dialog for spin configurations from another model.

"""
#  from visualbond.dialogs.finddialog import FindDialog
import logging
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    TOP,
    Button,
    Entry,
    Frame,
    Label,
    LabelFrame,
    OptionMenu,
    PanedWindow,
    StringVar,
    Toplevel,
    X,
    messagebox,
)
from tkinter import filedialog as fdlg
from tkinter.scrolledtext import ScrolledText


from spectrojotometer.magnetic_model import (
    compatible_configurations,
    map_configurations,
    site_map,
)
from spectrojotometer.model_io import magnetic_model_from_file, parse_spin_configurations

logger = logging.getLogger(__name__)


class ImportConfigWindow(Toplevel):
    """
    Window for importing configurations.
    """

    def __init__(self, app):
        self.app = app
        Toplevel.__init__(self, app.root)
        #  self.root = Toplevel(app.root)
        self.dictatoms = []
        self.transient(app.root)
        self.model = app.model
        self.configurations = ([], [], [])
        self.models = {}
        self.parameters = {}
        self.energy_scale = 1.0

        controls1 = LabelFrame(self, text="Parameters", padx=5, pady=5)

        row = Frame(controls1)
        Label(row, text="model file:").pack(side=LEFT)
        self.selected_model = StringVar()
        self.selected_model.trace_add("write", self.onmodelselect)
        self.optmodels = OptionMenu(row, self.selected_model, "[other model]")
        self.optmodels.pack(side=LEFT, fill=X)
        self.sitemap = Entry(row)
        self.sitemap.pack(side=RIGHT)
        Label(row, text="site map").pack(side=RIGHT)
        row.pack(side=TOP, fill=X)

        row = Frame(controls1)
        Label(row, text="Length tolerance:").pack(side=LEFT)
        self.tol = Entry(row, validate="key", validatecommand=self.app.vcmdf)
        self.tol.insert(10, 0.1)
        self.tol.pack(side=LEFT)
        row.pack(side=TOP, fill=X)
        controls1.pack(side=TOP, fill=X)

        #  controls2 = Frame(self, padx=5, pady=5)
        controls2 = PanedWindow(self, orient=HORIZONTAL)
        controls2.pack(side=TOP, fill=BOTH, expand=1)
        controls2l = LabelFrame(controls2, text="inputs", padx=5, pady=5)
        self.inputconfs = ScrolledText(controls2l, height=10, width=80, undo="True")
        self.inputconfs.pack()
        self.inputconfs.tag_configure("sel", background="black", foreground="gray")
        buttons = Frame(controls2l)
        Button(
            buttons,
            text="Load Configuration from File",
            command=self.configs_from_file,
        ).pack(side=RIGHT)
        Button(buttons, text="Import", command=self.map_confs).pack(side=LEFT)
        buttons.pack(side=BOTTOM, fill=X)

        #  controls2l.pack(side=LEFT, fill=Y)
        controls2.add(controls2l)
        controls2r = LabelFrame(controls2, text="in main model", padx=5, pady=5)
        self.outputconfs = ScrolledText(controls2r, height=10, width=80)
        self.outputconfs.config(state=DISABLED)
        self.outputconfs.pack()
        self.outputconfs.tag_configure("sel", background="black", foreground="gray")
        #  controls2r.pack(side=RIGHT, fill=Y)
        controls2.add(controls2r)
        #  controls2.pack(side=TOP, fill=X)

        framebts = Frame(self)
        Button(
            framebts,
            text="Send to main application",
            command=self.send_to_application,
        ).pack(side=LEFT)
        Button(framebts, text="Close", command=self.close_window).pack(side=RIGHT)
        framebts.pack(side=BOTTOM, fill=X)
        self.grab_set()
        app.root.wait_window(self)

    def onmodelselect(self, *args):
        """On model select"""
        if self.selected_model.get() == "[other model]":
            filename = fdlg.askopenfilename(
                initialdir=self.app.datafolder + "/",
                title="Select file to open",
                filetypes=(
                    ("cif files", "*.cif"),
                    ("Wien2k struct files", "*.struct"),
                    ("all files", "*.*"),
                ),
            )
            logger.debug("model file: %s", filename)
            if filename == "":
                return
            self.app.datafolder = str(Path(filename).parent)
            newmodel = magnetic_model_from_file(
                filename=filename, primitive_cell=self.app.primitive_cell_var.get()
            )
            self.models[filename] = newmodel
            menu = self.optmodels["menu"]
            menu.delete(0, "end")
            menu.add_command(
                label="[other model]",
                command=lambda value="[other model]": self.selected_model.set(
                    "[other model]"
                ),
            )
            logger.debug("updating menu")
            for key in self.models:
                menu.add_command(
                    label=key,
                    command=lambda value=key: self.selected_model.set(value),
                )
            self.selected_model.set(filename)
            self.update_idletasks()
            #  self.optmodels.set(filename)
        # xxxxxxxxx
        tol = float(self.tol.get())
        model1 = self.models[self.selected_model.get()]
        model2 = self.app.model
        size1 = len(model1.site_properties["coord_atomos"])
        size2 = len(model2.site_properties["coord_atomos"])
        self.energy_scale = float(size2) / float(size1)
        if self.energy_scale < 1.0:
            # Only the configurations periodic with the current cell can be
            # imported; map_confs checks each one.
            logger.info(
                "The cell of the current model is smaller than the one of %s: "
                "only its configurations periodic with the current cell can "
                "be imported.",
                self.selected_model.get(),
            )
        # For each site of the current model, the site of the other model at
        # the same position (-1 if none: map_confs then uses spin 0).
        dictatoms = site_map(model1, model2, tol)
        if -1 in dictatoms:
            logger.warning(
                "Sites %s of the current model have no counterpart in %s; "
                "they will get spin 0.",
                [i for i, j in enumerate(dictatoms) if j < 0],
                self.selected_model.get(),
            )
        self.dictatoms = list(dictatoms)
        self.sitemap.delete(0, END)
        self.sitemap.insert(0, dictatoms.__str__()[1:-1])

    def close_window(self):
        """close the window"""
        self.destroy()

    def send_to_application(self):
        """Sends the new configs to the application."""
        logger.debug("send_to_application")
        self.app.spinconfigs.insert(
            END, "\n\n#  From " + self.selected_model.get() + "\n"
        )
        self.app.spinconfigs.insert(END, self.outputconfs.get(1.0, END))
        self.app.spinconfigs.insert(END, "\n" + 20 * "# " + "\n\n")
        self.app.reload_configs(src_widget=self.app.spinconfigs)
        logger.debug("configurations sent to the application")

    def configs_from_file(self):
        """
        Read configs from a model
        """
        logger.debug("configs from file")
        if self.selected_model.get() == "[other model]":
            self.onmodelselect()
        filename = fdlg.askopenfilename(
            initialdir=self.app.datafolder + "/",
            title="Select file",
            filetypes=(("spin list", "*.spin"), ("all files", "*.*")),
        )
        if filename == "":
            return
        self.app.datafolder = str(Path(filename).parent)
        self.inputconfs.delete(1.0, END)
        with open(filename, "r") as filecfg:
            self.inputconfs.insert(END, "\n" + filecfg.read())
        logger.info("Configurations read from %s", filename)

    def map_confs(self):
        """
        Map the configurations of the other model (inputs) to the current
        one (outputs), with the site map shown (it can be edited).

        Each site of the current model takes the spin of its site in the
        other model, and the energies are scaled by the ratio of the
        numbers of sites. Configurations that the current cell cannot
        represent (not periodic with its lattice) are not imported: they
        are kept as comments, with the reason.
        """
        name = self.selected_model.get()
        if name not in self.models:
            messagebox.showerror("Import", "Choose the model of the configurations first.")
            return
        source, target = self.models[name], self.app.model
        try:
            mapping = [int(x) for x in self.sitemap.get().replace(",", " ").split()]
            tol = float(self.tol.get())
            # Checks the site map once (sites without counterpart, size).
            map_configurations(source, [], target, tol, mapping=mapping)
        except ValueError as error:
            messagebox.showerror(
                "Import",
                f"{error}.\n\nEdit the site map, or increase the length tolerance.",
            )
            return
        num_sites = len(source.site_properties["coord_atomos"])
        lines, skipped = [], 0
        for line in self.inputconfs.get(1.0, END).split("\n"):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                lines.append(line)
                continue
            try:
                energies, confs, comments = parse_spin_configurations(stripped)
            except ValueError:
                lines.append("# not a configuration: " + stripped)
                skipped += 1
                continue
            conf, comment = confs[0], comments[0].strip()
            if len(conf) != num_sites:
                lines.append(
                    f"# skipped, {len(conf)} values for {num_sites} sites: {stripped}"
                )
                skipped += 1
                continue
            if not compatible_configurations(source, [conf], target, tol)[0]:
                lines.append(
                    "# skipped, not periodic with the current cell: " + stripped
                )
                skipped += 1
                continue
            (mapped,) = map_configurations(source, [conf], target, tol, mapping=mapping)
            energy = energies[0] * self.energy_scale
            lines.append(f"{energy}\t{mapped}\t# {comment}".rstrip())
        if skipped:
            logger.warning(
                "%d configurations of %s were not imported (see the comments "
                "in the imported list).",
                skipped,
                name,
            )
        self.outputconfs.config(state=NORMAL)
        self.outputconfs.delete(1.0, END)
        self.outputconfs.insert(END, "\n".join(lines).strip("\n") + "\n")
        self.outputconfs.config(state=DISABLED)
