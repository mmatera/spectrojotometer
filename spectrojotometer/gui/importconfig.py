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


from spectrojotometer.magnetic_model import site_map
from spectrojotometer.model_io import magnetic_model_from_file

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
            messagebox.showinfo(
                "Different sizes",
                "#  alert: unit cell in model2 " + "is smaller than in model1.",
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
        """Map configurations from a model to another model"""
        self.dictatoms = [int(x) for x in self.sitemap.get().split(", ")]
        self.outputconfs.config(state=NORMAL)
        lines = self.inputconfs.get(1.0, END).split("\n")
        for line in lines:
            linestrip = line.strip()
            if linestrip == "" or linestrip[0] == "#":
                self.outputconfs.insert(END, line + "\n")
                continue

            fields = line.strip().split(maxsplit=1)
            if len(fields) == 1:
                self.outputconfs.insert(END, "# " + line + "\n")
                continue
            fields = [
                str(float(fields[0]) * self.energy_scale),
                fields[1].strip().split(sep="# ", maxsplit=1),
            ]
            if len(fields[1]) == 1:
                fields = [fields[0], fields[1][0], ""]
            else:
                fields = [fields[0], fields[1][0], fields[1][1]]

            configtxt = fields[1]
            config = []
            for c_s in configtxt:
                if c_s == "0":
                    config.append(0)
                if c_s == "1":
                    config.append(1)
            if len(config) < len(self.dictatoms):
                config = config + (len(self.dictatoms) - len(config)) * [0]
            config = [config[i] if i >= 0 else 0 for i in self.dictatoms]
            newline = str(fields[0]) + "\t" + str(config) + "  #  " + fields[2] + "\n"
            self.outputconfs.insert(END, newline)
        self.outputconfs.config(state=DISABLED)
