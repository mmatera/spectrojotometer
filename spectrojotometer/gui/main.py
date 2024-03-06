#!/usr/bin/env python3
#  from tkmessagebox import *
"""
Main window
"""
import logging
import os
import sys
import tempfile

#  from visualbond.dialogs.finddialog import FindDialog
import webbrowser
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    DISABLED,
    END,
    HORIZONTAL,
    INSERT,
    LEFT,
    NORMAL,
    RIGHT,
    SUNKEN,
    TOP,
    VERTICAL,
    YES,
    BooleanVar,
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Menu,
    OptionMenu,
    PanedWindow,
    PhotoImage,
    Radiobutton,
    StringVar,
    Tk,
    TkVersion,
    W,
    X,
    Y,
)
from tkinter import filedialog as fdlg
from tkinter import font, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import resource_filename

from spectrojotometer import __name__ as spectrojotometer_name
from spectrojotometer import __path__ as spectrojotometer_path
from spectrojotometer import __version__ as spectrojotometerversion
from spectrojotometer.model_io import confindex, magnetic_model_from_file

from .dialogs import FindDialog
from .importconfig import ImportConfigWindow
from .markers import textmarkers
from .validators import show_number, validate_float, validate_pinteger

logofilename = resource_filename(spectrojotometer_name, "logo.gif")
print(logofilename)

QUOTE = """#  BONDS GENERATOR 0.0:
#  1-Please open a .CIF file with the site positions to start...
#  2- Enter the parameters and Press Generate model to define
      effective couplings...
#  3- Press optimize configurations in order to determine the
      optimal configurations for the ab initio calculations
# 4- With the ab-initio energies press "Calculate model parameters"..
"""


class ApplicationGUI:
    """User Interface - Main window"""

    def __init__(self, root=Tk()):
        self.pages = {}
        self.application_title = "Visualbond 0.2"
        self.model = None
        self.configurations = ([], [], [])
        self.chisvals = None
        self.root = root
        logging.info(ttk.Style().theme_use("classic"))
        # Creating a Font object of "TkDefaultFont"
        self.defaultFont = font.nametofont("TkDefaultFont")
        self.textFont = font.nametofont("TkTextFont")
        self.fixedFont = font.nametofont("TkFixedFont")

        # Overriding default-font with custom settings
        # i.e changing font-family, size and weight
        self.defaultFont.configure(family="Arial", size=12, weight=font.BOLD)
        # Overriding default-font with custom settings
        # i.e changing font-family, size and weight
        self.fixedFont.configure(
            family="Courier",
            size=12,
            #                           weight=font.BOLD
        )
        # Overriding default-font with custom settings
        # i.e changing font-family, size and weight
        self.textFont.configure(
            family="Arial",
            size=10,
            #                           weight=font.BOLD
        )

        self.logo = PhotoImage(file=logofilename)
        self.vcmdi = (
            self.root.register(validate_pinteger),
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )
        self.vcmdf = (
            self.root.register(validate_float),
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )
        self.root.title("Bonds generator 0.2")
        self.datafolder = os.getcwd()
        self.tmpmodel = tempfile.NamedTemporaryFile(mode="w", suffix=".cif")
        self.tmpmodel.close()
        self.tmpconfig = tempfile.NamedTemporaryFile(mode="w", suffix=".spin")
        self.tmpconfig.close()
        self.buildmenus()
        self.parameters = {}
        self.outputformat = StringVar()
        self.outputformat.set("plain")

        paned_main_window = PanedWindow(self.root, orient=VERTICAL)
        # Install paned_main_window

        self.nb = ttk.Notebook(paned_main_window)

        self.build_page1()
        self.build_page2()
        self.build_page3()
        # self.build_page4()
        # self.nb.pack(fill=BOTH)

        # Status region
        # Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
        statusregion = Frame(paned_main_window, height=25, width=170)
        if True:
            logocvs = Canvas(statusregion, width=125, height=125)
            logocvs.pack(side=LEFT, fill=X)
            logocvs.create_image((64, 62), image=self.logo)
            self.status = ScrolledText(statusregion, height=10, width=170)
            # Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
            self.status.config(background="black", foreground="white")
            self.status.pack(fill=X)

        paned_main_window.add(self.nb)
        paned_main_window.add(statusregion)

        paned_main_window.pack(fill=BOTH, expand=True)
        paned_main_window.paneconfigure(self.nb, height=600)
        paned_main_window.paneconfigure(statusregion, height=200)

        # statusregion.pack(side=BOTTOM, fill=X)

        # Status Bar
        self.statusbar = Label(
            paned_main_window,
            text="No model loaded.",
            bd=4,
            relief=SUNKEN,
            anchor=W,
        )
        self.statusbar.pack(side=BOTTOM, fill=X)

        old_stdout = sys.stdout
        sys.stdout = self
        sys.stderr = self
        logging.info("Hola!")
        self.root.mainloop()
        sys.stdout = old_stdout
        logging.info("bye bye!")

    def buildmenus(self):
        self.root.bind_all("<Control-q>", self.close_app)
        self.root.bind_all("<Control-y>", self.call_redo)
        self.root.bind_all("<Control-f>", self.call_search)
        self.root.bind_all("<F1>", self.show_help)

        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        filemenu = Menu(self.menu)
        self.menu.add_cascade(label="File", menu=filemenu, accelerator="<Alt+f>")
        filemenu.add_command(label="Open model file...", command=self.open_model)
        filemenu.add_command(label="Import model file...", command=self.import_model)
        filemenu.add_command(label="Open config file...", command=self.import_configs)
        filemenu.add_separator()
        filemenu.add_command(label="Save model as ...", command=self.save_model)
        filemenu.add_command(
            label="Save configurations as ...", command=self.save_configs
        )
        filemenu.add_separator()
        filemenu.add_command(
            label="Exit", command=self.close_app, accelerator="<Control+q>"
        )

        editmenu = Menu(self.menu)
        self.menu.add_cascade(label="Edit", menu=editmenu, accelerator="<Control+e>")
        editmenu.add_command(
            label="Undo", command=self.call_undo, accelerator="<Control+Z>"
        )
        editmenu.add_command(
            label="Redo", command=self.call_redo, accelerator="<Control+Y>"
        )
        editmenu.add_separator()
        editmenu.add_command(
            label="Cut",
            command=lambda: self.nb.event_generate("<Control-x>"),
            accelerator="<Control+x>",
        )
        editmenu.add_command(
            label="Copy",
            command=lambda: self.nb.event_generate("<Control-c>"),
            accelerator="<Control+c>",
        )
        editmenu.add_command(
            label="Paste",
            command=lambda: self.nb.event_generate("<Control-v>"),
            accelerator="<Control+v>",
        )
        editmenu.add_separator()
        editmenu.add_command(
            label="Search", command=self.call_search, accelerator="<Control+f>"
        )
        editmenu.add_command(
            label="Replace",
            command=self.call_replace,
            accelerator="<Control+r>",
        )
        editmenu.add_separator()
        editmenu.add_command(
            label="Clean log",
            command=lambda: self.status.delete("1.0", END),
            accelerator="<Control+b>",
        )

        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="Documentation", command=self.show_help)
        helpmenu.add_command(label="About...", command=self.about)

    def build_page1(self):
        self.parameters["page1"] = {}
        self.pages["page1"] = Frame(self.nb)
        controls = Frame(self.pages["page1"], width=50)
        #  Controls for bonds
        controls2 = LabelFrame(controls, text="Add bonds", padx=5, pady=5)
        fields = ["Discretization", "rmin", "rmax"]
        defaultfields = ["0.02", "0.0", "4.9"]
        for i, field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor="w")
            ent = Entry(row, validate="key", validatecommand=self.vcmdf)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(0, defaultfields[i])
            self.parameters["page1"][field] = ent
        btn = Button(controls2, text="add bonds", command=self.add_bonds)
        btn.pack(side=BOTTOM)
        controls2.pack(side=TOP)

        controls1 = LabelFrame(controls, text="Grow lattice", padx=5, pady=5)
        fields = ["Lx", "Ly", "Lz"]
        defaultfields = ("1", "1", "1")
        for i, field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=12, text=field + ": ", anchor="w")
            ent = Entry(row, validate="key", validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10, defaultfields[i])
            self.parameters["page1"][field] = ent
        btn = Button(
            controls1,
            state=DISABLED,
            text="Grow unit cell",
            command=self.grow_unit_cell,
        )
        btn.pack(side=BOTTOM)
        #  TODO
        # controls1.pack()

        controls.pack(side=LEFT, fill=Y)

        self.modelcif = ScrolledText(self.pages["page1"], width=150, undo="True")
        self.modelcif.bind("<FocusOut>", self.reload_model)
        self.modelcif.pack(side=RIGHT, fill=Y)
        self.modelcif.tag_configure("sel", background="black", foreground="gray")
        self.modelcif.insert(END, QUOTE)
        self.pages["page1"]
        tools = Frame(self.pages["page1"])
        self.nb.add(self.pages["page1"], text="1. Define Model")

    def build_page2(self):
        self.parameters["page2"] = {}
        self.pages["page2"] = Frame(self.nb)
        controls = Frame(self.pages["page2"])

        #  Controls for loading configurations
        controls1 = LabelFrame(controls, text="Load", padx=5, pady=5)
        btn = Button(
            controls1,
            text="Load configs from other model",
            command=self.configs_from_other_model,
        )
        btn.pack()
        controls1.pack(side=TOP, fill=X)
        #  Controls for Optimize configurations
        controls2 = LabelFrame(controls, text="Optimize", padx=5, pady=5)
        fields = ["Number of configurations", "Bunch size", "Iterations"]
        defaultfields = ("10", "10", "100")
        for i, field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor="w")
            ent = Entry(row, validate="key", validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10, defaultfields[i])
            #             ents[field] = ent
            self.parameters["page2"][field] = ent
        btn = Button(controls2, text="Optimize", command=self.optimize_configs)
        btn.pack()
        btn2 = Button(
            controls2,
            text="Optimal Independent Set",
            command=self.optimal_independent_set,
        )
        btn2.pack()
        controls2.pack(side=TOP, fill=X)

        controls3 = LabelFrame(controls, text="Format", padx=5, pady=5)
        row = Frame(controls3)
        lab = Label(row, width=12, text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(
            row,
            self.outputformat,
            "plain",
            "latex",
            "Wolfram",
            command=self.print_full_equations,
        )
        optformat.pack(side=LEFT, fill=X)
        row.pack(side=TOP, fill=X)
        btnrecal = Button(
            controls3, text="Calculate", command=self.print_full_equations
        )
        btnrecal.pack(side=BOTTOM, fill=Y)
        controls3.pack(side=TOP, fill=X)
        controls.pack(side=LEFT, fill=Y)

        panels = PanedWindow(self.pages["page2"], orient=HORIZONTAL)
        panels.pack(fill=BOTH, expand=1)
        frameconfs = LabelFrame(
            panels, text="Configuration File", relief=SUNKEN, padx=5, pady=5
        )
        self.spinconfigs = ScrolledText(frameconfs, width=100, undo="True")
        self.spinconfigs.bind("<FocusOut>", self.reload_configs)
        self.spinconfigs.pack(side=LEFT, fill=Y)
        self.spinconfigs.tag_configure("sel", background="black", foreground="gray")
        panels.add(frameconfs)
        self.spinconfigs.insert(
            END,
            "#  Spin configurations definition file\n"
            + "#  Energy\t [config]\t\t #  "
            + "label / comment\n",
        )

        results = LabelFrame(panels, text="Results", padx=5, pady=5)
        eqpanel = LabelFrame(results, text="Equations", relief=SUNKEN, padx=5, pady=5)
        self.equationpanel = ScrolledText(eqpanel, width=20)
        self.equationpanel.tag_configure("sel", background="black", foreground="gray")
        self.equationpanel.config(state=DISABLED)
        self.equationpanel.pack(side=TOP, fill=BOTH)

        self.equationpanel.bind("<Button-1>", lambda ev: self.equationpanel.focus())
        eqpanel.pack(side=TOP, fill=BOTH)

        # results.pack(side=TOP, fill=BOTH)
        panels.add(results)

        self.nb.add(self.pages["page2"], text="2. Spin Configurations and Couplings ")
        self.nb.tab(self.pages["page2"], state="disabled")

    def build_page3(self):
        self.parameters["page3"] = {}
        self.pages["page3"] = Frame(self.nb)
        #         ents = {}

        # # # # # # # # # # # # # # # # # # # # # # # # # #
        controls = Frame(self.pages["page3"])
        #  Controls for loading configurations
        controls1 = LabelFrame(controls, text="Settings", width=10, padx=5, pady=5)
        fields = ["Energy tolerance"]
        defaultfields = ("0.001",)
        for i, field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=14, text=field + ": ", anchor="w")
            ent = Entry(row, width=5, validate="key", validatecommand=self.vcmdf)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT)
            row.pack(side=TOP, padx=5, pady=5)
            ent.insert(0, defaultfields[i])
            #             ents[field] = ent
            self.parameters["page3"][field] = ent

        row = LabelFrame(controls1, text="Error bound method", width=8, padx=5, pady=5)
        framemethod = Frame(row)
        bemode = BooleanVar()
        self.parameters["page3"]["usemc"] = bemode

        def enable_mcparams_inputs():
            self.nummcsteps.config(state="normal")
            self.mcsizefactor.config(state="normal")

        def disable_mcparams_inputs():
            self.nummcsteps.config(state="disabled")
            self.mcsizefactor.config(state="disabled")

        Radiobutton(
            framemethod,
            text="Quadratic bound",
            variable=bemode,
            value=False,
            command=disable_mcparams_inputs,
        ).pack(side=TOP)
        Radiobutton(
            framemethod,
            text="Monte Carlo",
            variable=bemode,
            value=True,
            command=enable_mcparams_inputs,
        ).pack(side=TOP)
        self.mcparams = Frame(framemethod)
        rowmc = Frame(self.mcparams)
        Label(rowmc, text="Num Samples:").pack(side=LEFT)
        self.nummcsteps = Entry(rowmc, width=5)
        self.nummcsteps.insert(0, "1000")
        self.nummcsteps.config(state=NORMAL)
        self.nummcsteps.pack(side=LEFT)
        rowmc.pack(side=TOP)
        rowmc = Frame(self.mcparams)
        Label(rowmc, text="Size Factor:").pack(side=LEFT)
        self.mcsizefactor = Entry(rowmc, width=5)
        self.mcsizefactor.insert(0, "1.")
        self.mcsizefactor.config(state=NORMAL)
        self.mcsizefactor.pack(side=LEFT)
        rowmc.pack(side=TOP)
        self.mcparams.pack(side=TOP)
        bemode.set(True)
        self.parameters["page3"]["mcsteps"] = self.nummcsteps
        self.parameters["page3"]["mcsizefactor"] = self.mcsizefactor
        framemethod.pack(side=TOP, fill=X)
        row.pack(side=TOP, fill=X)

        row = Frame(controls1)
        lab = Label(row, width=14, text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(
            row,
            self.outputformat,
            "plain",
            "latex",
            "wolfram",
            command=self.print_full_equations,
        )
        optformat.pack(side=LEFT, fill=X)
        row.pack(side=TOP, fill=X)
        btn = Button(
            controls1,
            text="Estimate Parameters",
            command=self.evaluate_couplings,
        )
        btn.pack()
        controls1.pack(side=TOP, fill=X)
        controls.pack(side=LEFT, fill=Y)

        panels = PanedWindow(self.pages["page3"], orient=HORIZONTAL)
        panels.pack(fill=BOTH, expand=1)
        # # # # # # #   ScrolledText  # # # # # # # # # # # # #

        self.spinconfigsenerg = ScrolledText(panels, undo="True")
        self.spinconfigsenerg.bind("<FocusOut>", self.reload_configs)
        # self.spinconfigsenerg.pack(side=LEFT, fill=Y)
        panels.add(self.spinconfigsenerg)
        self.spinconfigsenerg.tag_configure(
            "sel", background="black", foreground="gray"
        )
        self.spinconfigsenerg.insert(
            END,
            "# Spin configurations definition file"
            + "\n# Energy\t [config]\t\t"
            + " label / comment\n",
        )
        # # # # # #   Results
        results = LabelFrame(panels, text="Results", padx=5, pady=5)
        panelsr = PanedWindow(results, orient=VERTICAL)
        panelsr.pack(side=LEFT, fill=BOTH)

        eqpanel = LabelFrame(panelsr, text="Equations", relief=SUNKEN)
        self.equationpanel2 = ScrolledText(
            eqpanel, state=DISABLED, height=10, width=200
        )
        self.equationpanel2.pack(side=LEFT, fill=BOTH, expand=1)
        self.equationpanel2.tag_configure("sel", background="black", foreground="gray")
        self.equationpanel2.bind("<Button-1>", lambda ev: self.equationpanel2.focus())

        eqpanel.pack(fill=X, expand=1)
        panelsr.add(eqpanel)

        respanel = LabelFrame(
            panelsr,
            text="Determined Parameters",
            relief=SUNKEN,
            padx=5,
            pady=5,
            width=80,
        )
        self.resparam = ScrolledText(respanel, state=DISABLED, height=10, width=80)
        self.resparam.pack(fill=BOTH, expand=1)
        self.resparam.tag_configure("sel", background="black", foreground="gray")
        self.resparam.bind("<Button-1>", lambda ev: self.resparam.focus())
        panelsr.add(respanel)

        chipanel = LabelFrame(panelsr, text="Energy Errors", relief=SUNKEN)
        chibuttons = Frame(chipanel)
        chibuttons.pack(side=RIGHT)
        self.plotbutton = Button(
            chibuttons, text="plot", command=self.plot_delta_energies
        )
        self.plotbutton.config(state=DISABLED)
        self.plotbutton.pack(side=TOP)
        self.chis = ScrolledText(chipanel, state=DISABLED, height=10)
        self.chis.pack(side=LEFT, fill=BOTH)
        self.chis.tag_configure("sel", background="black", foreground="gray")
        self.chis.bind("<Button-1>", lambda ev: self.chis.focus())

        panelsr.add(chipanel)
        panels.add(results)

        self.nb.add(self.pages["page3"], text="3. Set energies and evaluate.")
        self.nb.tab(self.pages["page3"], state="disabled")

    def build_page4(self):
        self.pages["page4"] = Frame(self.nb)
        self.nb.add(self.pages["page4"], text="4. Evaluate parameters")
        self.parameters["page4"] = {}
        self.outputformat = StringVar()
        self.outputformat.set("Plain")
        ctrl = LabelFrame(
            self.pages["page4"],
            text="Evaluate and Show parameters",
            padx=5,
            pady=5,
        )
        OptionMenu(ctrl, self.outputformat, "Plain", "Latex", "Wolfram").pack(
            side=TOP, fill=X
        )
        btn = Button(ctrl, text="Evaluate couplings")
        btn.pack(side=TOP)
        ctrl.pack(side=RIGHT, fill=X)
        leftpanel = LabelFrame(
            self.pages["page4"],
            relief=SUNKEN,
            text="Configurations",
            padx=5,
            pady=5,
        )
        ScrolledText(leftpanel).pack(side=LEFT, fill=Y)
        leftpanel.pack(side=LEFT, fill=Y)
        rightpanel = LabelFrame(
            self.pages["page4"], relief=SUNKEN, text="Results", padx=5, pady=5
        )
        respanel = LabelFrame(
            rightpanel,
            text="Determined Parameters",
            relief=SUNKEN,
            padx=5,
            pady=5,
        )
        Label(respanel, text=20 * (80 * " " + "\n ")).pack(fill=BOTH)
        respanel.pack(side=BOTTOM, fill=X)
        eqpanel = LabelFrame(rightpanel, text="Equations", relief=SUNKEN)
        Label(eqpanel, text=20 * (80 * " " + "\n ")).pack(fill=BOTH)
        eqpanel.pack(side=BOTTOM, fill=X)
        rightpanel.pack(side=RIGHT, fill=BOTH)

    def about(self):
        aboutmsg = "Visualbond  " + "Version 0.2 - 2024\n"
        aboutmsg += "Python " + sys.version + "\n\n"
        aboutmsg += "spectrojotometer v. " + str(spectrojotometerversion) + "\n"
        aboutmsg += "tkinter v. " + str(TkVersion) + "\n"
        aboutmsg += "numpy v. " + np.__version__ + "\n"
        aboutmsg += "matplotlib v. " + matplotlib.__version__ + "\n\n"
        aboutmsg += "See condmat-ph/... "
        messagebox.showinfo("About", aboutmsg)

    def print_status(self, msg):
        logging.info(msg)

    def open_model(self, *_):
        filename = fdlg.askopenfilename(
            initialdir=self.datafolder + "/",
            title="Select file",
            filetypes=(
                ("cif files", "*.cif"),
                ("Wien2k struct files", "*.struct"),
                ("all files", "*.*"),
            ),
        )
        if filename == "":
            return
        self.datafolder = str(Path(filename).parent)
        self.model = magnetic_model_from_file(filename=filename)
        self.model.save_cif(self.tmpmodel.name)
        with open(filename, "r") as tmpf:
            modeltxt = tmpf.read()
            self.modelcif.delete("1.0", END)
            self.modelcif.insert(INSERT, modeltxt)

        self.nb.select(0)
        self.nb.tab(self.pages["page2"], state="normal")
        self.nb.tab(self.pages["page3"], state="normal")

    def import_model(self, *_):
        filename = fdlg.askopenfilename(
            initialdir=self.datafolder + "/",
            title="Select file",
            filetypes=(
                ("cif files", "*.cif"),
                ("Wien2k struct files", "*.struct"),
                ("all files", "*.*"),
            ),
        )
        if filename == "":
            return
        self.datafolder = str(Path(filename).parent)
        self.model = magnetic_model_from_file(filename=filename)
        self.model.save_cif(self.tmpmodel.name)
        self.statusbar.config(text="model loaded")
        with open(self.tmpmodel.name, "r") as tmpf:
            modeltxt = tmpf.read()
            self.modelcif.delete("1.0", END)
            self.modelcif.insert(INSERT, modeltxt)
        self.root.title(self.application_title + " - " + filename)
        self.nb.select(0)
        self.nb.tab(self.pages["page2"], state="normal")
        self.nb.tab(self.pages["page3"], state="normal")

    def import_configs(self, clean=True):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        filename = fdlg.askopenfilename(
            initialdir=self.datafolder + "/",
            title="Select file",
            filetypes=(("spin list", "*.spin"), ("all files", "*.*")),
        )
        if len(filename) == 0:
            logging.info("opening cancelled.")
            return

        self.datafolder = str(Path(filename).parent)
        if clean:
            self.spinconfigs.delete(1.0, END)

        with open(filename, "r") as tmpf:
            configstxt = tmpf.read()
            self.spinconfigs.delete("1.0", END)
            self.spinconfigs.insert(INSERT, configstxt)

        self.reload_configs(src_widget=self.spinconfigs)
        self.statusbar.config(text="config loaded")
        self.nb.select(1)
        logging.info("... done")

    def save_model(self):
        datafolder = self.datafolder
        filename = fdlg.asksaveasfilename(
            initialdir=datafolder + "/",
            title="Select file",
            filetypes=(("cif files", "*.cif"), ("all files", "*.*")),
        )
        if filename == "":
            return
        logging.info(filename.__repr__())
        if filename == "":
            return

        self.datafolder = str(Path(filename).parent)
        with open(filename, "w") as tmpf:
            tmpf.write(self.modelcif.get(1.0, END))
        self.print_status(filename)
        self.statusbar.config(text="model saved.")

    def save_configs(self):
        datafolder = self.datafolder
        filename = fdlg.asksaveasfilename(
            initialdir=datafolder + "/",
            title="Select file",
            filetypes=(("spin files", "*.spin"), ("all files", "*.*")),
        )
        if filename == "":
            return

        self.datafolder = str(Path(filename).parent)
        if self.nb.select() == self.nb.tabs()[2]:
            self.reload_configs(src_widget=self.spinconfigsenerg)
        else:
            self.reload_configs()
        with open(filename, "w") as tmpf:
            tmpf.write(self.spinconfigsenerg.get(1.0, END))
        self.print_status(filename)
        self.statusbar.config(text="configurations saved.")

    def print_full_equations(self, ev=None):
        eqformat = self.outputformat.get()
        if ev is None or isinstance(ev, str):
            logging.info("print_full_equation called without event")
        else:
            logging.info(ev.widget)
            self.reload_configs(ev)

        confs = self.configurations[1]
        labels = self.configurations[2]
        if len(confs) == 0 or len(self.model.bonds) == 0:
            return
        cm = self.model.coefficient_matrix(confs, False)
        equations = self.model.formatted_equations(
            cm, ensname=None, comments=labels, eq_format=eqformat
        )
        equations = (
            equations
            + "\n\n |"
            + textmarkers["Delta_symbol"][eqformat]
            + "J|/|"
            + textmarkers["Delta_symbol"][eqformat]
            + "E| < "
            + str(self.model.cost(confs))
        )
        self.equationpanel.config(state=NORMAL)
        self.equationpanel.delete(1.0, END)
        self.equationpanel.insert(END, equations)
        self.equationpanel.config(state=DISABLED)
        #  In the tab3, just those configurations with known energies.
        fullconfs = confs
        fulllabels = labels
        confs = []
        labels = []
        energs = self.configurations[0]
        for i, energ in enumerate(energs):
            if energ == energ:
                confs.append(fullconfs[i])
                labels.append(fulllabels[i])
        if len(confs) == 0:
            return
        cm = self.model.coefficient_matrix(confs, False)
        equations = self.model.formatted_equations(
            cm, ensname=None, comments=labels, eq_format=eqformat
        )
        equations = (
            equations
            + "\n\n |"
            + textmarkers["Delta_symbol"][eqformat]
            + "J|/|"
            + textmarkers["Delta_symbol"][eqformat]
            + "E| < "
            + str(self.model.cost(confs))
        )
        self.equationpanel2.config(state=NORMAL)
        self.equationpanel2.delete(1.0, END)
        self.equationpanel2.insert(END, equations)
        self.equationpanel2.config(state=DISABLED)

    def reload_configs(self, ev=None, src_widget=None):
        self.print_status("updating configs")
        if ev is not None:
            spinconfigs = ev.widget
        else:
            if src_widget is not None:
                spinconfigs = src_widget
            else:
                spinconfigs = self.spinconfigs
        confs = []
        labels = []
        energies = []

        conftxt = spinconfigs.get(1.0, END)
        for linnum, l in enumerate(conftxt.split(sep="\n")):
            ls = l.strip()
            if ls == "" or ls[0] == "#":
                continue
            fields = ls.split(maxsplit=1)
            try:
                energy = float(fields[0])
                ls = fields[1]
            except ValueError:
                self.print_status("Error at line " + str(linnum + 1))
                return
            newconf = []
            comment = ""
            for pos, c in enumerate(ls):
                if c == "#":
                    comment = ls[(pos + 1) :]
                    break
                elif c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            cell_size = self.model.lattice_properties["cell_size"]
            while len(newconf) < cell_size:
                newconf.append(0)
            if comment == "":
                comment = str(confindex(newconf))
            labels.append(comment)
            confs.append(newconf)
            energies.append(energy)
        self.configurations = (energies, confs, labels)
        with open(self.tmpconfig.name, "w"):
            for idx, nc in enumerate(confs):
                row = (
                    str(energies[idx]) + "\t" + str(nc) + "\t\t # " + labels[idx] + "\n"
                )
        self.print_full_equations()
        logging.info("updating window")
        if spinconfigs == self.spinconfigs:
            self.spinconfigsenerg.delete(1.0, END)
            self.spinconfigsenerg.insert(END, conftxt)
        else:
            self.spinconfigs.delete(1.0, END)
            self.spinconfigs.insert(END, conftxt)
        logging.info("listo...")

    def reload_model(self, ev):
        self.print_status("reload model")
        if self.model is None:
            return
        current_model = self.modelcif.get(1.0, END)
        newtmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".cif")
        newtmpfile.close()
        with open(newtmpfile.name, "w") as ff:
            ff.write(current_model)
        try:
            model = magnetic_model_from_file(filename=newtmpfile.name)
        except Exception:
            self.print_status("the model can not be loaded. Check the syntax.")
            self.statusbar.config(text="the model can not be loaded. Check the syntax.")
            os.remove(newtmpfile.name)
            return
        #  if everything works,
        self.model = model
        os.remove(self.tmpmodel.name)
        self.tmpmodel = newtmpfile
        self.statusbar.config(text="model updated")

    def grow_unit_cell(self):
        if self.model is None:
            self.print_status("Model is not defined. Please load a model")
            return
        self.print_status("growing cell...")
        messagebox.showinfo("Not implemented...", "Grow unit cell is not implemented.")
        # parms = self.parameters["page1"]
        # lx = int(parms["Lx"].get())
        # ly = int(parms["Ly"].get())
        # lz = int(parms["Lz"].get())

    #         self.model.generate_bonds(ranges=[[rmin, rmax]],
    #                 discretization=discr)
    #         self.model.save_cif(self.tmpmodel.name)
    #         with open(self.tmpmodel.name, "r") as tmpf:
    #             modeltxt=tmpf.read()
    #             self.modelcif.delete("1.0", END)
    #             self.modelcif.insert(INSERT, modeltxt)

    def optimal_independent_set(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        if len(self.model.bonds) == 0:
            self.print_status("Bonds must be defined before " + "run optimization.")
            return
        parms = self.parameters["page2"]
        n = int(parms["Number of configurations"].get())
        its = int(parms["Iterations"].get())
        us = max(int(parms["Bunch size"].get()), n)
        known = []
        self.reload_configs(src_widget=self.spinconfigs)
        newconfs, cn = self.model.optimize_independent_set(self.configurations[1])

        full_labels = [
            str(sum(k * 2**i for i, k in enumerate(c)))
            for c in self.configurations[1]
        ]
        labels = [str(confindex(c)) for c in newconfs]
        energs = [self.configurations[0][full_labels.index(l)] for l in labels]
        # self.configs=([float("nan") for i in newconfs], newconfs, labels)
        # eq_format = self.outputformat.get()
        self.spinconfigs.insert(END, "\n#  Subset of optimal configurations. ")
        self.spinconfigs.insert(END, "sqrt(l)/||A^-1|| " + str(cn) + ": \n")
        for idx, nc in enumerate(newconfs):
            row = (
                "# "
                + str(energs[idx])
                + "\t"
                + str(nc)
                + "\t\t # "
                + labels[idx]
                + "\n"
            )
            self.spinconfigs.insert(END, row)
        self.reload_configs(src_widget=self.spinconfigs)

    def optimize_configs(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        if len(self.model.bonds) == 0:
            self.print_status("Bonds must be defined before run optimization.")
            return
        parms = self.parameters["page2"]
        n = int(parms["Number of configurations"].get())
        its = int(parms["Iterations"].get())
        us = max(int(parms["Bunch size"].get()), n)
        self.reload_configs(src_widget=self.spinconfigs)
        known = []
        start = []
        for i, c in enumerate(self.configurations[1]):
            if np.isnan(self.configurations[0][i]):
                start.append(c)
            else:
                known.append(c)
        newconfs, cn = self.model.find_optimal_configurations(
            num_new_confs=n, start=start, known=known, its=its, update_size=us
        )
        labels = [str(confindex(c)) for c in newconfs]
        # self.configs=([float("nan") for i in newconfs], newconfs, labels)
        eqformat = self.outputformat.get()
        self.spinconfigs.insert(END, "\n#  New configurations. ")
        self.spinconfigs.insert(
            END,
            " |"
            + textmarkers["Delta_symbol"][eqformat]
            + "J|/|"
            + textmarkers["Delta_symbol"][eqformat]
            + "E| < "
            + str(cn)
            + ": \n",
        )
        for idx, nc in enumerate(newconfs):
            row = "nan \t" + str(nc) + "\t\t # " + labels[idx] + "\n"
            self.spinconfigs.insert(END, row)
        self.reload_configs(src_widget=self.spinconfigs)

    def add_bonds(self):
        if self.model is None:
            messagebox.showerror(
                "Error",
                "Model was not loaded. " + "Please load a model first.\n",
            )
            return
        self.print_status("adding bonds...")
        parms = self.parameters["page1"]
        rmin = float(parms["rmin"].get())
        rmax = float(parms["rmax"].get())
        discr = float(parms["Discretization"].get())
        self.model.generate_bonds(ranges=[[rmin, rmax]], discretization=discr)
        self.model.save_cif(self.tmpmodel.name)
        with open(self.tmpmodel.name, "r") as tmpf:
            modeltxt = tmpf.read()
            self.modelcif.delete("1.0", END)
            self.modelcif.insert(INSERT, modeltxt)

    def configs_from_other_model(self):
        if self.model is None:
            messagebox.showerror(
                "Error",
                "Model was not loaded. " + "Please load a model first.",
            )
            return
        self.print_status("importing configurations...")
        self.reload_configs(src_widget=self.spinconfigs)
        icw = ImportConfigWindow(self)
        self.print_status("...configurations imported")

    def evaluate_couplings(self):
        """Evaluate the couplings"""
        bond_names = list(self.model.bonds)
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return

        self.reload_configs(src_widget=self.spinconfigsenerg)

        tolerance = float(self.parameters["page3"]["Energy tolerance"].get())
        usemc = self.parameters["page3"]["usemc"].get()
        mcsteps = int(self.parameters["page3"]["mcsteps"].get())
        mcsizefactor = float(self.parameters["page3"]["mcsizefactor"].get())
        confs = []
        energs = []
        fmt = self.outputformat.get()
        logging.info("\n**Evaluating couplings")
        for it, c in enumerate(self.configurations[1]):
            en = self.configurations[0][it]
            if en == en:
                energs.append(en)
                confs.append(c)
        if len(confs) < len(self.model.bonds) + 1:
            self.print_status(
                "Number of known energies is not "
                + "enough to determine all the couplings\n"
            )
            messagebox.showerror(
                "Error",
                "Number of known energies is "
                + "not enough to determine all the couplings.",
            )
            resparmtxt = ""

        if usemc:
            js, jerr, chis, ar = self.model.compute_couplings(
                confs,
                energs,
                err_energs=tolerance,
                montecarlo=True,
                mcsteps=mcsteps,
                mcsizefactor=mcsizefactor,
            )
        else:
            js, jerr, chis, ar = self.model.compute_couplings(
                confs, energs, err_energs=tolerance, montecarlo=False
            )

        self.chisvals = chis
        offset_energy = js[-1]
        js.resize(js.size - 1)
        jmax = max(abs(js))

        resparmtxt = (
            "E"
            + textmarkers["sub_symbol"][fmt]
            + "0"
            + textmarkers["equal_symbol"][fmt]
            + str(offset_energy)
            + "\n\n"
        )
        if min(jerr) < 0:
            self.print_status(
                "Warning: error bounds suggest that the "
                + "model is not compatible with the data. "
                + "Try increasing the tolerance by means "
                + "of the parameter --tolerance [tol]."
            )
            incopatibletxt = (
                textmarkers["open_comment"][fmt]
                + " incompatible "
                + textmarkers["close_comment"][fmt]
                + textmarkers["separator_symbol"][fmt]
                + "\n"
            )
            for i, val in enumerate(js):
                if jerr[i] < 0:
                    resparmtxt = (
                        resparmtxt
                        + bond_names[i]
                        + " "
                        + textmarkers["equal_symbol"][fmt]
                        + "("
                        + show_number(val / jmax)
                        + ") "
                        + textmarkers["times_symbol"][fmt]
                        + " "
                        + show_number(jmax)
                        + incopatibletxt
                    )
                else:
                    resparmtxt = (
                        resparmtxt
                        + bond_names[i]
                        + " "
                        + textmarkers["equal_symbol"][fmt]
                        + "("
                        + show_number(val / jmax, tol=jerr[i] / jmax)
                        + textmarkers["plusminus_symbol"][fmt]
                        + ("%.2g" % (jerr[i] / jmax))
                        + ") "
                        + textmarkers["times_symbol"][fmt]
                        + " "
                        + ("%.3e" % jmax)
                        + textmarkers["separator_symbol"][fmt]
                        + "\n"
                    )
        else:
            for i, val in enumerate(js):
                resparmtxt = (
                    resparmtxt
                    + bond_names[i]
                    + " "
                    + textmarkers["equal_symbol"][fmt]
                    + "("
                    + show_number(val / jmax, tol=jerr[i] / jmax)
                    + " "
                    + textmarkers["plusminus_symbol"][fmt]
                    + " "
                    + show_number(jerr[i] / jmax)
                    + ") "
                    + textmarkers["times_symbol"][fmt]
                    + show_number(jmax)
                    + "\n"
                )

        if usemc:
            resparmtxt = (
                resparmtxt + "\n\n Monte Carlo acceptance rate:" + str(ar) + "\n"
            )

        # Inequations
        resparmtxt = resparmtxt + "\n\n region constraints:\n"
        ineqs = self.model.bound_inequalities(confs, energs, err_energs=tolerance)
        for ineq in ineqs:
            txtineq = ""
            coeff = ineq[0]
            for i, c in enumerate(coeff):
                if abs(c) < tolerance:
                    continue
                if abs(c - 1) < tolerance:
                    if txtineq != "":
                        txtineq += " +"
                    txtineq += bond_names[i]
                    continue

                if abs(c + 1) < tolerance:
                    txtineq += " -" + bond_names[i]
                    continue

                if txtineq != "" and c > 0:
                    txtineq += " +" + show_number(c, tolerance) + " "
                else:
                    txtineq += " " + show_number(c, tolerance) + " "
                txtineq += textmarkers["times_symbol"][fmt] + " "
                txtineq += bond_names[i]
            txtineq = textmarkers["open_mod"][fmt] + txtineq
            if ineq[1] < 0:
                txtineq += (
                    " + "
                    + show_number(-ineq[1])
                    + " "
                    + textmarkers["close_mod"][fmt]
                    + " < "
                )
            else:
                txtineq += (
                    " "
                    + show_number(-ineq[1])
                    + " "
                    + textmarkers["close_mod"][fmt]
                    + " < "
                )
            txtineq += show_number(ineq[2])
            resparmtxt += "\n\n" + txtineq

        self.resparam.config(state=NORMAL)
        self.resparam.delete(1.0, END)
        self.resparam.insert(END, resparmtxt)
        self.resparam.config(state=DISABLED)

        #  Update chi panel
        chitext = ""
        labels = self.configurations[2]
        for j, chi in enumerate(chis):
            chitext = (
                chitext
                + textmarkers["Delta_symbol"][fmt]
                + "E"
                + textmarkers["sub_symbol"][fmt]
                + str(j + 1)
                + "/"
                + textmarkers["Delta_symbol"][fmt]
                + "E"
                + textmarkers["equal_symbol"][fmt]
                + " "
            )
            chitext = chitext + (
                show_number(chi)
                + " "
                + textmarkers["open_comment"][fmt]
                + labels[j]
                + textmarkers["close_comment"][fmt]
                + textmarkers["separator_symbol"][fmt]
                + "\n"
            )
        self.chis.config(state=NORMAL)
        self.chis.delete(1.0, END)
        self.chis.insert(END, chitext)
        self.chis.config(state=DISABLED)
        self.plotbutton.config(state=NORMAL)

    def flush(self):
        pass

    def write(self, txt):
        self.status.insert(END, txt)

    def curr_edit(self):
        page = self.nb.tab(self.nb.select())["text"]

        if page == "1. Define Model":
            return self.modelcif
        elif page == "2. Spin Configurations and Couplings":
            return self.spinconfigs
        elif page == "3. Set energies and evaluate.":
            return self.spinconfigsenerg
        return None

    def call_undo(self, *_):
        edt = self.curr_edit()
        if edt is None:
            return
        edt.edit_undo()

    def call_redo(self, *_):
        edt = self.curr_edit()
        if edt is None:
            return
        edt.edit_redo()

    def call_cut(self, *_):
        pass

    def call_copy(self, *_):
        # current_focus =
        self.root.focus_get()
        self.nb.event_generate("<Control-c>")

    def call_paste(self, *_):
        pass

    def call_search(self, *_):
        edt = self.curr_edit()
        if edt is None:
            return
        FindDialog(self, edt, replace=False)
        return

        countVar = StringVar()
        pos = edt.search("J1", "1.0", stopindex=END, count=countVar)
        if pos == "":
            logging.info("Not found")
            return
        edt.tag_configure("search", background="green")
        edt.tag_add("search", pos, "%s + %sc" % (pos, countVar.get()))

    def call_replace(self, *_):
        edt = self.curr_edit()
        if edt is None:
            return
        FindDialog(self, edt, replace=True)

    def close_app(self, *_):
        self.root.destroy()

    def show_help(self, *_):
        # docpath = os.path.dirname(os.path.realpath(sys.argv[0])) +
        #                "/doc/tutorial.html"
        docpath = spectrojotometer_path[0] + "/doc/tutorial.html"
        webbrowser.open(docpath)

    def plot_delta_energies(self, *_):
        # energy_tolerance = float(
        #    self.parameters["page3"]["Energy tolerance"].get()
        # )
        full_labels = self.configurations[2]
        full_energs = self.configurations[0]
        labels = []
        for i, energ in enumerate(full_energs):
            if energ == energ:
                labels.append("#  " + full_labels[i])
        indices = list(range(len(labels)))
        # labels = args.get('labels', None)
        plt.plot(indices, [1.0 for i in indices])
        plt.plot(indices, [-1.0 for i in indices])
        plt.scatter(indices, self.chisvals)
        plt.xticks(indices, labels, rotation="vertical")
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.gcf().canvas.manager.set_window_title("Model errors")
        plt.show()


if __name__ == "__main__":
    ApplicationGUI()
    ApplicationGUI()
