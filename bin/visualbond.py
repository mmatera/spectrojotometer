#!/usr/bin/env python3
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
#  from tkmessagebox import *
import os
import tempfile
import sys
import numpy as np

import spectrojotometer
from spectrojotometer import __version__ as spectrojotometerversion
from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import magnetic_model_from_file, \
    read_spin_configurations_file, confindex


#  from visualbond.dialogs.finddialog import FindDialog
import webbrowser

import matplotlib
import matplotlib.pyplot as plt

#  ----------------------------------------------------

quote = """#  BONDS GENERATOR 0.0:
#  1-Please open a .CIF file with the site positions to start...
#  2- Enter the parameters and Press Generate model to define
      effective couplings...
#  3- Press optimize configurations in order to determine the
      optimal configurations for the ab initio calculations
# 4- With the ab-initio energies press "Calculate model parameters"..
"""

textmarkers = {}

textmarkers["separator_symbol"] = {"latex": "", "plain": "", "wolfram": ", ", }
textmarkers["Delta_symbol"] = {"latex": "\Delta ", "plain": "Delta",
                               "wolfram": "\[Delta]", }
textmarkers["times_symbol"] = {"latex": "", "plain": "*", "wolfram": "*", }
textmarkers["equal_symbol"] = {"latex": "=", "plain": "=", "wolfram": "==", }
textmarkers["open_comment"] = {"latex": "% ", "plain": "#  ",
                               "wolfram": "(*", }
textmarkers["close_comment"] = {"latex": "", "plain": "", "wolfram": "*)", }
textmarkers["sub_symbol"] = {"latex": "_", "plain": "", "wolfram": "", }
textmarkers["plusminus_symbol"] = {"latex": "\pm", "plain": "+/-",
                                   "wolfram": "\[PlusMinus]", }

textmarkers["open_mod"] = {"latex": "\left |", "plain": "|",
                                   "wolfram": "Abs[", }

textmarkers["close_mod"] = {"latex": "\right |", "plain": "|",
                                   "wolfram": "] ", }



logofilename = spectrojotometer.__file__[:-11] +  "logo.gif"



def show_number(val,tol=None):
    if tol is None:
        return "%.3g" % val
    else:
        tol = 10. ** int(np.log(tol)/np.log(10.)-1)
        if abs(val) < tol:
            return "0"
        val = int(val / tol) * tol
        return "%.3g" % val


def validate_pinteger(action, index, value_if_allowed,
                      prior_value, text, validation_type,
                      trigger_type, widget_name):
        if text == "":
            return True
        if all([l in '0123456789' for l in text]):
            return True
        else:
            return False


def validate_float(action, index, value_if_allowed,
                   prior_value, text, validation_type,
                   trigger_type, widget_name):
        if text == "":
            return True
        if all([l in '0123456789.+-' for l in text]):
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False


class FindDialog(Toplevel):
    def __init__(self, app, edt, replace=False):
        app.root.bind_all("<Escape>", self.closewin)
        self.app = app
        self.edt = edt
        Toplevel.__init__(self, app.root)
        #  self.root = Toplevel(app.root)
        self.transient(app.root)
        self.findstring = StringVar()
        self.findstring.trace("w", self.find_onchange)

        fieldsFrame = Frame(self)
        frmfind = LabelFrame(fieldsFrame, text="Find", padx=5, pady=5)
        fieldsFrame.pack(side=LEFT, fill=Y)
        row = Frame(fieldsFrame)
        Label(row, text="Find").pack()

        self.findentry = Entry(row, textvariable=self.findstring)
        self.findentry.pack()
        row.pack()

        if replace:
            row = Frame(fieldsFrame)
            Label(row, text="Replace").pack()
            self.replacetxt = StringVar()
            Entry(row, textvariable=self.replacetxt).pack()
            row.pack()

        buttonswFrame = Frame(self)
        Button(buttonswFrame, text="Find next",
               command=self.search_next).pack(side=TOP)
        Button(buttonswFrame, text="Find previous",
               command=self.search_previous).pack(side=TOP)
        if replace:
            Button(buttonswFrame, text="Replace",
                   command=self.replace).pack(side=TOP)
            Button(buttonswFrame, text="Replace all",
                   command=self.replace_all).pack(side=TOP)
        buttonswFrame.pack(side=RIGHT, fill=Y)

        self.grab_set()
        self.findentry.focus_set()
        app.root.wait_window(self)
        self.edt.tag_delete("search")
        self.edt.focus_set()

    def closewin(self, *arg):
        self.destroy()

    def find_onchange(self, *arg):
        self.currmatch = None
        self.edt.tag_remove("sel", 1., END)
        self.edt.tag_delete("search")
        self.edt.tag_configure("search", background="green")
        self.edt.tag_raise("sel")

        targettxt = self.findstring.get()
        print("looking for " + targettxt)
        if targettxt == "":
            return
        countVar = StringVar()
        start = "1.0"
        pos = self.edt.search(targettxt, start, stopindex=END, count=countVar)
        firstpos = pos
        while pos != "":
            end = "%s + %sc" % (pos, countVar.get())
            self.edt.tag_add("search", pos, end)
            pos = self.edt.search(targettxt, end,
                                  stopindex=END, count=countVar)

        if firstpos:
            self.currmatch = 0
            self.edt.mark_set("insert", firstpos)
            self.edt.tag_add("sel", self.edt.tag_ranges("search")[0],
                             self.edt.tag_ranges("search")[1])
        else:
            messagebox.showinfo("Find", "There are no matches.")

    def search_next(self):
        if self.currmatch is None:
            messagebox.showinfo("Find", "There are no matches.")
            return

        self.currmatch = self.currmatch + 1
        if 2 * self.currmatch == len(self.edt.tag_ranges("search")):
            messagebox.showinfo("Find next", "This is the last match")
            self.currmatch = self.currmatch - 1
            return

        self.edt.mark_set("insert",
                          self.edt.tag_ranges("search")[2 * self.currmatch])
        self.edt.see(self.edt.tag_ranges("search")[2 * self.currmatch])
        self.edt.tag_remove("sel", 1., END)
        self.edt.tag_add("sel",
                         self.edt.tag_ranges("search")[2 * self.currmatch],
                         self.edt.tag_ranges("search")[2 * self.currmatch + 1])

    def search_previous(self):
        if self.currmatch is None:
            messagebox.showinfo("Find", "There are no matches.")
            return
        if self.currmatch == 0:
            messagebox.showinfo("Find next", "This is the first match")
            return
        self.currmatch = self.currmatch - 1
        self.edt.mark_set("insert",
                          self.edt.tag_ranges("search")[2 * self.currmatch])
        self.edt.see(self.edt.tag_ranges("search")[2 * self.currmatch])
        self.edt.tag_remove("sel", 1., END)
        self.edt.tag_add("sel",
                         self.edt.tag_ranges("search")[2 * self.currmatch],
                         self.edt.tag_ranges("search")[2 * self.currmatch + 1])

    def replace(self):
        if self.currmatch is None:
            messagebox.showinfo("Find", "There are no matches.")
            return
        begin = self.edt.tag_ranges("search")[2 * self.currmatch]
        end = self.edt.tag_ranges("search")[2 * self.currmatch + 1]
        self.edt.delete(begin, end)
        self.edt.insert(begin, self.replacetxt.get())
        if self.currmatch > 0:
            self.currmatch = self.currmatch - 1
        self.search_next()

    def replace_all(self):
        if self.currmatch is None:
            messagebox.showinfo("Find", "There are no matches.")
        coordinates = []
        lpos = list(self.edt.tag_ranges("search"))
        lpos.reverse()

        while lpos:
            coordinates.append([lpos.pop(), lpos.pop()])

        for begin, end in coordinates:
            self.edt.delete(begin, end)
            self.edt.insert(begin, self.replacetxt.get())

        messagebox.showinfo("Replace all", "There are no more occurrences")


class ImportConfigWindow(Toplevel):
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

        controls1 = LabelFrame(self, text="Parameters", padx=5, pady=5)

        row = Frame(controls1)
        Label(row, text="model file:").pack(side=LEFT)
        self.selected_model = StringVar()
        self.selected_model.trace('w', self.onmodelselect)
        self.optmodels = OptionMenu(row, self.selected_model, "[other model]")
        self.optmodels.pack(side=LEFT, fill=X)
        self.sitemap  = Entry(row)
        self.sitemap.pack(side=RIGHT)
        Label(row,text="site map").pack(side=RIGHT)
        row.pack(side=TOP, fill=X)

        row = Frame(controls1)
        Label(row, text="Length tolerance:").pack(side=LEFT)
        self.tol = Entry(row, validate='key', validatecommand=self.app.vcmdf)
        self.tol.insert(10, .1)
        self.tol.pack(side=LEFT)
        row.pack(side=TOP, fill=X)
        controls1.pack(side=TOP, fill=X)

        #  controls2 = Frame(self, padx=5, pady=5)
        controls2 = PanedWindow(self, orient=HORIZONTAL)
        controls2.pack(side=TOP, fill=BOTH, expand=1)
        controls2l = LabelFrame(controls2, text="inputs", padx=5, pady=5)
        self.inputconfs = ScrolledText(controls2l, height=10, width=80,
                                       undo="True")
        self.inputconfs.pack()
        buttons = Frame(controls2l)
        Button(buttons, text="Load Configuration from File",
               command=self.configs_from_file).pack(side=RIGHT)
        Button(buttons, text="Import", command=self.map_confs).pack(side=LEFT)
        buttons.pack(side=BOTTOM, fill=X)

        #  controls2l.pack(side=LEFT, fill=Y)
        controls2.add(controls2l)
        controls2r = LabelFrame(controls2, text="in main model",
                                padx=5, pady=5)
        self.outputconfs = ScrolledText(controls2r, height=10, width=80)
        self.outputconfs.config(state=DISABLED)
        self.outputconfs.pack()

        #  controls2r.pack(side=RIGHT, fill=Y)
        controls2.add(controls2r)
        #  controls2.pack(side=TOP, fill=X)

        framebts = Frame(self)
        Button(framebts, text="Send to main application",
               command=self.send_to_application).pack(side=LEFT)
        Button(framebts, text="Close",
               command=self.close_window).pack(side=RIGHT)
        framebts.pack(side=BOTTOM, fill=X)
        self.grab_set()
        app.root.wait_window(self)

    def onmodelselect(self, *args):
        if self.selected_model.get() == "[other model]":
            filename = filedialog.askopenfilename(initialdir=self.app.datafolder + "/",
                                                  title="Select file to open",
                                                  filetypes=(("cif files", "*.cif"),
                                                             ("Wien2k struct files", "*.struct"),
                                                             ("all files", "*.*")))
            print(filename)
            if filename == "":
                return
            newmodel = magnetic_model_from_file(filename=filename)
            modellabel = filename
            self.models[filename] = newmodel
            menu = self.optmodels["menu"]
            menu.delete(0, "end")
            menu.add_command(label="[other model]",
                             command=lambda value="[other model]":
                             self.selected_model.set("[other model]"))
            print("Updating menu")
            for key in self.models:
                menu.add_command(label=key,
                                 command=lambda value=key:
                                 self.selected_model.set(value))
            self.selected_model.set(filename)
            self.update_idletasks()
            #  self.optmodels.set(filename)
        # xxxxxxxxx
        tol = float(self.tol.get())
        model1 = self.models[self.selected_model.get()]
        model2 = self.app.model
        size1 = len(model1.coord_atomos)
        self.scale_energy = float(len(model2.coord_atomos)) / float(size1)
        if self.scale_energy < 1.:
            messagebox.showinfo("Different sizes",
                                "#  alert: unit cell in model2 is smaller than in model1.")

        
        for p in model2.cood_atomos:
                print("\t",p)
        
        for k, p in enumerate(model1.supercell):
                print("\t", k % len(model1.coord_atomos) ,"->",p)
        dictatoms = [-1 for p in model2.coord_atomos]
        for i, p in enumerate(model2.coord_atomos):
            for j, q in enumerate(model1.supercell):
                if np.linalg.norm(p - q) < tol:
                    dictatoms[i] = j % size1
                    break
        self.dictatoms = [n for n in dictatoms]
        self.sitemap.delete(0,END)
        self.sitemap.insert(0,dictatoms.__str__()[1:-1])


    def close_window(self):
        self.destroy()

    def send_to_application(self):
        self.app.spinconfigs.insert(END, "\n\n#  From " +
                                    self.selected_model.get() + "\n")
        self.app.spinconfigs.insert(END, self.outputconfs.get(1.0, END))
        self.app.spinconfigs.insert(END, "\n" + 20 * "# " + "\n\n")
        self.app.reload_configs(src_widget=self.app.spinconfigs)

    def configs_from_file(self):
        if self.selected_model.get() == "[other model]":
            self.onmodelselect()
        filename = filedialog.askopenfilename(initialdir=self.app.datafolder + "/",
                                              title="Select file",
                                              filetypes=(("spin list", "*.spin"),
                                                         ("all files", "*.*")))
        if filename == "":
            return
        self.inputconfs.delete(1.0, END)
        with open(filename, "r") as filecfg:
            self.inputconfs.insert(END, "\n" + filecfg.read())

    def map_confs(self):
        self.dictatoms = [int(x) for x in self.sitemap.get().split(",")]
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
            fields = [str(float(fields[0]) * self.scale_energy),
                      fields[1].strip().split(sep="# ", maxsplit=1)]
            if len(fields[1]) == 1:
                fields = [fields[0], fields[1][0], ""]
            else:
                fields = [fields[0], fields[1][0], fields[1][1]]

            configtxt = fields[1]
            config = []
            for c in configtxt:
                if c == '0':
                    config.append(0)
                if c == '1':
                    config.append(1)
            if len(config) < len(self.dictatoms):
                config = config + (len(dictatoms) - len(config)) * [0]
            config = [config[i] if i >= 0 else 0 for i in self.dictatoms]
            newline = str(fields[0]) + "\t" + str(config) + \
                "  #  " + fields[2] + "\n"
            self.outputconfs.insert(END, newline)
        self.outputconfs.config(state=DISABLED)


class ApplicationGUI:
    def __init__(self):
        sys.stdout = self
        sys.stderr = self
        self.application_title = "Visualbond 0.1"
        self.model = None
        self.configurations = ([], [], [])
        self.chisvals = None
        self.root = Tk()
        self.logo = PhotoImage(file=logofilename)
        self.vcmdi = (self.root.register(validate_pinteger),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.vcmdf = (self.root.register(validate_float),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.root.title("Bonds generator 0.0")
        self.datafolder = os.getcwd()
        self.tmpmodel = tempfile.NamedTemporaryFile(mode="w", suffix=".cif")
        self.tmpmodel.close()
        self.tmpconfig = tempfile.NamedTemporaryFile(mode="w", suffix=".spin")
        self.tmpconfig.close()
        self.buildmenus()
        self.nb = ttk.Notebook(self.root)
        self.parameters = {}
        self.outputformat = StringVar()
        self.outputformat.set("plain")
        self.build_page1()
        self.build_page2()
        self.build_page3()
#         self.build_page4()
        self.nb.pack(expand=1, fill="both")
        Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
        statusregion = Frame(self.root,height=15,width=170)
        logocvs = Canvas(statusregion,width=125,height=125)
        logocvs.pack(side=LEFT,fill=X)
        logocvs.create_image((64,62),image=self.logo)
        self.status = ScrolledText(statusregion, height=15, width=170)
        # Frame(height=5, bd=1, relief=SUNKEN).pack(fill=X, padx=5, pady=5)
        self.status.config(background="black", foreground="white")
        self.status.pack(fill=X)
        statusregion.pack(side=BOTTOM,fill=X)
        
        self.statusbar = Label(self.root, text="No model loaded.", bd=1,
                               relief=SUNKEN, anchor=W)
        self.statusbar.pack(side=BOTTOM, fill=X)
        self.root.mainloop()

    def buildmenus(self):
        self.root.bind_all("<Control-q>", self.close_app)
        self.root.bind_all("<Control-y>", self.call_redo)
        self.root.bind_all("<Control-f>", self.call_search)
        self.root.bind_all("<F1>", self.show_help)

        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        filemenu = Menu(self.menu)
        self.menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open model file...",
                             command=self.import_model)
        filemenu.add_command(label="Open config file...",
                             command=self.import_configs)
        filemenu.add_command(label="Save model as ...",
                             command=self.save_model)
        filemenu.add_command(label="Save configurations as ...",
                             command=self.save_configs)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close_app,
                             accelerator="<Control+q>")

        editmenu = Menu(self.menu)
        self.menu.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Undo", command=self.call_undo,
                             accelerator="<Control+Z>")
        editmenu.add_command(label="Redo", command=self.call_redo,
                             accelerator="<Control+Y>")
        editmenu.add_separator()
        editmenu.add_command(label="Cut", command=lambda:
                             self.nb.event_generate('<Control-x>'),
                             accelerator="<Control+x>")
        editmenu.add_command(label="Copy", command=lambda:
                             self.nb.event_generate('<Control-c>'),
                             accelerator="<Control+c>")
        editmenu.add_command(label="Paste", command=lambda:
                             self.nb.event_generate('<Control-v>'),
                             accelerator="<Control+v>")
        editmenu.add_separator()
        editmenu.add_command(label="Search", command=self.call_search,
                             accelerator="<Control+F>")
        editmenu.add_command(label="Replace", command=self.call_replace,
                             accelerator="<Control+R>")

        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="Documentation",
                             command=self.show_help)
        helpmenu.add_command(label="About...", command=self.about)

    def build_page1(self):
        self.parameters["page1"] = {}
        self.page1 = Frame(self.nb)
        controls = Frame(self.page1, width=50)
        #  Controls for bonds
        controls2 = LabelFrame(controls, text="Add bonds", padx=5, pady=5)
        fields = ['Discretization', 'rmin', 'rmax']
        defaultfields = ["0.02", "0.0", "4.9"]
        for i, field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row, validate='key', validatecommand=self.vcmdf)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(0, defaultfields[i])
            self.parameters["page1"][field] = ent
        btn = Button(controls2, text="add bonds", command=self.add_bonds)
        btn.pack(side=BOTTOM)
        controls2.pack(side=TOP)

        controls1 = LabelFrame(controls, text="Grow lattice", padx=5, pady=5)
        fields = ['Lx', 'Ly', 'Lz']
        defaultfields = ("1", "1", "1")
        valorentry = []
        for i, field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row, validate='key', validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10, defaultfields[i])
            self.parameters["page1"][field] = ent
        btn = Button(controls1, state=DISABLED, text="Grow unit cell",
                     command=self.grow_unit_cell)
        btn.pack(side=BOTTOM)
        #  TODO
        # controls1.pack()

        controls.pack(side=LEFT, fill=Y)

        self.modelcif = ScrolledText(self.page1, width=150, undo="True")
        self.modelcif.bind("<FocusOut>", self.reload_model)
        self.modelcif.pack(side=RIGHT, fill=Y)
        self.modelcif.insert(END, quote)
        page1tools = Frame(self.page1)
        self.nb.add(self.page1, text="1. Define Model")

    def build_page2(self):
        self.parameters["page2"] = {}
        self.page2 = Frame(self.nb)
        controls = Frame(self.page2)

        #  Controls for loading configurations
        controls1 = LabelFrame(controls, text="Load", padx=5, pady=5)
        btn = Button(controls1, text="Load configs from other model",
                     command=self.configs_from_other_model)
        btn.pack()
        controls1.pack(side=TOP, fill=X)
        #  Controls for Optimize configurations
        controls2 = LabelFrame(controls, text="Optimize", padx=5, pady=5)
        fields = ['Number of configurations', 'Bunch size', 'Iterations']
        defaultfields = ("10", "10", "100")
        for i, field in enumerate(fields):
            row = Frame(controls2)
            lab = Label(row, width=12, text=field + ": ", anchor='w')
            ent = Entry(row, validate='key', validatecommand=self.vcmdi)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            ent.insert(10, defaultfields[i])
#             ents[field] = ent
            self.parameters["page2"][field] = ent
        btn = Button(controls2, text="Optimize",
                     command=self.optimize_configs)
        btn.pack()
        controls2.pack(side=TOP, fill=X)
        controls3 = LabelFrame(controls, text="Format", padx=5, pady=5)
        row = Frame(controls3)
        lab = Label(row, width=12, text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(row, self.outputformat, "plain",
                               "latex", "Wolfram",
                               command=self.print_full_equations)
        optformat.pack(side=LEFT, fill=X)
        row.pack(side=TOP, fill=X)
        btnrecal = Button(controls3, text="Calculate",
                     command=self.print_full_equations)
        btnrecal.pack(side=BOTTOM,fill=Y)
        controls3.pack(side=TOP, fill=X)
        controls.pack(side=LEFT, fill=Y)

        panels = PanedWindow(self.page2, orient=HORIZONTAL)
        panels.pack(fill=BOTH, expand=1)
        frameconfs = LabelFrame(panels, text="Configuration File",
                                relief=SUNKEN, padx=5, pady=5)
        self.spinconfigs = ScrolledText(frameconfs, width=100,
                                        undo="True")
        self.spinconfigs.bind("<FocusOut>",
                              self.reload_configs)
        self.spinconfigs.pack(side=LEFT, fill=Y)
        panels.add(frameconfs)
        self.spinconfigs.insert(END,
                                "#  Spin configurations definition file\n" +
                                "#  Energy\t [config]\t\t #  " +
                                "label / comment\n")

        results = LabelFrame(panels, text="Results", padx=5, pady=5)
        eqpanel = LabelFrame(results, text="Equations",
                             relief=SUNKEN, padx=5, pady=5)
        self.equationpanel = ScrolledText(eqpanel, width=20)
        self.equationpanel.config(state=DISABLED)
        self.equationpanel.pack(side=TOP, fill=BOTH)
        self.equationpanel.bind("<Button-1>",
                                lambda ev: self.equationpanel.focus())
        eqpanel.pack(side=TOP, fill=BOTH)

        # results.pack(side=TOP, fill=BOTH)
        panels.add(results)

        self.nb.add(self.page2, text="2. Spin Configurations and Couplings ")
        self.nb.tab(self.page2, state="disabled")

    def build_page3(self):
        self.parameters["page3"] = {}
        self.page3 = Frame(self.nb)
#         ents = {}

        # # # # # # # # # # # # # # # # # # # # # # # # # #
        controls = Frame(self.page3)
        #  Controls for loading configurations
        controls1 = LabelFrame(controls, text="Settings",
                               width=10, padx=5, pady=5)
        fields = ["Energy tolerance"]
        defaultfields = ("0.1",)
        for i, field in enumerate(fields):
            row = Frame(controls1)
            lab = Label(row, width=14, text=field + ": ", anchor='w')
            ent = Entry(row, width=5, validate='key',
                        validatecommand=self.vcmdf)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT)
            row.pack(side=TOP, padx=5, pady=5)
            ent.insert(0, defaultfields[i])
#             ents[field] = ent
            self.parameters["page3"][field] = ent

        row = LabelFrame(controls1,text="Error bound method",width=8, padx=5, pady=5)
        framemethod = Frame(row)
        bemode = BooleanVar()
        self.parameters["page3"]["usemc"] = bemode

        def enable_mcparams_inputs():
            self.nummcsteps.config(state="normal")
            self.mcsizefactor.config(state="normal")

            
        def disable_mcparams_inputs():
            self.nummcsteps.config(state="disabled")
            self.mcsizefactor.config(state="disabled")
        

                
        Radiobutton(framemethod, text="Quadratic bound", variable=bemode,\
                    value=False,\
                    command=disable_mcparams_inputs).pack(side=TOP)
        Radiobutton(framemethod, text="Monte Carlo", variable=bemode,\
                    value=True,\
                    command=enable_mcparams_inputs ).pack(side=TOP)
        self.mcparams = Frame(framemethod)
        rowmc = Frame(self.mcparams)
        Label(rowmc, text="Num Samples:").pack(side=LEFT)
        self.nummcsteps = Entry(rowmc, width=5)
        self.nummcsteps.insert(0,"1000")
        self.nummcsteps.config(state="disabled")
        self.nummcsteps.pack(side=LEFT)
        rowmc.pack(side=TOP)
        rowmc = Frame(self.mcparams)
        Label(rowmc, text="Size Factor:").pack(side=LEFT)
        self.mcsizefactor = Entry(rowmc, width=5)
        self.mcsizefactor.insert(0,"1.")
        self.mcsizefactor.config(state="disabled")
        self.mcsizefactor.pack(side=LEFT)
        rowmc.pack(side=TOP)
        self.mcparams.pack(side=TOP)
        bemode.set(0)
        self.parameters["page3"]["mcsteps"] = self.nummcsteps
        self.parameters["page3"]["mcsizefactor"] = self.mcsizefactor
        framemethod.pack(side=TOP,fill=X)
        row.pack(side=TOP,fill=X)

        row = Frame(controls1)
        lab = Label(row, width=14, text="Format", anchor="w")
        lab.pack(side=LEFT)
        optformat = OptionMenu(row, self.outputformat, "plain",
                               "latex", "wolfram",
                               command=self.print_full_equations)
        optformat.pack(side=LEFT, fill=X)
        row.pack(side=TOP, fill=X)

        
        btn = Button(controls1, text="Estimate Parameters",
                     command=self.evaluate_couplings)
        btn.pack()
        controls1.pack(side=TOP, fill=X)
        controls.pack(side=LEFT, fill=Y)

        panels = PanedWindow(self.page3, orient=HORIZONTAL)
        panels.pack(fill=BOTH, expand=1)
        # # # # # # #   ScrolledText  # # # # # # # # # # # # #

        self.spinconfigsenerg = ScrolledText(panels, undo="True")
        self.spinconfigsenerg.bind("<FocusOut>", self.reload_configs)
        # self.spinconfigsenerg.pack(side=LEFT, fill=Y)
        panels.add(self.spinconfigsenerg)
        self.spinconfigsenerg.insert(END,
                                     "#  Spin configurations definition file\n" +
                                     "#  Energy\t [config]\t\t #" +
                                     "  label / comment\n")
        # # # # # #   Results
        results = LabelFrame(panels, text="Results", padx=5, pady=5)
        panelsr = PanedWindow(results, orient=VERTICAL)
        panelsr.pack(side=LEFT, fill=BOTH)

        eqpanel = LabelFrame(panelsr, text="Equations", relief=SUNKEN)
        self.equationpanel2 = ScrolledText(eqpanel, state=DISABLED,
                                           height=10, width=200)
        self.equationpanel2.pack(side=LEFT, fill=BOTH, expand=1)
        self.equationpanel2.bind("<Button-1>",
                                 lambda ev: self.equationpanel2.focus())

        eqpanel.pack(fill=X, expand=1)
        panelsr.add(eqpanel)

        respanel = LabelFrame(panelsr, text="Determined Parameters",
                              relief=SUNKEN, padx=5, pady=5, width=80)
        self.resparam = ScrolledText(respanel, state=DISABLED,
                                     height=10, width=80)
        self.resparam.pack(fill=BOTH, expand=1)
        self.resparam.bind("<Button-1>",
                           lambda ev: self.resparam.focus())
        panelsr.add(respanel)

        chipanel = LabelFrame(panelsr, text="Energy Errors", relief=SUNKEN)
        chibuttons = Frame(chipanel)
        chibuttons.pack(side=RIGHT)
        self.plotbutton = Button(chibuttons, text="plot",
                                 command=self.plot_delta_energies)
        self.plotbutton.config(state=DISABLED)
        self.plotbutton.pack(side=TOP)
        self.chis = ScrolledText(chipanel, state=DISABLED, height=10)
        self.chis.pack(side=LEFT, fill=BOTH)
        self.chis.bind("<Button-1>",
                       lambda ev: self.chis.focus())

        panelsr.add(chipanel)
        panels.add(results)

        self.nb.add(self.page3, text="3. Set energies and evaluate.")
        self.nb.tab(self.page3, state="disabled")

    def build_page4(self):
        self.page4 = Frame(self.nb)
        self.nb.add(self.page4, text="4. Evaluate parameters")
        self.parameters["page4"] = {}
        self.outputformat = StringVar()
        self.outputformat.set("Plain")
        ctrl = LabelFrame(self.page4, text="Evaluate and Show parameters",
                          padx=5, pady=5)
        OptionMenu(ctrl, self.outputformat, "Plain", "Latex",
                   "Wolfram").pack(side=TOP, fill=X)
        btn = Button(ctrl, text="Evaluate couplings")
        btn.pack(side=TOP)
        ctrl.pack(side=RIGHT, fill=X)
        leftpanel = LabelFrame(self.page4, relief=SUNKEN,
                               text="Configurations", padx=5, pady=5)
        ScrolledText(leftpanel).pack(side=LEFT, fill=Y)
        leftpanel.pack(side=LEFT, fill=Y)
        rightpanel = LabelFrame(self.page4, relief=SUNKEN,
                                text="Results", padx=5, pady=5)
        respanel = LabelFrame(rightpanel, text="Determined Parameters",
                              relief=SUNKEN, padx=5, pady=5)
        Label(respanel, text=20 * (80 * " " + "\n ")).pack(fill=BOTH)
        respanel.pack(side=BOTTOM, fill=X)
        eqpanel = LabelFrame(rightpanel, text="Equations", relief=SUNKEN)
        Label(eqpanel, text=20 * (80 * " " + "\n ")).pack(fill=BOTH)
        eqpanel.pack(side=BOTTOM, fill=X)
        rightpanel.pack(side=RIGHT, fill=BOTH)

    def about(self):
        aboutmsg = "Visualbond  " + "Version 0.0 - 2017\n"
        aboutmsg += "Python " + sys.version + "\n\n"
        aboutmsg += "spectrojotometer v. " + str(spectrojotometerversion) + "\n"
        aboutmsg += "tkinter v. " + str(TkVersion) + "\n"
        aboutmsg += "numpy v. " + np.__version__ + "\n"
        aboutmsg += "matplotlib v. " + matplotlib.__version__ + "\n\n"
        aboutmsg += "See condmat-ph/... "
        messagebox.showinfo("About", aboutmsg)

    def print_status(self, msg):
        print(msg)

    def import_model(self, *args):
        filename = filedialog.askopenfilename(initialdir=self.datafolder + "/",
                                              title="Select file",
                                              filetypes=(("cif files", "*.cif"),
                                                         ("Wien2k struct files", "*.struct"),
                                                         ("all files", "*.*")))
        if filename == "":
            return
        self.model = magnetic_model_from_file(filename=filename)
        self.model.save_cif(self.tmpmodel.name)
        self.statusbar.config(text="model loaded")
        with open(self.tmpmodel.name, "r") as tmpf:
            modeltxt = tmpf.read()
            self.modelcif.delete("1.0", END)
            self.modelcif.insert(INSERT, modeltxt)
        self.root.title(self.application_title + " - " + filename)
        self.nb.select(0)
        self.nb.tab(self.page2, state="normal")
        self.nb.tab(self.page3, state="normal")

    def import_configs(self, clean=True):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        filename = filedialog.askopenfilename(initialdir=self.datafolder + "/",
                                              title="Select file",
                                              filetypes=(("spin list", "*.spin"),
                                                         ("all files", "*.*")))
        if len(filename) == 0:
            return
        self.configurations = read_spin_configurations_file(filename=filename,
                                                            model=self.model)
        confs = self.configurations[1]
        energies = self.configurations[0]
        labels = self.configurations[2]
        if clean:
            self.spinconfigs.delete(1.0, END)
        with open(self.tmpconfig.name, "w") as of:
            for idx, nc in enumerate(confs):
                row = (str(energies[idx]) + "\t" + str(nc) +
                       "\t\t # " + labels[idx] + "\n")
                of.write(row)
                self.spinconfigs.insert(INSERT, row)

        self.statusbar.config(text="config loaded")
        self.nb.select(1)
        self.reload_configs(src_widget=self.spinconfigs)

        
    def save_model(self):
        datafolder = os.getcwd()
        filename = filedialog.asksaveasfilename(initialdir=datafolder + "/",
                                                title="Select file",
                                                filetypes=(("cif files", "*.cif"),
                                                           ("all files", "*.*")))
        if filename == "":
            return
        print(filename.__repr__())
        if filename == "":
            return
        with open(filename, "w") as tmpf:
            tmpf.write(self.modelcif.get(1.0, END))
        self.print_status(filename)
        self.statusbar.config(text="model saved.")

    def save_configs(self):
        datafolder = os.getcwd()
        filename = filedialog.asksaveasfilename(initialdir=datafolder + "/",
                                                title="Select file",
                                                filetypes=(("spin files", "*.spin"),
                                                           ("all files", "*.*")))
        if filename == "":
            return
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
        confs = self.configurations[1]
        labels = self.configurations[2]
        if len(confs) == 0 or len(self.model.bond_lists) == 0:
            return
        cm = self.model.coefficient_matrix(confs, False)
        equations = self.model.formatted_equations(cm,
                                                   ensname=None,
                                                   comments=labels,
                                                   format=eqformat)
        equations = (equations + "\n\n |" +
                     textmarkers["Delta_symbol"][eqformat] +
                     "J|/|" + textmarkers["Delta_symbol"][eqformat] +
                     "E| < " +
                     str(self.model.inv_min_sv_from_config(confs)))
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
        equations = self.model.formatted_equations(cm,
                                                   ensname=None,
                                                   comments=labels,
                                                   format=eqformat)
        equations = (equations +
                     "\n\n |" + textmarkers["Delta_symbol"][eqformat] +
                     "J|/|" + textmarkers["Delta_symbol"][eqformat] + "E| < " +
                     str(self.model.inv_min_sv_from_config(confs)))
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
                    comment = ls[(pos + 1):]
                    break
                elif c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            while len(newconf) < self.model.cell_size:
                newconf.append(0)
            if comment == "":
                comment = str(confindex(newconf))
            labels.append(comment)
            confs.append(newconf)
            energies.append(energy)
        self.configurations = (energies, confs, labels)
        with open(self.tmpconfig.name, "w") as of:
            for idx, nc in enumerate(confs):
                row = str(energies[idx]) + "\t" + str(nc) + "\t\t # " + labels[idx] + "\n"
        self.print_full_equations()
        if spinconfigs == self.spinconfigs:
            self.spinconfigsenerg.delete(1.0, END)
            self.spinconfigsenerg.insert(END, conftxt)
        else:
            self.spinconfigs.delete(1.0, END)
            self.spinconfigs.insert(END, conftxt)
            

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
        except:
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
        parms = self.parameters["page1"]
        lx = int(parms["Lx"].get())
        ly = int(parms["Ly"].get())
        lz = int(parms["Lz"].get())
#         self.model.generate_bonds(ranges=[[rmin, rmax]], discretization=discr)
#         self.model.save_cif(self.tmpmodel.name)
#         with open(self.tmpmodel.name, "r") as tmpf:
#             modeltxt=tmpf.read()
#             self.modelcif.delete("1.0", END)
#             self.modelcif.insert(INSERT, modeltxt)

    def optimize_configs(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded.\n")
            return
        if len(self.model.bond_lists) == 0:
            self.print_status("Bonds must be defined before run optimization.")
            return
        parms = self.parameters["page2"]
        n = int(parms['Number of configurations'].get())
        its = int(parms["Iterations"].get())
        us = max(int(parms["Bunch size"].get()), n)
        known = []
        cn, newconfs = self.model.find_optimal_configurations(num_new_confs=n,
                                                              start=[],
                                                              known=known,
                                                              its=its, update_size=us)
        labels = [str(confindex(c)) for c in newconfs]
        # self.configs=([float("nan") for i in newconfs], newconfs, labels)
        eqformat = self.outputformat.get()
        self.spinconfigs.insert(END, "\n#  New configurations. ")
        self.spinconfigs.insert(END, " |" +
                                textmarkers["Delta_symbol"][eqformat] +
                                "J|/|" + textmarkers["Delta_symbol"][eqformat] +
                                "E| < " + str(cn) + ": \n")
        for idx, nc in enumerate(newconfs):
            row = "nan \t" + str(nc) + "\t\t # " + labels[idx] + "\n"
            self.spinconfigs.insert(END, row)
        self.reload_configs(src_widget=self.spinconfigs)

    def add_bonds(self):
        if self.model is None:
            messagebox.showerror("Error", "Model was not loaded. Please load a model first.\n")
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
            messagebox.showerror("Error", "Model was not loaded. Please load a model first.")
            return
        self.print_status("importing configurations...")
        self.reload_configs(src_widget=self.spinconfigs)
        icw = ImportConfigWindow(self)
        self.print_status("donde importing configurations")

    def evaluate_couplings(self):
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
        print("\n**Evaluating couplings")
        for it, c in enumerate(self.configurations[1]):
            en = self.configurations[0][it]
            if en == en:
                energs.append(en)
                confs.append(c)
        if len(confs) < len(self.model.bond_lists) + 1:
            self.print_status("Number of known energies is not enough to determine all the couplings\n")
            messagebox.showerror("Error", "Number of known energies is not enough to determine all the couplings.")
            resparmtxt = ""
        else:
            if  usemc:
                js, jerr, chis,ar = self.model.compute_couplings(confs,
                                                                 energs,
                                                                 err_energs=tolerance,montecarlo=True,
                                                                 mcsteps=mcsteps,mcsizefactor=mcsizefactor)
            else:
                js, jerr, chis,ar = self.model.compute_couplings(confs,
                                                                 energs,
                                                                 err_energs=tolerance,montecarlo=False)
                
            self.chisvals = chis
            offset_energy = js[-1]
            js.resize(js.size - 1)
            jmax = max(abs(js))

            resparmtxt = ("E" + textmarkers["sub_symbol"][fmt] + "0" +
                      textmarkers["equal_symbol"][fmt] + str(offset_energy) +
                      "\n\n")
            if min(jerr) < 0:
                self.print_status("Warning: error bounds suggest that  the model is not compatible with the data. Try increasing the tolerance by means of the parameter --tolerance [tol].")
                incopatibletxt = (textmarkers["open_comment"][fmt] +
                              " incompatible " +
                              textmarkers["close_comment"][fmt] +
                              textmarkers["separator_symbol"][fmt] + "\n")
                for i, val in enumerate(js):
                    if jerr[i] < 0:
                        resparmtxt = (resparmtxt + self.model.bond_names[i] + " " +
                                  textmarkers["equal_symbol"][fmt] + "(" +
                                  show_number(val / jmax) + ") " +
                                  textmarkers["times_symbol"][fmt] + " " +
                                  show_number(jmax) + incopatibletxt)
                    else:
                        resparmtxt = (resparmtxt + self.model.bond_names[i] + " " +
                                  textmarkers["equal_symbol"][fmt] + "(" +
                                  show_number(val / jmax,tol=jerr[i] / jmax) + textmarkers["plusminus_symbol"][fmt] +
                                   ("%.2g" % (jerr[i] / jmax)) +
                                  ") " + textmarkers["times_symbol"][fmt] + " " + ("%.3e" % jmax) +
                                  textmarkers["separator_symbol"][fmt] + "\n")
            else:
                for i, val in enumerate(js):
                    resparmtxt = (resparmtxt + self.model.bond_names[i] + " " +
                              textmarkers["equal_symbol"][fmt] +
                              "(" + show_number(val / jmax, tol=jerr[i] / jmax) + " " +
                              textmarkers["plusminus_symbol"][fmt] + " " +
                              show_number(jerr[i] / jmax) + ") " +
                              textmarkers["times_symbol"][fmt] +
                              show_number(jmax) + "\n")

            if usemc:
                resparmtxt = resparmtxt + "\n\n Monte Carlo acceptance rate:" + str(ar)+"\n"


        # Inequations
        resparmtxt = resparmtxt  + "\n\n region constraints:\n"
        ineqs = self.model.bound_inequations(confs,
                                             energs,
                                             err_energs=tolerance)
        for ineq in ineqs:
            txtineq = ""
            coeff = ineq[0]
            for i, c in enumerate(coeff):
                if abs(c) < tolerance:
                    continue
                if abs(c-1) < tolerance:
                    if txtineq != "":
                        txtineq += " +"
                    txtineq += self.model.bond_names[i]
                    continue

                if abs(c+1) < tolerance:
                    txtineq += " -" + self.model.bond_names[i]
                    continue

                if txtineq != "" and c > 0 :
                    txtineq += " +" + show_number(c, tolerance) + " "
                else:
                    txtineq +=  " " + show_number(c, tolerance) + " "
                txtineq += textmarkers["times_symbol"][fmt] + " "
                txtineq += self.model.bond_names[i]
            txtineq = textmarkers["open_mod"][fmt] + txtineq
            if ineq[1] < 0:
                txtineq += " + " + show_number(-ineq[1])  + " "  + textmarkers["close_mod"][fmt] + " < "
            else:
                txtineq +=  " " + show_number(-ineq[1]) +  " "  + textmarkers["close_mod"][fmt] + " < "
            txtineq +=  show_number(ineq[2]) 
            resparmtxt += "\n\n" + txtineq
                        
        self.resparam.config(state=NORMAL)
        self.resparam.delete(1.0, END)
        self.resparam.insert(END, resparmtxt)
        self.resparam.config(state=DISABLED)

        #  Update chi panel
        chitext = ""
        labels = self.configurations[2]
        for j, chi in enumerate(chis):
            chitext = (chitext + textmarkers["Delta_symbol"][fmt] +
                       "E" + textmarkers["sub_symbol"][fmt] + str(j + 1) +
                       "/" + textmarkers["Delta_symbol"][fmt] +
                       "E" +textmarkers["equal_symbol"][fmt] + " " +
                       show_number(chi) + " " + textmarkers["open_comment"][fmt] +
                       labels[j] +
                       textmarkers["close_comment"][fmt] +
                       textmarkers["separator_symbol"][fmt] + "\n")
        self.chis.config(state=NORMAL)
        self.chis.delete(1.0, END)
        self.chis.insert(END, chitext)
        self.chis.config(state=DISABLED)
        self.plotbutton.config(state=NORMAL)

    def write(self, txt):
        self.status.insert(INSERT, txt)

    def curr_edit(self):
        page = self.nb.tab(self.nb.select())["text"]
        
        if page == "1. Define Model":
            return (self.modelcif)
        elif page == "2. Spin Configurations and Couplings":
            return (self.spinconfigs)
        elif page == "3. Set energies and evaluate.":
            return (self.spinconfigsenerg)
        return None

    def call_undo(self, *args):
        edt = self.curr_edit()
        if edt is None:
            return
        edt.edit_undo()

    def call_redo(self, *args):
        edt = self.curr_edit()
        if edt is None:
            return
        edt.edit_redo()

    def call_cut(self, *args):
        pass

    def call_copy(self, *args):
        current_focus = self.root.focus_get()
        self.nb.event_generate('<Control-c>')
        pass

    def call_paste(self, *args):
        pass

    def call_search(self, *args):
        edt = self.curr_edit()
        if edt is None:
            return
        FindDialog(self, edt, replace=False)
        return
        countVar = StringVar()
        pos = edt.search("J1", "1.0", stopindex=END, count=countVar)
        if pos == "":
            print("Not found")
            return
        edt.tag_configure("search", background="green")
        edt.tag_add("search", pos, "%s + %sc" % (pos, countVar.get()))
        pass

    def call_replace(self, *args):
        edt = self.curr_edit()
        if edt is None:
            return
        FindDialog(self, edt, replace=True)
        pass

    def close_app(self, *args):
        self.root.destroy()

    def show_help(self, *args):
        #docpath = os.path.dirname(os.path.realpath(sys.argv[0])) +
        #                "/doc/tutorial.html"
        docpath = spectrojotometer.__path__[0] +  "/doc/tutorial.html"
        webbrowser.open(docpath)

    def plot_delta_energies(self, *args):
        energy_tolerance = float(self.parameters["page3"]["Energy tolerance"].get())
        fulllabels = self.configurations[2]
        fullenergs = self.configurations[0]
        labels = []
        for i, energ in enumerate(fullenergs):
            if energ == energ:
                labels.append("#  " + fulllabels[i])
        indices = [i for i in range(len(labels))]
        # labels = args.get('labels', None)
        plt.plot(indices, [1. for i in indices])
        plt.plot(indices, [-1. for i in indices])
        plt.scatter(indices, self.chisvals)
        plt.xticks(indices, labels, rotation='vertical')
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.gcf().canvas.set_window_title("Model errors")
        plt.show()


ApplicationGUI()
