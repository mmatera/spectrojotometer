"""Tk panel of visualbond for the Monte Carlo simulation of the Ising model."""

import json
import logging
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .couplings_editor import CouplingsEditor
from .mc_worker import MCWorker
from .scrollable import ScrollableFrame

logger = logging.getLogger(__name__)


class MonteCarloPanel(ttk.Frame):
    """Self-contained panel to run Monte Carlo simulations of the Ising model.

    Parameters
    ----------
    master : parent Tk widget
    get_model : callable or None
        Function without arguments returning the current MagneticModel
        (or None). It is called each time "Run" is pressed. If None, the
        data must be given with load_data().
    adapter : callable or None
        Function (model, supercell, J_overrides=None) -> (spins, bonds,
        J_vals, E0). By default, monte_carlo.adapter.magnetic_model_to_ising.
    """

    def __init__(self, master, get_model=None, adapter=None):
        super().__init__(master, padding=6)
        self.get_model = get_model
        self.adapter = adapter
        self.worker = None
        self.queue = queue.Queue()
        self.results = None
        self._plot_artists = {}

        self._current_model = None
        self._auto_sync = tk.BooleanVar(value=True)
        self._build_ui()
        # Show the couplings if a model is already loaded
        if get_model is not None:
            try:
                self.refresh_couplings(get_model())
            except Exception:
                pass
        self.after(100, self._poll_queue)

        
    # ------------------------------------------------------------------
    # Building the UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=260)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # The controls scroll: otherwise, in a short window, the last ones
        # (the couplings, Run, Save) are out of reach.
        self.controls = ScrollableFrame(self)
        self.controls.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left = self.controls.inner
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_params(left)
        self._build_plots(right)
        self._build_statusbar()

    def _build_params(self, parent):
        # --- Supercell ---
        f = ttk.LabelFrame(parent, text="Supercell")
        f.pack(fill="x", pady=(0, 4))
        row = ttk.Frame(f)
        row.pack(fill="x", padx=4, pady=4)
        self.var_Lx = tk.IntVar(value=2)
        self.var_Ly = tk.IntVar(value=2)
        self.var_Lz = tk.IntVar(value=2)
        for label, var in [("Lx", self.var_Lx), ("Ly", self.var_Ly),
                           ("Lz", self.var_Lz)]:
            ttk.Label(row, text=label).pack(side="left")
            ttk.Spinbox(row, from_=1, to=12, width=3, textvariable=var)\
                .pack(side="left", padx=(2, 8))

        # --- Temperatures ---
        f = ttk.LabelFrame(parent, text="Temperatures")
        f.pack(fill="x", pady=4)
        self.var_Tmin = tk.DoubleVar(value=0.5)
        self.var_Tmax = tk.DoubleVar(value=4.0)
        self.var_nT = tk.IntVar(value=20)
        for label, var in [("T_min", self.var_Tmin), ("T_max", self.var_Tmax),
                           ("n_T", self.var_nT)]:
            r = ttk.Frame(f)
            r.pack(fill="x", padx=4, pady=1)
            ttk.Label(r, text=label, width=6).pack(side="left")
            ttk.Entry(r, textvariable=var, width=10).pack(side="left")

        # --- Number of sweeps ---
        f = ttk.LabelFrame(parent, text="Sweeps")
        f.pack(fill="x", pady=4)
        self.var_equil = tk.IntVar(value=500)
        self.var_sweeps = tk.IntVar(value=2000)
        self.var_restarts = tk.IntVar(value=1)
        for label, var in [("equil", self.var_equil),
                           ("sweeps", self.var_sweeps),
                           ("restarts", self.var_restarts)]:
            r = ttk.Frame(f)
            r.pack(fill="x", padx=4, pady=1)
            ttk.Label(r, text=label, width=8).pack(side="left")
            ttk.Entry(r, textvariable=var, width=10).pack(side="left")

        # --- Algorithm ---
        f = ttk.LabelFrame(parent, text="Algorithm")
        f.pack(fill="x", pady=4)
        self.var_algo = tk.StringVar(value="auto")
        for opt in ["auto", "wolff", "pt", "metropolis"]:
            ttk.Radiobutton(f, text=opt, value=opt,
                            variable=self.var_algo).pack(anchor="w", padx=4)


        # --- Couplings ---
        self.editor = CouplingsEditor(parent, on_change=self._on_coupling_change)
        self.editor.pack(fill="both", expand=True, pady=4)

        # --- Auto-sync + Refresh ---
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(f, text="Auto-sync",
                        variable=self._auto_sync).pack(side="left")
        ttk.Button(f, text="Refresh",
                   command=self._on_manual_refresh).pack(side="right")            
        # --- Seed ---
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text="Seed").pack(side="left")
        self.var_seed = tk.StringVar(value="42")
        ttk.Entry(f, textvariable=self.var_seed, width=8).pack(side="left")

        # --- Buttons ---
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(10, 4))
        self.btn_run = ttk.Button(f, text="Run", command=self._on_run)
        self.btn_run.pack(fill="x", pady=2)
        self.btn_cancel = ttk.Button(f, text="Cancel",
                                     command=self._on_cancel, state="disabled")
        self.btn_cancel.pack(fill="x", pady=2)
        self.btn_save = ttk.Button(f, text="Save JSON",
                                   command=self._on_save, state="disabled")
        self.btn_save.pack(fill="x", pady=2)

    def _build_plots(self, parent):
        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
            from matplotlib.figure import Figure
        except ImportError:
            ttk.Label(parent, text="matplotlib is not available").pack()
            self.canvas = None
            return

        self.fig = Figure(figsize=(7, 5.5), dpi=100)
        self.ax_chi = self.fig.add_subplot(221)
        self.ax_m   = self.fig.add_subplot(222)
        self.ax_c   = self.fig.add_subplot(223)
        self.ax_u4  = self.fig.add_subplot(224)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

    def _build_statusbar(self):
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        bar.columnconfigure(0, weight=1)

        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.var_status).grid(
            row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(bar, mode="determinate", length=200)
        self.progress.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _on_run(self):
        if self.worker is not None and self.worker.is_alive():
            return

        model = self.get_model() if self.get_model is not None else self._current_model

        # Without a model, the data may have been given with load_data()
        if model is None and not hasattr(self, "_direct_data"):
            messagebox.showwarning(
                "Monte Carlo",
                "No magnetic model is loaded. Load one in the main window, "
                "or give the bonds and couplings with load_data().",
            )
            return

        # The couplings: the values of the editor
        J_overrides = None
        if hasattr(self, "editor") and self.editor._rows:
            J_overrides = self.editor.get_values()
            if J_overrides is None:
                messagebox.showerror(
                    "Monte Carlo",
                    "Some couplings have no valid value. Estimate them in "
                    "the main window, or type their values.",
                )
                return

        # The Ising model
        try:
            if hasattr(self, "_direct_data"):
                spins, bonds, J_vals, E0 = self._direct_data
            else:
                from ..monte_carlo.adapter import magnetic_model_to_ising
                adapter = self.adapter or magnetic_model_to_ising
                logger.debug("adapter: %s", adapter)
                supercell = (self.var_Lx.get(), self.var_Ly.get(),
                             self.var_Lz.get())
                logger.debug("building the Ising model")
                if J_overrides:
                    # The values of the editor (they were read and then
                    # ignored, so manual changes had no effect).
                    spins, bonds, J_vals, E0 = adapter(
                        model, supercell, J_overrides=J_overrides
                    )
                else:
                    spins, bonds, J_vals, E0 = adapter(model, supercell)
                logger.debug("Ising model built")
        except Exception as e:
            messagebox.showerror("Monte Carlo",
                                 f"Error building the model:\n{e}")
            return

        params = dict(
            spins=spins, bonds=bonds, J_vals=J_vals, E0=E0,
            T_min=self.var_Tmin.get(),
            T_max=self.var_Tmax.get(),
            n_temps=self.var_nT.get(),
            n_equil=self.var_equil.get(),
            n_sweeps=self.var_sweeps.get(),
            n_restarts=self.var_restarts.get(),
            algorithm=self.var_algo.get(),
            seed=self._parse_seed(),
        )

        self._clear_plots()
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.btn_save.config(state="disabled")
        self.progress.config(value=0, maximum=params["n_temps"])
        self.var_status.set("Running...")

        self.worker = MCWorker(params, self.queue)
        self.worker.start()

    def _on_cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.var_status.set("Cancelling...")

    def _on_save(self):
        if self.results is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._serialize(self.results), f, indent=2)
            self.var_status.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save", f"Error:\n{e}")

    # ------------------------------------------------------------------
    # Polling the queue (main thread)
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_message(self, msg):
        kind = msg[0]
        if kind == "start":
            pass
        elif kind == "progress":
            _, done, total, text = msg
            self.progress.config(maximum=total, value=done)
            self.var_status.set(text)
        elif kind == "done":
            self.results = msg[1]
            self._on_done()
        elif kind == "error":
            self.var_status.set("Error")
            messagebox.showerror("Monte Carlo", msg[1])
            self._reset_buttons()
        elif kind == "cancelled":
            self.var_status.set("Cancelled")
            self._reset_buttons()

    def _on_done(self):
        self._reset_buttons()
        r = self.results
        Tc = r.get("Tc_peak")
        cw = r.get("curie_weiss")
        parts = [f"N = {r['N']}", f"algo = {r['algorithm']}"]
        if r.get("frustrated"):
            parts.append("FRUSTRATED")
        if Tc is not None:
            parts.append(f"Tc = {Tc:.3f}")
        if cw:
            parts.append(f"Theta_CW = {cw['theta_CW']:+.3f}")
        self.var_status.set("  |  ".join(parts))
        self.btn_save.config(state="normal")
        self._plot_results()

    def _reset_buttons(self):
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _clear_plots(self):
        for ax in [self.ax_chi, self.ax_m, self.ax_c, self.ax_u4]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _plot_results(self):
        if self.canvas is None or self.results is None:
            return
        r = self.results
        T = np.asarray(r["T"])

        ax = self.ax_chi
        ax.clear()
        ax.plot(T, 1/np.array(r["chi_connected"]), "o-", label=r"$1/\chi_{conn}$")
        ax.plot(T, 1/np.array(r["chi"]), "s--", alpha=0.5, label=r"$1/\chi$")
        if r.get("Tc_peak") is not None:
            ax.axvline(r["Tc_peak"], color="r", ls=":", alpha=0.6,
                       label=f"$T_c$={r['Tc_peak']:.2f}")
        ax.set_xlabel("T")
        ax.set_ylabel(r"$1/\chi$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = self.ax_m
        ax.clear()
        ax.plot(T, r["M"], "o-", color="C1")
        ax.set_xlabel("T")
        ax.set_ylabel(r"$\langle|M|\rangle/N$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax = self.ax_c
        ax.clear()
        ax.plot(T, r["C"], "o-", color="C2")
        ax.set_xlabel("T")
        ax.set_ylabel("C")
        ax.grid(True, alpha=0.3)

        ax = self.ax_u4
        ax.clear()
        ax.plot(T, r["U4"], "o-", color="C3")
        ax.axhline(2.0 / 3.0, ls=":", color="grey", alpha=0.6)
        ax.set_xlabel("T")
        ax.set_ylabel(r"$U_4$")
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_data(self, spins, bonds, J_vals, E0=0.0):
        """Load the data of an Ising model directly, without a
        MagneticModel (for tests and standalone demos).
        """
        self._direct_data = (
            np.asarray(spins, dtype=np.int8),
            np.asarray(bonds, dtype=np.int64),
            np.asarray(J_vals, dtype=np.float64),
            float(E0),
        )

    def _parse_seed(self):
        s = self.var_seed.get().strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    @staticmethod
    def _serialize(obj):
        """Convert NumPy arrays and types to JSON-serializable ones."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: MonteCarloPanel._serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MonteCarloPanel._serialize(x) for x in obj]
        if isinstance(obj, tuple):
            return [MonteCarloPanel._serialize(x) for x in obj]
        return obj

    # ------------------------------------------------------------------
    # Synchronizing the couplings
    # ------------------------------------------------------------------

    def refresh_couplings(self, model=None):
        """Show the couplings of the model (their names and estimated
        values) in the editor.

        Called by visualbond after estimating the couplings and, with
        auto-sync on, when entering this tab; also by the Refresh button.
        """
        if model is None:
            if self.get_model is None:
                return
            model = self.get_model()
        self._current_model = model
        if model is None or not getattr(model, "bonds", None):
            self.editor.set_couplings({})
            return
        self.editor.set_couplings(model.bonds, reset_baseline=True)

    def _on_manual_refresh(self):
        self.refresh_couplings()

    def _on_coupling_change(self, name, value):
        """Called by the editor when a value is confirmed."""
        # Feedback in the status bar
        modified = self.editor.is_modified()
        if modified:
            self.var_status.set(f"Couplings edited by hand (last: {name} = {value:+.4f})")
        else:
            self.var_status.set("Couplings as in the model")
