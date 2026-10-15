"""Panel Tkinter para el modulo Monte Carlo Ising."""

import json
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from .mc_worker import MCWorker


class MonteCarloPanel(ttk.Frame):
    """Panel autocontenido para correr Monte Carlo Ising.

    Parameters
    ----------
    master : tk widget padre
    get_model : callable o None
        Funcion sin argumentos que devuelve el MagneticModel actual (o
        None si no hay). Se llama cada vez que se aprieta "Run". Si es
        None, el panel asume que el usuario proveera datos via
        load_data().
    adapter : callable o None
        Funcion (model, supercell) -> (spins, bonds, J_vals, E0). Por
        default usa spectrojotometer.monte_carlo.adapter.
    """

    def __init__(self, master, get_model=None, adapter=None):
        super().__init__(master, padding=6)
        self.get_model = get_model
        self.adapter = adapter
        self.worker = None
        self.queue = queue.Queue()
        self.results = None
        self._plot_artists = {}

        self._build_ui()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Construccion de UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=260)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))
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

        # --- Temperaturas ---
        f = ttk.LabelFrame(parent, text="Temperaturas")
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

        # --- Presupuesto ---
        f = ttk.LabelFrame(parent, text="Presupuesto MC")
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

        # --- Algoritmo ---
        f = ttk.LabelFrame(parent, text="Algoritmo")
        f.pack(fill="x", pady=4)
        self.var_algo = tk.StringVar(value="auto")
        for opt in ["auto", "wolff", "pt", "metropolis"]:
            ttk.Radiobutton(f, text=opt, value=opt,
                            variable=self.var_algo).pack(anchor="w", padx=4)

        # --- Seed ---
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text="Seed").pack(side="left")
        self.var_seed = tk.StringVar(value="42")
        ttk.Entry(f, textvariable=self.var_seed, width=8).pack(side="left")

        # --- Botones ---
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
                FigureCanvasTkAgg, NavigationToolbar2Tk)
            from matplotlib.figure import Figure
        except ImportError:
            ttk.Label(parent, text="matplotlib no disponible").pack()
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

        self.var_status = tk.StringVar(value="Listo")
        ttk.Label(bar, textvariable=self.var_status).grid(
            row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(bar, mode="determinate", length=200)
        self.progress.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # Callbacks de botones
    # ------------------------------------------------------------------

    def _on_run(self):
        if self.worker is not None and self.worker.is_alive():
            return

        # Obtener modelo
        model = self.get_model() if self.get_model is not None else None
        if model is None and not hasattr(self, "_direct_data"):
            messagebox.showwarning(
                "Monte Carlo",
                "No hay modelo magnetico cargado. Cargar uno en la GUI "
                "principal o usar load_data() para proveer bonds y J.",
            )
            return

        # Preparar datos Ising
        try:
            if hasattr(self, "_direct_data"):
                spins, bonds, J_vals, E0 = self._direct_data
            else:
                from ..monte_carlo.adapter import magnetic_model_to_ising
                adapter = self.adapter or magnetic_model_to_ising
                print("adapter:", adapter)
                supercell = (self.var_Lx.get(), self.var_Ly.get(),
                             self.var_Lz.get())
                print("create adapter")
                spins, bonds, J_vals, E0 = adapter(model, supercell)
                print("done")
        except Exception as e:
            messagebox.showerror("Monte Carlo",
                                 f"Error preparando el modelo:\n{e}")
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
        self.var_status.set("Corriendo...")

        self.worker = MCWorker(params, self.queue)
        self.worker.start()

    def _on_cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.var_status.set("Cancelando...")

    def _on_save(self):
        if self.results is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._serialize(self.results), f, indent=2)
            self.var_status.set(f"Guardado: {path}")
        except Exception as e:
            messagebox.showerror("Guardar", f"Error:\n{e}")

    # ------------------------------------------------------------------
    # Polling del queue (main thread)
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
            self.var_status.set("Cancelado")
            self._reset_buttons()

    def _on_done(self):
        self._reset_buttons()
        r = self.results
        Tc = r.get("Tc_peak")
        cw = r.get("curie_weiss")
        parts = [f"N = {r['N']}", f"algo = {r['algorithm']}"]
        if r.get("frustrated"):
            parts.append("FRUSTRADO")
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
    # Utilidades
    # ------------------------------------------------------------------

    def load_data(self, spins, bonds, J_vals, E0=0.0):
        """Carga datos Ising directamente, sin pasar por MagneticModel.

        Util para tests y para el demo standalone.
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
        """Convierte arrays y tipos NumPy a JSON-serializables."""
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
