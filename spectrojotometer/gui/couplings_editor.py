"""
Editor of the coupling constants used by the Monte Carlo panel.

It shows one row per coupling of the model (name, distance, number of
bonds, and an editable value of J). The values of the model are the
baseline: is_modified() tells whether the user changed any of them, and
get_values() returns the values to simulate with.
"""

import logging
import math
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


def _format_value(value):
    """Text of a value in its entry: empty when it is not known (nan)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(value) else f"{value:.6g}"


def _parse_value(text):
    """The value typed in an entry, or None if it is not a finite number."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class CouplingsEditor(ttk.LabelFrame):
    """
    Parameters
    ----------
    master : the parent widget.
    on_change : called as on_change(name, value) when the user confirms a
        valid value of a coupling (leaving its entry, or with Enter).
    """

    def __init__(self, master, on_change=None, **kwargs):
        kwargs.setdefault("text", "Couplings (J)")
        super().__init__(master, **kwargs)
        self.on_change = on_change
        # name -> {"var": StringVar, "entry": Entry, "baseline": float}
        self._rows = {}
        self._table = ttk.Frame(self)
        self._table.pack(fill="both", expand=True, padx=4, pady=2)
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(
            buttons, text="Reset to model values", command=self.reset
        ).pack(side="right")
        self._empty_label = ttk.Label(
            self._table, text="No couplings: load a model with bonds."
        )
        self._empty_label.grid(row=0, column=0, sticky="w")

    def set_couplings(self, bonds, reset_baseline=False):
        """
        Show the couplings of bonds (MagneticModel.bonds: name ->
        {"distance", "bonds", "value"}).

        With reset_baseline, the values of bonds become the baseline and
        the entries show them. Otherwise, the baseline of a coupling that
        was already shown is kept, and so is the value typed by the user.
        """
        previous = {
            name: (row["var"].get(), row["baseline"])
            for name, row in self._rows.items()
        }
        for child in self._table.winfo_children():
            child.destroy()
        self._rows = {}
        if not bonds:
            self._empty_label = ttk.Label(
                self._table, text="No couplings: load a model with bonds."
            )
            self._empty_label.grid(row=0, column=0, sticky="w")
            return

        for col, title in enumerate(("Name", "Distance", "Bonds", "J")):
            ttk.Label(self._table, text=title).grid(
                row=0, column=col, sticky="w", padx=2
            )
        for row_index, (name, info) in enumerate(bonds.items(), start=1):
            model_value = info.get("value", float("nan"))
            if reset_baseline or name not in previous:
                text, baseline = _format_value(model_value), model_value
            else:
                text, baseline = previous[name]
            var = tk.StringVar(value=text)
            ttk.Label(self._table, text=name).grid(
                row=row_index, column=0, sticky="w", padx=2
            )
            distance = info.get("distance")
            ttk.Label(
                self._table,
                text="" if distance is None else f"{float(distance):.4f}",
            ).grid(row=row_index, column=1, sticky="w", padx=2)
            ttk.Label(self._table, text=str(len(info.get("bonds", [])))).grid(
                row=row_index, column=2, sticky="w", padx=2
            )
            entry = ttk.Entry(self._table, textvariable=var, width=12)
            entry.grid(row=row_index, column=3, sticky="we", padx=2)
            entry.bind("<FocusOut>", lambda _e, n=name: self._confirm(n))
            entry.bind("<Return>", lambda _e, n=name: self._confirm(n))
            self._rows[name] = {"var": var, "entry": entry, "baseline": baseline}

    def get_values(self):
        """
        {name: J} with the values of the entries, or None if some entry is
        empty or not a finite number.
        """
        values = {}
        for name, row in self._rows.items():
            value = _parse_value(row["var"].get())
            if value is None:
                return None
            values[name] = value
        return values

    def is_modified(self):
        """Whether some value differs from the baseline (the model's)."""
        for row in self._rows.values():
            value = _parse_value(row["var"].get())
            baseline = _parse_value(row["baseline"])
            if value is None and baseline is None:
                continue
            if value is None or baseline is None:
                return True
            if not math.isclose(value, baseline, rel_tol=1e-9, abs_tol=1e-12):
                return True
        return False

    def reset(self):
        """Show the baseline values again."""
        for row in self._rows.values():
            row["var"].set(_format_value(row["baseline"]))
        if self.on_change is not None and self._rows:
            name = next(iter(self._rows))
            value = _parse_value(self._rows[name]["var"].get())
            if value is not None:
                self.on_change(name, value)

    def _confirm(self, name):
        value = _parse_value(self._rows[name]["var"].get())
        if value is None:
            logger.warning("The value of %s is not a number.", name)
            return
        if self.on_change is not None:
            self.on_change(name, value)
