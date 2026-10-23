"""
A vertically scrollable container for Tk widgets.

Put the widgets in ``ScrollableFrame(parent).inner``. The scrollbar is shown
only when the content does not fit, and the mouse wheel scrolls it while the
pointer is over the container.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A frame whose content (the ``inner`` frame) scrolls vertically.

    The width of the container follows the width its content requests, so
    nothing is clipped horizontally; only the height is limited by the
    space available.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        background = ttk.Style().lookup("TFrame", "background") or None
        self.canvas = tk.Canvas(
            self, highlightthickness=0, borderwidth=0, background=background
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # The wheel scrolls this container only while the pointer is over it.
        self.bind("<Enter>", self._bind_wheel)
        self.bind("<Leave>", self._unbind_wheel)

    # Sizes ---------------------------------------------------------------

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(
            scrollregion=self.canvas.bbox("all"),
            width=self.inner.winfo_reqwidth(),
        )
        self._update_scrollbar()

    def _on_canvas_configure(self, event):
        # The content takes the whole width of the canvas.
        self.canvas.itemconfigure(self._window, width=event.width)
        self._update_scrollbar()

    def needs_scrolling(self):
        """Whether the content is taller than the visible area."""
        return self.inner.winfo_reqheight() > self.canvas.winfo_height() > 1

    def _update_scrollbar(self):
        if self.needs_scrolling():
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    # Mouse wheel ---------------------------------------------------------

    def _bind_wheel(self, _event=None):
        self.bind_all("<MouseWheel>", self._on_wheel)  # Windows, macOS
        self.bind_all("<Button-4>", self._on_wheel)  # X11
        self.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event=None):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.unbind_all(sequence)

    def _on_wheel(self, event):
        if not self.needs_scrolling():
            return
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        elif event.delta:
            # Windows: multiples of 120; macOS: small values.
            step = -int(event.delta / 120) if abs(event.delta) >= 120 else -event.delta
        else:
            return
        self.canvas.yview_scroll(step, "units")
