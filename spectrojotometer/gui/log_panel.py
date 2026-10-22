"""
Show the messages of the ``logging`` module in a Tk text widget.

The library reports progress and diagnostics through ``logging``. The GUI
attaches a handler to the ``spectrojotometer`` logger that puts each record
in a queue, and the Tk event loop moves them to the text widget. Tk widgets
must only be touched from the main thread, so this also works for messages
emitted from worker threads (e.g. the Monte Carlo worker).
"""

import logging
import queue
from tkinter import END

PACKAGE_LOGGER = "spectrojotometer"

_LEVEL_TAGS = {
    logging.WARNING: ("warning", "#ffd75f"),
    logging.ERROR: ("error", "#ff6b6b"),
    logging.CRITICAL: ("error", "#ff6b6b"),
}


class _PanelFormatter(logging.Formatter):
    """Plain messages for INFO, prefixed by the level otherwise."""

    def format(self, record):
        text = super().format(record)
        if record.levelno == logging.INFO:
            return text
        return f"{record.levelname}: {text}"


class QueueLogHandler(logging.Handler):
    """Put formatted records in a queue. Safe to call from any thread."""

    def __init__(self, out_queue):
        super().__init__()
        self.queue = out_queue
        self.setFormatter(_PanelFormatter("%(message)s"))

    def emit(self, record):
        try:
            self.queue.put((record.levelno, self.format(record)))
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


class LogPanel:
    """
    Connect the ``spectrojotometer`` logger to a Tk text widget.

    Parameters
    ----------
    root : the Tk root (or any widget), used to schedule the polling.
    text_widget : the Text / ScrolledText where messages are shown.
    level : initial level of the package logger.
    poll_ms : polling interval of the queue, in milliseconds.
    """

    def __init__(self, root, text_widget, level=logging.INFO, poll_ms=100):
        self.root = root
        self.text = text_widget
        self.poll_ms = poll_ms
        self.queue = queue.Queue()
        self.handler = QueueLogHandler(self.queue)
        self.logger = logging.getLogger(PACKAGE_LOGGER)
        self.logger.addHandler(self.handler)
        self.set_level(level)
        for tag, color in _LEVEL_TAGS.values():
            self.text.tag_configure(tag, foreground=color)
        self._after_id = None
        self._poll()

    def set_level(self, level):
        """Set the level of the package logger (e.g. logging.DEBUG)."""
        self.logger.setLevel(level)

    def drain(self):
        """Move the pending messages to the text widget."""
        while True:
            try:
                levelno, text = self.queue.get_nowait()
            except queue.Empty:
                return
            tag = _LEVEL_TAGS.get(levelno, (None, None))[0]
            self.text.insert(END, text + "\n", (tag,) if tag else ())
            self.text.see(END)

    def _poll(self):
        self.drain()
        self._after_id = self.root.after(self.poll_ms, self._poll)

    def close(self):
        """Detach the handler and stop polling."""
        self.logger.removeHandler(self.handler)
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:  # the root may already be destroyed
                pass
            self._after_id = None
