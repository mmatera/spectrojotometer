"""
spectrojotometer: effective magnetic models from ab initio energies.

The package reports progress and diagnostics through the standard
``logging`` module, under the ``spectrojotometer`` logger, and never
configures logging by itself. Applications decide where the messages go,
e.g. ``logging.basicConfig(level=logging.INFO)`` in a script.
"""
import logging

__version__ = "0.3.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
