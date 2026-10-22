Changes
=======



Unreleased
----------

* **Fixed** ``import spectrojotometer.model_io`` failing with
  ``AttributeError: module 'importlib' has no attribute 'util'`` in a
  regular (non-editable) installation, which made the library, the
  command line tools and visualbond unusable after ``pip install``.
  ``tests/test_imports.py`` imports every module in a fresh interpreter.
* The library reports through ``logging`` (one logger per module, under
  ``spectrojotometer``) instead of printing, and no longer calls
  ``logging.basicConfig`` on import, which configured logging for any
  program importing it. Diagnostic dumps are at the ``DEBUG`` level.
* visualbond shows the log messages in its panel through a thread-safe
  handler, instead of redirecting ``sys.stdout``/``sys.stderr``.
  Messages that were lost before now reach the panel: warnings from the
  library and the GUI (e.g. incompatible or undetermined couplings), and
  messages from worker threads. *Edit → Detailed log messages* shows the
  diagnostics. Errors in callbacks are shown in the panel and in the
  terminal.
* Fixed "Send to application" and "Import configs from file" in the
  configuration import window, which failed calling the nonexistent
  ``logging.message``.
* The command line tools accept ``-v`` / ``-vv``.
* **Fixed** the Monte Carlo error bars (``montecarlo_box``), which
  depended on the energy units: the sampling radius scaled as the square
  of the tolerance. With energies in eV they were underestimated by
  orders of magnitude. Points are now drawn uniformly in an ellipsoid
  that is guaranteed to contain the compatibility region, so the
  results are invariant under a change of units.
* **Fixed** ``bound_inequalities``, which could return a spurious
  constraint with huge coefficients depending on the sign chosen by the
  singular value decomposition.
* New default method for the error bars: the exact bounding box of the
  compatibility region, by linear programming
  (``compute_couplings(method="exact")``, ``tools.compatibility_box``).
  ``method="montecarlo"`` and ``method="ellipse"`` give inner and outer
  estimates of it. The ``montecarlo=`` argument is still accepted. The
  Tkinter GUI offers the three methods, with the exact box as default.
* ``compute_couplings`` and ``montecarlo_box`` accept ``rng`` (seed or
  ``numpy.random.Generator``) for reproducible Monte Carlo results.
  Undetermined couplings are consistently reported with a ``nan`` error
  bar.
* ``scipy`` is now an explicit dependency (it was already required
  through ``pymatgen``).
* ``evaluate_cc`` works again (it used removed attributes, and the
  command line tools failed to import ``eprint``), and accepts
  ``--method`` and ``--mcsteps``.
* Documented the effective model, the compatibility region and the
  error-bound methods in the README and the tutorial.
* Added ``primitive_cell`` parameter to ``magnetic_model_from_file`` /
  ``magnetic_model_from_cif``: when the CIF's symmetry loop lists pure
  translations (a centered Bravais lattice: F, I, C, A or B), atoms can
  now be kept as the CIF's asymmetric unit instead of being expanded to
  the conventional cell, with the model's Bravais vectors switched to a
  primitive basis. Also exposed as a checkbox in the Tkinter GUI and as
  a field in the visualbondweb API.
* Bonds related by lattice centering (not just plain translations) are
  now found automatically when generating bonds from a CIF, completing
  the symmetries read from the file with the centering implied by the
  space group's Hermann-Mauguin symbol when available.
* ``MagneticModel`` now stores ``space_group_symbol`` (and, when
  relevant, ``symmetries``) in ``lattice_properties`` for informational/
  round-trip purposes.


0.2 (beta)
-----------

* Improving code organization.
* generate bonds in systems with peridic bound conditions.




0.0 (alpha only)
----------------

- Initial version
