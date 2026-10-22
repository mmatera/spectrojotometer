Changes
=======



0.3.0 (unreleased)
------------------

* Packaging:

  * The command line tools work after ``pip install``: they moved from
    ``bin/`` to ``spectrojotometer.cli`` (also runnable as
    ``python -m spectrojotometer.cli.<tool>``); four of the six entry
    points pointed to nonexistent modules. The command names do not
    change. ``visualbond`` is installed as a GUI script (no console
    window on Windows).
  * Only the ``spectrojotometer`` package is installed; ``bin``,
    ``tests`` and ``examples`` were installed as top-level packages.
  * The tutorial shown by visualbond (Help -> Documentation) includes
    its images.
  * Metadata: version 0.3.0 (single source: ``spectrojotometer.__version__``,
    also shown by visualbond), license GPL-2.0-or-later, Python >= 3.11
    (required by numpy, scipy and pymatgen), project URLs, README as long
    description. ``argparse`` is no longer listed as a dependency.
  * ``winebuild`` uses ``pip install`` and the new location of the
    visualbond script, and bundles the documentation.
  * The CI builds the sdist and the wheel, installs the wheel in a clean
    environment and runs the installed commands.

* Importing ``spectrojotometer.gui`` no longer opens a Tk window.
* ``ApplicationGUI`` no longer enters the Tk event loop from its
  constructor: call ``ApplicationGUI().run()``. The window can now be
  built and driven in tests (``tests/test_gui.py``).
* Monte Carlo simulation of the fitted model (``spectrojotometer.monte_carlo``,
  and the Monte Carlo tab of visualbond): Ising Metropolis, Wolff (with a
  gauge transformation for unfrustrated antiferromagnets) and parallel
  tempering, detection of frustration, critical temperature. New
  dependency: numba. Along with its integration:

  * The adapter lost the lattice offsets of the bonds (it stripped the
    prefix of the packed offset): bonds to neighbouring cells were placed
    inside the same cell (37.5% of the bonds of a 2x2x2 supercell of
    ``cromita_ortogonal``). It also accepts offsets as tuples, takes the
    number of sites from the model, and rejects couplings without a value.
  * The values of the couplings editor of the Monte Carlo tab were read
    and ignored. The editor itself (``gui/couplings_editor.py``) was
    missing and has been rewritten.
  * ``compute_couplings`` stores the estimated couplings in
    ``model.bonds[name]["value"]`` (``nan`` until estimated, or if
    undetermined).
  * The synchronization of the couplings when entering the Monte Carlo
    tab never ran.
  * With several Metropolis replicas, the warning about disagreeing
    replicas was always issued (it compared different temperatures).
  * ``run_ising_mc`` documented.
  * The Monte Carlo tab, its messages and the docstrings of the module
    in English, like the rest of the package. The docstring of the
    adapter described the wrong encoding of the offsets ('100' for +1
    along a, instead of '1_100'), the one that caused the bug above.
  * The controls of the Monte Carlo tab scroll (``gui/scrollable.py``):
    in a short window, the last ones (the couplings, Run, Save) were out
    of reach.

* visualbond with Tk 9: the import window and the search dialog failed
  (``trace`` replaced by ``trace_add``). Python 3.14 added to the CI.

* Documentation:

  * Installation instructions (Python >= 3.11, editable install for
    development, running the GUI tests on a headless machine).
  * ``map_configs`` and ``print_equations`` are documented like the
    other commands; every ``--help`` shown is the actual one.
  * The documented workflow works: with ``--rmax 4.9``, the example of
    ``bond_generator`` defined more couplings than a 3-site cell can
    determine, and the ``--known`` example had a spurious argument. It
    is now run by the tests.
  * ``README.rst``: the logo was in Markdown syntax, and the title and
    the link to the Windows version differed from ``README.md``.
  * ``tests/README.md`` describes every test file (it only covered the
    readers of model files, and described the pymatgen-based reader as
    an optional prototype).
  * visualbond: the optimal subset showed its cost as
    ``sqrt(l)/||A^-1||``; it now shows ``|ΔJ|/|ΔE| <``, like the rest.

* Mapping configurations between models, in the library:
  ``site_map(source, target, tol)`` and ``map_configurations(source,
  confs, target, tol, fill=None)``. Sites match up to a translation of
  the source lattice (before, only within the supercell). A site of the
  target without counterpart is an error unless ``fill`` is given;
  before, it silently took the spin of the last site. Used by
  ``map_configs`` and by visualbond's import window.
* The command line tools work again: ``bond_generator`` (stale
  attributes), ``print_equations`` (wrong argument name), ``map_configs``
  (stale attributes, and a TypeError; its output is now a configuration
  file, with the energies scaled by the ratio of cell sizes; new
  ``--tol`` and ``--fill``). The import window of visualbond works
  again (stale attributes, a misspelt one). Every command is now run
  for real in the tests and in the CI on the installed package.
* Choosing configurations, one consistent measure: the largest error of
  each coupling per unit error in the energies.

  * New ``MagneticModel.error_bounds(confs, method)``: exact, by linear
    programming, or the closed-form ellipse bound (always larger, about a
    thousand times faster). ``cost(confs)`` is now its maximum, exact.
    Before, three different quantities were used (sqrt(N)/sigma_min of
    the matrix with and without the base energy column), none of them
    the actual error.
  * ``optimize_independent_set`` and ``find_optimal_configurations``:
    backward elimination plus exchanges, scored by the ellipse bound.
    In tests on rings with 4-5 couplings, the chosen sets have 14-66%
    smaller error bars for a given number of configurations. Without a
    given number, the size is the smallest one that determines every
    coupling, enlarged while each configuration reduces the bound by at
    least 5% (``min_improvement``): about 30% fewer configurations, with
    about 30% smaller errors, than before.
  * When there is nothing new to add, the cost of the known set is
    reported (it was ``inf``).
  * ``optimize-configurations`` works again (stale attributes, swapped
    return values, missing ``--known``), and chooses the number of
    configurations automatically unless ``-n`` is given.

* English names: ``coefficient_matrix(confs, centered=...)`` (was
  ``normalizar``) and ``MagneticModel.remove_bond_by_name`` (was
  ``remover_bond_by_name``). The old names still work, with a
  ``DeprecationWarning``. The key ``"coord_atomos"`` of
  ``site_properties`` is unchanged.
* ``MagneticModel``: the constructor no longer modifies the
  ``g_lande_factors`` and ``spin_repr`` lists it receives. Its
  ``ranges`` and ``discretization`` arguments never had an effect and
  are deprecated (use ``generate_bonds``). The constructor was split
  into small helpers (complexity 26 -> 5), with tests that fix its
  behaviour.

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
