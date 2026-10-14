Changes
=======



Unreleased
----------

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
