# Tests for loading modules (CIF / struct)

This battery covers `spectrojotometer/model_io.py`: the functions that read  WIEN2k `.struct` files or CIF files, and build a `MagneticModel`.

## How to run

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Organization

- `conftest.py` — fixtures of paths (`examples_dir`, `fixtures_dir`)
  and a helper (`bond_length` / `all_bond_lengths`) that recompute the
  *real* distance between two atoms in a model already loaded, from its
  cartesian positions and the Bravais lattice, instead of trust in the 
  "declared" value in the input file.
- `fixtures/` — synthetic minimal geometries, chosen in a way that cartesian positions and bond distances can be verified by hand (differently from the files in  `examples/`, that are real data but not necesarily self consistents).
- `test_cif_loading.py` — `.cif` file parsing: atoms, species,
  symmetry expansions vs  `primitive_cell=True`, bonds.
- `test_struct_loading.py` — parsing `.struct` files (WIEN2k).
- `test_dispatch.py` — `magnetic_model_from_file`: check that
  the right delegated function is used according to the extension
  (case insensitive).
- `test_symmetry_helpers.py` — low-level pure functions:
  `parse_symmetry`, `centering_letter_from_symbol`,
  `expand_symmetries_with_centering`, `normalize_bond`,
  `pack_offset`/`unpack_offset`.
- `test_geometry_consistency.py` — cross invariants:: safe with
  `save_cif` and reread  must reproduce the geometry; conventional cells and primitives of a same structure must describe the same atomic density.

## pymatgen-based prototype battery

`spectrojotometer/model_io_pymatgen.py` is an experimental prototype
of `magnetic_model_from_cif` built on top of pymatgen's CIF parser.
It has its own battery, which needs `pip install -e ".[pymatgen]"`:

- `test_cif_loading_pymatgen.py` — same checks as
  `test_cif_loading.py`, but exercising the prototype on its own
  (not compared against anything), so a bug shared by both
  implementations doesn't go unnoticed.
- `test_cif_parser_equivalence.py` — compares the current
  implementation against the prototype, file by file (same atoms,
  same cell, same bonds, via `assert_models_equivalent` in
  `conftest.py`). Runs over the 8 CIFs used elsewhere in this suite,
  including the two that used to break the line-by-line reader
  (`h2o.cif`, `fe2o_synthetic.cif`) and a new one exercising the
  non-standard `_atom_site_g_factor`/`_atom_site_spin` columns.

If pymatgen isn't installed, both files skip themselves
(`pytest.importorskip`) without affecting the rest of the suite.

### Another bug found along the way

While building the `synthetic_bond_via_symmetry.cif` fixture (a bond
declared with `_geom_bond_site_symmetry_2` pointing at a
non-identity symmetry operator, e.g. `"2_555"`), another pre-existing
bug showed up — shared by both implementations, since the prototype
reuses `cif_read_loop_bonds`/`generate_atoms_by_symmetries` unchanged:
`cif_read_loop_bonds` builds the replica's atom label as
`<label>_<sym>` with `sym = cif_index - 1`, but
`generate_atoms_by_symmetries` had labeled that same replica as
`<label>_<sym + 1>`. Since the labels don't match, the bond is
silently dropped (0 bonds instead of 1) instead of failing loudly.
Documented as `xfail` in
`test_bond_via_non_identity_symmetry_is_currently_dropped`
(parametrized over both implementations) in
`test_cif_parser_equivalence.py`.

Neither of these was fixed here — the goal was to finish the
prototype's test battery, not to fix the code.

