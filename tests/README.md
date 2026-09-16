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

