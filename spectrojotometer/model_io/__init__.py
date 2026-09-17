"""
model_io
Tools to load models from files.

This used to be a single ~1400-line module. It's now a small package
split by responsibility:

- `common`   -- things shared by both readers (Bravais-vector
                construction from cell parameters, spin-configuration
                files) plus the default set of magnetic species.
- `cif`      -- everything CIF-specific: symmetry parsing/expansion
                (including centering), atom and bond loops, primitive
                cell reduction.
- `struct`   -- the WIEN2k `.struct` reader.
- `dispatch` -- `magnetic_model_from_file`, which picks `cif` or
                `struct` based on the filename extension.

Everything that used to be importable as `spectrojotometer.model_io.X`
still is: this file re-exports the full previous public surface, so
existing code (`from spectrojotometer.model_io import magnetic_model_from_cif`,
etc.) keeps working unchanged.
"""
import logging

logging.basicConfig(level=logging.INFO)

from .common import (
    DEFAULT_MAGNETIC_ATOMS,
    confindex,
    read_bravais_vectors,
    read_spin_configurations_file,
)
from .cif import (
    centering_letter_from_symbol,
    cif_read_loop_atoms,
    cif_read_loop_bonds,
    cif_read_loop_bonds_compact,
    cif_read_loop_symmetries,
    expand_symmetries_with_centering,
    find_atom_offset_by_symmetry,
    frac_offset_to_primitive_int,
    generate_atoms_by_symmetries,
    generate_bonds_by_symmetries,
    magnetic_model_from_cif,
    normalize_bond,
    parse_symmetry,
    primitive_vectors_from_symmetries,
    primitive_vectors_from_symmetries_reason,
)
from .struct import magnetic_model_from_wk2_struct
from .dispatch import magnetic_model_from_file

__all__ = [
    "DEFAULT_MAGNETIC_ATOMS",
    "centering_letter_from_symbol",
    "cif_read_loop_atoms",
    "cif_read_loop_bonds",
    "cif_read_loop_bonds_compact",
    "cif_read_loop_symmetries",
    "confindex",
    "expand_symmetries_with_centering",
    "find_atom_offset_by_symmetry",
    "frac_offset_to_primitive_int",
    "generate_atoms_by_symmetries",
    "generate_bonds_by_symmetries",
    "magnetic_model_from_cif",
    "magnetic_model_from_file",
    "magnetic_model_from_wk2_struct",
    "normalize_bond",
    "parse_symmetry",
    "primitive_vectors_from_symmetries",
    "primitive_vectors_from_symmetries_reason",
    "read_bravais_vectors",
    "read_spin_configurations_file",
]
