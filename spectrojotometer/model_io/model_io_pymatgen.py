"""
model_io_pymatgen
`magnetic_model_from_cif_pymatgen`: a `magnetic_model_from_cif`
implementation built on top of pymatgen's CIF parser, used by
`dispatch.magnetic_model_from_file` as the default CIF reader (see
`dispatch.USE_PYMATGEN_CIF_READER`) whenever pymatgen is installed.
`cif.magnetic_model_from_cif` -- the original, hand-written
line-by-line parser -- is still available for direct use and as a
fallback when pymatgen isn't installed.

Design: let pymatgen do the part that is really "parsing a CIF
properly" (tokenizing `loop_` rows while respecting quotes, resolving
the cell, and computing the symmetry operators -- including centering,
which pymatgen derives from the space-group symbol instead of the
letter-based heuristic in `cif.expand_symmetries_with_centering`) and
reuse, UNCHANGED, all of spectrojotometer's own domain logic that
already lives in `cif.py` (reading atoms, reading bonds with their
`J0`/`J1` coupling labels, expanding atoms/bonds by symmetry, reducing
to a primitive cell).

To reuse those functions without touching them, `_loop_labels_entries`
takes a pymatgen `CifBlock` (which stores each `loop_` as columns,
already correctly tokenized) and reconstructs the same "labels,
entries" row-oriented format (see `cif.cif_read_loop_symmetries`) that
those functions expect.
"""
from typing import Optional

import numpy as np
from pymatgen.io.cif import CifFile, CifParser

from spectrojotometer.magnetic_model import MagneticModel
from .common import DEFAULT_MAGNETIC_ATOMS
from .cif import (
    cif_read_loop_atoms,
    cif_read_loop_bonds,
    cif_read_loop_bonds_compact,
    generate_atoms_by_symmetries,
    generate_bonds_by_symmetries,
    primitive_vectors_from_symmetries,
    primitive_vectors_from_symmetries_reason,
)


def _loop_labels_entries(block, tag_hint: str) -> tuple:
    """
    Find, among `block.loops`, the group of columns that contains
    `tag_hint`, and return it as (labels, entries) in the same
    row-oriented format already used by `cif_read_loop_atoms`,
    `cif_read_loop_bonds` and `cif_read_loop_bonds_compact`.

    Parameters
    ----------
    block : pymatgen.io.cif.CifBlock
        The CIF block to search (typically the single block of a
        spectrojotometer CIF file).
    tag_hint : str
        A CIF tag (e.g. `"_atom_site_fract_x"`) expected to belong to
        the loop of interest.

    Returns
    -------
    labels : list of str
        The tags of the loop that contains `tag_hint`, in their
        original column order.
    entries : list of list
        The rows of that loop, each a list with one value per label,
        aligned with `labels`. `([], [])` if no loop in `block`
        contains `tag_hint` (e.g. a CIF with no bonds block).
    """
    for group in block.loops:
        if tag_hint in group:
            labels = list(group)
            n_rows = len(block.data[group[0]])
            entries = [[block.data[tag][row] for tag in labels] for row in range(n_rows)]
            return labels, entries
    return [], []


def magnetic_model_from_cif_pymatgen(
    filename: str,
    magnetic_atoms: tuple = DEFAULT_MAGNETIC_ATOMS,
    bond_names: Optional[list] = None,
    primitive_cell: bool = False,
) -> MagneticModel:
    """
    Build a `MagneticModel` from a CIF file, the same way
    `cif.magnetic_model_from_cif` does, but delegating the cell,
    `loop_` tokenizing, and symmetry-operator (including centering)
    parsing to pymatgen instead of the hand-written line-by-line
    reader -- see the module docstring for the rationale, and
    `cif.magnetic_model_from_cif`'s own docstring for the meaning of
    `magnetic_atoms`, `bond_names` and `primitive_cell`, which are
    identical here.

    Parameters
    ----------
    filename : str
        The CIF file to read.
    magnetic_atoms : tuple, optional
        The set of atom species to be included.
        The default is ("Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V").
    bond_names : Optional[list], optional
        Currently unused (kept for signature compatibility with
        `cif.magnetic_model_from_cif`): bonds are always named from
        the CIF's own `_geom_bond_label` column, or auto-named by
        distance when that column is absent. The default is None.
    primitive_cell : bool, optional
        If True, reduce the model to a primitive cell (atoms are kept
        as the CIF's asymmetric unit, without symmetry expansion, and
        the lattice is replaced by a primitive basis derived from the
        symmetry operators) instead of the conventional cell. The
        default is False.

    Returns
    -------
    MagneticModel
        The model.

    Raises
    ------
    ValueError
        If pymatgen could not resolve a lattice for `filename`, or (with
        `primitive_cell=True`) if no primitive basis could be built
        from the symmetry operators -- see
        `cif.primitive_vectors_from_symmetries_reason` for the reason
        reported in the message.
    """
    cif_file = CifFile.from_file(filename)
    _, block = next(iter(cif_file.data.items()))
    parser = CifParser(filename)

    lattice_block = parser.get_lattice(block)
    if lattice_block is None:
        raise ValueError(f"{block} could not produce a lattice_block")
    conventional_vectors = np.array(lattice_block.matrix)
    symmetries = [
        (np.array(op.rotation_matrix), np.array(op.translation_vector))
        for op in parser.get_symops(block)
    ]
    space_group_symbol = (
        block.data.get("_symmetry_space_group_name_h-m")
        or block.data.get("_space_group_name_h-m_alt")
        or block.data.get("_space_group_name_h-m_ref")
    )

    # --- atoms (same function as the hand-written reader) --------------
    atom_labels, atom_entries = _loop_labels_entries(block, "_atom_site_fract_x")
    (
        atomlabels,
        magnetic_species,
        magnetic_positions,
        g_lande_factors,
        spin_repr,
    ) = cif_read_loop_atoms(atom_labels, atom_entries, magnetic_atoms)

    if not primitive_cell:
        (
            atomlabels,
            magnetic_species,
            magnetic_positions,
            g_lande_factors,
            spin_repr,
        ) = generate_atoms_by_symmetries(
            symmetries,
            atomlabels,
            magnetic_species,
            magnetic_positions,
            g_lande_factors,
            spin_repr,
        )

    # --- bonds (same functions as the hand-written reader) --------------
    bond_atom_labels, bond_entries = _loop_labels_entries(
        block, "_geom_bond_atom_site_label_1"
    )
    bond_labels, bond_distances, bondlists = None, [], None
    primitive_bravais_vectors = None

    if bond_atom_labels:
        if primitive_cell:
            primitive_basis = primitive_vectors_from_symmetries(
                symmetries, conventional_vectors
            )
            if primitive_basis is None:
                reason = primitive_vectors_from_symmetries_reason(
                    symmetries, conventional_vectors
                )
                raise ValueError(
                    f"cannot build a primitive-cell model from {filename}: {reason}"
                )
            primitive_frac_basis, primitive_bravais_vectors = primitive_basis
            bond_labels, bond_distances, bondlists = cif_read_loop_bonds_compact(
                bond_atom_labels,
                bond_entries,
                atomlabels,
                symmetries,
                primitive_frac_basis,
            )
        else:
            bond_labels, bond_distances, bondlists = cif_read_loop_bonds(
                bond_atom_labels, bond_entries, atomlabels
            )
            # Unlike the hand-written reader, there's no need to call
            # `cif.expand_symmetries_with_centering` here:
            # `parser.get_symops` already includes the centering
            # translations.
            generate_bonds_by_symmetries(
                symmetries, bond_labels, bond_distances, bondlists, magnetic_positions
            )

    # --- final assembly (identical to the hand-written reader) ---------
    if primitive_cell:
        if primitive_bravais_vectors is None:
            primitive_basis = primitive_vectors_from_symmetries(
                symmetries, conventional_vectors
            )
            if primitive_basis is None:
                reason = primitive_vectors_from_symmetries_reason(
                    symmetries, conventional_vectors
                )
                raise ValueError(
                    f"cannot build a primitive-cell model from {filename}: {reason}"
                )
            _, primitive_bravais_vectors = primitive_basis
        bravais_vectors = primitive_bravais_vectors
    else:
        bravais_vectors = conventional_vectors

    if len(magnetic_positions) != 0:
        magnetic_positions = np.array(magnetic_positions).dot(
            np.array(conventional_vectors)
        )

    return MagneticModel(
        magnetic_positions,
        bravais_vectors,
        bond_lists=bondlists,
        bond_names=bond_labels,
        bond_distances=(
            [float(d) for d in bond_distances] if bondlists is not None else None
        ),
        magnetic_species=magnetic_species,
        g_lande_factors=g_lande_factors,
        spin_repr=spin_repr,
        symmetries=None,
        space_group_symbol=space_group_symbol,
    )
