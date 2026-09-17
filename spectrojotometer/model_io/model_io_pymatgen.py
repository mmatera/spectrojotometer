"""
Prototipo de `magnetic_model_from_cif` construido sobre el parser de
CIF de pymatgen, para evaluar cuánto del lector línea-por-línea de
`model_io.py` se podría retirar.

Idea del diseño: dejar que pymatgen se encargue de la parte que
realmente es "parsear un CIF de verdad" (tokenizar filas de `loop_`
respetando comillas, resolver la celda, y calcular los operadores de
simetría -- incluyendo el centrado, que pymatgen deriva del símbolo de
grupo espacial en vez de una heurística por letra) y reutilizar, SIN
MODIFICAR, toda la lógica de dominio específica de spectrojotometer
que ya existe en `model_io.py` (leer átomos, leer enlaces con sus
etiquetas de acoplamiento `J0`/`J1`, expandir átomos/enlaces por
simetría, reducir a celda primitiva).

Para poder reutilizar esas funciones sin tocarlas, `_loop_labels_entries`
reconstruye, a partir de un `CifBlock` de pymatgen (que guarda cada
loop_ como columnas, ya perfectamente tokenizadas), el mismo formato
"labels, entries" (una lista de filas) que esas funciones esperan.
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
    Encuentra, dentro de `block.loops`, el grupo de columnas que
    contiene `tag_hint`, y lo devuelve como (labels, entries) en el
    mismo formato "lista de filas" que ya usan `cif_read_loop_atoms`,
    `cif_read_loop_bonds` y `cif_read_loop_bonds_compact`.

    Devuelve ([], []) si ningún loop_ del bloque tiene esa columna
    (p.ej. un CIF sin bloque de enlaces).
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
    Igual que `model_io.magnetic_model_from_cif`, pero delegando en
    pymatgen la lectura de la celda, el tokenizado de cada `loop_` y
    el cálculo de los operadores de simetría (incluido el centrado).
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

    # --- átomos (misma función que la versión actual) -----------------
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

    # --- enlaces (mismas funciones que la versión actual) --------------
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
            # A diferencia de la versión actual, no hace falta
            # `expand_symmetries_with_centering`: `parser.get_symops`
            # ya incluye las traslaciones de centrado.
            generate_bonds_by_symmetries(
                symmetries, bond_labels, bond_distances, bondlists, magnetic_positions
            )

    # --- ensamblado final (idéntico a la versión actual) ---------------
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
