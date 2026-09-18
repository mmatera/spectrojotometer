"""
cif
Everything needed to read a CIF file and build a MagneticModel from
it: parsing symmetry operators, atom and bond loops, expanding atoms
and bonds by symmetry (including centering), and reducing to a
primitive cell.
"""
from typing import Optional, Tuple, Union
from itertools import combinations
import logging
import numpy as np

from ..magnetic_model import MagneticModel
from ..tools import (
    format_symmetry_operator,
    pack_offset,
    split_cif_loop_line,
    unpack_offset,
    unpack_symmetry_and_offset,
)
from .common import DEFAULT_MAGNETIC_ATOMS, read_bravais_vectors

logging.basicConfig(level=logging.INFO)


def find_atom_offset_by_symmetry(p, symop, magnetic_positions) -> tuple:
    """
    Apply a symmetry operator to a fractional position and identify
    which already-known magnetic site it lands on (up to a lattice
    translation).

    Used by `generate_bonds_by_symmetries` to find the symmetry image
    of a bond's endpoints among the atoms already expanded by
    `generate_atoms_by_symmetries`: since every image of a magnetic
    atom is already present in `magnetic_positions` (mod 1), the
    symmetry-transformed point must coincide with one of them up to an
    integer cell offset.

    Parameters
    ----------
    p : array-like
        The fractional position (length-3) to transform.
    symop : tuple
        A `(rotation_matrix, translation_vector)` pair, as returned by
        `parse_symmetry` / `cif_read_loop_symmetries`.
    magnetic_positions : list of array-like
        The fractional positions of every magnetic atom currently in
        the (already symmetry-expanded) model.

    Returns
    -------
    j : int
        The index into `magnetic_positions` of the site closest to the
        transformed point, using the minimum-image convention (i.e.
        comparing positions modulo 1 along each axis).
    offset : numpy.ndarray
        The integer lattice offset (length-3) such that
        `magnetic_positions[j] + offset` equals the transformed point
        `symop[0].dot(p) + symop[1]` (assuming the nearest match found
        is, in fact, an exact image and not just the closest one).
    """
    sp = symop[0].dot(p) + symop[1]
    j = sorted(
        [
            (k, np.linalg.norm((q - sp + 0.5) % 1 - 0.5))
            for k, q in enumerate(magnetic_positions)
        ],
        key=lambda u: u[1],
    )[0][0]
    offset = np.array(sp - magnetic_positions[j], dtype=int)
    return j, offset

def normalize_bond(
    src: int, dest: int, offset: Union[str, list]
) -> Tuple[int, int, str]:
    """
    Brings a bond to its normalized form:

    Parameters
    ----------
    src : int
        The source atom.
    dest : int
        the target atom.
    offset : Union[str, list]
        The offset of the cell where the target atom is.

    Returns
    -------
    src: int
        The source atom
    dest: int
        The target atom
    offset: str
        The encoded offset of the cell where the target atom is.
    """
    if isinstance(offset, str):
        offset = unpack_offset(offset)
    if src > dest:
        src, dest = dest, src
        offset = -offset
    return src, dest, pack_offset(offset)

def parse_symmetry(strsymm: str) -> Tuple[list, list]:
    """
    Parse a symmetry specification.

    Parameters
    ----------
    strsymm : str
        The symmetry specification to be parsed.

    Returns
    -------
    w: list

    offset: list

    """
    strsymm = strsymm.strip()
    strsymm.split(", ")
    rows = strsymm.strip().split(sep=", ")
    trows = []
    offset = []
    for row in rows:
        row = row.strip()
        terms = row.split(sep="-")
        for i, term in enumerate(terms):
            terms[i] = "-" + term
        if terms[0] == "-":
            terms = terms[1:]
        else:
            terms[0] = terms[0][1:]
        terms = sum([term.split(sep="+") for term in terms], [])
        terms = [term.strip() for term in terms]
        trow = [0, 0, 0, 0]
        for term in terms:
            if term[-1] == "x":
                trow[0] = term[:-1]
            elif term[-1] == "y":
                trow[1] = term[:-1]
            elif term[-1] == "z":
                trow[2] = term[:-1]
            else:
                trow[3] = term
        for i in range(4):
            if trow[i] == 0:
                continue
            if trow[i] == "":
                trow[i] = 1
                continue
            if trow[i] == "-":
                trow[i] = -1
                continue
            factors = trow[i].split(sep="/")
            if len(factors) == 1:
                trow[i] = float(factors[0].strip())
            else:
                trow[i] = float(factors[0].strip()) / float(factors[1].strip())
        trows.append(trow)
    w = np.array(trows)
    offset = w[:, -1]
    w = w[:, :-1]
    return w, offset

_CENTERING_TRANSLATIONS = {
    "P": [(0.0, 0.0, 0.0)],
    "A": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],
    "B": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],
    "C": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],
    "I": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    "F": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
    # Rhombohedral, obverse setting, hexagonal axes.
    "R": [(0.0, 0.0, 0.0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],
}

def centering_letter_from_symbol(symbol: Optional[str]) -> str:
    """
    Extract the lattice-centering letter (P, A, B, C, I, F or R) from a
    Hermann-Mauguin space-group symbol, e.g. read from
    `_symmetry_space_group_name_H-M` or `_space_group_name_H-M_alt`
    in a CIF file.

    Parameters
    ----------
    symbol : Optional[str]
        The Hermann-Mauguin symbol (e.g. "F d -3 m"), or None if not
        available.

    Returns
    -------
    str
        The centering letter. Defaults to "P" (primitive, i.e. no
        extra translations) when the symbol is missing or not
        recognized.
    """
    if not symbol:
        return "P"
    symbol = symbol.strip().strip("'\"")
    if not symbol:
        return "P"
    letter = symbol[0].upper()
    return letter if letter in _CENTERING_TRANSLATIONS else "P"

def expand_symmetries_with_centering(
    symmetries: list, space_group_symbol: Optional[str]
) -> list:
    """
    Complete a (possibly partial) list of symmetry operators read from a
    CIF file with the lattice-centering translations implied by the
    space group's Bravais lattice type (P, A, B, C, I, F or R).

    Many CIF files only list the coset representatives of the point
    group in `_space_group_symop_operation_xyz` /
    `_symmetry_equiv_pos_as_xyz`, and leave the centering translations
    implicit in the space group symbol. For example, a spinel like
    ZnCr2O4 (space group Fd-3m, FCC Bravais lattice) is sometimes
    described with only the 48 point-group operators instead of the
    full 192. Without the centering translations, bonds related by an
    FCC centering translation (e.g. (0,½,½)) are treated as
    inequivalent, so the reduction of independent coupling constants
    is incomplete.

    NOTE: this is meant to be used only to complete the symmetries fed
    to `generate_bonds_by_symmetries` (equivalence of couplings), not
    the ones fed to `generate_atoms_by_symmetries`: atomic positions
    are taken exactly as listed in the CIF and are never expanded.

    This is a no-op (returns `symmetries` unchanged) for primitive
    lattices, for an empty/unknown symbol, or when `symmetries` is
    empty.

    Parameters
    ----------
    symmetries : list
        A list of (rotation_matrix, translation_vector) pairs, as
        returned by `cif_read_loop_symmetries`.
    space_group_symbol : Optional[str]
        The Hermann-Mauguin space-group symbol, if available.

    Returns
    -------
    list
        The symmetry list, completed with centering translations.
    """
    centerings = _CENTERING_TRANSLATIONS[centering_letter_from_symbol(space_group_symbol)]
    if len(centerings) == 1 or not symmetries:
        return symmetries

    expanded = []
    seen = set()
    for rot, trans in symmetries:
        for cvec in centerings:
            new_trans = (np.array(trans) + np.array(cvec)) % 1.0
            key = (tuple(np.round(np.array(rot).flatten(), 6)), tuple(np.round(new_trans, 6)))
            if key in seen:
                continue
            seen.add(key)
            expanded.append((rot, new_trans))
    return expanded

def cif_read_loop_symmetries(labels: list, entries: tuple) -> list:
    """
    Parse the symmetry-operators loop of a CIF file into
    `(rotation, translation)` pairs.

    Parameters
    ----------
    labels : list
        The column tags of a single `loop_` block, in order, as
        collected while reading the CIF (row-oriented format: see
        `cif_read_loop_atoms` for the same convention). Only used
        here to locate the column holding the symmetry-operator
        strings, under either of its two standard CIF names:
        `_symmetry_equiv_pos_as_xyz` or
        `_space_group_symop_operation_xyz`.
    entries : tuple
        The rows of that same `loop_` block, each a list with one
        value per label. If `labels` doesn't correspond to a
        symmetries loop (neither expected tag is present), `entries`
        is ignored.

    Returns
    -------
    list
        One `(rotation_matrix, translation_vector)` pair per row, as
        returned by `parse_symmetry` on the corresponding entry.
        Empty if `labels` has neither of the two symmetry-operator
        tags.
    """
    symmetries = []
    for i, t in enumerate(labels):
        if t == "_symmetry_equiv_pos_as_xyz":
            symmdefcol = i
        elif t == "_space_group_symop_operation_xyz":
            symmdefcol = i
        else:
            # label is not a symmetry operation
            continue
        for j, entry in enumerate(entries):
            symmetries.append(parse_symmetry(entry[symmdefcol]))
    return symmetries

def cif_read_loop_atoms(labels: list, entries: list, magnetic_atoms: tuple) -> tuple:
    """
    Parse the `_atom_site` loop of a CIF file and keep only the atoms
    whose species is in `magnetic_atoms`.

    Parameters
    ----------
    labels : list
        The column tags of the `_atom_site` `loop_` block, in order
        (row-oriented format, same convention used throughout this
        module: see `cif_read_loop_symmetries`). Expected/recognized
        tags are `_atom_site_label`, `_atom_site_type_symbol`,
        `_atom_site_fract_x`/`_y`/`_z`, and the non-standard
        `_atom_site_g_factor`/`_atom_site_spin` extensions used by
        spectrojotometer to store the Landé g-factor and spin
        representation per site.
    entries : list
        The rows of that loop, each a list with one value per label
        (mutated in place: the fractional-coordinate fields of
        entries that pass the `magnetic_atoms` filter have any
        trailing CIF uncertainty annotation, e.g. `"1.234(5)"`,
        stripped and replaced by the corresponding float).
    magnetic_atoms : tuple
        The set of species symbols (e.g. `("Cu", "Mn", ...)`) to keep;
        every other atom in the loop is silently dropped.

    Returns
    -------
    atomlabels : dict
        Maps each kept atom's `_atom_site_label` to its index in the
        other returned lists (in the order they were kept).
    magnetic_species : list of str
        The species symbol of each kept atom.
    magnetic_positions : list of numpy.ndarray
        The fractional coordinates (length-3) of each kept atom.
    g_factors : list
        The `_atom_site_g_factor` value of each kept atom, or `"."`
        if that column isn't present in `labels`.
    spin_repr : list
        The `_atom_site_spin` value of each kept atom, or `"."` if
        that column isn't present in `labels`.
    """
    logging.info("atom positions found")
    magnetic_positions = []
    magnetic_species = []
    atomlabels = {}
    g_factors = []
    spin_repr = []
    col_g = -1
    col_s = -1
    for i, t in enumerate(labels):
        if t == "_atom_site_fract_x":
            xcol = i
        elif t == "_atom_site_fract_y":
            ycol = i
        elif t == "_atom_site_fract_z":
            zcol = i
        elif t == "_atom_site_type_symbol":
            tcol = i
        elif t == "_atom_site_label":
            labelcol = i
        elif t == "_atom_site_g_factor":
            col_g = i
        elif t == "_atom_site_spin":
            col_s = i
    idxma = 0
    for i, entry in enumerate(entries):
        if entry[tcol] in magnetic_atoms:
            for cc in (xcol, ycol, zcol):
                entry[cc] = float(entry[cc].split("(", maxsplit=1)[0])

            magnetic_positions.append(np.array([entry[xcol], entry[ycol], entry[zcol]]))
            magnetic_species.append(entry[tcol])
            if col_g != -1:
                g_factors.append(entry[col_g])
            else:
                g_factors.append(".")
            if col_s != -1:
                spin_repr.append(entry[col_s])
            else:
                spin_repr.append(".")
            atomlabels[entry[labelcol]] = idxma
            idxma = idxma + 1
    return (
        atomlabels,
        magnetic_species,
        magnetic_positions,
        g_factors,
        spin_repr,
    )

def cif_read_loop_bonds(labels: list, entries: list, atomlabels: list) -> tuple:
    """
    Parse the `_geom_bond` loop of a CIF file into normalized bonds,
    grouped by magnetic coupling.

    Parameters
    ----------
    labels : list
        The column tags of the `_geom_bond` `loop_` block, in order
        (row-oriented format; see `cif_read_loop_symmetries`).
        Expected tags: `_geom_bond_atom_site_label_1`/`_2`,
        `_geom_bond_distance`, and optionally
        `_geom_bond_label` (the coupling name, e.g. `"J0"`) and
        `_geom_bond_site_symmetry_1`/`_2`.
    entries : list
        The rows of that loop, each a list with one value per label
        (mutated in place: the distance field of every processed row
        has any trailing CIF uncertainty annotation stripped).
    atomlabels : dict
        Maps an atom label (as it appears in
        `_geom_bond_atom_site_label_1`/`_2`, possibly followed by
        `"_<symmetry index>"` when a bond's second atom is a symmetry
        image, see below) to its index in the model, as returned/
        extended by `cif_read_loop_atoms` and
        `generate_atoms_by_symmetries`.

    Notes
    -----
    When present, `_geom_bond_site_symmetry_1`/`_2` are decoded with
    `unpack_symmetry_and_offset`; a non-identity symmetry index `sym`
    is appended to the corresponding atom label as `"_" + str(sym)`
    before the `atomlabels` lookup, matching the labels
    `generate_atoms_by_symmetries` assigns to symmetry-generated
    atoms. A bond whose (possibly suffixed) label isn't found in
    `atomlabels` is skipped, with a message logged at INFO level
    (this is the expected outcome for bonds to atoms that were
    filtered out by `magnetic_atoms`, but can also hide a labeling
    mismatch -- see the tests for a known case).

    If `_geom_bond_label` is absent, bonds are grouped automatically
    by identical declared distance instead, with generated names
    `"J1"`, `"J2"`, etc. in order of appearance.

    Returns
    -------
    bond_labels : list of str
        The coupling names, ordered to match `bond_distances` and
        `bondlists`.
    bond_distances : list of str
        The declared `_geom_bond_distance` for each coupling (the
        first one seen for that label; not recomputed from the atomic
        positions).
    bondlists : list of list of tuple
        For each coupling, the list of its bonds as
        `(src, dest, offset)` triples in `normalize_bond` form (`src`
        and `dest` are indices into the model, `offset` is a packed
        cell-offset string).
    """
    logging.info("Reading bonds from cif")
    jlabelcol = None
    bondlists = []
    bond_distances = []
    bond_labels = {}
    sym1col = -1
    sym2col = -1
    for i, t in enumerate(labels):
        if t == "_geom_bond_atom_site_label_1":
            at1col = i
        if t == "_geom_bond_atom_site_label_2":
            at2col = i
        if t == "_geom_bond_distance":
            distcol = i
        if t == "_geom_bond_label":
            jlabelcol = i
        if t == "_geom_bond_site_symmetry_1":
            sym1col = i
        if t == "_geom_bond_site_symmetry_2":
            sym2col = i

    logging.info(
        {
            "at1col": at1col,
            "at2col": at2col,
            "distcol": distcol,
            "jlabelcol": jlabelcol,
        }
    )
    for en in entries:
        label1 = en[at1col]
        label2 = en[at2col]
        outbond1 = np.array([0, 0, 0])
        outbond2 = np.array([0, 0, 0])
        if sym1col != -1:
            sym, outbond1 = unpack_symmetry_and_offset(en[sym1col])
            if sym != 0:
                label1 = label1 + "_" + str(sym)
        if sym2col != -1:
            sym, outbond2 = unpack_symmetry_and_offset(en[sym2col])
            if sym != 0:
                label2 = label2 + "_" + str(sym)
        if not (label1 in atomlabels and label2 in atomlabels):
            msg = f"{[label1, label2]} not in atomlabels"
            logging.info(msg)
            continue
        outbond = np.array(outbond2) - np.array(outbond1)
        newbond = normalize_bond(atomlabels[label1], atomlabels[label2], outbond)
        en[distcol] = en[distcol].split(sep="(", maxsplit=1)[0]
        if jlabelcol is None:
            if en[distcol] not in bond_distances:
                bond_distances.append(en[distcol])
                bondlabel = "J" + str(len(bond_distances))
                bond_labels[bondlabel] = len(bond_labels)
                bondlists.append([])
            bs = bond_distances.index(en[distcol])
            bondlists[bs].append(newbond)
        else:
            bondlabel = en[jlabelcol]
            if bond_labels.get(bondlabel) is None:
                bond_labels[bondlabel] = len(bondlists)
                bond_distances.append(en[distcol])
                bondlists.append([])

            bondlists[bond_labels[bondlabel]].append(newbond)
    bond_labels = sorted(
        [(value, la) for la, value in bond_labels.items()], key=lambda x: x[0]
    )
    bond_labels = [x[1] for x in bond_labels]
    return bond_labels, bond_distances, bondlists

def primitive_vectors_from_symmetries(
    symmetries: list, conventional_vectors
) -> Optional[tuple]:
    """
    Given a list of symmetry operators that are all pure translations
    (i.e. describe a centered Bravais lattice: F, I, C, A or B), find a
    primitive basis for the full lattice (conventional cell + centering
    translations).

    This is used to build a *compact* model (only the atoms of the
    asymmetric unit) whose periodicity is expressed with a primitive
    Bravais basis, instead of expanding the atoms to fill the
    conventional cell.

    Parameters
    ----------
    symmetries : list
        (rotation, translation) pairs. All rotations must be the
        identity (a purely translational coset) for this to apply.
    conventional_vectors : array-like
        The 3 conventional (e.g. cubic/orthorhombic) Cartesian lattice
        vectors, as returned by `read_bravais_vectors`.

    Returns
    -------
    Optional[tuple]
        `(primitive_frac_vectors, primitive_cartesian_vectors)` if a
        valid primitive basis was found, or `None` if the symmetries
        are not pure translations, or no valid sublattice basis of the
        expected index could be found (in which case the caller should
        fall back to expanding atoms with `generate_atoms_by_symmetries`
        instead). Call `primitive_vectors_from_symmetries_reason` for a
        human-readable explanation of why `None` was returned.
    """
    result, _ = _primitive_vectors_from_symmetries_impl(symmetries, conventional_vectors)
    return result

def primitive_vectors_from_symmetries_reason(
    symmetries: list, conventional_vectors
) -> str:
    """
    Same computation as `primitive_vectors_from_symmetries`, but returns
    a human-readable explanation instead of the basis itself. Useful to
    build an informative error message when `primitive_cell=True` fails.
    """
    _, reason = _primitive_vectors_from_symmetries_impl(symmetries, conventional_vectors)
    return reason

def _primitive_vectors_from_symmetries_impl(symmetries: list, conventional_vectors):
    conventional_vectors = np.array(conventional_vectors)
    translations = []
    for k, (rot, trans) in enumerate(symmetries):
        if not np.allclose(rot, np.eye(3), atol=1e-6):
            reason = (
                f"symmetry operator #{k + 1} ({format_symmetry_operator(rot, trans)}) "
                "has a non-identity rotation part; primitive_cell=True only "
                "supports lattices whose symmetries are pure translations "
                "(centering: F, I, C, A or B), not point-group rotations."
            )
            return None, reason
        translations.append(np.array(trans) % 1.0)

    uniq = []
    for t in translations:
        if any(np.allclose(t, u, atol=1e-6) for u in uniq):
            continue
        uniq.append(t)
    n_cosets = len(uniq)
    if n_cosets <= 1:
        # Nothing to reduce: the lattice already has no extra centering
        # translations beyond the identity (e.g. a CIF that was already
        # saved from a compact/primitive-cell model). The "primitive"
        # basis is then just the conventional vectors themselves.
        return (np.eye(3), conventional_vectors), ""

    candidates = [t for t in uniq if not np.allclose(t, 0, atol=1e-6)]
    tried_dets = []
    for combo in combinations(candidates, 3):
        mat = np.array(combo)
        det = abs(np.linalg.det(mat))
        tried_dets.append(det)
        if det < 1e-6:
            continue
        if abs(det - 1.0 / n_cosets) < 1e-3:
            return (mat, mat.dot(conventional_vectors)), ""

    coset_list = ", ".join(
        str(tuple(round(float(x), 4) for x in t)) for t in candidates
    )
    reason = (
        f"found {n_cosets} distinct translations (besides identity: "
        f"{coset_list}), but no combination of 3 of them spans a sublattice "
        f"of the expected index (1/{n_cosets} of the conventional cell "
        f"volume; closest determinants tried: "
        f"{[round(d, 4) for d in tried_dets]}). This can happen if the "
        "symmetries mix more than one centering type, if fewer than 3 "
        "independent translations are listed (e.g. I or C centering, "
        "which need to be combined with the conventional vectors, not "
        "supported yet), or if there's a numerical inconsistency in the "
        "listed translations."
    )
    return None, reason

def frac_offset_to_primitive_int(delta_frac, primitive_frac_basis, tol: float = 1e-3):
    """
    Express a fractional displacement (given in conventional-cell
    fractional coordinates) as an integer combination of a primitive
    lattice basis.

    Parameters
    ----------
    delta_frac : array-like
        The displacement, in conventional fractional coordinates.
    primitive_frac_basis : array-like
        3 primitive lattice vectors, in conventional fractional
        coordinates (as returned by `primitive_vectors_from_symmetries`).
    tol : float, optional
        Maximum deviation from an integer allowed for the result to be
        accepted. The default is 1e-3.

    Returns
    -------
    Optional[np.ndarray]
        The integer coefficients, or `None` if `delta_frac` is not (to
        within `tol`) an integer combination of the given basis.
    """
    mat = np.array(primitive_frac_basis).T
    try:
        coeffs = np.linalg.solve(mat, np.array(delta_frac, dtype=float))
    except np.linalg.LinAlgError:
        return None
    rounded = np.round(coeffs)
    if np.max(np.abs(coeffs - rounded)) > tol:
        return None
    return rounded.astype(int)

def cif_read_loop_bonds_compact(
    labels: list,
    entries: list,
    atomlabels: dict,
    symmetries: list,
    primitive_frac_basis,
) -> tuple:
    """
    Like `cif_read_loop_bonds`, but for a *compact* model: atoms are
    exactly the ones declared in `_atom_site` (the asymmetric unit,
    never expanded), and `_geom_bond_site_symmetry_2` (or `_1`) is
    resolved using the actual symmetry operator it refers to (an index
    into `symmetries`, following the standard CIF convention), rather
    than by looking up a separately-labeled symmetry-image atom.

    Bonds whose relative symmetry is not a pure translation, or whose
    resulting displacement is not an integer combination of the
    primitive lattice (`primitive_frac_basis`), are skipped with a
    warning: they cannot be represented in a 4-atom/primitive-cell
    model and require expanding the atoms instead (the default
    behaviour of `magnetic_model_from_cif`).
    """
    logging.info("Reading bonds from cif (compact/primitive mode)")
    jlabelcol = None
    bondlists = []
    bond_distances = []
    bond_labels = {}
    sym1col = -1
    sym2col = -1
    for i, t in enumerate(labels):
        if t == "_geom_bond_atom_site_label_1":
            at1col = i
        if t == "_geom_bond_atom_site_label_2":
            at2col = i
        if t == "_geom_bond_distance":
            distcol = i
        if t == "_geom_bond_label":
            jlabelcol = i
        if t == "_geom_bond_site_symmetry_1":
            sym1col = i
        if t == "_geom_bond_site_symmetry_2":
            sym2col = i

    def resolve_delta(code):
        sym, conv_offset = unpack_symmetry_and_offset(code)
        if sym < 0 or sym >= len(symmetries):
            msg = f"symmetry index {sym + 1} in '{code}' is out of range"
            logging.warning(msg)
            return None
        rot, trans = symmetries[sym]
        if not np.allclose(rot, np.eye(3), atol=1e-6):
            msg = (
                f"symmetry operator {sym + 1} used in '{code}' is not a pure "
                "translation; bonds related by point-group rotations cannot "
                "be folded into a compact/primitive-cell model"
            )
            logging.warning(msg)
            return None
        return np.array(trans) + np.array(conv_offset)

    for en in entries:
        label1 = en[at1col]
        label2 = en[at2col]
        delta1 = np.zeros(3)
        delta2 = np.zeros(3)
        if sym1col != -1:
            delta1 = resolve_delta(en[sym1col])
            if delta1 is None:
                continue
        if sym2col != -1:
            delta2 = resolve_delta(en[sym2col])
            if delta2 is None:
                continue
        if not (label1 in atomlabels and label2 in atomlabels):
            msg = f"{[label1, label2]} not in atomlabels"
            logging.info(msg)
            continue
        offset = frac_offset_to_primitive_int(delta2 - delta1, primitive_frac_basis)
        if offset is None:
            msg = (
                f"the offset between {label1} and {label2} "
                f"({delta2 - delta1}) is not an integer combination of the "
                "primitive lattice vectors; skipping this bond"
            )
            logging.warning(msg)
            continue

        newbond = normalize_bond(atomlabels[label1], atomlabels[label2], offset)
        en[distcol] = en[distcol].split(sep="(", maxsplit=1)[0]
        if jlabelcol is None:
            if en[distcol] not in bond_distances:
                bond_distances.append(en[distcol])
                bondlabel = "J" + str(len(bond_distances))
                bond_labels[bondlabel] = len(bond_labels)
                bondlists.append([])
            bs = bond_distances.index(en[distcol])
            bondlists[bs].append(newbond)
        else:
            bondlabel = en[jlabelcol]
            if bond_labels.get(bondlabel) is None:
                bond_labels[bondlabel] = len(bondlists)
                bond_distances.append(en[distcol])
                bondlists.append([])
            bondlists[bond_labels[bondlabel]].append(newbond)

    bond_labels = sorted(
        [(value, la) for la, value in bond_labels.items()], key=lambda x: x[0]
    )
    bond_labels = [x[1] for x in bond_labels]
    return bond_labels, bond_distances, bondlists

def generate_atoms_by_symmetries(
    symmetries: tuple,
    atomlabels: tuple,
    magnetic_species: tuple,
    magnetic_positions: tuple,
    g_factors: tuple,
    spin_repr: tuple,
):
    """
    Expand the asymmetric-unit atoms (as read by `cif_read_loop_atoms`)
    to the full conventional cell, by applying every symmetry operator
    to every atom and discarding images that coincide (mod 1, within
    1e-3) with an atom already placed.

    A no-op (returns the inputs unchanged) when `symmetries` has 0 or
    1 elements (nothing to expand with, or only the identity).

    Parameters
    ----------
    symmetries : tuple
        `(rotation, translation)` pairs, as returned by
        `cif_read_loop_symmetries` (optionally completed with
        centering translations, though centering is normally only
        added for bonds via `expand_symmetries_with_centering`, not
        here -- see that function's docstring).
    atomlabels : dict
        Maps each asymmetric-unit atom's label to its index in the
        other arguments, as returned by `cif_read_loop_atoms`.
        Mutated in place: for every new image kept (generated by the
        `(s + 1)`-th symmetry operator, `s > 0`), an extra entry
        `"<original label>_<s + 1>"` is added, pointing at the new
        atom's index. This lets `cif_read_loop_bonds` resolve a bond
        declared via `_geom_bond_site_symmetry_2` against the right
        symmetry image.
    magnetic_species : list of str
        The species of each asymmetric-unit atom.
    magnetic_positions : list of numpy.ndarray
        The fractional coordinates of each asymmetric-unit atom.
    g_factors : list
        The Landé g-factor of each asymmetric-unit atom (or `"."`).
    spin_repr : list
        The spin representation of each asymmetric-unit atom (or
        `"."`).

    Returns
    -------
    atomlabels : dict
        The (possibly mutated) input `atomlabels`, now also covering
        the newly generated atoms.
    magnetic_species : list of str
        The species of every atom after expansion, in the order they
        were generated (asymmetric unit first, then each symmetry
        operator's new images).
    magnetic_positions : list of numpy.ndarray
        The fractional coordinates of every atom after expansion,
        aligned with `magnetic_species`.
    g_factors : list
        The Landé g-factor of every atom after expansion (each image
        inherits its source atom's value).
    spin_repr : list
        The spin representation of every atom after expansion (each
        image inherits its source atom's value).
    """
    if len(symmetries) > 1:
        magnetic_positions2 = []
        magnetic_species2 = []
        g_factors2 = []
        spin_repr2 = []

        idxma = len(atomlabels)
        for s, sym in enumerate(symmetries):
            for i, r in enumerate(magnetic_positions):
                newposition = sym[0].dot(r) + sym[1]
                already_in_list = False
                for pos in magnetic_positions2:
                    if np.linalg.norm((newposition - pos + 0.5) % 1 - 0.5) < 1.0e-3:
                        already_in_list = True
                        break
                if already_in_list:
                    continue
                magnetic_positions2.append(newposition)
                magnetic_species2.append(magnetic_species[i])
                g_factors2.append(g_factors[i])
                spin_repr2.append(spin_repr[i])
                atomlabel = [key for key, val in atomlabels.items() if val == i][0]
                if s > 0:
                    atomlabel += "_" + str(s + 1)
                    atomlabels[atomlabel] = idxma
                    idxma = idxma + 1
        magnetic_positions = magnetic_positions2
        magnetic_species = magnetic_species2
        g_factors = g_factors2
        spin_repr = spin_repr2
    return (
        atomlabels,
        magnetic_species,
        magnetic_positions,
        g_factors,
        spin_repr,
    )

def generate_bonds_by_symmetries(
    symmetries, bond_labels, bonddistances, bondlists, magnetic_positions
):
    """
    Fill in, for every already-declared coupling, all its
    symmetry-equivalent bonds -- i.e. find the images of each declared
    bond under every symmetry operator (besides the identity) and add
    any that aren't already present, so that all bonds related by the
    space group end up sharing the same coupling label.

    Meant to be called after the model's atoms have already been
    expanded to the full cell (`generate_atoms_by_symmetries`), so
    that every symmetry image of a bond's endpoints already exists in
    `magnetic_positions` and can be located with
    `find_atom_offset_by_symmetry`.

    Parameters
    ----------
    symmetries : list
        `(rotation, translation)` pairs. The identity (assumed to be
        `symmetries[0]`) is skipped; every other operator is applied
        to every existing bond of every coupling.
    bond_labels : list
        The coupling names, as returned by `cif_read_loop_bonds`
        (only used here to keep `bondlists` and `bonddistances`
        aligned with it; not otherwise read or modified).
    bonddistances : list
        The declared distance of each coupling, aligned with
        `bond_labels` (read but not modified: newly found symmetry
        images are assumed to share their coupling's distance).
    bondlists : list of list of tuple
        For each coupling (aligned with `bond_labels`), its bonds as
        `(src, dest, offset)` triples in `normalize_bond` form.
        **Mutated in place**: every newly found, not-yet-present
        symmetry image of a bond is appended to the corresponding
        list.
    magnetic_positions : list of numpy.ndarray
        The fractional coordinates of every atom in the (already
        expanded) model, used to resolve each symmetry image to an
        atom index via `find_atom_offset_by_symmetry`.

    Returns
    -------
    None
        `bondlists` is extended in place; nothing is returned.

    Notes
    -----
    A symmetry image whose resulting offset falls outside the
    representable range (any component below -5 or above 4, the
    supercell-offset limits of `pack_offset`/`unpack_offset`) is
    skipped with a warning logged, rather than producing a bond with
    an incorrect or truncated offset.
    """
    for s in range(len(symmetries) - 1):
        symop = symmetries[s + 1]
        for bidx, blst in enumerate(bondlists):
            for bnd in blst:
                i, j, offset = bnd
                offset = unpack_offset(offset)
                i, offseti = find_atom_offset_by_symmetry(
                    magnetic_positions[i], symop, magnetic_positions
                )
                j, offsetj = find_atom_offset_by_symmetry(
                    magnetic_positions[j] + offset, symop, magnetic_positions
                )
                if any(offseti != 0):
                    # The source must be inside the cell
                    i, offseti, j, offsetj = j, offsetj, i, offseti
                if any(offseti != 0):
                    # both atoms are outside the cell
                    continue
                offset = offsetj - offseti
                if any(offset < -5) or any(offset > 4):
                    msg = f"the bond is too long. offset= {offset}"
                    logging.warning(msg)
                    continue

                newbond = normalize_bond(i, j, offset)
                if newbond not in blst:
                    blst.append(newbond)

def magnetic_model_from_cif(
    filename: str,
    magnetic_atoms: tuple = DEFAULT_MAGNETIC_ATOMS,
    bond_names: Optional[list] = None,
    primitive_cell: bool = False,
) -> MagneticModel:
    """
    Parameters
    ----------
    filename : str
        The name of the file to read.
    magnetic_atoms : tuple, optional
        The set of atoms to be considered magnetic.
        The default is ("Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V").
    bond_names : Optional[list], optional
        The names of the bonds.  The default is None, meaning that the bonds
        are named automatically.
    primitive_cell : bool, optional
        If False (the default), atoms are expanded to fill the
        conventional cell using the CIF's symmetry operators (the
        historical behaviour).
        If True, atoms are taken exactly as declared in `_atom_site`
        (never expanded), and the model's Bravais vectors are switched
        to a primitive basis built from the symmetry operators, which
        must all be pure translations (i.e. describe a centered lattice
        such as F, I, C, A or B — not a general point-group symmetry).
        Bonds must then reference symmetry images using the standard
        CIF `_geom_bond_site_symmetry_2` code (e.g. "2_555" for the
        second listed symmetry operator, no extra cell shift), not by
        inventing extra atom labels beyond those in `_atom_site`.

    Returns
    -------
    MagneticModel
        a MagneticModel.

    """
    bravais_params = {}
    magnetic_positions = None
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []
    symmetries = []
    space_group_symbol = None
    primitive_bravais_vectors = None
    g_lande_factors = None
    spin_repr = None

    # CIF tags (outside of a loop_ block) that may carry the Hermann-Mauguin
    # space-group symbol, used to infer the lattice centering (P, A, B, C,
    # I, F, R) when the symmetries loop only lists coset representatives.
    space_group_name_tags = (
        "_symmetry_space_group_name_h-m",
        "_space_group_name_h-m_alt",
        "_space_group_name_h-m_ref",
    )

    msg = f"loaading model from{filename}"
    logging.info(msg)
    with open(filename, "r") as src:
        # `pending_line` lets an inner block (below) hand back a line it
        # read ahead but did not consume, so the next outer iteration can
        # still process it. This matters for CIF files where consecutive
        # `loop_` blocks are not separated by a blank line.
        pending_line = None
        while True:
            if pending_line is not None:
                line = pending_line
                pending_line = None
            else:
                line = src.readline()
                if line == "":
                    break
            listrip = line.strip()
            if listrip[:13] == "_cell_length_":
                varvals = listrip[13:].split()
                varvals[1] = varvals[1].split(sep="(", maxsplit=1)[0]
                bravais_params[varvals[0]] = float(varvals[1])
            elif listrip[:12] == "_cell_angle_":
                varvals = line[12:].strip().split()
                varvals[1] = varvals[1].split(sep="(", maxsplit=1)[0]
                bravais_params[varvals[0]] = float(varvals[1]) * 3.1415926 / 180.0
            elif listrip[:1] == "_":
                tag_value = listrip.split(None, 1)
                if tag_value and tag_value[0].lower() in space_group_name_tags:
                    if len(tag_value) > 1:
                        space_group_symbol = tag_value[1].strip().strip("'\"")
            elif listrip[:5] == "loop_":
                labels = []
                entries = []
                ls = src.readline()
                listrip = ls.strip()
                if listrip == "":
                    break
                if listrip != "" and line[0] == "#":
                    continue
                while listrip[0] == "_" or listrip[0] == "#":
                    if listrip != "" and line[0] == "#":
                        continue
                    labels.append(listrip.split()[0])
                    line = src.readline()
                    listrip = line.strip()
                while listrip != "" and listrip[0] != "_" and listrip[:5] != "loop_":
                    newentry = split_cif_loop_line(listrip)
                    entries.append(newentry)
                    ls = src.readline()
                    listrip = ls.strip()
                if listrip != "":
                    # `listrip` is either a new tag ("_...") or the start
                    # of the next `loop_` block, read ahead while looking
                    # for the end of this loop's data rows. Since it was
                    # never appended as an entry, hand it back so the next
                    # outer iteration processes it instead of losing it.
                    pending_line = listrip + "\n"

                # if the block contains symmetries
                if (
                    "_symmetry_equiv_pos_as_xyz" in labels
                    or "_space_group_symop_operation_xyz" in labels
                ):
                    symmetries = cif_read_loop_symmetries(labels, entries)

                # if the block contains the set of atoms
                if "_atom_site_fract_x" in labels:
                    (
                        atomlabels,
                        magnetic_species,
                        magnetic_positions,
                        g_lande_factors,
                        spin_repr,
                    ) = cif_read_loop_atoms(labels, entries, magnetic_atoms)
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

                # If the block contains the set of bonds
                if "_geom_bond_atom_site_label_1" in labels:
                    if primitive_cell:
                        # Atoms stay exactly as declared (the asymmetric
                        # unit). Bonds are resolved against a primitive
                        # Bravais basis built from the (pure-translation)
                        # symmetry operators, using the standard CIF
                        # `_geom_bond_site_symmetry_2` convention (a
                        # symmetry-operator index, not an extra atom
                        # label) to refer to symmetry images.
                        conventional_vectors = read_bravais_vectors(bravais_params)
                        primitive_basis = primitive_vectors_from_symmetries(
                            symmetries, conventional_vectors
                        )
                        if primitive_basis is None:
                            reason = primitive_vectors_from_symmetries_reason(
                                symmetries, conventional_vectors
                            )
                            raise ValueError(
                                f"cannot build a primitive-cell model from {filename}: "
                                f"{reason}"
                            )
                        primitive_frac_basis, primitive_bravais_vectors = (
                            primitive_basis
                        )
                        (
                            bond_labels,
                            bond_distances,
                            bondlists,
                        ) = cif_read_loop_bonds_compact(
                            labels,
                            entries,
                            atomlabels,
                            symmetries,
                            primitive_frac_basis,
                        )
                    else:
                        (
                            bond_labels,
                            bond_distances,
                            bondlists,
                        ) = cif_read_loop_bonds(labels, entries, atomlabels)
                        # Atoms are taken exactly as given in the CIF (no
                        # expansion). For finding *equivalent bonds*, however,
                        # we do want the full space-group symmetry, completing
                        # whatever the CIF's symmetry loop lists with the
                        # lattice-centering translations implied by the space
                        # group symbol (e.g. F for an FCC lattice like the
                        # spinel structure of ZnCr2O4). This only groups
                        # existing bonds into equivalence classes; it never
                        # creates new atoms.
                        bond_symmetries = expand_symmetries_with_centering(
                            symmetries, space_group_symbol
                        )
                        generate_bonds_by_symmetries(
                            bond_symmetries,
                            bond_labels,
                            bond_distances,
                            bondlists,
                            magnetic_positions,
                        )

    conventional_vectors = read_bravais_vectors(bravais_params)
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
                    f"cannot build a primitive-cell model from {filename}: "
                    f"{reason}"
                )
            _, primitive_bravais_vectors = primitive_basis
        bravais_vectors = primitive_bravais_vectors
    else:
        bravais_vectors = conventional_vectors
    if len(magnetic_positions)!=0:
        magnetic_positions = np.array(magnetic_positions).dot(np.array(conventional_vectors))
    for msg in (
        f"    magnetic species: {magnetic_species}",
        f"    spin representation: {spin_repr}",
        f"   lande factor: {g_lande_factors}",
        f"    positions: {magnetic_positions}",
        f"    bondlabels: {bond_labels}",
        f"   bondlists: {bondlists}",
    ):
        logging.info(msg)

    model = MagneticModel(
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
        # NOTE: `symmetries` here must always be expressed in the same
        # fractional frame as `bravais_vectors`. The CIF's own symmetries
        # (identity + centering) are defined relative to the *conventional*
        # cell; once reduced to a primitive cell (primitive_cell=True) or
        # fully expanded (the default), no further symmetry is needed to
        # describe the model, so we deliberately do NOT forward them here
        # (that would make `save_cif` write a symmetry loop that no longer
        # matches the stored lattice vectors). `space_group_symbol` is kept
        # purely as informational metadata; it is not used by `save_cif`.
        symmetries=None,
        space_group_symbol=space_group_symbol,
    )

    return model
