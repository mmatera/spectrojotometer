"""
Tools used along the code.
"""
import re

import numpy as np

# Matches a single-quoted field, a double-quoted field, or a run of
# non-whitespace characters. Used to tokenize a CIF ``loop_`` data
# line while keeping quoted fields (which may contain spaces, as in
# symmetry operators like 'x, y, z') together as a single token.
_CIF_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def split_cif_loop_line(line: str) -> list:
    """
    Split a single ``loop_`` data line of a CIF file into its fields,
    the way a real CIF parser would: whitespace-separated, except
    that a field enclosed in single or double quotes is kept as one
    token even if it contains spaces (surrounding quotes are
    stripped from the returned token).

    This matters for files (e.g. produced by pymatgen) where the
    symmetry-operators loop has an extra numeric id column before the
    quoted operator, such as::

        1  'x, y, z'

    A naive ``line.split()`` would break the quoted operator into
    three separate tokens (``"'x,"``, ``"y,"``, ``"z'"``); this
    function returns ``["1", "x, y, z"]`` instead.

    Parameters
    ----------
    line : str
        One data line of a CIF ``loop_`` block (already stripped of
        the trailing newline is not required).

    Returns
    -------
    list of str
        The fields of the line, with any enclosing quotes removed.
    """
    tokens = _CIF_TOKEN_RE.findall(line.strip())
    fields = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
            fields.append(token[1:-1])
        else:
            fields.append(token)
    return fields



def box_ellipse(coeff_matrix, radius):
    """
    Determines the minimum edge half-length of the rectangular box
    containing the ellipsoid defined by
    x^t * A * x - radius**2=0, with
    faces parallel to the coordinate axes.
    A = coeff_matrix^t * coeff_matrix
    """
    coeff_matrix = np.array(coeff_matrix)
    a_matrix = coeff_matrix.transpose().dot(coeff_matrix)
    size = len(a_matrix)
    widths = []
    for i in range(size):
        setperp = list(range(i)) + list(range(i + 1, size))
        right_sv = a_matrix[i, setperp]
        a_22 = a_matrix[setperp][:, setperp]
        try:
            a_perpinv = np.linalg.inv(a_22)
            gamma = a_perpinv.dot(right_sv)
            gamma = gamma.dot(right_sv)
            widths.append(radius / np.sqrt(a_matrix[i, i] - gamma))
        except np.linalg.linalg.LinAlgError:
            widths.append(1.0e300)
    return widths


def normalize_configurations(configurations):
    """
    Normalize a set of spin configurations.
    Using the symmetry of inversion, ensures that the majority
    of the spins are 0.
    """
    nconfs = {}
    for spin_conf in configurations:
        if float(sum(spin_conf)) / len(spin_conf) < 0.5:
            lbl = str(sum(s * 2**n for n, s in enumerate(spin_conf)))
            nconfs[lbl] = spin_conf
        else:
            negate_spins = [1 - s for s in spin_conf]
            lbl = str(sum(i * 2**n for n, i in enumerate(negate_spins)))
            nconfs[lbl] = negate_spins
    return list(nconfs.values())


def offset_orientation(offset: list) -> int:
    """
    Determine the orientation of an offset vector.
    To avoid double-counting in the generation of bonds in a supercell,
    we need to establish a convention about the orientation of a vector.

    The convention is that a vector is positive if the last
    non-zero coordinate is positive.

    """
    if len(offset) > 3:
        return None
    for coord in offset[::-1]:
        if coord == 0:
            continue
        return 1 if coord > 0 else -1
    return 0


def format_symmetry_operator(rot, trans, tol: float = 1e-6) -> str:
    """
    Format a (rotation, translation) symmetry operator as a CIF-style
    string such as "x, y+1/2, z+1/2". This is the inverse of
    `model_io.parse_symmetry`.

    Parameters
    ----------
    rot : array-like
        3x3 rotation/point-group matrix (entries expected to be 0, 1 or
        -1, as for any crystallographic operator).
    trans : array-like
        Length-3 translation vector, in fractional coordinates.
    tol : float, optional
        Numerical tolerance. The default is 1e-6.

    Returns
    -------
    str
        The CIF-style symmetry operator string.
    """
    var_names = ("x", "y", "z")
    rot = np.array(rot, dtype=float)
    trans = np.array(trans, dtype=float)

    def format_frac(val: float) -> str:
        for denom in (1, 2, 3, 4, 6):
            num = val * denom
            if abs(num - round(num)) < 1e-4:
                num = int(round(num))
                return str(num) if denom == 1 else f"{num}/{denom}"
        return f"{val:.6f}"

    rows = []
    for i in range(3):
        terms = []
        for j in range(3):
            coeff = rot[i][j]
            if abs(coeff) < tol:
                continue
            sign = "+" if coeff > 0 else "-"
            mag = abs(coeff)
            term = (
                var_names[j]
                if abs(mag - 1) < tol
                else f"{format_frac(mag)}{var_names[j]}"
            )
            terms.append((sign, term))
        shift = trans[i] % 1.0
        if shift > 0.5 + tol:
            shift -= 1.0
        if abs(shift) > tol:
            terms.append(("+" if shift > 0 else "-", format_frac(abs(shift))))
        if not terms:
            rows.append("0")
            continue
        row_str = ""
        for k, (sign, term) in enumerate(terms):
            row_str += (sign if (k > 0 or sign == "-") else "") + term
        rows.append(row_str)
    return ", ".join(rows)


def pack_offset(r_list: list) -> str:
    """
    Parameters
    ----------
    r_list : list
        a list of indices identifying a supercell

    Returns
    -------
    str :
        the offset representation.

    """
    # Codificación del offset:
    # El offset se especifica con `1_` seguido de
    # digitos, cada uno correspondiente a un índice.
    # Si el dígito es >5, el offset en la coordenada
    # es su complemento a 10. Por ejemplo,
    # la celda en la posición 1,-1,1 tiene
    # una clave 1_191
    r_list = np.array(r_list)
    if all(r_list == 0):
        return "."
    offset_key = "1_" + "".join(str(int(coord + 10) % 10) for coord in r_list)
    return offset_key


def unpack_offset(encoded_offset: str) -> np.ndarray:
    """
    Parameters
    ----------
    encoded_offset : str
        an encoded relative position for a cell.

    Returns
    -------
    np.array
        the offset the represented cell.
    """
    if encoded_offset == ".":
        return np.array([0, 0, 0])
    encoded_offset_parts = encoded_offset.split("_")
    if len(encoded_offset_parts) == 1:
        return np.array([0, 0, 0])
    encoded_offset = encoded_offset_parts[1]
    result = np.array([(int(c)+5)% 10 - 5 for c in encoded_offset], dtype=int)
    return result


def unpack_symmetry_and_offset(encoded_offset: str) -> tuple:
    """
    Parameters
    ----------
    encoded_offset : str
        The encoded offset, as used in a CIF file.

    Returns
    -------
    symmetry index : int
        An index used in CIF files. Seems to be always 1.
    offset : np.ndarray
        the offset.
    """
    if encoded_offset == ".":
        return 0, np.array([0, 0, 0])
    encoded_offset_parts = encoded_offset.split("_")
    if len(encoded_offset_parts) == 1:
        return int(encoded_offset_parts[0]) - 1, np.array([0, 0, 0])
    return (
        int(encoded_offset_parts[0]) - 1,
        np.array([int(c) - 5 for c in encoded_offset_parts[1]], dtype=int),
    )
