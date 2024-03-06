"""
Tools used along the code.
"""
import numpy as np


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
    result = np.array([int(c) - 5 for c in encoded_offset], dtype=int)
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
