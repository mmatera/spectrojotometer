"""
struct
Reader for WIEN2k .struct files.
"""

import re

import numpy as np

from ..magnetic_model import MagneticModel
from .common import DEFAULT_MAGNETIC_ATOMS, read_bravais_vectors

DEGREE_TO_RAD = np.pi / 180

def magnetic_model_from_wk2_struct(
        filename: str,
        magnetic_atoms: tuple = DEFAULT_MAGNETIC_ATOMS,
        bond_names: list | None = None,
        primitive_cell: bool = False,
) -> MagneticModel:
    """
    Build a `MagneticModel` from a WIEN2k `.struct` file: read the
    lattice parameters and the fractional positions of every atom
    whose species is in `magnetic_atoms`.

    The file follows the fixed-column format of WIEN2k; the atom blocks
    may number their atoms as `ATOM   1:` or `ATOM  -1:`. Lengths are in
    bohr unless the mode line says `unit=ang`; the model is in angstrom,
    like the ones read from CIF files. Each block lists an inequivalent
    site and its `MULT` symmetry-equivalent replicas. For centred
    lattices (types B, F, CXY, CYZ, CXZ) WIEN2k lists only one of the
    atoms related by the centring. By default (primitive_cell=False) the
    reader adds the others, and the model has the conventional cell, like
    the CIF reader. With primitive_cell=True, the model has the primitive
    cell of the centred lattice, with one site for each set of sites
    related by the centring (for P and H lattices, the cell is the same).
    Rhombohedral (R) lattices are not supported. The element is read
    from the name of the site (e.g. `Cr1`, `CR1`, `O1`).

    The reader does not read or build any bonds: the returned model
    always has an empty `bonds` dict,
    and `bond_names` is accepted only for signature compatibility with
    `magnetic_model_from_cif`/`magnetic_model_from_file` -- it has no
    effect here.

    Parameters
    ----------
    filename : str
        The file to read.
    magnetic_atoms : tuple, optional
        The set of atoms species to be included.
        The default is ("Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V").
    bond_names : Optional[list], optional
        Unused; accepted only for API compatibility with
        `magnetic_model_from_cif`. The default is None.

    Returns
    -------
    MagneticModel
        The model, with an empty `bonds` dict and, if no atom in the
        file matches `magnetic_atoms`, an empty set of sites.
    """
    with open(filename) as fin:
        lines = fin.read().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{filename}: too short for a WIEN2k .struct file")
    title = lines[0].strip()
    lattice_type = lines[1][:4].strip().upper()
    if lattice_type.startswith("R"):
        raise ValueError(
            f"{filename}: rhombohedral (R) lattices are not supported by the "
            ".struct reader; convert the file to CIF"
        )
    if lattice_type not in _CENTERING:
        raise ValueError(f"{filename}: unknown lattice type {lattice_type!r}")
    # Lengths are in bohr, unless the mode line says unit=ang.
    unit = re.search(r"unit\s*=\s*([A-Za-z]+)", lines[2], re.IGNORECASE)
    scale = 1.0 if unit and unit.group(1).lower().startswith("ang") else BOHR_IN_ANGSTROM

    a, b, c, alpha, beta, gamma = _lattice_parameters(lines[3])
    bravais_vectors = read_bravais_vectors(
        {
            "a": a * scale,
            "b": b * scale,
            "c": c * scale,
            "alpha": alpha * DEGREE_TO_RAD,
            "beta": beta * DEGREE_TO_RAD,
            "gamma": gamma * DEGREE_TO_RAD,
        }
    )

    magnetic_positions = []
    magnetic_species: list[str] = []
    i = 4
    while i < len(lines):
        if not lines[i].lstrip().startswith("ATOM"):
            i += 1
            continue
        # ATOM line, MULT line, MULT - 1 lines of equivalent positions,
        # the name line and the three lines of the local rotation matrix.
        positions = [_position(lines[i], filename, i)]
        mult = re.search(r"MULT\s*=\s*(\d+)", lines[i + 1])
        if mult is None:
            raise ValueError(f"{filename}, line {i + 2}: MULT not found")
        mult = int(mult.group(1))
        for k in range(mult - 1):
            positions.append(_position(lines[i + 2 + k], filename, i + 2 + k))
        name_line = i + 1 + mult
        species = _species(lines[name_line], filename, name_line)
        i = name_line + 4
        if species not in magnetic_atoms:
            continue
        for position in positions:
            # Centred lattices list one of the atoms related by centring.
            for shift in _CENTERING[lattice_type]:
                magnetic_positions.append((np.array(position) + shift) % 1.0)
                magnetic_species.append(species)

    magnetic_positions, magnetic_species = _unique_sites(
        magnetic_positions, magnetic_species
    )
    if len(magnetic_positions) != 0:
        magnetic_positions = np.array(magnetic_positions).dot(np.array(bravais_vectors))
    if primitive_cell:
        bravais_vectors = list(
            np.array(_PRIMITIVE[lattice_type]) @ np.array(bravais_vectors)
        )
        if len(magnetic_positions) != 0:
            # One site for each set related by the centring: unique
            # modulo the primitive lattice.
            primitive = np.array(bravais_vectors)
            frac = np.linalg.solve(primitive.T, np.array(magnetic_positions).T).T
            kept, kept_species = _unique_sites(list(frac), magnetic_species)
            magnetic_positions = np.array(kept) @ primitive
            magnetic_species = kept_species
    return MagneticModel(
        magnetic_positions,
        bravais_vectors,
        bond_lists=None,
        bond_names=None,
        magnetic_species=magnetic_species,
        model_label=title,
    )


# WIEN2k writes lengths in bohr (CODATA 2018 value).
BOHR_IN_ANGSTROM = 0.529177210903

# Centring translations of each lattice type of WIEN2k.
_CENTERING = {
    "P": [np.zeros(3)],
    "H": [np.zeros(3)],
    "B": [np.zeros(3), np.array([0.5, 0.5, 0.5])],
    "F": [
        np.zeros(3),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.5, 0.0]),
    ],
    "CXY": [np.zeros(3), np.array([0.5, 0.5, 0.0])],
    "CYZ": [np.zeros(3), np.array([0.0, 0.5, 0.5])],
    "CXZ": [np.zeros(3), np.array([0.5, 0.0, 0.5])],
}

# Primitive vectors of each centred lattice, as rows of coefficients of
# the conventional vectors (a, b, c).
_PRIMITIVE = {
    "P": np.eye(3),
    "H": np.eye(3),
    "B": np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    "F": np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    "CXY": np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    "CYZ": np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, -0.5, 0.5]]),
    "CXZ": np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.5]]),
}

_COORDINATES = re.compile(
    r"X\s*=\s*([-+]?\d*\.?\d+)\s*Y\s*=\s*([-+]?\d*\.?\d+)\s*Z\s*=\s*([-+]?\d*\.?\d+)"
)


def _lattice_parameters(line):
    """a, b, c, alpha, beta, gamma: written as 6F10.6 (fixed columns, so
    two values may have no space between them)."""
    try:
        values = [float(line[10 * k : 10 * k + 10]) for k in range(6)]
    except ValueError:
        values = [float(x) for x in line.split()[:6]]
    if len(values) != 6:
        raise ValueError(f"cannot read the lattice parameters from {line!r}")
    return values


def _position(line, filename, index):
    match = _COORDINATES.search(line)
    if match is None:
        raise ValueError(f"{filename}, line {index + 1}: no X=, Y=, Z= coordinates")
    return [float(x) for x in match.groups()]


def _species(line, filename, index):
    """The element of a name line (e.g. 'Zn1', 'Cr 2', 'O1', 'CR1')."""
    from pymatgen.core.periodic_table import Element

    letters = re.match(r"\s*([A-Za-z]+)", line[:10])
    if letters is None:
        raise ValueError(f"{filename}, line {index + 1}: no element name")
    letters = letters.group(1)
    for size in (2, 1):
        symbol = letters[:size].capitalize()
        if len(letters) >= size and Element.is_valid_symbol(symbol):
            return symbol
    raise ValueError(f"{filename}, line {index + 1}: unknown element {letters!r}")


def _unique_sites(positions, species, tol=1e-4):
    """Drop repeated sites (fractional coordinates modulo 1)."""
    kept_positions, kept_species = [], []
    for position, kind in zip(positions, species):
        if any(
            np.allclose(((position - other + 0.5) % 1.0) - 0.5, 0.0, atol=tol)
            for other in kept_positions
        ):
            continue
        kept_positions.append(position)
        kept_species.append(kind)
    return kept_positions, kept_species
