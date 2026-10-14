"""
struct
Reader for WIEN2k .struct files.
"""
from typing import Optional
import numpy as np

from ..magnetic_model import MagneticModel
from .common import DEFAULT_MAGNETIC_ATOMS, read_bravais_vectors

DEGREE_TO_RAD = 3.1415926 / 180

def magnetic_model_from_wk2_struct(
        filename: str,
        magnetic_atoms: tuple = DEFAULT_MAGNETIC_ATOMS,
        bond_names: Optional[list] = None,
) -> MagneticModel:
    """
    Build a `MagneticModel` from a WIEN2k `.struct` file: read the
    lattice parameters and the fractional positions of every atom
    whose species is in `magnetic_atoms`.

    Unlike `cif.magnetic_model_from_cif`, this reader does not expand
    atoms by symmetry (a `.struct` file already lists every atom in
    the conventional cell, one block per inequivalent site plus its
    `MULT` symmetry-equivalent replicas) and does not read or build
    any bonds: the returned model always has an empty `bonds` dict,
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
    bravais_params = {}
    magnetic_positions = []
    bravais_vectors = None
    # labels = None
    # entries = None
    magnetic_species:list[str] = []
    # bond_labels:list[str] = None
    # bondlists:list[tuple] = None
    # bond_distances:list[float] = []

    with open(filename) as fin:
        title = fin.readline()
        fin.readline()  # size
        fin.readline()  # not any clue
        bravais = fin.readline()
        for line in fin:
            sl = line.strip()
            if sl[:4] == "ATOM":
                positions = []
                if sl[4] == " ":
                    sl = list(sl)
                    sl[4] = "-"
                    sl = "".join(sl)
                if sl[5] == " ":
                    sl = list(sl)
                    sl[5] = "-"
                    sl = "".join(sl)
                fields = sl.split()
                # idxatom = fields[0][4:-1]
                positions.append(
                    [
                        float(fields[1][3:]),
                        float(fields[2][3:]),
                        float(fields[3][3:]),
                    ]
                )
                mult = int(fin.readline().strip().split()[1])
                mult = mult - 1
                for k in range(mult):
                    sl = fin.readline()
                    fields = sl.split()
                    positions.append(
                        [
                            float(fields[1][3:]),
                            float(fields[2][3:]),
                            float(fields[3][3:]),
                        ]
                    )

                atomlabelfield = fin.readline().strip()
                if atomlabelfield[1] == " ":
                    atomspecies = atomlabelfield[0]
                else:
                    atomspecies = atomlabelfield[:2]
                # atomlabel = atomspecies + idxatom
                lrm = fin.readline()  # Rotation matrix
                lrm = lrm + fin.readline()
                lrm = lrm + fin.readline()

                if atomspecies not in magnetic_atoms:
                    continue
                for p in positions:
                    magnetic_positions.append(p)
                    magnetic_species.append(atomspecies)

        bravais_fields = bravais.strip().split()
        bravais_params["a"] = float(bravais_fields[0])
        bravais_params["b"] = float(bravais_fields[1])
        bravais_params["c"] = float(bravais_fields[2])
        bravais_params["alpha"] = float(bravais_fields[3]) * DEGREE_TO_RAD
        bravais_params["beta"] = float(bravais_fields[4]) * DEGREE_TO_RAD
        bravais_params["gamma"] = float(bravais_fields[5]) * DEGREE_TO_RAD

        # Same formula as the CIF reader's (this used to be duplicated
        # here almost line for line): building the Bravais basis from
        # (a, b, c, alpha, beta, gamma) is identical for .cif and
        # .struct.
        bravais_vectors = read_bravais_vectors(bravais_params)

    if len(magnetic_positions)!=0:
        magnetic_positions = np.array(magnetic_positions).dot(np.array(bravais_vectors))
    model = MagneticModel(
        magnetic_positions,
        bravais_vectors,
        bond_lists=None,
        bond_names=None,
        magnetic_species=magnetic_species,
        model_label=title,
    )

    return model
