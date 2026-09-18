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


    Parameters
    ----------
    filename : str
        The file to read.
    magnetic_atoms : tuple, optional
        The set of atoms species to be included.
        The default is ("Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V").
    bond_names : Optional[list], optional
        Names to be used for the couplings.
        The default is None: use automatic names .

    Returns
    -------
    MagneticModel
        The model.

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

        # Misma fórmula que en el lector de CIF (era código duplicado
        # línea por línea): construir la base de Bravais a partir de
        # (a, b, c, alpha, beta, gamma) es idéntico para .cif y .struct.
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
