"""
common
Utilities shared by the CIF and .struct readers, plus a couple of
helpers (spin-configuration files, config indices) that are not tied
to either file format.
"""
import numpy as np

from ..magnetic_model import MagneticModel

DEFAULT_MAGNETIC_ATOMS = tuple(
    (
        "Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V",
    )
)


def read_bravais_vectors(bravais_params: dict) -> list:
    """
    Build a Bravais' basis from its parameters.

    Parameters
    ----------
    bravais_params : dict
        The parameters that defines a Bravais' basis.

    Returns
    -------
    bravais_vectors: list
        the Bravais' basis.

    """
    def value_or_pi_half(x):
        return 3.1415926*.5 if x is None else x

    bravais_vectors = []
    if bravais_params.get("a") is not None:
        bravais_vectors.append(np.array([bravais_params.get("a"), 0, 0]))

    if bravais_params.get("b") is not None:
        gamma = value_or_pi_half(bravais_params.get("gamma"))
        bravais_vectors.append(
            np.array(
                [
                    bravais_params.get("b") * np.cos(gamma),
                    bravais_params.get("b") * np.sin(gamma),
                    0,
                ]
            )
        )

    if bravais_params.get("c") is not None:
        alpha:float = value_or_pi_half(bravais_params.get("alpha"))
        beta:float = value_or_pi_half(bravais_params.get("beta"))
        x = np.cos(alpha)
        y = np.cos(beta) - x * np.cos(gamma)
        y = y / np.sin(gamma)
        z = bravais_params.get("c") * np.sqrt(1 - x * x - y * y)
        x = bravais_params.get("c") * x
        y = bravais_params.get("c") * y
        bravais_vectors.append(np.array([x, y, z]))
    return bravais_vectors

def confindex(c: list) -> int:
    """Compute the spin configuration label"""
    return sum([i * 2**n for n, i in enumerate(c)])

def read_spin_configurations_file(filename: str, model: MagneticModel) -> tuple:
    """
    Read a set of spin configurations relative to a model
    from a file

    Parameters
    ----------
    filename : str
        the file to read.
    model : MagneticModel
        the reference model.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    configuration_list = []
    energy_list = []
    comments = []
    with open(filename, "r") as stream:
        for line in stream:
            ls = line.strip()
            if ls == "" or ls[0] == "#":
                continue
            fields = ls.split(maxsplit=1)
            energy = float(fields[0])
            ls = fields[1]
            newconf = []
            comment = ""
            for pos, c in enumerate(ls):
                if c == "#":
                    comment = ls[(pos + 1) :]
                    break
                if c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            while len(newconf) < model.cell_size:
                newconf.append(0)
            comments.append(comment)
            configuration_list.append(newconf)
            energy_list.append(energy)
    return (energy_list, configuration_list, comments)
