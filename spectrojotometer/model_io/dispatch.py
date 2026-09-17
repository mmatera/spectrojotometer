"""
dispatch
magnetic_model_from_file: picks the right reader (cif vs struct)
based on the filename extension.
"""
from typing import Optional
import logging

from ..magnetic_model import MagneticModel
from .common import DEFAULT_MAGNETIC_ATOMS
from .cif import magnetic_model_from_cif
from .struct import magnetic_model_from_wk2_struct


def magnetic_model_from_file(
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
        Only used for CIF files. See `magnetic_model_from_cif`. Ignored
        (with a warning if set) for `.struct` files, which don't support
        this reduction.

    Returns
    -------
    MagneticModel
        a MagneticModel.
    """
    if filename[-4:] == ".cif" or filename[-4:] == ".CIF":
        return magnetic_model_from_cif(
            filename, magnetic_atoms, bond_names, primitive_cell=primitive_cell
        )
    if filename[-7:] == ".struct" or filename[-7:] == ".STRUCT":
        if primitive_cell:
            logging.warning(
                "primitive_cell=True is not supported for .struct files; "
                "ignoring it."
            )
        return magnetic_model_from_wk2_struct(filename, magnetic_atoms, bond_names)
    logging.error("unknown file format")
    raise ValueError("Unknown file format for {filename}.")
