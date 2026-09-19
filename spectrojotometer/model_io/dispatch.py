"""
dispatch
magnetic_model_from_file: picks the right reader (cif vs struct)
based on the filename extension.

CIF reading is delegated to `model_io_pymatgen.magnetic_model_from_cif_pymatgen`
by default (see `USE_PYMATGEN_CIF_READER` below), not to the
hand-written line-by-line parser in `cif.py`. `cif.magnetic_model_from_cif`
is kept around (and still exported from `spectrojotometer.model_io`)
for direct use and comparison, but it is no longer what
`magnetic_model_from_file` -- and therefore the GUI and the
command-line scripts -- actually calls for a `.cif` file.
"""
from typing import Optional
import importlib
import logging


from ..magnetic_model import MagneticModel
from .common import DEFAULT_MAGNETIC_ATOMS
from .struct import magnetic_model_from_wk2_struct

# Prefer the pymatgen-based CIF reader over the hand-written one in
# `cif.py`. Set this to False to force `cif.magnetic_model_from_cif`
# even when pymatgen is installed.
USE_PYMATGEN_CIF_READER = True

if USE_PYMATGEN_CIF_READER and importlib.util.find_spec("pymatgen"):
    from .model_io_pymatgen import (
        magnetic_model_from_cif_pymatgen as magnetic_model_from_cif,
    )
    from .model_io_pymatgen import magnetic_model_from_cif_pymatgen
else:
    if USE_PYMATGEN_CIF_READER:
        logging.warning(
            "pymatgen is not available; falling back to the internal "
            "(non-pymatgen) CIF reader in model_io.cif."
        )
        USE_PYMATGEN_CIF_READER = False
    from .cif import magnetic_model_from_cif as magnetic_model_from_cif
    # No pymatgen available: expose the same reader under both names,
    # rather than leaving `magnetic_model_from_cif_pymatgen` undefined.
    magnetic_model_from_cif_pymatgen = magnetic_model_from_cif


def magnetic_model_from_file(
    filename: str,
    magnetic_atoms: tuple = DEFAULT_MAGNETIC_ATOMS,
    bond_names: Optional[list] = None,
    primitive_cell: bool = False,
) -> MagneticModel:
    """
    Build a `MagneticModel` from a crystal-structure file, picking the
    right reader from the filename's extension: `.cif`/`.CIF` go to
    `magnetic_model_from_cif` (see the module docstring for which
    implementation that actually is), `.struct`/`.STRUCT` (WIEN2k) go
    to `magnetic_model_from_wk2_struct`.

    Parameters
    ----------
    filename : str
        The name of the file to read.
    magnetic_atoms : tuple, optional
        The set of atoms to be considered magnetic.
        The default is ("Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V").
    bond_names : Optional[list], optional
        The names of the bonds.  The default is None, meaning that the bonds
        are named automatically. Ignored for `.struct` files, which don't
        read bonds at all.
    primitive_cell : bool, optional
        Only used for CIF files. See `magnetic_model_from_cif`. Ignored
        (with a warning if set) for `.struct` files, which don't support
        this reduction.

    Returns
    -------
    MagneticModel
        a MagneticModel.

    Raises
    ------
    ValueError
        If `filename` doesn't end in `.cif`, `.CIF`, `.struct` or
        `.STRUCT`.
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
    raise ValueError(f"Unknown file format for {filename}.")
