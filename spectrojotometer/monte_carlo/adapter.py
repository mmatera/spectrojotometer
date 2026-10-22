"""From a MagneticModel to the arrays of the Ising simulation.

Expected structure of model.bonds:

    {
        'nn1': {
            'distance': 2.9386,
            'bonds': [(0, 1, '.'), (1, 2, '.')],
            'value': J1,
        },
        ...
    }

Each element of 'bonds' is (i, j, tag). The tag is the lattice offset
of the image of site j bonded to site i, as written by
spectrojotometer.tools.pack_offset (and read by unpack_offset):

    '.'       -> (0, 0, 0)         same cell
    '1_100'   -> (+1, 0, 0)        +1 along a
    '1_900'   -> (-1, 0, 0)        -1 along a
    '1_190'   -> (+1, -1, 0)

The whole tag must be decoded: without its '1_' prefix it means no
offset. Tuples of three integers are accepted too.

The couplings of the model are already in the Ising convention (the fit
projects the Heisenberg Hamiltonian onto the collinear basis and fits its
diagonal part): no S^2 factor is applied.
"""

import numpy as np

from ..tools import unpack_offset

# ---------------------------------------------------------------------------
# Arrays from model.bonds
# ---------------------------------------------------------------------------


def _extract_bond_arrays(model, J_overrides=None):
    """(bonds_cell, offset_cell, J_cell, n_sites) from model.bonds.

    Parameters
    ----------
    J_overrides : dict, optional
        {bond_name: J_value}, replacing the 'value' of the model for the
        couplings listed; the others use the value of the model.
    """
    bonds_dict = model.bonds
    J_overrides = J_overrides or {}

    i_list, j_list = [], []
    off_list = []
    J_list = []

    for name, info in bonds_dict.items():
        if name in J_overrides:
            J = float(J_overrides[name])
        elif "value" in info:
            J = float(info["value"])
        else:
            raise KeyError(
                f"The coupling {name!r} has no 'value' and is not in "
                f"J_overrides. Available keys: {list(info.keys())}"
            )
        if not np.isfinite(J):
            raise ValueError(
                f"The coupling {name!r} has no value (J = {J}): estimate the "
                "couplings first, or give its value in J_overrides."
            )
        for bond in info["bonds"]:
            if len(bond) != 3:
                raise ValueError(
                    f"Unexpected bond format: {bond!r}. "
                    "Expected (i, j, tag)."
                )
            i, j, tag = bond
            off = _offset_of(tag)
            i_list.append(int(i))
            j_list.append(int(j))
            off_list.append(off)
            J_list.append(J)

    if not i_list:
        bonds_cell = np.zeros((0, 2), dtype=np.int64)
        offset_cell = np.zeros((0, 3), dtype=np.int64)
        J_cell = np.zeros((0,), dtype=np.float64)
    else:
        bonds_cell = np.column_stack([i_list, j_list]).astype(np.int64)
        offset_cell = np.asarray(off_list, dtype=np.int64)
        J_cell = np.asarray(J_list, dtype=np.float64)

    n_sites = getattr(model, "n_sites", None)
    if n_sites is None:
        n_sites = getattr(model, "natoms", None)
    if n_sites is None and hasattr(model, "site_properties"):
        # MagneticModel: every site counts, also the ones without bonds.
        n_sites = len(model.site_properties["coord_atomos"])
    if n_sites is None:
        if bonds_cell.size == 0:
            raise ValueError(
                "Cannot infer n_sites: the model has no 'n_sites', "
                "'natoms' or 'site_properties', and model.bonds is empty."
            )
        n_sites = int(bonds_cell.max()) + 1

    return bonds_cell, offset_cell, J_cell, int(n_sites)


def _offset_of(tag):
    """
    The lattice offset (3 integers) of a bond, from the tag stored in
    MagneticModel.bonds: a packed string as written by pack_offset (e.g.
    '1_100' for +1 along a; '.' for no offset), or a sequence of three
    integers.

    The whole string must be decoded: stripping its 'N_' prefix (as this
    function used to do) loses the offset, and bonds to neighbouring cells
    end up inside the same cell.
    """
    if tag is None or (isinstance(tag, str) and tag in (".", "")):
        return np.zeros(3, dtype=np.int64)
    if isinstance(tag, str):
        return np.asarray(unpack_offset(tag), dtype=np.int64)
    offset = np.asarray(tag, dtype=np.int64)
    if offset.shape != (3,):
        raise ValueError(f"Unexpected bond offset: {tag!r}")
    return offset


# ---------------------------------------------------------------------------
# Tiling the supercell
# ---------------------------------------------------------------------------

def _tile_bonds(bonds_cell, offset_cell, J_cell, N_cell, supercell):
    """Expand the bonds and couplings of the unit cell to the supercell.

    Parameters
    ----------
    bonds_cell : (M, 2) int64
        Indices (i, j) in the unit cell.
    offset_cell : (M, 3) int64
        Offset, in cells, of site j with respect to site i.
    J_cell : (M,) float64
    N_cell : int
        Number of sites of the unit cell.
    supercell : (3,) tuple

    Returns
    -------
    N : int
    bonds : (M_super, 2) int64
    J_vals : (M_super,) float64
    """
    Lx, Ly, Lz = map(int, supercell)
    n_cells = Lx * Ly * Lz
    N = N_cell * n_cells

    def cell_index(ix, iy, iz):
        ix %= Lx
        iy %= Ly
        iz %= Lz
        return (ix * Ly + iy) * Lz + iz

    def site_index(cell, atom):
        return cell * N_cell + atom

    bonds_list = []
    J_list = []
    for ix in range(Lx):
        for iy in range(Ly):
            for iz in range(Lz):
                cell0 = cell_index(ix, iy, iz)
                for k in range(len(bonds_cell)):
                    i_atom = int(bonds_cell[k, 0])
                    j_atom = int(bonds_cell[k, 1])
                    ox, oy, oz = map(int, offset_cell[k])
                    cell1 = cell_index(ix + ox, iy + oy, iz + oz)
                    i_glob = site_index(cell0, i_atom)
                    j_glob = site_index(cell1, j_atom)
                    if i_glob == j_glob:
                        continue  # self-loop: a constant contribution
                    bonds_list.append((i_glob, j_glob))
                    J_list.append(J_cell[k])

    if bonds_list:
        bonds = np.asarray(bonds_list, dtype=np.int64)
        J_vals = np.asarray(J_list, dtype=np.float64)
    else:
        bonds = np.zeros((0, 2), dtype=np.int64)
        J_vals = np.zeros((0,), dtype=np.float64)

    return N, bonds, J_vals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def magnetic_model_to_ising(model, supercell=(1, 1, 1), J_overrides=None):
    """(spins, bonds, J_vals, E0) of the Ising model on a supercell of a
    MagneticModel.

    Parameters
    ----------
    model : MagneticModel
    supercell : tuple(int, int, int)
    J_overrides : dict, optional
        {bond_name: J_value}, replacing the 'value' of the model (the one
        stored by compute_couplings). A coupling without a value (nan) is
        an error.

    Returns
    -------
    spins : (N,) int8
    bonds : (M, 2) int64
    J_vals : (M,) float64
    E0 : float
    """
    bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(
        model, J_overrides=J_overrides
    )
    N, bonds, J_vals = _tile_bonds(
        bonds_cell, offset_cell, J_cell, n_sites, supercell
    )
    spins = np.ones(N, dtype=np.int8)
    E0 = float(getattr(model, "E0", 0.0))
    return spins, bonds, J_vals, E0



# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def inspect_bonds(model):
    """Print a summary of model.bonds: useful to check how the offsets and
    the couplings are read before running a simulation.
    """
    print("=" * 60)
    print(f"[adapter] Model: {type(model).__name__}")
    if hasattr(model, "site_properties"):
        n_sites = len(model.site_properties["coord_atomos"])
    else:
        n_sites = getattr(model, "n_sites", getattr(model, "natoms", "unknown"))
    print(f"[adapter] Sites in the unit cell: {n_sites}")
    print(f"[adapter] Couplings: {len(model.bonds)}")

    all_tags = set()
    for name, info in model.bonds.items():
        tags = sorted({str(b[2]) for b in info["bonds"]})
        all_tags.update(tags)
        print(f"  - {name!r}: dist={info.get('distance', '?')}, "
              f"J={info.get('value')}, n_bonds={len(info['bonds'])}, "
              f"tags={tags}")

    print(f"[adapter] Distinct tags: {sorted(all_tags)}")

    try:
        bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(model)
        print(f"[adapter] Read {len(bonds_cell)} bonds in the unit cell")
        print(f"[adapter] Distinct offsets: "
              f"{sorted(set(map(tuple, offset_cell)))}")
    except Exception as e:
        print(f"[adapter] Error reading the bonds: {e}")
    print("=" * 60)
