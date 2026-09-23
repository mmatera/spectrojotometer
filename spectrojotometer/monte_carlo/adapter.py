"""Adaptador de MagneticModel a la representacion Ising.

Estructura esperada de model.bonds:

    {
        'nn1': {
            'distance': 2.9386,
            'bonds': [(0, 1, '.'), (1, 2, '.')],
            'value': J1,
        },
        ...
    }

Cada elemento de 'bonds' es (i, j, tag). El tag codifica la imagen
periodica del atomo j respecto del atomo i y se decodifica con
spectrojotometer.tools.unpack_offset:

    '.'      -> (0, 0, 0)         misma celda
    '100'    -> (+1, 0, 0)        +1 en a
    '900'    -> (-1, 0, 0)        -1 en a
    '190'    -> (+1, -1, 0)
    etc.

Los J del modelo ya estan en convencion Ising (el fit de Spectrojotometer
proyecta el Hamiltoniano de Heisenberg sobre la base colineal y ajusta
la parte diagonal). No se aplica ningun factor S^2.
"""

import numpy as np

from ..tools import unpack_offset


# ---------------------------------------------------------------------------
# Extraccion de arrays desde model.bonds
# ---------------------------------------------------------------------------

def _extract_bond_arrays(model):
    """Extrae (bonds_cell, offset_cell, J_cell, n_sites) del model.bonds.

    Returns
    -------
    bonds_cell : (M, 2) int64
    offset_cell : (M, 3) int64
    J_cell : (M,) float64
    n_sites : int
    """
    bonds_dict = model.bonds

    i_list, j_list = [], []
    off_list = []
    J_list = []

    for name, info in bonds_dict.items():
        if "value" not in info:
            raise KeyError(
                f"El bond {name!r} no tiene clave 'value'. "
                f"Claves disponibles: {list(info.keys())}"
            )
        J = float(info["value"])
        print(name,"=",J)
        for bond in info["bonds"]:
            if len(bond) != 3:
                raise ValueError(
                    f"Bond con formato inesperado: {bond!r}. "
                    "Se espera (i, j, tag)."
                )
            i, j, tag = bond
            off = unpack_offset(tag)
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

    # n_sites de la celda unidad
    n_sites = getattr(model, "n_sites", None)
    if n_sites is None:
        n_sites = getattr(model, "natoms", None)
    if n_sites is None:
        if bonds_cell.size == 0:
            raise ValueError(
                "No se puede inferir n_sites: model no tiene 'n_sites' "
                "ni 'natoms', y model.bonds esta vacio."
            )
        n_sites = int(bonds_cell.max()) + 1

    return bonds_cell, offset_cell, J_cell, int(n_sites)


# ---------------------------------------------------------------------------
# Tiling a la supercell
# ---------------------------------------------------------------------------

def _tile_bonds(bonds_cell, offset_cell, J_cell, N_cell, supercell):
    """Expande bonds/J de la celda unidad a la supercell.

    Parameters
    ----------
    bonds_cell : (M, 2) int64
        Indices (i, j) en la celda unidad.
    offset_cell : (M, 3) int64
        Offset en celdas del atomo j respecto del atomo i.
    J_cell : (M,) float64
    N_cell : int
        Numero de sitios de la celda unidad.
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
                        continue  # self-loop: contribucion constante
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
# API publica
# ---------------------------------------------------------------------------

def magnetic_model_to_ising(model, supercell=(1, 1, 1)):
    """Extrae (spins, bonds, J_vals, E0) de un MagneticModel.

    Parameters
    ----------
    model : MagneticModel
    supercell : tuple(int, int, int)

    Returns
    -------
    spins : (N,) int8
    bonds : (M, 2) int64
    J_vals : (M,) float64
    E0 : float
    """
    bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(model)

    N, bonds, J_vals = _tile_bonds(
        bonds_cell, offset_cell, J_cell, n_sites, supercell
    )

    spins = np.ones(N, dtype=np.int8)
    E0 = float(getattr(model, "E0", 0.0))

    return spins, bonds, J_vals, E0


# ---------------------------------------------------------------------------
# Diagnostico
# ---------------------------------------------------------------------------

def inspect_bonds(model):
    """Imprime un resumen de la estructura de model.bonds.

    Util para verificar que el adapter interpreta correctamente los tags
    de offset y los J antes de correr el Monte Carlo.
    """
    print("=" * 60)
    print(f"[adapter] Modelo: {type(model).__name__}")
    print(f"[adapter] n_sites (celda unidad): "
          f"{getattr(model, 'n_sites', getattr(model, 'natoms', 'NO DEFINIDO'))}")
    print(f"[adapter] Tipos de bond: {len(model.bonds)}")

    all_tags = set()
    for name, info in model.bonds.items():
        tags = sorted({b[2] for b in info["bonds"]})
        all_tags.update(tags)
        print(f"  - {name!r}: dist={info.get('distance', '?')}, "
              f"J={info['value']}, n_bonds={len(info['bonds'])}, "
              f"tags={tags}")

    print(f"[adapter] Tags unicos: {sorted(all_tags)}")

    try:
        bonds_cell, offset_cell, J_cell, n_sites = _extract_bond_arrays(model)
        print(f"[adapter] Parse exitoso: {len(bonds_cell)} bonds en la celda")
        print(f"[adapter] Offsets unicos encontrados: "
              f"{sorted(set(map(tuple, offset_cell)))}")
    except Exception as e:
        print(f"[adapter] ERROR en parse: {e}")
    print("=" * 60)
