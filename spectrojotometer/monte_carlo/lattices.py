"""Constructores de redes para tests y ejemplos."""

import numpy as np


def build_square_lattice_2d(L, J=1.0):
    N = L * L
    spins = np.ones(N, dtype=np.int8)

    def idx(x, y):
        return (x % L) * L + (y % L)

    
    bonds = []
    for x in range(L):
        for y in range(L):
            i = idx(x, y)
            bonds.append((i, idx(x + 1, y)))
            bonds.append((i, idx(x, y + 1)))
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.full(len(bonds), float(J), dtype=np.float64)
    return spins, bonds, J_vals, 0.0


def build_triangular_lattice_2d(L, J=-1.0):
    """Triangular (frustrada si J < 0)."""
    N = L * L
    spins = np.ones(N, dtype=np.int8)

    def idx(x, y):
        return (x % L) * L + (y % L)
    
    bonds = []
    for x in range(L):
        for y in range(L):
            i = idx(x, y)
            bonds.append((i, idx(x + 1, y)))
            bonds.append((i, idx(x, y + 1)))
            bonds.append((i, idx(x + 1, y + 1)))
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.full(len(bonds), float(J), dtype=np.float64)
    return spins, bonds, J_vals, 0.0


def build_square_lattice_3d(L, J=1.0):
    N = L * L * L
    spins = np.ones(N, dtype=np.int8)

    def idx(x, y, z):
        return ((x % L) * L + (y % L)) * L + (z % L)

    bonds = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                bonds.append((i, idx(x + 1, y, z)))
                bonds.append((i, idx(x, y + 1, z)))
                bonds.append((i, idx(x, y, z + 1)))
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.full(len(bonds), float(J), dtype=np.float64)
    return spins, bonds, J_vals, 0.0


def build_chain_1d(L, J=1.0, periodic=True):
    N = L
    spins = np.ones(N, dtype=np.int8)
    bonds = []
    for i in range(L):
        j = (i + 1) % L if periodic else i + 1
        if j < L:
            bonds.append((i, j))
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.full(len(bonds), float(J), dtype=np.float64)
    return spins, bonds, J_vals, 0.0
