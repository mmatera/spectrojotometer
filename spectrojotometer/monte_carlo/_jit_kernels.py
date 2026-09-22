"""Kernels Numba para Monte Carlo Ising.

Todos los kernels operan sobre arrays planos (formato CSR para las
listas de vecinos). El RNG es SplitMix64 propio, con estado en un
array np.uint64 de longitud 1, para que cada instancia tenga su
propio stream independiente del RNG global de Numba o NumPy.

Si Numba no esta instalado, se usa un decorador no-op y los kernels
funcionan igual en Python puro (mucho mas lentos).
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(func):
            return func

        return deco


# ---------------------------------------------------------------------------
# RNG (SplitMix64)
# ---------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TWO53 = 1.0 / 9007199254740992.0  # 2**-53


@njit(cache=True, inline="always")
def _splitmix64(state):
    """Avanza el estado y devuelve un uint64 pseudoaleatorio."""
    state[0] = state[0] + _GOLDEN
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _rand_double(state):
    """Uniforme en [0, 1) usando los 53 bits altos."""
    return float(_splitmix64(state) >> np.uint64(11)) * _TWO53


@njit(cache=True, inline="always")
def _rand_int(state, n):
    """Entero uniforme en [0, n)."""
    return int(_splitmix64(state) % np.uint64(n))


# ---------------------------------------------------------------------------
# Energia y campo local
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _interaction_energy_kernel(spins, bonds, J_vals):
    """Devuelve sum_{<i,j>} J_ij s_i s_j (sin E0)."""
    e = 0.0
    for k in range(bonds.shape[0]):
        i = bonds[k, 0]
        j = bonds[k, 1]
        e += J_vals[k] * float(spins[i]) * float(spins[j])
    return e


@njit(cache=True, fastmath=True, inline="always")
def _local_field(spins, nbr_idx, nbr_J, nbr_start, i):
    """Campo molecular h_i = sum_j J_ij s_j."""
    h = 0.0
    a = nbr_start[i]
    b = nbr_start[i + 1]
    for k in range(a, b):
        h += nbr_J[k] * float(spins[nbr_idx[k]])
    return h


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

@njit(cache=True)
def _metropolis_sweep_kernel(spins, nbr_idx, nbr_J, nbr_start, beta, state):
    """Un barrido completo: N intentos de flip con sitio aleatorio."""
    N = spins.shape[0]
    for _ in range(N):
        i = _rand_int(state, N)
        h = _local_field(spins, nbr_idx, nbr_J, nbr_start, i)
        si = spins[i]
        dE = 2.0 * si * h
        if dE <= 0.0:
            spins[i] = -si
        else:
            if _rand_double(state) < np.exp(-beta * dE):
                spins[i] = -si


# ---------------------------------------------------------------------------
# Wolff
# ---------------------------------------------------------------------------

@njit(cache=True)
def _wolff_step_kernel(spins, nbr_idx, nbr_J, nbr_start, beta, state,
                       cluster_mask, stack):
    """Un paso de Wolff: crecer cluster desde una semilla aleatoria y voltearlo.

    Importante: la arista (u, v) solo se activa si spins[u] == spins[v].
    Esta es la condicion de alineacion del algoritmo de Wolff: sin ella
    el cluster atraviesa paredes de dominio y desordena el sistema.
    """
    N = spins.shape[0]
    for k in range(N):
        cluster_mask[k] = False

    seed_site = _rand_int(state, N)
    cluster_mask[seed_site] = True
    stack[0] = seed_site
    top = 1
    size = 1
    cluster_spin = spins[seed_site]

    while top > 0:
        top -= 1
        u = stack[top]
        a = nbr_start[u]
        b = nbr_start[u + 1]
        for k in range(a, b):
            v = nbr_idx[k]
            if cluster_mask[v]:
                continue
            # Condicion de alineacion: solo agregar vecinos del mismo signo
            # que el spin del cluster (equivalente a spins[u] porque todo
            # el cluster comparte el signo pre-flip).
            if spins[v] != cluster_spin:
                continue
            J = nbr_J[k]
            p_add = 1.0 - np.exp(-2.0 * beta * J)
            if _rand_double(state) < p_add:
                cluster_mask[v] = True
                stack[top] = v
                top += 1
                size += 1

    for k in range(N):
        if cluster_mask[k]:
            spins[k] = -spins[k]

    return size

# ---------------------------------------------------------------------------
# Warmup (opcional, llamar antes de benchmarks)
# ---------------------------------------------------------------------------

def warmup():
    """Fuerza la compilacion de todos los kernels con datos triviales."""
    if not HAS_NUMBA:
        return
    spins = np.ones(4, dtype=np.int8)
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    J_vals = np.ones(4, dtype=np.float64)
    nbr_idx = np.array([1, 3, 0, 2, 1, 3, 0, 2], dtype=np.int64)
    nbr_J = np.ones(8, dtype=np.float64)
    nbr_start = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    state = np.array([_GOLDEN], dtype=np.uint64)
    mask = np.zeros(4, dtype=np.bool_)
    stack = np.zeros(4, dtype=np.int64)

    _interaction_energy_kernel(spins, bonds, J_vals)
    _metropolis_sweep_kernel(spins, nbr_idx, nbr_J, nbr_start, 1.0, state)
    _wolff_step_kernel(spins, nbr_idx, nbr_J, nbr_start, 1.0, state, mask, stack)
