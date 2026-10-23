"""Detection of magnetic frustration from the balance of signs (Harary)."""

from collections import deque

import numpy as np


def detect_frustration(bonds, J_vals, N):
    """Determine whether the signed graph of the couplings is frustrated.

    A graph whose edges are labelled +-1 is *balanced* if there is an
    assignment tau_i in {+1, -1} such that tau_i * tau_j * sign(J_ij) = +1
    for every edge. Equivalently: every cycle contains an even number of
    negative edges.

    Parameters
    ----------
    bonds : array (M, 2)
    J_vals : array (M,)
    N : int
        Number of sites.

    Returns
    -------
    frustrated : bool
    gauge : np.ndarray (N,) int8 or None
        If not frustrated, tau such that tau_i * tau_j * sign(J_ij) = +1.
    odd_cycle : list[int] or None
        If frustrated, a cycle with an odd number of negative edges.
    """
    bonds = np.asarray(bonds, dtype=np.int64)
    J_vals = np.asarray(J_vals, dtype=np.float64)
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError("bonds must have shape (M, 2)")

    adj = [[] for _ in range(N)]
    for (i, j), J in zip(bonds, J_vals):
        s = 1 if J >= 0 else -1
        adj[int(i)].append((int(j), s))
        adj[int(j)].append((int(i), s))

    tau = np.zeros(N, dtype=np.int8)
    parent = -np.ones(N, dtype=np.int64)

    for start in range(N):
        if tau[start] != 0:
            continue
        tau[start] = 1
        q = deque([start])
        while q:
            u = q.popleft()
            for v, s in adj[u]:
                expected = tau[u] * s
                if tau[v] == 0:
                    tau[v] = expected
                    parent[v] = u
                    q.append(v)
                elif tau[v] != expected:
                    cycle = _reconstruct_cycle(u, v, parent)
                    return True, None, cycle

    return False, tau, None


def _reconstruct_cycle(u, v, parent):
    """Rebuild the cycle u -> ... -> LCA <- ... <- v -> u."""
    path_u = []
    x = u
    while x != -1:
        path_u.append(int(x))
        x = int(parent[x])

    path_v = []
    x = v
    while x != -1:
        path_v.append(int(x))
        x = int(parent[x])

    set_u = set(path_u)
    lca = next((n for n in path_v if n in set_u), None)
    if lca is None:
        return [int(u), int(v), int(u)]

    idx_u = path_u.index(lca)
    idx_v = path_v.index(lca)
    cycle = path_u[: idx_u + 1] + path_v[:idx_v][::-1] + [int(u)]
    return cycle
