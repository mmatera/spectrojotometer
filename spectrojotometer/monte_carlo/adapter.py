"""Adaptador de MagneticModel a la representacion Ising.

Los J del fit ya estan en convencion Ising: Spectrojotometer proyecta el
Hamiltoniano de Heisenberg sobre la base colineal y ajusta la parte
diagonal. No aplicar reescalado S^2.
"""



def magnetic_model_to_ising(model, supercell=(1, 1, 1)):
    """Extrae (spins, bonds, J_vals, E0) de un MagneticModel.

    Debe adaptarse a la API real de MagneticModel. Los pasos esperados son:

    1. Expandir el modelo a la supercell.
    2. Extraer la lista de bonds con sus indices en la supercell.
    3. Extraer los J ajustados (ya en convencion Ising).
    4. Extraer E0.

    Returns
    -------
    spins : (N,) int8
    bonds : (M, 2) int64
    J_vals : (M,) float64
    E0 : float
    """
    # TODO: ajustar a la API real. Ejemplo esquematico:
    # model_super = model.build_supercell(supercell)
    # bonds = np.asarray(model_super.bond_indices, dtype=np.int64)
    # J_vals = np.asarray(model_super.J_values, dtype=np.float64)
    # E0 = float(getattr(model_super, "E0", 0.0))
    # N = model_super.n_sites
    # spins = np.ones(N, dtype=np.int8)
    # return spins, bonds, J_vals, E0
    raise NotImplementedError(
        "Adaptar magnetic_model_to_ising() a la API real de MagneticModel."
    )
