# coding: utf-8
"""
Magnetic Model class
"""
from typing import Optional
import logging
import numpy as np
import numpy.linalg as la

from .tools import box_ellipse, compatibility_box, format_symmetry_operator, offset_orientation, pack_offset

logger = logging.getLogger(__name__)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3
# from matplotlib.lines import Line2D

# ******************************************************************************


SYMBOLS = {
    "plain": {
        "times_symbol": "*",
        "equal_symbol": "=",
        "open_comment": "# ",
        "close_comment": "",
        "sub_symb": "_",
    },
    "wolfram": {
        "times_symbol": "*",
        "equal_symbol": "==",
        "open_comment": "(* ",
        "close_comment": " *)",
        "sub_symb": "",
    },
    "latex": {
        "times_symbol": "",
        "equal_symbol": "=",
        "open_comment": "% ",
        "close_comment": "",
        "sub_symb": "_",
    },
}


def supercell_iterator(supercell_size, coord_atomos, bravais_vectors):
    """Iterator over subcells in the supercell.

    supercell_size can be:
      - an int  : same size along all axes (backward-compatible)
      - a list  : [sx, sy, sz] — independent size per Bravais axis
    """

    # Basis vectors must be np.arrays.
    bravais_vectors = [np.array(vector) for vector in bravais_vectors]

    # If no bravais_vectors, the supercell is just the cell.
    if len(bravais_vectors) == 0:
        yield ([], coord_atomos)
        return

    # Normalise to a per-axis list, one entry per Bravais vector.
    n = len(bravais_vectors)
    if np.isscalar(supercell_size):
        sizes = [int(supercell_size)] * n
    else:
        sizes = [int(s) for s in supercell_size]
        if len(sizes) < n:
            sizes += [sizes[-1]] * (n - len(sizes))  # pad with last value
        sizes = sizes[:n]

    current = [-s for s in sizes]
    max_indx = n
    while True:
        offset = sum(coeff * v_b for coeff, v_b in zip(current, bravais_vectors))
        yield (current[:], [site + offset for site in coord_atomos])
        # Advance to next replica (mixed-radix counter)
        indx = 0
        while True:
            if indx == max_indx:
                return
            sub_idx = current[indx] + 1
            if sub_idx <= sizes[indx]:
                current[indx] = sub_idx
                break
            current[indx] = -sizes[indx]
            indx += 1


def build_supercell(supercell_size, coord_atomos, bravais_vectors):
    """Build a list of offsets for the supercell"""

    return np.array(
        [
            cell[1]
            for cell in supercell_iterator(
                supercell_size, coord_atomos, bravais_vectors
            )
        ]
    )


class MagneticModel:
    """
    MagneticModel represents a set of magnetic atoms
    bound by pairwise Heisenberg-like interactions.
    """
    cell_size:int
    model_label:str
    onfly:bool
    site_properties: dict
    lattice_properties:dict
    bonds: dict

    def __init__(
        self,
        atomic_pos,
        bravais_lat,
        bond_lists=None,
        bond_names=None,
        bond_distances=None,
        ranges=None,
        supercell_size=1,
        discretization:float=0.02,
        magnetic_species=None,
        onfly:bool=True,
        model_label="default",
        g_lande_factors:Optional[float]=None,
        spin_repr:Optional[float]=None,
        symmetries=None,
        space_group_symbol=None,
    ):
        self.model_label = model_label
        self.onfly = onfly
        self.cell_size = 0
        logger.debug("supercell size: %s", supercell_size)
        # Normalise: accept int or [sx, sy, sz]; clamp each axis to 1–4
        if np.isscalar(supercell_size):
            supercell_size = min(4, max(1, int(supercell_size)))
        else:
            supercell_size = [min(4, max(1, int(s))) for s in supercell_size]

        logger.debug("atomic positions: %s", atomic_pos)
        if magnetic_species is None:
            magnetic_species = ["X"] * len(atomic_pos)

        logger.debug("magnetic species: %s", magnetic_species)

        if g_lande_factors is None:
            g_lande_factors = [2.0] * len(magnetic_species)
        else:
            g_str:str
            indx:int
            for indx, g_str in enumerate(g_lande_factors):
                g_lande_factors[indx] = 2.0 if g_str == "." else float(g_str)

        if spin_repr is None:
            spin_repr = [0.5 for i in magnetic_species]
        else:
            for indx, s_max in enumerate(spin_repr):
                spin_repr[indx] = 0.5 if s_max == "." else float(s_max)

        if ranges is None:
            maxdist = 0
            dist = 0
            for p_vec in atomic_pos:
                for q_vec in atomic_pos:
                    dist = np.linalg.norm(p_vec - q_vec)
                    if dist > maxdist:
                        maxdist = dist
            if maxdist == 0:
                maxdist = 1
            ranges = [[0, maxdist]]
        elif np.isscalar(ranges):
            ranges = [[0, float(ranges)]]
        elif len(ranges) > 0 and np.isscalar(ranges[0]):
            ranges = [list(ranges)]

        if bond_lists is not None:
            logger.debug("got %d couplings", len(bond_lists))
            if bond_distances is None:
                bond_distances = [0] * len(bond_lists)
            elif len(bond_distances) != len(bond_lists):
                raise ValueError(
                    "bond_distances must have the same length as bond_lists "
                    f"({len(bond_distances)} != {len(bond_lists)})"
                )
            if bond_names is None:
                bond_names = ["J" + str(i) for i in range(len(bond_lists))]
        else:
            bond_names = []
            bond_lists = []
            bond_distances = []

        self.site_properties = site_properties = {}
        site_properties["coord_atomos"] = atomic_pos
        site_properties["g_lande_factors"] = g_lande_factors
        site_properties["spin_repr"] = spin_repr
        site_properties["magnetic_species"] = magnetic_species

        self.lattice_properties = lattice_properties = {}
        lattice_properties["bravais_vectors"] = bravais_lat
        lattice_properties["cell_size"] = len(atomic_pos)
        lattice_properties["supercell_size"] = supercell_size
        # Symmetry operators actually used to build this model (empty for a
        # plain P1 description). Kept mainly so `save_cif` can write a
        # faithful CIF back out instead of always collapsing to P1, and so
        # downstream tools (e.g. the web interface) can tell how the model
        # was built.
        lattice_properties["symmetries"] = (
            list(symmetries) if symmetries is not None else []
        )
        lattice_properties["space_group_symbol"] = space_group_symbol

        self.bonds = {
            b_name: {"distance": b_distance, "bonds": b_list, "value":np.nan}
            for b_name, b_distance, b_list in zip(
                bond_names, bond_distances, bond_lists
            )
        }

    def remover_bond_by_name(self, name):
        """
        Remove the bond with name <name>
        """
        try:
            self.bonds.pop(name)
        except KeyError:
            msg = "bond " + name + " not found"
            logger.warning(msg)
            return

    remove_bond_by_name = remover_bond_by_name

    def generate_bonds(self, discretization, ranges, bond_names=None):
        """
        En base a la celda unidad, la red de Bravais y el tamaño de la
        "super-celda" (cuantas copias de la celda unidad se consideran
        como entorno) calcula los bonds,
        esto es, los pares de sitios magnéticos que interactúan según cada
        una de las constantes de acoplamiento a determinar.
        """
        atom_type = self.site_properties["magnetic_species"]
        cell_size = self.lattice_properties["cell_size"]
        coord_atomos = self.site_properties["coord_atomos"]
        supercell_size = self.lattice_properties["supercell_size"]
        bonds = self.bonds

        if bond_names is None:
            bond_names = []

        def is_in_range(val):
            for min_r, max_r in ranges:
                if min_r < val < max_r:
                    return True
            return False

        def normalize_key(bond_distance):
            return round(bond_distance / discretization) * discretization

        # Collect bonds
        new_bonds = {}
        for supercell_idx, sites_sc in supercell_iterator(
            supercell_size,
            coord_atomos,
            self.lattice_properties["bravais_vectors"],
        ):
            for i, pos_i in enumerate(coord_atomos):
                for j, pos_j in enumerate(sites_sc):
                    logger.debug("pair %s, supercell %s", [i, j], supercell_idx)
                    rel_pos = pos_j - pos_i
                    orientation = offset_orientation(rel_pos)
                    if orientation < 0:
                        continue
                    distance = np.linalg.norm(rel_pos)
                    logger.debug("   distance %s", distance)
                    if not is_in_range(distance):
                        continue

                    logger.debug("bond between %s and %s (%s)", i, j, supercell_idx)
                    bound_type = tuple(sorted({atom_type[i], atom_type[j]}))
                    key = (normalize_key(distance), bound_type)
                    offset_key = pack_offset(supercell_idx)
                    new_bonds.setdefault(key, []).append((i, j % cell_size, offset_key))

        bn_index = 0
        while len(new_bonds) > len(bond_names):
            while "J" + str(bn_index + 1) in bond_names:
                bn_index = bn_index + 1
            logger.debug("adding %s", "J" + str(bn_index + 1))
            bond_names.append("J" + str(bn_index + 1))

        # Update the bonds dict.
        logger.debug("current bonds: %s", bonds)
        logger.debug("new bond names: %s", bond_names)

        for name, dist_type in zip(bond_names, sorted(new_bonds)):
            dist = dist_type[0]
            if name in bonds:
                if abs(bonds[name]["distance"] - dist) > discretization:
                    while name in bonds:
                        name = "J" + name
                else:
                    bonds[name]["bonds"].extend(new_bonds[dist_type])
                    continue
            bonds[name] = {"distance": dist, "bonds": new_bonds[dist_type]}

    def formatted_equations(
        self, coeff_matrix, ensname=None, comments=None, eq_format="plain"
    ):
        """Format the equations"""

        res = ""
        symb = SYMBOLS[eq_format]
        jsname = []
        jsname[:] = self.bonds
        jsname.append("E" + symb["sub_symb"] + "0")
        if ensname is None:
            ensname = [
                "E" + symb["sub_symb"] + str(i + 1) for i in range(len(coeff_matrix))
            ]
        for indx, row in enumerate(coeff_matrix):
            eq_str = ""
            for k, coeff in enumerate(row):
                coeff_round = round(coeff * 100) / 100.0
                if coeff > 0:
                    if eq_str != "":
                        eq_str = eq_str + " + "
                    else:
                        eq_str = "  "
                    if coeff_round != 1:
                        eq_str = eq_str + str(coeff_round) + " " + symb["times_symbol"]
                    eq_str = eq_str + " " + jsname[k]
                elif coeff < 0:
                    if eq_str != "":
                        eq_str = eq_str + " "
                    eq_str = eq_str + "- "
                    if coeff_round != -1:
                        eq_str = eq_str + str(-coeff_round) + " " + symb["times_symbol"]
                    eq_str = eq_str + " " + jsname[k]
            if comments is not None:
                res = (
                    res
                    + eq_str
                    + symb["equal_symbol"]
                    + ensname[indx]
                    + "  "
                    + symb["open_comment"]
                    + comments[indx]
                    + symb["close_comment"]
                    + "\n\n"
                )
            else:
                res = res + eq_str + symb["equal_symbol"] + ensname[indx] + "\n\n"
        return res

    def print_equations(
        self, coeff_matrix, ensname=None, comments=None, eq_format="plain"
    ):
        """Print the equations found"""
        print("\n\n# Equations: \n============\n\n")
        print(self.formatted_equations(coeff_matrix, ensname, comments, eq_format))

    def coefficient_matrix(self, configs, normalizar=False):
        """
        Return the matrix defining the equation system
        connecting coupling constants with simulated energies
        for a given set of configurations configs.
        """
        logger.debug("compute the coefficient matrix")

        rawcm = [
            np.array(
                [
                    -sum((-1) ** sc[b[0]] * (-1) ** sc[b[1]] for b in bondfamily)
                    for sc in configs
                ]
            )
            for bondfamily in (bd["bonds"] for bd in self.bonds.values())
        ]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("raw coefficient matrix:\n%s", "\n".join(str(row) for row in rawcm))
        if normalizar:
            coeff_matrix = [v - np.average(v) for v in rawcm]
        else:
            coeff_matrix = rawcm

        coeff_matrix = np.array(
            coeff_matrix
            +
            # Energía no magnética
            [np.array([1.0 for sc in configs])]
        ).transpose()
        return coeff_matrix

    def generate_configurations_onfly(self):
        """Generate spin configurations"""
        size = self.lattice_properties["cell_size"]
        for c in range(2 ** ((size - 1))):
            yield [c >> i & 1 for i in range(size - 1, -1, -1)]

    def generate_random_configurations(self, num_confs=10):
        """Generate spin configurations"""
        size = self.lattice_properties["cell_size"]
        rng = np.random.default_rng()
        # The first spin is fixed to 0 (global spin inversion symmetry),
        # the others are drawn uniformly. Drawing bits directly avoids
        # integer overflow for large cells.
        for _ in range(num_confs):
            bits = rng.integers(0, 2, size=max(size - 1, 0))
            yield [0] * min(size, 1) + [int(b) for b in bits]

    def check_independence(self, conf, setconfs, tol=0.1):
        """
        Check that the equation associated to conf is distinct from the
        ones associated to the configurations in setconfs (i.e. its row of
        the coefficient matrix is farther than tol from all of them).

        Note: this is a distinctness test, not a rank test. Rank
        deficiencies are handled later in optimize_independent_set.
        """
        if len(conf) == 0:
            return False
        if len(setconfs) == 0:
            return True
        a_coeffs = self.coefficient_matrix(setconfs, False)
        r0_coeffs = self.coefficient_matrix([conf], False)[0]
        return bool(np.min(np.linalg.norm(a_coeffs - r0_coeffs, axis=1)) >= tol)

    def add_independent_confs(self, confs, newconfs):
        """
        Produce new independent configurations and add
        them to the list.
        """
        for conf in newconfs:
            if self.check_independence(conf, confs):
                confs.append(conf)
        return confs

    def get_independent_confs(self, confs):
        """
        Choose the subset of configurations that produces
        independent equations.
        """
        res = []
        if len(confs) == 0:
            return res
        for c in confs:
            if self.check_independence(c, res):
                res.append(c)
        logger.debug("independent confs: %s", res)
        return res

    def cost(self, confs):
        """
        This function evaluates a bound on the error associated
        to a set of configurations.
        """
        a = self.coefficient_matrix(confs, False)
        lp = len(a)
        sv = np.linalg.svd(a, full_matrices=False, compute_uv=False)[-1]
        return np.sqrt(lp) / sv

    def optimize_independent_set(self, confs, length=None, forced=None, tol=1.0e-6):
        """
        Given a set of configurations confs, returns

        (res, cost)

        where res is the subset of configurations (not including the forced
        ones) that optimizes the cost function
        sqrt(len(res) + len(forced)) / sigma_min(coefficient_matrix).

        If the optional parameter length is provided, then it tries to optimize
        the cost function for a fixed size length (the set may be enlarged
        if that is needed to recover the full rank of the information).

        If the set only provides partial information, the cost is np.inf.
        """
        if len(confs) == 0:
            return [], np.inf

        forced = list(forced) if forced is not None else []
        lenforced = len(forced)

        candidates = [
            c
            for c in self.get_independent_confs(confs)
            if self.check_independence(c, forced)
        ]
        if not candidates:
            return [], np.inf
        curr = forced + candidates

        a = self.coefficient_matrix(curr, False)
        u, sv, _ = np.linalg.svd(a, full_matrices=False)
        k = int(np.sum(sv > tol))  # numerical rank
        if k == 0:
            logger.info(
                "optimize_independent_set: the set of configurations"
                " does not provide any information."
            )
            return [], np.inf

        # Compare with the number of unknowns (bonds + base energy).
        partialinfo = k < a.shape[1]
        if partialinfo:
            logger.info(
                "optimize_independent_set: the set of configurations just"
                " provides partial information. "
                "Optimizing the set for this information."
            )
        u = u[:, :k]

        # Sort the non-forced configurations by leverage (row norm of U).
        order = sorted(
            range(lenforced, len(curr)), key=lambda i: -np.linalg.norm(u[i])
        )
        curr_sorted = forced + [curr[i] for i in order]
        a_new = a[order]  # non-forced rows, sorted by weight

        def smin(rows):
            """k-th singular value of the rows (0 if there are fewer)."""
            s = np.linalg.svd(np.asarray(rows), full_matrices=False, compute_uv=False)
            return s[k - 1] if len(s) >= k else 0.0

        def result(npick):
            return curr_sorted[lenforced : lenforced + npick]

        ared = list(a[:lenforced])
        if length is not None:
            lp = min(max(length, 0), len(a_new))
        else:
            lp = min(max(k - lenforced, 0), len(a_new))
        ared += list(a_new[:lp])

        # Enlarge until the rank-k information is recovered.
        s = smin(ared) if ared else 0.0
        while s <= tol and lp < len(a_new):
            ared.append(a_new[lp])
            lp += 1
            s = smin(ared)

        if s <= tol or partialinfo:
            logger.info(
                "Information is not complete. "
                "Picking the subset that provides the maximal information."
            )
            return result(lp), np.inf

        if length is not None:
            cost = np.sqrt(lp + lenforced) / s
            logger.info("optimize_independent_set for fixed length. cost=%s", cost)
            return result(lp), cost

        # No length provided and complete information: enlarge the set
        # while that reduces the cost function.
        cost_full = np.sqrt(len(curr)) / sv[k - 1]
        cost = np.sqrt(lp + lenforced) / s
        while lp < len(a_new) and cost > cost_full:
            ared.append(a_new[lp])
            lp += 1
            cost = np.sqrt(lp + lenforced) / smin(ared)

        while lp < len(a_new):
            s2 = smin(ared + [a_new[lp]])
            new_cost = np.sqrt(lp + lenforced + 1) / s2 if s2 > 0 else np.inf
            if new_cost >= cost:
                break
            cost = new_cost
            ared.append(a_new[lp])
            lp += 1

        return result(lp), cost

    def find_optimal_configurations(
        self,
        start=None,
        num_new_confs=None,
        known=None,
        its=100,
        update_size=1,
    ):
        """
        Find an optimal set of configurations in order to estimate the
        coupling constants with the best accuracy.
        """
        if num_new_confs is None:
            lmax = max(len(self.bonds) + 1, 1) + update_size
        else:
            lmax = max(num_new_confs, len(self.bonds) + 1, 1)

        if known is None:
            known = []

        logger.info("--------------find_optimal_set ----------------")

        if start is None:
            repres = self.generate_random_configurations(max(update_size, lmax))
            last_better = self.get_independent_confs(list(repres))
        else:
            last_better = self.get_independent_confs(start)

        for _ in range(its):
            repres = self.generate_random_configurations(update_size)
            last_better = self.get_independent_confs(last_better + list(repres))
            last_better = [c for c in last_better if self.check_independence(c, known)]
            if len(last_better) >= lmax:
                break

        if len(last_better) < lmax:
            logger.info(
                "not enough configurations. Trying to optimize "
                + "the subset and return it."
            )
            return self.optimize_independent_set(
                last_better, num_new_confs, forced=known
            )

        last_better, cost = self.optimize_independent_set(
            last_better, num_new_confs, forced=known
        )
        for _ in range(its):
            repres = self.generate_random_configurations(update_size)
            new_try = self.get_independent_confs(last_better + list(repres))
            new_try, newcost = self.optimize_independent_set(
                new_try, num_new_confs, forced=known
            )
            if newcost < cost:
                cost = newcost
                last_better = new_try

        return last_better, cost

    def compute_couplings(
        self,
        confs,
        energs,
        err_energs=0.01,
        printeqs=False,
        montecarlo=None,
        mcsteps=None,
        mcsizefactor=1.0,
        rng=None,
        method=None,
    ):
        """
        compute_couplings
        Given a set of configurations, and the energies calculated from
        the ab-initio tools, estimates the values of the coupling
        constants from the proposed model.


        confs: list of magnetic configurations
        energs: energies evaluated for each magnetic configuration
        err_energs: estimation of the maximum convergence error in energies
        printeqs: print the corresponding equations.

        method: how to estimate the box containing the compatibility
            region:
            "exact": exact bounding box, by linear programming
                (see tools.compatibility_box). Default.
            "montecarlo": Monte Carlo sampling (see montecarlo_box). Its
                box is always inside the exact one, and underestimates it
                as the number of couplings grows.
            "ellipse": bounding box of the ellipsoid that contains the
                region. Always contains the exact box.
        montecarlo: kept for backward compatibility. If given (and method
            is None), True means "montecarlo" and False means "ellipse".
        mcsteps, mcsizefactor, rng: passed to montecarlo_box.

        Return values:
        --------------
        js, deltaJ, model_chi, acceptance_rate

        js: the estimated values of the couplings. The last entry is
            the average energy over the configurations (the columns of
            the coupling constants are centered, which decouples them
            from the base energy).

        deltaJ: half widths of a box centered at js that contains the
            compatibility region
            {j : max_i |(A.j - energs)_i| <= err_energs}.

        model_chi: (A.js_ls - energs) / err_energs, with js_ls the least
            squares solution.

        acceptance_rate: Monte Carlo acceptance rate (1.0 for the other
            methods, 0.0 if the model is incompatible).

        Undetermined couplings are returned with js = 0 and deltaJ = nan.
        If the model is incompatible with the energies, all the entries
        of deltaJ are -1.
        """
        print("Efind jssss")
        if printeqs:
            coeffs = self.coefficient_matrix(confs, normalizar=False)
            print("\n# Configurations:\n=================\n")
            for c in confs:
                print(c)
            self.print_equations(coeffs)

        coeffs = self.coefficient_matrix(confs, normalizar=True)

        # The choice in the way the equation is written allows to decouple
        # the determination of the coupling constants from the base
        # energy. This implies that the condition number associated to the
        # coupling constants should be evaluated from the reduced set of
        # equations.
        rcoeffs = coeffs[:, 0:-1]
        singularvalues = np.linalg.svd(rcoeffs, compute_uv=False)
        logger.debug("singular values of the coupling columns: %s", singularvalues)
        cond_number = np.sqrt(len(rcoeffs)) / max(min(singularvalues), 1.0e-9)
        if printeqs:
            print(f"\nInverse of the minimum singular value: {cond_number}\n")

        if method is None:
            method = (
                "exact" if montecarlo is None
                else ("montecarlo" if montecarlo else "ellipse")
            )
        if method == "montecarlo":
            return self.montecarlo_box(
                coeffs,
                energs,
                err_energs,
                1000 if mcsteps is None else mcsteps,
                mcsizefactor,
                rng=rng,
            )
        if method == "exact":
            return self.exact_box(coeffs, energs, err_energs)
        if method != "ellipse":
            raise ValueError(f"unknown method {method!r}")
        # Otherwise, it gives an estimation in terms of the bigger ellipse.
        # js = (A^t A )^{-1} A^t En, i.e. least squares solution
        resolvent = np.linalg.pinv(coeffs.transpose().dot(coeffs), rcond=1.0e-6).dot(
            coeffs.transpose()
        )
        js = resolvent.dot(energs)
        model_chi = (coeffs.dot(js) - energs) / err_energs
        rr = len(model_chi) - sum(model_chi**2)
        if rr < 0:
            deltaJ = [-1 for i in js]
        else:
            rr = np.sqrt(rr) * err_energs
            deltaJ = box_ellipse(coeffs, rr)
        # Update values of the coupling constants:
        print("store values", js, "in", self.bonds.keys())
        for j_label ,j_val in zip(self.bonds, js):
            print(j_label,"->", j_val)
            self.bonds[j_label]["value"]=j_val

        return (js, deltaJ, model_chi, 1.0)

    def bound_inequalities(self, confs, energs, err_energs=0.01, rtol=1.0e-3):
        """
        Compute the compatibility regions for the coupling constants.

        Returns a list of tuples (v, offset, delta), one for each direction
        in the space of couplings that is determined by the equations, such
        that the compatibility region satisfies

            |v . J - offset| <= delta.

        v is normalized so that its largest component is 1 in absolute
        value. Directions associated to singular values smaller than
        rtol times the largest one are considered undetermined and
        skipped. Returns [] if the model is incompatible.

        The constraints follow from projecting the ellipsoid
        |A.(J - J_ls)|^2 <= N err^2 - |r|^2, which contains the
        compatibility region, onto the right singular vectors of A.
        """
        coeffs = self.coefficient_matrix(confs, normalizar=True)
        energs = np.asarray(energs, dtype=float)
        # The columns of the couplings are centered, so they are orthogonal
        # to the (constant) column of the base energy: the base energy
        # decouples exactly, and we can work only with the couplings.
        rcoeffs = coeffs[:, :-1]
        if rcoeffs.shape[1] == 0:
            return []
        u, singularvalues, vh = np.linalg.svd(rcoeffs, full_matrices=False)
        if singularvalues[0] <= 0:
            return []
        k = int(np.sum(singularvalues > rtol * singularvalues[0]))
        u, singularvalues, vh = u[:, :k], singularvalues[:k], vh[:k]
        # Least squares solution for the couplings, and residual of the
        # full system (the base energy is fitted by the mean energy).
        j0s = vh.T.dot(u.T.dot(energs) / singularvalues)
        residual = rcoeffs.dot(j0s) - (energs - np.mean(energs))
        err = len(energs) * err_energs**2 - residual.dot(residual)
        if err < 0:
            return []
        err = np.sqrt(err)

        ineqs = []
        for v, s_val in zip(vh, singularvalues):
            maxcoeff = v[np.argmax(np.abs(v))]  # keeps a canonical sign
            ineqs.append(
                (v / maxcoeff, v.dot(j0s) / maxcoeff, err / s_val / abs(maxcoeff))
            )
        return ineqs

    def exact_box(self, coeffs, energs, tol):
        """
        Exact bounding box of the compatibility region

            P = {j : max_i |(coeffs . j - energs)_i| <= tol},

        computed by linear programming (see tools.compatibility_box).

        Returns (js, d_js, model_chi, 1.0) with the same conventions as
        montecarlo_box: js is the center of the box, d_js its half widths
        (nan for undetermined parameters, -1 for all of them if the model
        is incompatible), and model_chi the normalized residual of the
        least squares solution.
        """
        coeffs = np.asarray(coeffs, dtype=float)
        energs = np.asarray(energs, dtype=float)
        j_ls = np.linalg.lstsq(coeffs, energs, rcond=None)[0]
        model_chi = (coeffs.dot(j_ls) - energs) / tol
        box = compatibility_box(coeffs, energs, tol)
        if box is None:
            logger.warning("The model is incompatible with the energies.")
            return j_ls, -np.ones(len(j_ls)), model_chi, 0.0
        lower, upper = box
        determined = np.isfinite(lower) & np.isfinite(upper)
        js = np.zeros(len(lower))
        d_js = np.full(len(lower), np.nan)
        js[determined] = 0.5 * (upper[determined] + lower[determined])
        d_js[determined] = 0.5 * (upper[determined] - lower[determined])
        return js, d_js, model_chi, 1.0

    def montecarlo_box(
        self, coeffs, energs, tol, numtry=1000, sizefactor=1.0, rng=None
    ):
        """
        Estimate the bounding box of the compatibility region

            P = {j : max_i |(coeffs . j - energs)_i| <= tol}

        by rejection sampling.

        Since |v|_2^2 <= N |v|_inf^2, P is contained in the ellipsoid

            |coeffs . (j - j_ls)|^2 <= N tol^2 - |r|^2 = R^2,

        where j_ls is the least-squares solution and r its residual
        (r is orthogonal to the image of coeffs). Points are drawn
        uniformly inside that ellipsoid and kept if they belong to P, so
        the accepted points are uniformly distributed in P. Everything
        scales linearly with (energs, tol): the result does not depend on
        the energy units, and the acceptance rate is the volume ratio
        vol(P) / vol(ellipsoid).

        Parameters
        ----------
        coeffs : (N, p) array, coefficient matrix.
        energs : (N,) array, energies.
        tol : float, maximum admissible error in each energy.
        numtry : int, number of sampled points.
        sizefactor : float, scale of the sampling ellipsoid relative to the
            one that is guaranteed to contain P. Values < 1 may miss parts
            of P; values > 1 only lower the acceptance rate.
        rng : None, int or numpy.random.Generator, source of randomness.

        Returns
        -------
        (js, d_js, model_chi, acceptance_rate)

        js : center of the box. Undetermined parameters are set to 0.
        d_js : half widths of the box (np.nan for undetermined
               parameters, -1 for all of them if the model is incompatible).
        model_chi : (coeffs . js_ls - energs) / tol, with js_ls the least
               squares solution.
        acceptance_rate : fraction of accepted points.
        """
        rng = np.random.default_rng(rng)
        coeffs = np.asarray(coeffs, dtype=float)
        energs = np.asarray(energs, dtype=float)
        numcalc, numparm = coeffs.shape

        # Work in the row space of coeffs: directions in its kernel are not
        # determined by the equations.
        u, sv, vh = la.svd(coeffs, full_matrices=False)
        k = int(np.sum(sv > 1.0e-6))
        u, sv, vh = u[:, :k], sv[:k], vh[:k]
        # A parameter is determined iff its axis lies in the row space.
        determined = np.linalg.norm(vh, axis=0) > 1.0 - 1.0e-6

        # Least squares solution (minimal norm) and its residual.
        j_ls = vh.T.dot(u.T.dot(energs) / sv)
        residual = coeffs.dot(j_ls) - energs
        model_chi = residual / tol
        radius2 = numcalc * tol**2 - residual.dot(residual)

        def incompatible(msg):
            logger.warning(msg)
            return j_ls, -np.ones(numparm), model_chi, 0.0

        if radius2 <= 0 or k == 0:
            return incompatible("The model is incompatible with the energies.")

        # Uniform samples in the unit k-ball: Gaussian direction, radius
        # distributed as U^(1/k). Then map them onto the ellipsoid.
        z = rng.normal(size=(numtry, k))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z *= rng.random(numtry)[:, None] ** (1.0 / k)
        radius = sizefactor * np.sqrt(radius2)
        samples = j_ls + (radius * z / sv).dot(vh)
        samples = np.vstack([samples, j_ls])

        accepted = samples[
            np.max(np.abs(samples.dot(coeffs.T) - energs), axis=1) <= tol
        ]
        rate = len(accepted) / (numtry + 1.0)
        logger.info("Monte Carlo acceptance rate=%s", rate)

        if len(accepted) == 0:
            return incompatible(
                "No compatible point was found: the model seems "
                "incompatible. Try with a larger number of points."
            )

        if len(accepted) == 1:
            logger.info("errors estimated by boxing the ellipse")
            js = j_ls.copy()
            d_js = np.array(box_ellipse(coeffs, np.sqrt(radius2)))
        else:
            lower = accepted.min(axis=0)
            upper = accepted.max(axis=0)
            js = 0.5 * (upper + lower)
            d_js = 0.5 * (upper - lower)

        if len(jtrys) == 1:
            print("errors  estimated by boxing the  ellipse")
            rr = np.sqrt(numcalc * tol**2 - esqerr**2)
            djs = box_ellipse(coeffs, rr)
            return j0s, djs, (e0s - energs) / tol, 1.0 / (numtry + 1.0)
        print(len(jtrys))
        print("Accepted rate=", len(jtrys) / (numtry + 1.0))

        j0s = np.average(jtrys, 0)
        jtrys = [abs(js - j0s) for js in jtrys]
        d_js = np.max(np.array(jtrys), 0)
        # If the model is singular, discard the coefficients that
        # couldn't be determined, setting them to 0.
        if issingular:
            for i, d_j in enumerate(d_js):
                if d_j > 100:
                    d_js[i] = np.nan
                    j0s[i] = 0

        e0s = coeffs.dot(j0s)

        print("store values", j0s, "in", self.bonds.keys())
        for j_label ,j_val in zip(self.bonds, j0s):
            print(j_label,"->", j_val)
            self.bonds[j_label]["value"]=j_val
        
        return (
            j0s,
            d_js,
            (e0s - energs)[:numcalc] / tol,
            len(jtrys) / (numtry + 1.0),
        )

    def save_cif(self, filename, bond_names=None):
        """Save the model in a CIF file"""
        bravais_vectors = [
            np.asarray(v, dtype=float) for v in self.lattice_properties["bravais_vectors"]
        ]
        if bond_names is None:
            bond_names = tuple(self.bonds)
        msg = "save cif" + filename
        logger.info(msg)
        model_name = "data_magnetic_model_1"

        symmetries = self.lattice_properties.get("symmetries") or []
        n_dim = len(bravais_vectors)
        var_names = ["z"] if n_dim == 1 else (["x", "y"] if n_dim == 2 else ["x", "y", "z"])
        if symmetries:
            symop_lines = "\n".join(
                f"'{format_symmetry_operator(rot, trans)}'" for rot, trans in symmetries
            )
        else:
            symop_lines = f"'{', '.join(var_names)}'"

        with open(filename, "w") as fileout:
            head = (
                f"""
# ======================================================================

# CRYSTAL DATA

# ----------------------------------------------------------------------

{model_name}

_chemical_name_common                  """
                + self.model_label
                + "\n"
            )
            fileout.write(head)
            # bbn = ["a", "b", "c"]
            # bbnang = ["alpha", "beta", "gamma"]

            if len(bravais_vectors) == 1:
                fileout.write(
                    "_cell_length_a \t\t\t"
                    + str(np.linalg.norm(bravais_vectors[0]))
                    + "\n\n\n"
                )
                fileout.write(
                    "loop_\n _space_group_symop_operation_xyz" + "\t\t\t\n" + symop_lines + "\n"
                )
            elif len(bravais_vectors) == 2:
                a_norm = np.linalg.norm(bravais_vectors[0])
                b_norm = np.linalg.norm(bravais_vectors[1])
                gamma = float(
                    np.degrees(
                        np.arccos(
                            bravais_vectors[0].dot(bravais_vectors[1])
                            / (a_norm * b_norm)
                        )
                    )
                )
                fileout.write("_cell_length_a \t\t\t" + str(a_norm) + "\n")
                fileout.write("_cell_length_b \t\t\t" + str(b_norm) + "\n")
                fileout.write("_cell_length_gamma \t\t\t" + str(round(gamma, 6)) + "\n\n")
                fileout.write(
                    "loop_\n _space_group_symop_operation_xyz" + "\n" + symop_lines + "\n\n"
                )

            elif len(bravais_vectors) == 3:
                a_norm = np.linalg.norm(bravais_vectors[0])
                b_norm = np.linalg.norm(bravais_vectors[1])
                c_norm = np.linalg.norm(bravais_vectors[2])
                gamma = float(
                    np.degrees(
                        np.arccos(
                            bravais_vectors[0].dot(bravais_vectors[1])
                            / (a_norm * b_norm)
                        )
                    )
                )
                # alpha: angle between b and c; beta: angle between a and c.
                alpha = float(
                    np.degrees(
                        np.arccos(
                            bravais_vectors[1].dot(bravais_vectors[2])
                            / (b_norm * c_norm)
                        )
                    )
                )
                beta = float(
                    np.degrees(
                        np.arccos(
                            bravais_vectors[0].dot(bravais_vectors[2])
                            / (a_norm * c_norm)
                        )
                    )
                )
                fileout.write("_cell_length_a \t\t\t" + str(a_norm) + "\n")
                fileout.write("_cell_length_b \t\t\t" + str(b_norm) + "\n")
                fileout.write("_cell_length_c \t\t\t" + str(c_norm) + "\n")
                fileout.write("_cell_angle_alpha \t\t\t" + str(round(alpha, 6)) + "\n")
                fileout.write("_cell_angle_beta \t\t\t" + str(round(beta, 6)) + "\n")
                fileout.write("_cell_angle_gamma \t\t\t" + str(round(gamma, 6)) + "\n\n")
                fileout.write(
                    "loop_\n _space_group_symop_operation_xyz" + "\t\t\t\n" + symop_lines + "\n\n"
                )

            fileout.write("# Atom positions \n\n")
            fileout.write(
                "loop_\n   _atom_site_label\n"
                + "_atom_site_occupancy\n"
                + "_atom_site_fract_x\n"
                + "_atom_site_fract_y\n"
                + "_atom_site_fract_z\n"
                + "_atom_site_adp_type\n"
                + "_atom_site_B_iso_or_equiv\n"
                + "_atom_site_spin\n"
                + "_atom_site_g_factor\n"
                + "_atom_site_type_symbol\n"
            )

            coord_atoms = np.asarray(self.site_properties["coord_atomos"], dtype=float)
            if len(coord_atoms) != 0 and len(bravais_vectors) > 0:
                # pinv handles 1D/2D lattices (non-square basis) as well.
                bravais_coords = coord_atoms.dot(np.linalg.pinv(np.array(bravais_vectors)))
                # CIF always expects three fractional coordinates.
                bravais_coords = np.pad(
                    bravais_coords, ((0, 0), (0, 3 - bravais_coords.shape[1]))
                )
            else:
                bravais_coords = np.zeros((len(coord_atoms), 3))

            magnetic_species = self.site_properties["magnetic_species"]
            logger.debug("magnetic species %s", magnetic_species)
            g_lande_factors = self.site_properties["g_lande_factors"]
            spin_repr = self.site_properties["spin_repr"]
            for i, pos in enumerate(bravais_coords):
                line = (
                    f"{magnetic_species[i]}{i+1}\t 1.\t"
                    f"{round(1e5 * pos[0]) * 1e-5}\t"
                    f"{round(1e5 * pos[1]) * 1e-5}\t"
                    f"{round(1e5 * pos[2]) * 1e-5}\t"
                    f"Biso \t 1\t{spin_repr[i]}\t"
                    f"{g_lande_factors[i]}\t"
                    f"{magnetic_species[i]} \n"
                )

                fileout.write(line)
            fileout.write("   \n")

            if len(self.bonds) > 0:
                fileout.write("# Bonds  \n")
                fileout.write("loop_\n")
                fileout.write("_geom_bond_atom_site_label_1\n")
                fileout.write("_geom_bond_atom_site_label_2\n")
                fileout.write("_geom_bond_distance\n")
                fileout.write("_geom_bond_label\n")
                fileout.write("_geom_bond_site_symmetry_2\n")
                for k, bond_name in enumerate(self.bonds):
                    bond_list = self.bonds[bond_name]["bonds"]
                    distance = self.bonds[bond_name]["distance"]
                    logger.debug("%s bond list: %s", k, bond_list)
                    for src, dest, offset in bond_list:
                        line = (
                            f"{magnetic_species[src]}{src+1}\t"
                            f"{magnetic_species[dest]}{dest+1}\t"
                            f"{distance}\t"
                            f"{bond_name}\t"
                            f"{offset}\n"
                        )
                        fileout.write(line)
                fileout.write("   \n")
        return True
