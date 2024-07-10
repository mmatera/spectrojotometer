# coding: utf-8
"""
Magnetic Model class
"""

import logging
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from .tools import box_ellipse, offset_orientation, pack_offset

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
    """Iterator over subcells in the supercell."""

    # Basis vectors must be np.arrays.
    bravais_vectors = [np.array(vector) for vector in bravais_vectors]

    # If no bravais_vectors , the supercell is just the cell.
    if len(bravais_vectors) == 0:
        yield ([], coord_atomos)
        return

    current = [-supercell_size] * len(bravais_vectors)
    max_indx = len(current)
    while True:
        offset = sum(coeff * v_b for coeff, v_b in zip(current, bravais_vectors))
        yield (current, [site + offset for site in coord_atomos])
        # Update the indices of the supercell
        indx = 0
        # Moves the position to the next replica.
        while True:
            if indx == max_indx:
                return
            sub_idx = current[indx] + 1
            if sub_idx <= supercell_size:
                current[indx] = sub_idx
                break
            current[indx] = -supercell_size
            indx = indx + 1


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

    def __init__(
        self,
        atomic_pos,
        bravais_lat,
        bond_lists=None,
        bond_names=None,
        ranges=None,
        supercell_size=1,
        discretization=0.02,
        magnetic_species=None,
        onfly=True,
        model_label="default",
        g_lande_factors=None,
        spin_repr=None,
    ):
        self.model_label = model_label
        self.onfly = onfly

        print("supercell size:", supercell_size)
        supercell_size = min(4, supercell_size)

        print("atomic_pos", atomic_pos)
        if magnetic_species is None:
            magnetic_species = ["X"] * len(atomic_pos)

        print("magnetic_species", magnetic_species)

        if g_lande_factors is None:
            g_lande_factors = [2.0] * len(magnetic_species)
        else:
            for indx, g_str in enumerate(g_lande_factors):
                g_lande_factors[indx] = 2.0 if g_str == "." else float(g_str)

        if spin_repr is None:
            spin_repr = [0.5 for i in magnetic_species]
        else:
            for indx, s_max in enumerate(spin_repr):
                spin_repr[indx] = 0.5 if s_max == "." else float(s_max)

        if ranges is None:
            maxdist = 0
            for p_vec in atomic_pos:
                for q_vec in atomic_pos:
                    dist = np.linalg.norm(p_vec - q_vec)
                    if dist > maxdist:
                        maxdist = dist
            ranges = [[0, dist]]
        elif isinstance(ranges, float):
            ranges = [[0, ranges]]
        elif isinstance(ranges, list) and isinstance(ranges[0], float):
            ranges = [ranges]

        if bond_lists is not None:
            print("recibí", len(bond_lists), "couplings")
            bond_distances = [0] * len(bond_lists)
            if bond_names is None:
                bond_names = ["J" + str(i) for i in range(len(bond_lists))]
        else:
            bond_names = []
            if bond_names is not None:
                bond_names[:] = bond_names
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

        self.bonds = {
            b_name: {"distance": b_distance, "bonds": b_list}
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
            logging.error(msg)
            return

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
                    print([i, j], supercell_idx)
                    rel_pos = pos_j - pos_i
                    orientation = offset_orientation(rel_pos)
                    if orientation < 0:
                        continue
                    distance = np.linalg.norm(rel_pos)
                    print("   distance", distance)
                    if not is_in_range(distance):
                        continue

                    print("bond between ", i, "and ", j, "(", supercell_idx, ")")
                    bound_type = tuple(set([atom_type[i], atom_type[j]]))
                    key = (normalize_key(distance), bound_type)
                    offset_key = pack_offset(supercell_idx)
                    new_bonds.setdefault(key, []).append((i, j % cell_size, offset_key))

        bn_index = 0
        while len(new_bonds) > len(bond_names):
            while "J" + str(bn_index + 1) in bond_names:
                bn_index = bn_index + 1
            print("adding ", "J" + str(bn_index + 1))
            bond_names.append("J" + str(bn_index + 1))

        # Update the bonds dict.
        print("current bonds:", bonds)
        print("new bond names:", bond_names)

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
        logging.info("compute the coefficient matrix")

        rawcm = [
            np.array(
                [
                    -sum((-1) ** sc[b[0]] * (-1) ** sc[b[1]] for b in bondfamily)
                    for sc in configs
                ]
            )
            for bondfamily in (bd["bonds"] for bd in self.bonds.values())
        ]
        print("raw cm\n", "  \n".join(str(row) for row in rawcm))
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
        for s_val in np.random.random_integers(0, 2 ** ((size - 1)), num_confs):
            yield [s_val >> i & 1 for i in range(size - 1, -1, -1)]

    def check_independence(self, conf, setconfs):
        """
        Check the linear independence between the equations
        associated to the configurations.
        """
        if conf == []:
            return False
        if setconfs == []:
            return True
        a_coeffs = self.coefficient_matrix(setconfs, False)
        r0_coeffs = self.coefficient_matrix([conf], False)[0]
        for r_coeffs in a_coeffs:
            if np.linalg.norm(r_coeffs - r0_coeffs) < 0.1:
                return False
        return True

    def add_independent_confs(self, confs, newconfs):
        """
        Produce new independent configurations and add
        them to the list.
        """
        for conf in newconfs:
            if self.check_independence(conf, confs):
                confs.apppend(conf)
        return confs

    def get_independent_confs(self, confs):
        """
        Choose the subset of configurations that produces
        independent equations.
        """
        res = []
        if confs == []:
            return res
        for c in confs:
            if self.check_independence(c, res):
                res.append(c)
        print("  independent confs:", res)
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

    def optimize_independent_set(self, confs, length=None, forced=None):
        """
        Given a set of configurations confs, returns

        (res, cost)

        where res is the subset of configurations that optimizes the
        cost function    sqrt(len(res))/|| coefficient_matrix(res)^+ ||

        If the optional parameter l is provided, then it tries to optimize
        the cost function for a fixed size length.

        """
        partialinfo = False
        cost = np.nan
        if confs == []:
            return [], np.Infinity

        if forced is None:
            forced = []
            lenforced = 0
        else:
            lenforced = len(forced)

        ids0 = [sum(k * 2**i for i, k in enumerate(c)) for c in forced + confs]
        confs = self.get_independent_confs(confs)
        curr = list(forced)
        for c in confs:
            if self.check_independence(c, forced):
                curr.append(c)

        print("curr:")
        for c in curr:
            print(c)
        idscurr = [sum(k * 2**i for i, k in enumerate(c)) for c in curr]
        idscurr = [ids0.index(k) for k in idscurr]

        a = self.coefficient_matrix(curr, False)
        ared = []
        for q in a[:lenforced]:
            ared.append(q)

        print("a=", a)
        u, sv = np.linalg.svd(a, full_matrices=False)[:2]
        k = len(sv)
        while sv[k - 1] < 1.0e-6:
            k = k - 1

        if k == 0:
            logging.info(
                "optimize_independent_set: the set of configurations"
                + " does not provide any information."
            )
            return ([], np.Infinity)

        if k < len(sv):
            logging.info(
                "optimize_independent_set: the set of configurations just"
                + " provides partial information. "
                + "Optimizing the set for this information."
            )
            partialinfo = True
            sv = sv[k - 1]
            u = u[:, :k]
        else:
            sv = sv[-1]
            cost = np.sqrt(len(curr)) / sv

        weights = [(np.linalg.norm(r), i) for i, r in enumerate(u) if i >= lenforced]
        weights = sorted(weights, key=lambda x: -x[0])
        # Sorting the configurations by weights
        curr = forced + [curr[w[1]] for w in weights]
        # subblock of a without forced, sorted by weights
        a = np.array([a[w[1]] for w in weights])

        if length is not None:
            lp = min(length, len(a))
        else:
            lp = k

        # Look for the minimal subset that has the same information than
        # the original of length >= lp.

        for i in range(lp):
            ared.append(a[i])

        while lp < len(curr):
            sv = np.linalg.svd(np.array(ared), full_matrices=False, compute_uv=False)[
                k - 1
            ]
            if sv > 1.0e-6:
                break

            ared.append(a[lp])
            lp = lp + 1
        # If we are dealing with a set of configurations
        # with just partial info,
        # this is the best we can return.
        if partialinfo:
            print(
                "Information is not complete."
                + "Picking the subset that provides the maximal information."
            )
            return (curr[lenforced : lp + lenforced], np.Infinity)
        if length is not None:
            msg = (
                "------------\n"
                "optimize_independent_set for fix length."
                " cost="
                f"{np.sqrt(lp) / sv}"
            )
            logging.info(msg)
            return (
                curr[lenforced : lp + lenforced],
                np.sqrt(lp + lenforced) / sv,
            )

        # If  l was not provided, and the set is informationally complete,
        # try to enlarge it to reduce the cost function
        newcost = np.sqrt(lp + lenforced) / sv
        while lp < len(a) and newcost > cost:
            nr = a[lp]
            sv = np.linalg.svd(
                np.array(ared + [nr]),
                full_matrices=False,
                compute_uv=False,
            )[k - 1]
            newcost = np.sqrt(lp + lenforced + 1) / sv
            ared.append(nr)
            lp = lp + 1

        cost = newcost
        while lp < len(a):
            nr = a[lp]
            sv = np.linalg.svd(
                np.array(ared + [nr]),
                full_matrices=False,
                compute_uv=False,
            )[k - 1]
            newcost = np.sqrt(lp + lenforced + 1) / sv
            if newcost >= cost:
                break

            cost = newcost
            ared.append(nr)
            lp = lp + 1

        return (curr[lenforced : lp + lenforced], cost)

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

        logging.info("--------------find_optimal_set ----------------")

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
            logging.info(
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
        montecarlo=True,
        mcsteps=None,
        mcsizefactor=1.0,
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

        Return values:
        --------------
        js, deltaJ, model_chi

        js: the estimated values of the js, according to the least square
            rule

        deltaJ: a list with the sizes of a box centered at js,
        that contains the compatibility region. This is chosen as the
        bounding box of the ellipsoid
        |A.(j-js)|^2 + (chi^2-len(energies)) = 0.

        model_chi: the difference between the input energies, and
        those evaluated with the model with couplings js.

        """

        if printeqs:
            coeffs = self.coefficient_matrix(confs, normalizar=False)
            logging.info("\n# Configurations:\n=================\n\n")
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
        print(singularvalues)
        cond_number = np.sqrt(len(rcoeffs)) / max(min(singularvalues), 1.0e-9)
        if printeqs:
            msg = "\nInverse of the minimum singular value: " f"{cond_number}\n\n"
            logging.info(msg)

        # If Montecarlo, uses the montecarlo routine to estimate the box.
        if montecarlo:
            return self.montecarlo_box(
                coeffs, energs, err_energs, mcsteps, mcsizefactor
            )
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
        return (js, deltaJ, model_chi, 1.0)

    def bound_inequalities(self, confs, energs, err_energs=0.01):
        """
        Compute the compatibility regions for the coupling constants
        """
        print("confs: ", confs)
        print("energs: ", energs)
        print("err: ", err_energs)
        coeffs = self.coefficient_matrix(confs, normalizar=True)
        u, singularvalues, vh = np.linalg.svd(coeffs)
        s0 = singularvalues[0]
        resolvent = []
        vhr = []
        for i, si in enumerate(singularvalues):
            if si / s0 < 1e-3:
                break
            resolvent.append(u[:, i] / si)
            vhr.append(vh[i])
        vh = np.array(vhr)
        vhr = None
        j0s = np.array(vh).transpose().dot(np.array(resolvent).dot(energs))
        err = (
            len(energs) * err_energs**2
            - np.linalg.norm(energs - coeffs.dot(j0s)) ** 2
        )
        if err < 0:
            return []

        err = np.sqrt(err)
        ineqs = []
        for i, v in enumerate(vh):
            # singular vector associated to the energy must be skipped
            if abs(v[-1] - 1) < 1.0e-9:
                continue
            offset = v.dot(j0s)
            delta = err / singularvalues[i]
            v = v[:-1]
            maxcoeff = max(abs(v))
            ineq = (v / maxcoeff, offset / maxcoeff, delta / maxcoeff)
            ineqs.append(ineq)
        return ineqs

    def montecarlo_box(self, coeffs, energs, tol, numtry=1000, sizefactor=1.0):
        """
        Build an error box by Monte Carlo sampling parameters.
        """
        numcalc = len(energs)
        numparm = len(coeffs[0])
        issingular = False
        # Check that the number of equations is equal or large than the
        # number of parameters to be determined. Otherwise, add trivial
        # equations.
        if numparm > numcalc:
            coeffs = np.array(list(coeffs) + [np.zeros(numparm)] * (numparm - numcalc))
            energs = np.array(list(energs) + [0.0] * (numparm - numcalc))

        # Look for the singular value decomposition
        u, sv, v = la.svd(coeffs, full_matrices=True, compute_uv=True)
        # Check if the system is determined and discard vanishing
        # singular values/vectors.
        k = len(sv)
        svr = np.array(list(sv))
        if svr[-1] < 1.0e-6:
            issingular = True
            while svr[-1] < 1.0e-6:
                svr = svr[:-1]
                k = k - 1

        ur = u[:, :k]
        vr = v[:k]
        # center of the elliptical region
        j0s = vr.transpose().dot((1.0 / svr) * (ur.transpose().dot(energs)))
        e0s = coeffs.dot(j0s)
        esqerr = la.norm(e0s - energs)
        scale = sizefactor * (numcalc * tol**2 - esqerr**2) / np.sqrt(4.0 * numparm)
        if scale <= 0:
            print("scale < 0. Model is incompatible")
            return (
                j0s,
                np.array([-1 for j in j0s]),
                (e0s - energs)[:numcalc] / tol,
                0.0,
            )
        v = v.transpose()
        k = len(sv)
        ur = None
        vr = None
        svr = None

        # Generate the random set on the elliptic upper bound
        jtrys = np.array(
            [rnd.normal(0, 1, numtry) / max(s / scale, 1.0e-9) for s in sv]
        )
        jtrys = v.dot(jtrys).transpose()
        # If the Minumum Square point is compatible,
        # we add it to the random set
        if max(abs(energs - e0s)) < tol:
            jtrys = np.array([j0s + j for j in jtrys] + [j0s])
        else:
            jtrys = np.array([j0s + j for j in jtrys])
        # We keep just the compatible points
        jtrys = jtrys[np.array([max(abs(coeffs.dot(j) - energs)) < tol for j in jtrys])]

        # It it resuls incompatible, show a warning.
        if len(jtrys) == 0:
            print(
                "model seems incompatible. Try with a larger "
                + "number of points or a different sizefactor."
            )
            return (
                j0s,
                np.array([-1 for j in j0s]),
                (e0s - energs)[:numcalc] / tol,
                0.0,
            )

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
        return (
            j0s,
            d_js,
            (e0s - energs)[:numcalc] / tol,
            len(jtrys) / (numtry + 1.0),
        )

    def save_cif(self, filename, bond_names=None):
        """Save the model in a CIF file"""
        bravais_vectors = self.lattice_properties["bravais_vectors"]
        if bond_names is None:
            bond_names = tuple(self.bonds)
        msg = "save cif" + filename
        logging.info(msg)
        model_name = "data_magnetic_model_1"

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
                    "loop_\n _space_group_symop_operation_xyz" + "\t\t\t\n'z'\n"
                )
            elif len(bravais_vectors) == 2:
                a_norm = np.linalg.norm(bravais_vectors[0])
                b_norm = np.linalg.norm(bravais_vectors[1])
                gamma = round(
                    180
                    / 3.1415926
                    * bravais_vectors[0].dot(bravais_vectors[1])
                    / (a_norm * b_norm)
                )
                fileout.write("_cell_length_a \t\t\t" + str(a_norm) + "\n")
                fileout.write("_cell_length_b \t\t\t" + str(b_norm) + "\n")
                fileout.write("_cell_length_gamma \t\t\t" + str(gamma) + "\n\n")
                fileout.write(
                    "loop_\n _space_group_symop_operation_xyz" + "\n'x, y'\n\n"
                )

            elif len(bravais_vectors) == 3:
                a_norm = np.linalg.norm(bravais_vectors[0])
                b_norm = np.linalg.norm(bravais_vectors[1])
                c_norm = np.linalg.norm(bravais_vectors[2])
                gamma = round(
                    180
                    / 3.1415926
                    * np.arccos(
                        bravais_vectors[0].dot(bravais_vectors[1]) / (a_norm * b_norm)
                    )
                )
                alpha = round(
                    180
                    / 3.1415926
                    * np.arccos(
                        bravais_vectors[0].dot(bravais_vectors[2]) / (a_norm * c_norm)
                    )
                )
                beta = round(
                    180
                    / 3.1415926
                    * np.arccos(
                        bravais_vectors[1].dot(bravais_vectors[2]) / (c_norm * b_norm)
                    )
                )
                fileout.write("_cell_length_a \t\t\t" + str(a_norm) + "\n")
                fileout.write("_cell_length_b \t\t\t" + str(b_norm) + "\n")
                fileout.write("_cell_length_c \t\t\t" + str(c_norm) + "\n")
                fileout.write("_cell_angle_alpha \t\t\t" + str(alpha) + "\n")
                fileout.write("_cell_angle_beta \t\t\t" + str(beta) + "\n")
                fileout.write("_cell_angle_gamma \t\t\t" + str(gamma) + "\n\n")
                fileout.write(
                    "loop_\n _space_group_symop_operation_xyz" + "\t\t\t\n'x, y, z'\n\n"
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

            bravais_coords = self.site_properties["coord_atomos"].dot(
                np.linalg.inv(np.array(bravais_vectors))
            )

            magnetic_species = self.site_properties["magnetic_species"]
            print("magnetic species", magnetic_species)
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
                    print(k, " bondlist: ", bond_list)
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
