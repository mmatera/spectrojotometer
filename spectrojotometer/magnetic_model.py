# coding: utf-8


from __future__ import print_function
import sys

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from .tools import *
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3
# from matplotlib.lines import Line2D

# ******************************************************************************


class MagneticModel:
    def __init__(self, atomic_pos,
                 bravais_lat,
                 bond_lists=None,
                 bond_names=None,
                 ranges=None,
                 supercell_size=2,
                 discretization=0.02,
                 magnetic_species=None,
                 onfly=True, model_label="default",
                 g_lande_factors=None,
                 spin_repr=None):
        self.model_label = model_label
        self.cell_size = len(atomic_pos)
        self.supercell_size = supercell_size
        self.coord_atomos = atomic_pos
        self.bravais_vectors = bravais_lat
        self.onfly = onfly
        self.g_lande_factors = g_lande_factors
        self.spin_repr = spin_repr

        if magnetic_species is None:
            self.magnetic_species = ["X" for i in atomic_pos]
        else:
            self.magnetic_species = []
            self.magnetic_species[:] = magnetic_species

        if g_lande_factors is None:
            self.g_lande_factors = [2. for i in self.magnetic_species]
        else:
            for i, g in enumerate(self.g_lande_factors):
                if g == ".":
                    self.g_lande_factors[i] = 2.

        if spin_repr is None:
            self.spin_repr = [.5 for i in self.magnetic_species]
        else:
            for i, g in enumerate(self.spin_repr):
                if g == ".":
                    self.spin_repr[i] = .5

        if len(bravais_lat) == 1:
            self.supercell = np.array([[site[idx] +
                                        i * self.bravais_vectors[0][idx]
                                        for idx in range(3)]
                                       for i in range(-supercell_size,
                                                      supercell_size + 1)
                                       for site in self.coord_atomos])

        elif len(bravais_lat) == 2:
            self.supercell = np.array([[site[idx] + i *
                                        self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx]
                                        for idx in range(3)]
                                       for i in range(-supercell_size,
                                                      supercell_size + 1)
                                       for j in range(-supercell_size,
                                                      supercell_size + 1)
                                       for site in self.coord_atomos])
        elif len(bravais_lat) == 3:
            self.supercell = np.array([[site[idx] + i *
                                        self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx]
                                        + k * self.bravais_vectors[2][idx]
                                        for idx in range(3)]
                                       for i in range(-supercell_size,
                                                      supercell_size + 1)
                                       for j in range(-supercell_size,
                                                      supercell_size + 1)
                                       for k in range(-supercell_size,
                                                      supercell_size + 1)
                                       for site in self.coord_atomos])
        else:
            self.supercell = self.coord_atomos

        if ranges is None:
            maxdist = 0
            for p in self.coord_atomos:
                for q in self.coord_atomos:
                    d = np.linalg.norm(p - q)
                    if d > maxdist:
                        maxdist = d
            ranges = [[0, d]]
        elif type(ranges) is float:
            ranges = [[0, ranges]]
        elif type(ranges) is list and type(ranges[0]) is float:
            ranges = [ranges]

        if bond_lists is not None:
            self.bond_lists = bond_lists
            bond_distances = [0 for i in bond_lists]
            if bond_names is not None:
                self.bond_names = bond_names
            else:
                self.bond_names = ["J" + str(i)
                                   for i in range(len(self.bond_lists))]
        else:
            self.bond_names = []
            if bond_names is not None:
                self.bond_names[:] = bond_names
            self.bond_lists = []
            self.bond_distances = []
            self.discretization = discretization
            self.generate_bonds(discretization, ranges)

# Handling bonds
    def remover_bond(self, idx):
        """
        Remove the idx-esim bond
        """
        self.bond_distances.pop(idx)
        self.bond_names.pop(idx)
        self.bond_lists.pop(idx)

    def remover_bond_by_name(self, name):
        """
        Remove the bond with name <name>
        """
        try:
            idx = self.bond_names.index(name)
        except ValueError:
            eprint("bond " + name + " not found")
            return
        self.remover_bond(idx)

    def generate_bonds(self, discretization, ranges, bond_names=None):
        """
        En base a la celda unidad, la red de Bravais y el tamaño de la
        "super-celda" (cuantas copias de la celda unidad se consideran
        como entorno) calcula los bonds,
        esto es, los pares de sitios magnéticos que interactúan según cada
        una de las constantes de acoplamiento a determinar.
        """
        cell_size = self.cell_size
        coord_atomos = self.coord_atomos
        bravais_vectors = self.bravais_vectors
        supercell = self.supercell
        old_bond_lists = self.bond_lists
        old_bond_distances = self.bond_distances
        old_bond_names = self.bond_names

        bond_lists = []
        bond_distances = []
        bond_type = []
        bond_names = []
        atom_type = self.magnetic_species

        def is_in_range(val):
            for r in ranges:
                if r[0] < val < r[1]:
                    return True
            return False

        for d, bt in sorted([(np.linalg.norm(q-p),
                              set([atom_type[i], atom_type[j]]))
                             for i, q in enumerate(self.coord_atomos)
                             for j, p in enumerate(self.coord_atomos)]):
            dr = round(d/discretization) * discretization
            if not is_in_range(dr):
                continue
            if dr != 0 and (dr not in bond_distances or bt not in bond_type):
                bond_distances.append(dr)
                bond_lists.append([])
                bond_type.append(bt)

        # supercell=np.array([p for p in supercell if p[2]>0])
        for p, x in enumerate(coord_atomos):
            for q, y in enumerate(self.supercell):
                qred = q % cell_size
                offset = qred - q
                bravais_lat = self.bravais_vectors
                if len(bravais_lat) == 1:
                    offset = np.array([offset + 5, 5, 5])
                elif len(bravais_lat) == 2:
                    lsc = 2 * self.supercell_size + 1
                    offset = np.array([offset % lsc + 5, int(offset / lsc)
                                       + 5, 5])
                elif len(bravais_lat) == 3:
                    lsc = 2 * self.supercell_size + 1
                    offset = np.array([offset % lsc + 5,
                                       int(offset / lsc) % lsc + 5,
                                       5 + int(offset / lsc**2)])
                else:
                    offset = np.array([5, 5, 5])

                if offset[0] == offset[1] == offset[2] == 5:
                    offset = "."
                elif ((0 <= offset[0] <= 9) and (0 <= offset[1] <= 9) and
                      (0 <= offset[2] <= 9)):
                    offset = str(offset[0]) + str(offset[1]) + str(offset[2])
                else:
                    offset = "."
                if(p < qred):
                    d = x - y
                    d = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
                    if not is_in_range(d):
                        continue
                    bt = set([atom_type[p], atom_type[qred]])
                    for i in range(len(bond_distances)):
                        if np.abs(d - bond_distances[i]) < \
                           discretization and bt == bond_type[i]:
                            bond_lists[i].append((p, qred, offset))

        self.bond_lists = old_bond_lists + bond_lists
        self.bond_distances = old_bond_distances + bond_distances

        nnames = len(self.bond_names)
        while len(self.bond_names) < len(self.bond_lists):
            while "J" + str(nnames + 1) in self.bond_names:
                nnames = nnames + 1
            self.bond_names.append("J" + str(nnames + 1))
        return

    def formatted_equations(self, cm, ensname=None,
                            comments=None, format="plain"):
        res = ""
        if format == "latex":
            times_symbol = ""
        else:
            times_symbol = "*"
        if format == "plain" or format == "latex":
            equal_symbol = "="
        else:
            equal_symbol = "=="

        if format == "plain":
            open_comment = "# "
            close_comment = ""
        elif format == "latex":
            open_comment = "% "
            close_comment = ""
        elif format == "wolfram":
            open_comment = "(*"
            close_comment = "*)"
        else:
            open_comment = "# "
            close_comment = ""

        if format == "latex":
            sub_symb = "_"
        else:
            sub_symb = ""

        jsname = []
        jsname[:] = self.bond_names
        jsname.append("E" + sub_symb + "0")
        if ensname is None:
            ensname = ["E" + sub_symb + str(i+1) for i in range(len(cm))]
        for i, row in enumerate(cm):
            eq = ""
            for k, c in enumerate(row):
                cr = round(c * 100)/100.
                if c > 0:
                    if eq != "":
                        eq = eq + " + "
                    else:
                        eq = "  "
                    if cr != 1:
                        eq = eq + str(cr) + " " + times_symbol
                    eq = eq + " " + jsname[k]
                elif c < 0:
                    if eq != "":
                        eq = eq + " "
                    eq = eq + "- "
                    if cr != -1:
                        eq = eq + str(-cr) + " " + times_symbol
                    eq = eq + " " + jsname[k]
            if comments is not None:
                res = (res + eq + equal_symbol + ensname[i] +
                       "  " + open_comment + comments[i] +
                       close_comment + "\n\n")
            else:
                res = res + eq + equal_symbol + ensname[i] + "\n\n"
        return res

    def print_equations(self, cm, ensname=None,
                        comments=None, format="plain"):
        print("\n\n# Equations: \n============\n\n")
        print(self.formatted_eqnations(self, cm,
                                       ensname, comments, format))

    def coefficient_matrix(self, configs, normalizar=False):
        """
        Return the matrix defining the equation system
        connecting coupling constants with simulated energies
        for a given set of configurations configs.
        """
        rawcm = [np.array([-sum([(-1)**sc[b[0]] * (-1)**sc[b[1]]
                                 for b in bondfamily])
                           for sc in configs])
                 for bondfamily in self.bond_lists]
        if normalizar:
            cm = [v - np.average(v) for v in rawcm]
        else:
            cm = rawcm

        cm = np.array(cm +
                      # Energía no magnética
                      [np.array([1. for sc in configs])]).transpose()
        return cm

    def generate_configurations_onfly(self):
        size = self.cell_size
        for c in range(2**((size-1))):
            yield [c >> i & 1 for i in range(size-1, -1, -1)]

    def generate_random_configurations(self, t=10):
        size = self.cell_size
        for c in np.random.random_integers(0, 2**((size-1)), t):
            yield [c >> i & 1 for i in range(size-1, -1, -1)]

    def check_independence(self, conf, setconfs):
        if conf == []:
            return False
        if setconfs == []:
            return True
        a = self.coefficient_matrix(setconfs, False)
        r0 = self.coefficient_matrix([conf], False)[0]
        for r in a:
            if np.linalg.norm(r-r0) < .1:
                return False
        return True

    def add_independent_confs(self, confs, newconfs):
        for c in newconfs:
            if self.check_independence(c, confs):
                confs.apppend(c)
        return confs

    def get_independent_confs(self, confs):
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
        sv = np.linalg.svd(a, full_matrices=False,
                           compute_uv=False)[-1]
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

        ids0 = [sum([k*2**i for i, k in enumerate(c)]) for c in forced + confs]
        confs = self.get_independent_confs(confs)
        curr = [f for f in forced]
        for c in confs:
            if self.check_independence(c, forced):
                curr.append(c)

        print("curr:")
        for c in curr:
            print(c)
        idscurr = [sum([k*2**i for i, k in enumerate(c)]) for c in curr]
        idscurr = [ids0.index(k) for k in idscurr]

        a = self.coefficient_matrix(curr, False)
        ared = []
        for q in a[:lenforced]:
            ared.append(q)

        print("a=", a)
        u, sv, v = np.linalg.svd(a, full_matrices=False)
        v = None
        k = len(sv)
        while sv[k-1] < 1.e-6:
            k = k - 1

        if k == 0:
            eprint("optimize_independent_set: the set of configurations" +
                   " does not provide any information.")
            return ([], np.Infinity)

        if k < len(sv):
            eprint("optimize_independent_set: the set of configurations just" +
                   " provides partial information. " +
                   "Optimizing the set for this information.")
            partialinfo = True
            sv = sv[k - 1]
            u = u[:, :k]
        else:
            sv = sv[-1]
            cost = np.sqrt(len(curr))/sv

        weights = [(np.linalg.norm(r), i) for i, r in enumerate(u)
                   if i >= lenforced]
        weights = sorted(weights, key=lambda x: -x[0])
        print("idscurr=", idscurr)
        print("weights: ")
        for w in weights:
            print(w)
            # print(idscurr[w[1]], "->", w[0])
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
            sv = np.linalg.svd(np.array(ared),
                               full_matrices=False,
                               compute_uv=False)[k-1]
            eprint("------------------------------------------------", sv)
            if sv > 1.e-6:
                break
            else:
                ared.append(a[lp])
                lp = lp + 1
        # If we are dealing with a set of configurations
        # with just partial info,
        # this is the best we can return.
        if partialinfo:
            print("Information is not complete." +
                  "Picking the subset that provides the maximal information.")
            return (curr[lenforced:lp+lenforced], np.Infinity)
        elif length is not None:
            eprint("------------ optimize_independent_set for fix length." +
                   " cost=", np.sqrt(lp) / sv)
            return (curr[lenforced:lp+lenforced], np.sqrt(lp+lenforced) / sv)
        else:
            # If  l was not provided, and the set is informationally complete,
            # try to enlarge it to reduce the cost function
            newcost = np.sqrt(lp + lenforced) / sv
            while lp < len(a) and newcost > cost:
                nr = a[lp]
                sv = np.linalg.svd(np.array(ared + [nr]),
                                   full_matrices=False,
                                   compute_uv=False)[k-1]
                newcost = np.sqrt(lp + lenforced + 1) / sv
                ared.append(nr)
                lp = lp + 1

            cost = newcost
            while lp < len(a):
                nr = a[lp]
                sv = np.linalg.svd(np.array(ared + [nr]),
                                   full_matrices=False,
                                   compute_uv=False)[k-1]
                newcost = np.sqrt(lp + lenforced + 1)/sv
                if newcost >= cost:
                    break
                else:
                    cost = newcost
                    ared.append(nr)
                    lp = lp + 1

            return (curr[lenforced:lp+lenforced], cost)

    def find_optimal_configurations(self, start=None, num_new_confs=None,
                                    known=None, its=100, update_size=1):

        if num_new_confs is None:
            lmax = max(len(self.bond_lists) + 1, 1) + update_size
        else:
            lmax = max(num_new_confs, len(self.bond_lists) + 1, 1)

        if known is None:
            known = []

        eprint("--------------find_optimal_set ----------------")

        if start is None:
            repres = self.generate_random_configurations(
                max(update_size, lmax))
            last_better = self.get_independent_confs([q for q in repres])
        else:
            last_better = self.get_independent_confs(start)

        for i in range(its):
            repres = self.generate_random_configurations(update_size)
            last_better = self.get_independent_confs(last_better +
                                                     [q for q in repres])
            last_better = [c for c in last_better
                           if self.check_independence(c, known)]
            if len(last_better) >= lmax:
                break

        if len(last_better) < lmax:
            eprint("not enough configurations. Trying to optimize " +
                   "the subset and return it.")
            return self.optimize_independent_set(last_better,
                                                 num_new_confs,
                                                 forced=known)

        last_better, cost = self.optimize_independent_set(last_better,
                                                          num_new_confs,
                                                          forced=known)
        for i in range(its):
            repres = self.generate_random_configurations(update_size)
            new_try = self.get_independent_confs(last_better +
                                                 [q for q in repres])
            new_try, newcost = self.optimize_independent_set(new_try,
                                                             num_new_confs,
                                                             forced=known)
            if newcost < cost:
                cost = newcost
                last_better = new_try

        return last_better, cost

    def compute_couplings(self, confs, energs, err_energs=.01,
                          printeqs=False, montecarlo=True,
                          mcsteps=None, mcsizefactor=1.):
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
            eprint("\n# Configurations:\n=================\n\n")
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
        cond_number = np.sqrt(len(rcoeffs)) / max(min(singularvalues), 1.e-9)
        if printeqs:
            eprint("\nInverse of the minimum singular value: ",
                   cond_number, "\n\n")

        # If Montecarlo, uses the montecarlo routine to estimate the box.
        if montecarlo:
            return self.montecarlo_box(coeffs, energs, err_energs,
                                       mcsteps, mcsizefactor)
        else:
            # Otherwise, it gives an estimation in terms of the bigger ellipse.
            # js = (A^t A )^{-1} A^t En, i.e. least squares solution
            resolvent = np.linalg.pinv(coeffs.transpose().dot(coeffs),
                                       rcond=1.e-6).dot(coeffs.transpose())
            js = resolvent.dot(energs)
            model_chi = (coeffs.dot(js)-energs)/err_energs
            rr = (len(model_chi) - sum(model_chi**2))
            if rr < 0:
                deltaJ = [-1 for i in js]
            else:
                rr = np.sqrt(rr) * err_energs
                deltaJ = box_ellipse(coeffs, rr)
            return (js, deltaJ, model_chi, 1.)

    def bound_inequalities(self, confs, energs, err_energs=.01):
        print("confs: ", confs)
        print("energs: ", energs)
        print("err: ", err_energs)
        coeffs = self.coefficient_matrix(confs, normalizar=True)
        u, singularvalues, vh = np.linalg.svd(coeffs)
        s0 = singularvalues[0]
        resolvent = []
        vhr = []
        for i in range(len(singularvalues)):
            si = singularvalues[i]
            if si/s0 < 1e-3:
                break
            resolvent.append(u[:, i]/si)
            vhr.append(vh[i])
        vh = np.array(vhr)
        vhr = None
        j0s = np.array(vh).transpose().dot(np.array(resolvent).dot(energs))
        err = len(energs) * err_energs**2 - \
            np.linalg.norm(energs-coeffs.dot(j0s))**2
        if err < 0:
            return []

        err = np.sqrt(err)
        ineqs = []
        for i, v in enumerate(vh):
            # singular vector associated to the energy must be skipped
            if abs(v[-1] - 1) < 1.e-9:
                continue
            offset = v.dot(j0s)
            delta = err/singularvalues[i]
            v = v[:-1]
            maxcoeff = max(abs(v))
            ineq = (v/maxcoeff, offset/maxcoeff, delta/maxcoeff)
            ineqs.append(ineq)
        return ineqs

    def montecarlo_box(self, coeffs, energs, tol, numtry=1000, sizefactor=1.):
        numcalc = len(energs)
        numparm = len(coeffs[0])
        issingular = False
        # Check that the number of equations is equal or large than the
        # number of parameters to be determined. Otherwise, add trivial
        # equations.
        if(numparm > numcalc):
            coeffs = np.array([c for c in coeffs] +
                              [np.zeros(numparm)
                               for i in range(numparm - numcalc)])
            energs = np.array([c for c in energs] +
                              [0.
                               for i in range(numparm - numcalc)])

        # Look for the singular value decomposition
        u, sv, v = la.svd(coeffs, full_matrices=True, compute_uv=True)
        # Check if the system is determined and discard vanishing
        # singular values/vectors.
        k = len(sv)
        svr = np.array([s for s in sv])
        if svr[-1] < 1.e-6:
            issingular = True
            while svr[-1] < 1.e-6:
                svr = svr[:-1]
                k = k - 1

        ur = u[:, :k]
        vr = v[:k]
        # center of the elliptical region
        j0s = vr.transpose().dot((1./svr)*(ur.transpose().dot(energs)))
        e0s = coeffs.dot(j0s)
        esqerr = la.norm(e0s - energs)
        scale = sizefactor * (numcalc * tol**2 - esqerr**2)/np.sqrt(4.*numparm)
        if scale <= 0:
            print("scale < 0. Model is incompatible")
            return (j0s, np.array([-1 for j in j0s]),
                    (e0s-energs)[:numcalc]/tol, 0.)
        v = v.transpose()
        k = len(sv)
        ur = None
        vr = None
        svr = None

        # Generate the random set on the elliptic upper bound
        jtrys = np.array([rnd.normal(0, 1, numtry)/max(s/scale, 1.e-9)
                          for s in sv])
        jtrys = v.dot(jtrys).transpose()
        # If the Minumum Square point is compatible,
        # we add it to the random set
        if max(abs(energs - e0s)) < tol:
            jtrys = np.array([j0s + j for j in jtrys] + [j0s])
        else:
            jtrys = np.array([j0s + j for j in jtrys])
        # We keep just the compatible points
        jtrys = jtrys[np.array([max(abs(coeffs.dot(j)-energs)) < tol
                                for j in jtrys])]

        # It it resuls incompatible, show a warning.
        if len(jtrys) == 0:
            print("model seems incompatible. Try with a larger " +
                  "number of points or a different sizefactor.")
            return (j0s, np.array([-1 for j in j0s]),
                    (e0s-energs)[:numcalc]/tol, 0.)

        if len(jtrys) == 1:
            print("errors  estimated by boxing the  ellipse")
            rr = np.sqrt(numcalc*tol**2-esqerr**2)
            djs = box_ellipse(coeffs, rr)
            return j0s, djs, (e0s-energs)/tol, 1./(numtry+1.)
        print(len(jtrys))
        print("Accepted rate=", len(jtrys)/(numtry+1.))

        j0s = np.average(jtrys, 0)
        jtrys = [abs(js-j0s) for js in jtrys]
        djs = np.max(np.array(jtrys), 0)
        # If the model is singular, discard the coefficients that
        # couldn't be determined, setting them to 0.
        if issingular:
            for i, dj in enumerate(djs):
                if dj > 100:
                    djs[i] = np.nan
                    j0s[i] = 0

        e0s = coeffs.dot(j0s)
        return j0s, djs, (e0s-energs)[:numcalc] / tol, len(jtrys)/(numtry+1.)

    def save_cif(self, filename, bond_names=None):
        bravais_vectors = self.bravais_vectors

        with open(filename, "w") as fileout:
            head = """
# ======================================================================

# CRYSTAL DATA

# ----------------------------------------------------------------------

data_magnetic_model_1

_chemical_name_common                  """ + self.model_label + "\n"
            fileout.write(head)
            bbn = ["a", "b", "c"]
            bbnang = ["alpha", "beta", "gamma"]

            if len(bravais_vectors) == 1:
                fileout.write("_cell_length_a \t\t\t" +
                              str(np.linalg.norm(bravais_vectors[0]))+"\n\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz" +
                              "\t\t\t\n\'z\'\n")
            elif len(bravais_vectors) == 2:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                gamma = round(180 / 3.1415926 *
                              bravais_vectors[0].dot(bravais_vectors[1])/(a*b))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz" +
                              "\n\'x, y\'\n\n")

            elif len(bravais_vectors) == 3:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                c = np.linalg.norm(bravais_vectors[2])
                gamma = round(180 / 3.1415926 *
                              np.arccos(bravais_vectors[0].dot(
                                  bravais_vectors[1])/(a*b)))
                alpha = round(180 / 3.1415926 *
                              np.arccos(bravais_vectors[0].dot(
                                  bravais_vectors[2])/(a*c)))
                beta = round(180 / 3.1415926 *
                             np.arccos(bravais_vectors[1].dot(
                                 bravais_vectors[2])/(c * b)))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_c \t\t\t" + str(c)+"\n")
                fileout.write("_cell_angle_alpha \t\t\t" + str(alpha)+"\n")
                fileout.write("_cell_angle_beta \t\t\t" + str(beta)+"\n")
                fileout.write("_cell_angle_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz" +
                              "\t\t\t\n\'x, y, z\'\n\n")

            fileout.write("# Atom positions \n\n")
            fileout.write("loop_\n   _atom_site_label\n" +
                          "_atom_site_occupancy\n" +
                          "_atom_site_fract_x\n" +
                          "_atom_site_fract_y\n" +
                          "_atom_site_fract_z\n" +
                          "_atom_site_adp_type\n" +
                          "_atom_site_B_iso_or_equiv\n" +
                          "_atom_site_spin\n" +
                          "_atom_site_g_factor\n" +
                          "_atom_site_type_symbol\n")

            bravaiscoords = self.coord_atomos.dot(np.linalg.inv(
                                                    np.array(bravais_vectors)))

            for i, pos in enumerate(bravaiscoords):
                fileout.write(self.magnetic_species[i] + str(i+1) +
                              "\t 1.\t" +
                              str(round(1E5*pos[0])*1E-5) + "\t" +
                              str(round(1E5*pos[1])*1E-5) + "\t" +
                              str(round(1E5*pos[2])*1E-5) + "\t" +
                              "Biso \t" +
                              "1 \t" +
                              str(self.spin_repr[i]) + "\t" +
                              str(self.g_lande_factors[i]) + "\t" +
                              self.magnetic_species[i] + " \n")
            fileout.write("   \n")

            if len(self.bond_lists) > 0:
                fileout.write("# Bonds  \n")
                fileout.write("loop_\n")
                fileout.write("_geom_bond_atom_site_label_1\n")
                fileout.write("_geom_bond_atom_site_label_2\n")
                fileout.write("_geom_bond_distance\n")
                fileout.write("_geom_bond_label\n")
                fileout.write("_geom_bond_site_symmetry_2\n")
                for k, bl in enumerate(self.bond_names):
                    print(k, " bondlist: ", self.bond_lists[k])
                    for a, b, c in self.bond_lists[k]:
                        fileout.write(self.magnetic_species[a] + str(a+1) +
                                      "\t" +
                                      self.magnetic_species[b] + str(b+1) +
                                      "\t" + str(self.bond_distances[k]) +
                                      "\t" + str(bl) + "\t" + c + "\n")
                fileout.write("   \n")
        return True
