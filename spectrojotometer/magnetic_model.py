"""
Magnetic Model class
"""
import logging
import warnings

import numpy as np
import numpy.linalg as la

from .tools import (
    box_ellipse,
    compatibility_box,
    format_symmetry_operator,
    offset_orientation,
    pack_offset,
)

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


def supercell_iterator(supercell_size, atomic_positions, bravais_vectors):
    """Iterator over subcells in the supercell.

    supercell_size can be:
      - an int  : same size along all axes (backward-compatible)
      - a list  : [sx, sy, sz] — independent size per Bravais axis
    """

    # Basis vectors must be np.arrays.
    bravais_vectors = [np.array(vector) for vector in bravais_vectors]

    # If no bravais_vectors, the supercell is just the cell.
    if len(bravais_vectors) == 0:
        yield ([], atomic_positions)
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
        yield (current[:], [site + offset for site in atomic_positions])
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


def build_supercell(supercell_size, atomic_positions, bravais_vectors):
    """Build a list of offsets for the supercell"""

    return np.array(
        [
            cell[1]
            for cell in supercell_iterator(
                supercell_size, atomic_positions, bravais_vectors
            )
        ]
    )



def _position_site_map(source, target, tol):
    """site_map by comparing positions: both models in the same frame."""
    src_pos = np.asarray(source.site_properties["coord_atomos"], dtype=float)
    tgt_pos = np.asarray(target.site_properties["coord_atomos"], dtype=float)
    bravais = np.asarray(source.lattice_properties["bravais_vectors"], dtype=float)
    result = []
    for point in tgt_pos:
        best, best_dist = -1, tol
        for index, site in enumerate(src_pos):
            diff = point - site
            if bravais.size:
                # Closest lattice vector: integer combination of the
                # Bravais vectors (least squares, then rounded).
                coeffs = np.linalg.lstsq(bravais.T, diff, rcond=None)[0]
                diff = diff - bravais.T @ np.round(coeffs)
            dist = np.linalg.norm(diff)
            if dist < best_dist:
                best, best_dist = index, dist
        result.append(best)
    return result


def _to_pymatgen_structure(model):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice(np.asarray(model.lattice_properties["bravais_vectors"], dtype=float)),
        list(model.site_properties["magnetic_species"]),
        np.asarray(model.site_properties["coord_atomos"], dtype=float),
        coords_are_cartesian=True,
    )


def _structure_site_maps(source, target):
    """
    (forward, reverse) from a single alignment found by pymatgen's
    StructureMatcher, which allows a different origin, orientation or
    choice of cell (and a supercell of either model):

    forward: for each target site, a source site at the same position;
    reverse: for each source site, a target site at the same position.
    Both are all -1 if the structures do not match.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    s_src, s_tgt = _to_pymatgen_structure(source), _to_pymatgen_structure(target)
    n_src, n_tgt = len(s_src), len(s_tgt)
    matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=True)
    # The matcher enlarges the second structure into a supercell of the
    # first one; the images of each site are consecutive in that supercell.
    try:
        if n_tgt >= n_src:
            found = matcher.get_transformation(s_tgt, s_src)
        else:
            found = matcher.get_transformation(s_src, s_tgt)
    except ValueError:
        found = None
    if found is None:
        return [-1] * n_tgt, [-1] * n_src
    supercell, _, mapping = found
    images = int(round(abs(np.linalg.det(supercell))))
    if n_tgt >= n_src:
        forward = [-1 if m is None else int(m) // images for m in mapping[:n_tgt]]
        reverse = [-1] * n_src
        for t, s in enumerate(forward):
            if s >= 0 and reverse[s] < 0:
                reverse[s] = t
    else:
        reverse = [-1 if m is None else int(m) // images for m in mapping[:n_src]]
        forward = [-1] * n_tgt
        for s, t in enumerate(reverse):
            if t >= 0 and forward[t] < 0:
                forward[t] = s
    return forward, reverse


def site_maps(source, target, tol=0.1, method="auto"):
    """
    (forward, reverse, method_used), from a single alignment:
    forward[t] is the source site of each target site (as site_map),
    reverse[s] a target site of each source site, and method_used the
    method actually used ("position" or "structure"; see site_map).
    """
    if method not in ("auto", "position", "structure"):
        raise ValueError(f"unknown method {method!r}")
    if method in ("auto", "position"):
        forward = _position_site_map(source, target, tol)
        if method == "position" or min(forward, default=0) >= 0:
            return forward, _position_site_map(target, source, tol), "position"
    matched, reverse = _structure_site_maps(source, target)
    if method == "auto" and min(matched, default=0) < 0:
        # No complete match either: the partial map by positions tells
        # which sites are missing.
        return forward, _position_site_map(target, source, tol), "position"
    return matched, reverse, "structure"


def site_map(source, target, tol=0.1, method="auto"):
    """
    For each site of the target model, the index of the site of the source
    model at the same position, up to a translation of the source lattice;
    -1 if there is none.

    method:
        "position": compare the positions (Cartesian coordinates), within
            tol: both models must be described in the same frame (same
            origin and orientation of the axes).
        "structure": align the two structures with pymatgen's
            StructureMatcher, which allows a different origin,
            orientation or choice of cell. Among alignments related by a
            symmetry of the structures, any one may be chosen.
        "auto" (default): "position" if it finds every site, otherwise
            "structure" if it does; else the partial map by positions.
    """
    return site_maps(source, target, tol, method)[0]


def compatible_configurations(source, confs, target, tol=0.1, method="auto"):
    """
    For each configuration of the source model, whether the target model
    can represent it: whether the configuration is periodic with the
    lattice of the target.

    When the cell of the target is a supercell of the source one, every
    configuration is. When the target cell is smaller (e.g. the source is
    a supercell of it), only the configurations that repeat with the
    lattice of the target are: in the others, two source sites that are
    the same site of the target have different spins, and no
    configuration of the target describes them (its energy, scaled by the
    ratio of cell sizes, would be the energy of a different
    configuration).

    Sites of the source without counterpart in the target are not checked.
    method: as in site_map.
    """
    # For each target site, its source site; and, for each source site, a
    # target site at the same position (modulo the target lattice). Both
    # from the same alignment.
    forward, reverse, _ = site_maps(source, target, tol, method)
    result = []
    for conf in confs:
        ok = True
        for src, tgt in enumerate(reverse):
            if tgt < 0 or forward[tgt] < 0:
                continue
            if conf[src] != conf[forward[tgt]]:
                ok = False
                break
        result.append(ok)
    return result


def map_configurations(
    source, confs, target, tol=0.1, fill=None, mapping=None, method="auto"
):
    """
    Translate spin configurations of the source model into configurations
    of the target model, using site_map: each site of the target takes the
    spin of the source site at the same position.

    mapping: the site map to use, one index of a source site per site of
    the target (by default, site_map(source, target, tol, method)).

    Raises ValueError if some site of the target has no counterpart in the
    source, unless fill (0 or 1) is given: then those sites take that
    value. Also if a configuration does not have one value per site of the
    source model, or if the target cannot represent it (see
    compatible_configurations).
    """
    if mapping is None:
        mapping = site_map(source, target, tol, method)
    else:
        mapping = [int(j) for j in mapping]
    num_sites = len(source.site_properties["coord_atomos"])
    num_target = len(target.site_properties["coord_atomos"])
    if len(mapping) != num_target:
        raise ValueError(
            f"the site map has {len(mapping)} entries; the target model has "
            f"{num_target} sites"
        )
    if any(j >= num_sites for j in mapping):
        raise ValueError(
            f"the site map refers to sites beyond the {num_sites} of the "
            "source model"
        )
    missing = [i for i, j in enumerate(mapping) if j < 0]
    if missing and fill is None:
        raise ValueError(
            f"sites {missing} of the target model have no counterpart in the "
            f"source model (tolerance {tol})"
        )
    for k, conf in enumerate(confs):
        if len(conf) != num_sites:
            raise ValueError(
                f"configuration {conf} has {len(conf)} values; the source "
                f"model has {num_sites} sites"
            )
    incompatible = [
        k
        for k, ok in enumerate(
            compatible_configurations(source, confs, target, tol, method)
        )
        if not ok
    ]
    if incompatible:
        raise ValueError(
            f"configurations {incompatible} are not periodic with the lattice "
            "of the target model, which cannot represent them (see "
            "compatible_configurations)"
        )
    return [[conf[j] if j >= 0 else fill for j in mapping] for conf in confs]

def _clamp_supercell_size(supercell_size):
    """An int, or one value per axis; each clamped to 1..4."""
    logger.debug("supercell size: %s", supercell_size)
    if np.isscalar(supercell_size):
        return min(4, max(1, int(supercell_size)))
    return [min(4, max(1, int(size))) for size in supercell_size]


def _per_site_values(values, num_sites, default):
    """
    A new list of floats with one value per site. ``None`` gives the
    default for every site; a "." entry (CIF for "not given") gives the
    default for that site.
    """
    if values is None:
        return [default] * num_sites
    return [default if value == "." else float(value) for value in values]


def _bonds_dict(bond_lists, bond_names, bond_distances):
    """
    The ``bonds`` dict: name -> {"distance": ..., "bonds": [...],
    "value": ...}. The value of each coupling is np.nan until it is
    estimated (see MagneticModel.compute_couplings).
    """
    if bond_lists is None:
        return {}
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
    return {
        name: {"distance": distance, "bonds": bonds, "value": np.nan}
        for name, distance, bonds in zip(bond_names, bond_distances, bond_lists)
    }


def _ellipse_error_bounds(centered_coeffs):
    """
    sqrt(N) * sqrt([(Ac^T Ac)^+]_kk) for each column k of the centered
    coefficients Ac (N rows); np.inf for the couplings Ac does not
    determine (column k outside its row space).
    """
    centered_coeffs = np.asarray(centered_coeffs, dtype=float)
    num_rows, num_cols = centered_coeffs.shape
    if num_rows == 0 or num_cols == 0:
        return np.full(num_cols, np.inf)
    _, sv, vh = np.linalg.svd(centered_coeffs, full_matrices=False)
    rank = int(np.sum(sv > 1.0e-6))
    vh, sv = vh[:rank], sv[:rank]
    determined = np.linalg.norm(vh, axis=0) > 1.0 - 1.0e-6
    widths = np.sqrt(num_rows) * np.linalg.norm(vh / sv[:, None], axis=0)
    return np.where(determined, widths, np.inf)


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
        discretization=None,
        magnetic_species=None,
        onfly: bool = True,
        model_label="default",
        g_lande_factors: list | None = None,
        spin_repr: list | None = None,
        symmetries=None,
        space_group_symbol=None,
    ):
        """
        Parameters
        ----------
        atomic_pos : list of arrays, positions of the magnetic sites.
        bravais_lat : list of arrays, Bravais vectors.
        bond_lists, bond_names, bond_distances : the couplings, as lists of
            bonds ``(src, dest, offset)``, with their names (default
            ``J0, J1, ...``) and distances (default 0).
        supercell_size : int or [sx, sy, sz], each value clamped to 1..4.
        magnetic_species : names of the species (default "X").
        g_lande_factors, spin_repr : per site values (default 2.0 and 1/2);
            strings are accepted, with "." meaning "not given". The given
            lists are not modified.
        symmetries, space_group_symbol : how the model was built, kept for
            save_cif and for information.
        ranges, discretization : deprecated, ignored. Use generate_bonds.
        """
        if ranges is not None or discretization is not None:
            warnings.warn(
                "The ranges and discretization arguments of MagneticModel "
                "have no effect; call generate_bonds(ranges, discretization) "
                "to generate the bonds.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model_label = model_label
        self.onfly = onfly
        self.cell_size = 0
        num_sites = len(atomic_pos)
        logger.debug("atomic positions: %s", atomic_pos)

        if magnetic_species is None:
            magnetic_species = ["X"] * num_sites
        logger.debug("magnetic species: %s", magnetic_species)
        num_species = len(magnetic_species)

        self.site_properties = {
            "coord_atomos": atomic_pos,
            "g_lande_factors": _per_site_values(g_lande_factors, num_species, 2.0),
            "spin_repr": _per_site_values(spin_repr, num_species, 0.5),
            "magnetic_species": magnetic_species,
        }
        self.lattice_properties = {
            "bravais_vectors": bravais_lat,
            "cell_size": num_sites,
            "supercell_size": _clamp_supercell_size(supercell_size),
            # Symmetry operators used to build this model (empty for a plain
            # P1 description). Kept so that save_cif can write a faithful
            # CIF, and so that other tools can tell how it was built.
            "symmetries": list(symmetries) if symmetries is not None else [],
            "space_group_symbol": space_group_symbol,
        }
        self.bonds = _bonds_dict(bond_lists, bond_names, bond_distances)

    def remove_bond_by_name(self, name):
        """
        Remove the bond with name <name>
        """
        try:
            self.bonds.pop(name)
        except KeyError:
            msg = "bond " + name + " not found"
            logger.warning(msg)
            return

    def remover_bond_by_name(self, name):
        """Deprecated: use remove_bond_by_name."""
        warnings.warn(
            "remover_bond_by_name is deprecated; use remove_bond_by_name",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.remove_bond_by_name(name)

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
        atomic_positions = self.site_properties["coord_atomos"]
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
            atomic_positions,
            self.lattice_properties["bravais_vectors"],
        ):
            for i, pos_i in enumerate(atomic_positions):
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
            bonds[name] = {
                "distance": dist,
                "bonds": new_bonds[dist_type],
                "value": np.nan,
            }

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

    def coefficient_matrix(self, configs, centered=False, normalizar=None):
        """
        Return the matrix defining the equation system
        connecting coupling constants with simulated energies
        for a given set of configurations configs.

        If centered is True, the column of each coupling is shifted to
        zero average, which makes it orthogonal to the last (constant)
        column, the one of the base energy. normalizar is the deprecated
        name of centered.
        """
        if normalizar is not None:
            warnings.warn(
                "the normalizar argument of coefficient_matrix is deprecated; "
                "use centered",
                DeprecationWarning,
                stacklevel=2,
            )
            centered = normalizar
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
        if centered:
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
        for c in range(2 ** (size - 1)):
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

    def error_bounds(self, confs, method="exact"):
        """
        Largest error of each coupling constant per unit error in the
        energies, for the set of configurations confs.

        If every energy is known within a tolerance t, the coupling J_k is
        known within t * error_bounds(confs)[k]: these are the half widths
        of the box containing the compatibility region (see
        compute_couplings) for zero residual and unit tolerance. They are a
        property of the configurations alone, so they can be computed
        before the energies.

        method:
            "exact": by linear programming (the base energy is a free
                parameter). Never increases when configurations are added.
            "ellipse": the bound from the ellipsoid that contains the
                region, sqrt(N) * sqrt([(Ac^T Ac)^-1]_kk), with Ac the
                centered coefficients of the couplings. Closed form, about a
                thousand times faster, and always >= the exact value.

        Returns an array with one entry per coupling, in the order of
        self.bonds; np.inf for couplings the configurations do not
        determine.
        """
        num_bonds = len(self.bonds)
        if len(confs) == 0:
            return np.full(num_bonds, np.inf)
        coeffs = self.coefficient_matrix(confs, centered=True)
        if method == "ellipse":
            return _ellipse_error_bounds(coeffs[:, :num_bonds])
        if method != "exact":
            raise ValueError(f"unknown method {method!r}")
        lower, upper = compatibility_box(coeffs, np.zeros(len(confs)), 1.0)
        return 0.5 * (upper - lower)[:num_bonds]

    def cost(self, confs, method="exact"):
        """
        Largest error of a coupling constant per unit error in the energies
        (the maximum of error_bounds): if every energy is known within t,
        no coupling has an error larger than t * cost(confs). np.inf if the
        configurations do not determine all the couplings.
        """
        bounds = self.error_bounds(confs, method)
        return float(np.max(bounds)) if len(bounds) else 0.0

    def _select_configurations(self, candidates, forced, length, min_improvement):
        """
        Choose configurations among candidates, after the forced ones,
        scored by the maximal ellipse error bound (see error_bounds).

        1. Backward elimination: starting from every candidate, remove one
           at a time the one whose removal hurts the score least. This
           gives a good set for each size, down to the smallest one that
           determines every coupling.
        2. The size: length new configurations if given (or the smallest
           size that determines every coupling, if larger); otherwise the
           smallest size, enlarged while each extra configuration reduces
           the score by at least min_improvement (a fraction).
        3. Exchange (Fedorov): replace a chosen configuration by one left
           out while that improves the score.

        Returns (chosen, score): the chosen candidates and their score
        (np.inf if the candidates do not determine every coupling).
        """
        num_bonds = len(self.bonds)
        num_forced = len(forced)
        rows = self.coefficient_matrix(forced + candidates, False)[:, :num_bonds]
        forced_idx = list(range(num_forced))
        all_candidates = list(range(num_forced, len(rows)))

        def bounds(indices):
            if len(indices) == 0:
                return np.full(num_bonds, np.inf)
            sub = rows[indices]
            return _ellipse_error_bounds(sub - sub.mean(axis=0))

        def score(chosen):
            if num_bonds == 0:
                return 0.0
            return float(np.max(bounds(forced_idx + chosen)))

        def result(chosen):
            return [candidates[i - num_forced] for i in chosen]

        full_score = score(all_candidates)
        if not np.isfinite(full_score):
            # Partial information: return what determines most couplings.
            return result(all_candidates), np.inf

        # 1. Backward elimination, keeping the best set of each size.
        current = list(all_candidates)
        path = {len(current): (list(current), full_score)}
        while current:
            best = None
            for i in current:
                trial = [j for j in current if j != i]
                trial_score = score(trial)
                if best is None or trial_score < best[1]:
                    best = (trial, trial_score)
            if not np.isfinite(best[1]):
                break  # removing any configuration loses information
            current = best[0]
            path[len(current)] = (list(current), best[1])
        min_size = min(path)

        # 2. The size.
        if length is not None:
            size = min(max(length, min_size), max(path))
        else:
            size = min_size
            while size + 1 in path and path[size + 1][1] <= (
                1.0 - min_improvement
            ) * path[size][1]:
                size += 1
        chosen, chosen_score = path[size]

        # 3. Exchanges.
        improved = True
        while improved:
            improved = False
            for pos in range(len(chosen)):
                for j in all_candidates:
                    if j in chosen:
                        continue
                    trial = chosen[:pos] + [j] + chosen[pos + 1 :]
                    trial_score = score(trial)
                    if trial_score < chosen_score * (1.0 - 1.0e-9):
                        chosen, chosen_score = trial, trial_score
                        improved = True
        return result(chosen), chosen_score

    def optimize_independent_set(
        self, confs, length=None, forced=None, tol=1.0e-6, min_improvement=0.05
    ):
        """
        Choose, among confs, the configurations that best determine the
        coupling constants, together with the forced (already known) ones.

        Returns (res, cost): res are the chosen configurations (not
        including the forced ones), and cost = self.cost(forced + res), the
        largest error of a coupling per unit error in the energies (np.inf
        if the information is partial; then res provides as much as
        possible).

        If length is given, res has length configurations (or more, if
        they are needed to determine every coupling). Otherwise, it starts
        from the minimal set that determines every coupling, and adds
        configurations while each one reduces the error bound by at least
        min_improvement (5% by default).

        The choice is greedy, scored by the ellipse bound (see
        error_bounds), which is fast and ranks sets very close to the exact
        bound. tol is kept for backward compatibility.
        """
        forced = list(forced) if forced is not None else []
        if len(confs) == 0:
            # Nothing to choose from: the cost is the one of the forced set.
            return [], self.cost(forced)
        candidates = [
            c
            for c in self.get_independent_confs(confs)
            if self.check_independence(c, forced)
        ]
        if not candidates:
            # Nothing new to add: the cost is the one of the forced set.
            return [], self.cost(forced)
        chosen, score = self._select_configurations(
            candidates, forced, length, min_improvement
        )
        if not np.isfinite(score):
            logger.info(
                "optimize_independent_set: the configurations do not determine "
                "every coupling; returning the ones that provide most information."
            )
            return chosen, np.inf
        return chosen, self.cost(forced + chosen)

    def find_optimal_configurations(
        self,
        start=None,
        num_new_confs=None,
        known=None,
        its=100,
        update_size=1,
        min_improvement=0.05,
    ):
        """
        Search for a set of new configurations that, together with the known
        ones, determines the coupling constants with the smallest errors.

        Random configurations are proposed its times; each time the best
        subset of the current ones plus the proposals is selected (see
        optimize_independent_set). If num_new_confs is given, the sets have
        that size; otherwise their size follows the same rule as in
        optimize_independent_set: an extra configuration must reduce the
        error bound by at least min_improvement.

        Returns (configurations, cost), with cost = self.cost(known +
        configurations), or np.inf if the information is partial.
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
                last_better, num_new_confs, forced=known,
                min_improvement=min_improvement,
            )

        # Sets of different sizes are compared with the same rule used to
        # choose the size: score * (1 / (1 - min_improvement)) ** size.
        penalty = 1.0 / (1.0 - min_improvement)

        def select(pool):
            candidates = [c for c in pool if self.check_independence(c, known)]
            if not candidates:
                return [], self.cost(known, method="ellipse")
            chosen, score = self._select_configurations(
                candidates, known, num_new_confs, min_improvement
            )
            if num_new_confs is None:
                score = score * penalty ** len(chosen)
            return chosen, score

        last_better, score = select(last_better)
        for _ in range(its):
            repres = self.generate_random_configurations(update_size)
            new_try, new_score = select(
                self.get_independent_confs(last_better + list(repres))
            )
            if new_score < score:
                score, last_better = new_score, new_try

        if not np.isfinite(score):
            return last_better, np.inf
        return last_better, self.cost(known + last_better)

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

        The estimated couplings are also stored in the model, as
        self.bonds[name]["value"] (np.nan for undetermined couplings), so
        that, e.g., a Monte Carlo simulation can use them.
        """
        # print("Efind jssss")

        if printeqs:
            coeffs = self.coefficient_matrix(confs, centered=False)
            print("\n# Configurations:\n=================\n")
            for c in confs:
                print(c)
            self.print_equations(coeffs)

        coeffs = self.coefficient_matrix(confs, centered=True)

        if printeqs:
            print("\nLargest error of each coupling per unit error in the energies:")
            for name, bound in zip(self.bonds, self.error_bounds(confs)):
                print(f"   {name}: {bound}")
            print()

        if method is None:
            method = (
                "exact" if montecarlo is None
                else ("montecarlo" if montecarlo else "ellipse")
            )
        if method == "montecarlo":
            return self._store_coupling_values(
                self.montecarlo_box(
                    coeffs,
                    energs,
                    err_energs,
                    1000 if mcsteps is None else mcsteps,
                    mcsizefactor,
                    rng=rng,
                )
            )
        if method == "exact":
            return self._store_coupling_values(
                self.exact_box(coeffs, energs, err_energs)
            )
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
        return self._store_coupling_values((js, deltaJ, model_chi, 1.0))

    def _store_coupling_values(self, result):
        """
        Store the couplings estimated by compute_couplings in
        self.bonds[name]["value"]: np.nan for the undetermined ones (nan
        error bar). Returns result unchanged.
        """
        js, d_js = result[0], result[1]
        for name, value, error in zip(self.bonds, js, d_js):
            self.bonds[name]["value"] = float(value) if np.isfinite(error) else np.nan
        logger.debug(
            "stored coupling values: %s",
            {name: bond["value"] for name, bond in self.bonds.items()},
        )
        return result

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
        coeffs = self.coefficient_matrix(confs, centered=True)
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

        js[~determined] = 0.0
        d_js = np.where(determined, d_js, np.nan)
        return js, d_js, model_chi, rate

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
