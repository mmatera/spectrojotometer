from .magnetic_model import *
import numpy as np


def pack_offset(r):
    r = np.array(r, dtype=int)
    if any(r > 4) or any(r < -5):
        print("offset too big!")
        raise Exception("overflow")
    if all(r == 0):
        return "."
    r = r + 5
    res = "1_" + str(r[0]) + str(r[1]) + str(r[2])
    return res


def unpack_offset(r):
    if r == ".":
        return np.array([0, 0, 0])
    r = r.split("_")
    if len(r) == 1:
        return np.array([0, 0, 0])
    r = r[1]
    r = np.array([int(c) - 5 for c in r], dtype=int)
    return r


def unpack_symmetry_and_offset(r):
    if r == ".":
        return 0, np.array([0, 0, 0])
    r = r.split("_")
    if len(r) == 1:
        return int(r[0]) - 1, np.array([0, 0, 0])
    return int(r[0]) - 1, np.array([int(c) - 5 for c in r[1]], dtype=int)


def find_atom_offset_by_symmetry(p, symop, magnetic_positions):
    sp = symop[0].dot(p) + symop[1]
    j = sorted([(k, np.linalg.norm((q - sp + .5) % 1 - .5))
                for k, q in enumerate(magnetic_positions)],
               key=lambda u: u[1])[0][0]
    offset = np.array(sp - magnetic_positions[j], dtype=int)
    return j, offset


def normalize_bond(i, j, offset):
    if i > j:
        i, j = j, i
        offset = -offset
    elif i == j:
        if offset[2] < 0:
            offset = -offset
        elif offset[2] == 0:
            if offset[1] < 0:
                offset = - offset
            elif offset[1] == 0 and \
                 offset[0] < 0:
                offset = - offset
    return i, j, pack_offset(offset)


def parse_symmetry(strsymm):
    strsymm = strsymm.strip()
    strsymm.split(", ")
    rows = strsymm.strip().split(sep=", ")
    trows = []
    offset = []
    for row in rows:
        row = row.strip()
        terms = row.split(sep="-")
        for i, term in enumerate(terms):
            terms[i] = "-" + term
        if terms[0] == "-":
            terms = terms[1:]
        else:
            terms[0] = terms[0][1:]
        terms = sum([term.split(sep="+") for term in terms], [])
        terms = [term.strip() for term in terms]
        trow = [0, 0, 0, 0]
        for term in terms:
            if term[-1] == 'x':
                trow[0] = term[:-1]
            elif term[-1] == 'y':
                trow[1] = term[:-1]
            elif term[-1] == 'z':
                trow[2] = term[:-1]
            else:
                trow[3] = term
        for i in range(4):
            if trow[i] == 0:
                continue
            if trow[i] == "":
                trow[i] = 1
                continue
            if trow[i] == "-":
                trow[i] = -1
                continue
            factors = trow[i].split(sep="/")
            if len(factors) == 1:
                trow[i] = float(factors[0].strip())
            else:
                trow[i] = float(factors[0].strip())/float(factors[1].strip())
        trows.append(trow)
    w = np.array(trows)
    offset = w[:, -1]
    w = w[:, :-1]
    return w, offset


def magnetic_model_from_file(filename,
                             magnetic_atoms=["Mn", "Fe", "Co",
                                             "Ni", "Dy", "Tb",
                                             "Eu", "Cu", "V", "Ti", "Cr"],
                             bond_names=None):
    if filename[-4:] == ".cif" or filename[-4:] == ".CIF":
        return magnetic_model_from_cif(
            filename, magnetic_atoms, bond_names)
    if filename[-7:] == ".struct" or filename[-7:] == ".STRUCT":
        return magnetic_model_from_wk2_struct(
            filename, magnetic_atoms,
            bond_names)
    eprint("unknown file format")
    return -1


def cif_read_loop_symmetries(labels, entries):
    symmetries = []
    for i, t in enumerate(labels):
        if t == "_symmetry_equiv_pos_as_xyz":
            symmdefcol = i
        if t == "_space_group_symop_operation_xyz":
            symmdefcol = i
        for i, entry in enumerate(entries):
            symmetries.append(parse_symmetry(entry[symmdefcol]))
    return symmetries


def cif_read_loop_atoms(labels, entries, magnetic_atoms):
    print("atom positions found")
    magnetic_positions = []
    magnetic_species = []
    atomlabels = {}
    g_factors = []
    spin_repr = []
    col_g = -1
    col_s = -1
    for i, t in enumerate(labels):
        if t == "_atom_site_fract_x":
            xcol = i
        elif t == "_atom_site_fract_y":
            ycol = i
        elif t == "_atom_site_fract_z":
            zcol = i
        elif t == "_atom_site_type_symbol":
            tcol = i
        elif t == "_atom_site_label":
            labelcol = i
        elif t == "_atom_site_g_factor":
            col_g = i
        elif t == "_atom_site_spin":
            col_s = i
    idxma = 0
    for i, entry in enumerate(entries):
        if entry[tcol] in magnetic_atoms:
            for cc in (xcol, ycol, zcol):
                entry[cc] = float(entry[cc].split("(", maxsplit=1)[0])

            magnetic_positions.append(np.array([entry[xcol],
                                                entry[ycol],
                                                entry[zcol]]))
            magnetic_species.append(entry[tcol])
            if col_g != -1:
                g_factors.append(entry[col_g])
            else:
                g_factors.append(".")
            if col_s != -1:
                spin_repr.append(entry[col_s])
            else:
                spin_repr.append(".")
            atomlabels[entry[labelcol]] = idxma
            idxma = idxma + 1
    return atomlabels, magnetic_species, \
        magnetic_positions, g_factors, spin_repr


def cif_read_loop_bonds(labels, entries, atomlabels):
    print("Reading bonds from cif")
    print("atomlabels")
    jlabelcol = None
    bondlists = []
    bond_distances = []
    bond_labels = {}
    sym1col = -1
    sym2col = -1
    for i, t in enumerate(labels):
        if t == "_geom_bond_atom_site_label_1":
            at1col = i
        if t == "_geom_bond_atom_site_label_2":
            at2col = i
        if t == "_geom_bond_distance":
            distcol = i
        if t == "_geom_bond_label":
            jlabelcol = i
        if t == "_geom_bond_site_symmetry_1":
            sym1col = i
        if t == "_geom_bond_site_symmetry_2":
            sym2col = i

    for en in entries:
        label1 = en[at1col]
        label2 = en[at2col]
        outbond1 = np.array([0, 0, 0])
        outbond2 = np.array([0, 0, 0])
        if sym1col != -1:
            sym,  outbond1 = unpack_symmetry_and_offset(en[sym1col])
            if sym != 0:
                label1 = label1 + "_" + str(sym)
        if sym2col != -1:
            sym,  outbond2 = unpack_symmetry_and_offset(en[sym2col])
            if sym != 0:
                label2 = label2 + "_" + str(sym)
        if not(label1 in atomlabels and label2 in atomlabels):
            continue
        outbond = np.array(outbond2) - np.array(outbond1)
        newbond = normalize_bond(atomlabels[label1],
                                 atomlabels[label2], outbond)
        en[distcol] = en[distcol].split(sep="(", maxsplit=1)[0]
        if jlabelcol is None:
            if en[distcol] not in bond_distances:
                bond_distances.append(en[distcol])
                bondlabel = "J" + str(len(bond_distances))
                bond_labels[bondlabel] = len(bond_labels)
                bondlists.append([])
            bs = bond_distances.index(en[distcol])
            bondlists[bs].append(newbond)
        else:
            bondlabel = en[jlabelcol]
            if bond_labels.get(bondlabel) is None:
                bond_labels[bondlabel] = len(bondlists)
                bond_distances.append(en[distcol])
                bondlists.append([])

            bondlists[bond_labels[bondlabel]].append(newbond)
    bond_labels = sorted([(bond_labels[la], la)
                          for la in bond_labels],
                         key=lambda x: x[0])
    bond_labels = [x[1] for x in bond_labels]
    return bond_labels, bond_distances, bondlists


def generate_atoms_by_symmetries(symmetries, atomlabels, magnetic_species,
                                 magnetic_positions,
                                 g_factors, spin_repr):
    if len(symmetries) > 1:
        magnetic_positions2 = []
        magnetic_species2 = []
        g_factors2 = []
        spin_repr2 = []

        idxma = len(atomlabels)
        for s, sym in enumerate(symmetries):
            for i, r in enumerate(magnetic_positions):
                newposition = sym[0].dot(r)+sym[1]
                already_in_list = False
                for pos in magnetic_positions2:
                    if np.linalg.norm(
                            (newposition - pos + .5) % 1 - .5) < 1.e-3:
                        already_in_list = True
                        break
                if already_in_list:
                    continue
                magnetic_positions2.append(newposition)
                magnetic_species2.append(magnetic_species[i])
                g_factors2.append(g_factors[i])
                spin_repr2.append(spin_repr[i])
                atomlabel = [key for key, val in atomlabels.items()
                             if val == i][0]
                if s > 0:
                    atomlabel += "_" + str(s+1)
                    atomlabels[atomlabel] = idxma
                    idxma = idxma + 1
        magnetic_positions = magnetic_positions2
        magnetic_species = magnetic_species2
        g_factors = g_factors2
        spin_repr = spin_repr2
    return atomlabels, magnetic_species, magnetic_positions, \
            g_factors, spin_repr


def read_bravais_vectors(bravais_params):
    bravais_vectors = []
    if bravais_params.get('a') is not None:
        bravais_vectors.append(np.array([bravais_params.get('a'), 0, 0]))

    if bravais_params.get('b') is not None:
        gamma = bravais_params.get('gamma')
        if gamma is None:
            gamma = 3.1415926 * .5
        bravais_vectors.append(
            np.array([bravais_params.get('b')*np.cos(gamma),
                      bravais_params.get('b') * np.sin(gamma), 0]))

    if bravais_params.get('c') is not None:
        alpha = bravais_params.get('alpha')
        beta = bravais_params.get('beta')
        if alpha is None:
            alpha = 3.1415926*.5
        if beta is None:
            beta = 3.1415926*.5
        x = np.cos(alpha)
        y = np.cos(beta) - x * np.cos(gamma)
        y = y/np.sin(gamma)
        z = bravais_params.get('c') * np.sqrt(1-x*x-y*y)
        x = bravais_params.get('c') * x
        y = bravais_params.get('c') * y
        bravais_vectors.append(np.array([x, y, z]))
    return bravais_vectors


def generate_bonds_by_symmetries(symmetries,
                                 bond_labels,
                                 bonddistances,
                                 bondlists,
                                 magnetic_positions):
    for s in range(len(symmetries) - 1):
        symop = symmetries[s+1]
        for bidx, blst in enumerate(bondlists):
            for bnd in blst:
                i, j, offset = bnd
                offset = unpack_offset(offset)
                i, offseti = find_atom_offset_by_symmetry(
                    magnetic_positions[i], symop, magnetic_positions)
                j, offsetj = find_atom_offset_by_symmetry(
                    magnetic_positions[j] + offset, symop, magnetic_positions)
                if any(offseti != 0):
                    # The source must be inside the cell
                    i, offseti, j, offsetj = j, offsetj, i, offseti
                if any(offseti != 0):
                    # both atoms are outside the cell
                    continue
                offset = offsetj - offseti
                if any(offset < -5) or any(offset > 4):
                    print("the bond is too long. offset= ", offset)
                    continue

                newbond = normalize_bond(i, j, offset)
                if not (newbond in blst):
                    blst.append(newbond)


def magnetic_model_from_cif(filename,
                            magnetic_atoms=["Mn", "Fe", "Co", "Ni",
                                            "Dy", "Tb", "Eu", "Cu", "V"],
                            bond_names=None):
    bravais_params = {}
    magnetic_positions = None
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []
    symmetries = []
    g_lande_factors = None
    spin_repr = None

    with open(filename, "r") as src:
        for line in src:
            listrip = line.strip()
            if listrip[:13] == '_cell_length_':
                varvals = listrip[13:].split()
                varvals[1] = varvals[1].split(sep="(", maxsplit=1)[0]
                bravais_params[varvals[0]] = float(varvals[1])
            elif listrip[:12] == '_cell_angle_':
                varvals = line[12:].strip().split()
                varvals[1] = varvals[1].split(sep="(", maxsplit=1)[0]
                bravais_params[varvals[0]] = float(varvals[1]) * 3.1415926/180.
            elif listrip[:5] == 'loop_':
                labels = []
                entries = []
                ls = src.readline()
                listrip = ls.strip()
                if listrip == '':
                    break
                if listrip != "" and line[0] == "#":
                    continue
                while listrip[0] == '_' or listrip[0] == '#':
                    if listrip != "" and line[0] == "#":
                        continue
                    labels.append(listrip.split()[0])
                    line = src.readline()
                    listrip = line.strip()
                while listrip != '':
                    newentry = listrip.strip()
                    if newentry[0] in ('"', "'"):
                        newentry = [newentry[1:-1]]
                    else:
                        newentry = newentry.split()
                    entries.append(newentry)
                    ls = src.readline()
                    listrip = ls.strip()

                # if the block contains symmetries
                if '_symmetry_equiv_pos_as_xyz' in labels or \
                '_space_group_symop_operation_xyz' in labels:
                    symmetries = cif_read_loop_symmetries(labels, entries)

                # if the block contains the set of atoms
                if '_atom_site_fract_x' in labels:
                    atomlabels, magnetic_species, magnetic_positions, \
                    g_lande_factors, spin_repr = cif_read_loop_atoms(
                        labels, entries, magnetic_atoms)
                    atomlabels, magnetic_species, magnetic_positions, \
                    g_lande_factors, spin_repr = generate_atoms_by_symmetries(
                        symmetries,
                        atomlabels,
                        magnetic_species,
                        magnetic_positions,
                        g_lande_factors, spin_repr)

                # If the block contains the set of bonds
                if '_geom_bond_atom_site_label_1' in labels:
                    bond_labels, bond_distances, bondlists = cif_read_loop_bonds(
                        labels, entries, atomlabels)
                    generate_bonds_by_symmetries(symmetries,
                                                 bond_labels,
                                                 bond_distances,
                                                 bondlists,
                                                 magnetic_positions)

    bravais_vectors = read_bravais_vectors(bravais_params)
    magnetic_positions = np.array(magnetic_positions).dot(
        np.array(bravais_vectors))
    print("    magnetic species: ", magnetic_species)
    print("    spin representation: ", spin_repr)
    print("    lande factor: ", g_lande_factors)
    print("    positions: ", magnetic_positions)
    print("    bondlabels", bond_labels)
    print("    bondlists", bondlists)

    model = MagneticModel(magnetic_positions, bravais_vectors,
                          bond_lists=bondlists,
                          bond_names=bond_labels,
                          magnetic_species=magnetic_species,
                          g_lande_factors=g_lande_factors,
                          spin_repr=spin_repr)
    model.bond_distances = [float(d) for d in bond_distances]
    return model


def magnetic_model_from_wk2_struct(filename,
                                   magnetic_atoms=["Mn", "Fe", "Co",
                                                   "Ni", "Dy", "Tb",
                                                   "Eu", "V"],
                                   bond_names=None):
    bravais_params = {}
    magnetic_positions = []
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []

    with open(filename) as fin:
        title = fin.readline()
        fin.readline()         # size
        fin.readline()         # not any clue
        bravais = fin.readline()
        for l in fin:
            sl = l.strip()
            if sl[:4] == "ATOM":
                positions = []
                if sl[4] == " ":
                    sl = list(sl)
                    sl[4] = "-"
                    sl = "".join(sl)
                if sl[5] == " ":
                    sl = list(sl)
                    sl[5] = "-"
                    sl = "".join(sl)
                fields = sl.split()
                idxatom = fields[0][4:-1]
                positions.append([float(fields[1][3:]),
                                  float(fields[2][3:]),
                                  float(fields[3][3:])])
                mult = int(fin.readline().strip().split()[1])
                mult = mult - 1
                for k in range(mult):
                    sl = fin.readline()
                    fields = sl.split()
                    positions.append([float(fields[1][3:]),
                                      float(fields[2][3:]),
                                      float(fields[2][3:])])

                atomlabelfield = fin.readline().strip()
                if atomlabelfield[1] == " ":
                    atomspecies = atomlabelfield[0]
                else:
                    atomspecies = atomlabelfield[:2]
                atomlabel = atomspecies + idxatom
                lrm = fin.readline()              # Rotation matrix
                lrm = lrm + fin.readline()
                lrm = lrm + fin.readline()

                if atomspecies not in magnetic_atoms:
                    continue
                for p in positions:
                    magnetic_positions.append(p)
                    magnetic_species.append(atomspecies)

        bravais_fields = bravais.strip().split()
        bravais_params["a"] = float(bravais_fields[0])
        bravais_params["b"] = float(bravais_fields[1])
        bravais_params["c"] = float(bravais_fields[2])
        bravais_params["alpha"] = float(bravais_fields[3]) * 3.1415926/180
        bravais_params["beta"] = float(bravais_fields[4]) * 3.1415926/180
        bravais_params["gamma"] = float(bravais_fields[5]) * 3.1415926/180

        bravais_vectors = []
        if bravais_params.get('a') is not None:
            bravais_vectors.append(np.array([bravais_params.get('a'), 0, 0]))

        if bravais_params.get('b') is not None:
            gamma = bravais_params.get('gamma')
            if gamma is None:
                gamma = 3.1415926*.5
            bravais_vectors.append(np.array([bravais_params.get('b') *
                                             np.cos(gamma),
                                             bravais_params.get('b') *
                                             np.sin(gamma), 0]))

        if bravais_params.get('c') is not None:
            alpha = bravais_params.get('alpha')
            beta = bravais_params.get('beta')
            if alpha is None:
                alpha = 3.1415926*.5
            if beta is None:
                beta = 3.1415926*.5
            x = np.cos(alpha)
            y = np.cos(beta) - x * np.cos(gamma)
            y = y/np.sin(gamma)
            z = bravais_params.get('c') * np.sqrt(1-x*x-y*y)
            x = bravais_params.get('c') * x
            y = bravais_params.get('c') * y
            bravais_vectors.append(np.array([x, y, z]))

    magnetic_positions = np.array(magnetic_positions).dot(
        np.array(bravais_vectors))
    model = MagneticModel(magnetic_positions, bravais_vectors,
                          bond_lists=None,
                          bond_names=None,
                          magnetic_species=magnetic_species,
                          model_label=title)
    model.bond_distances = [float(d) for d in bond_distances]
    return model


def confindex(c):
    return sum([i*2**n for n, i in enumerate(c)])


def read_spin_configurations_file(filename, model):
    configuration_list = []
    energy_list = []
    comments = []
    with open(filename, "r") as f:
        for l in f:
            ls = l.strip()
            if ls == "" or ls[0] == "#":
                continue
            fields = ls.split(maxsplit=1)
            energy = float(fields[0])
            ls = fields[1]
            newconf = []
            comment = ""
            for pos, c in enumerate(ls):
                if c == "#":
                    comment = ls[(pos+1):]
                    break
                elif c == "0":
                    newconf.append(0)
                elif c == "1":
                    newconf.append(1)
            while len(newconf) < model.cell_size:
                eprint("filling empty places")
                newconf.append(0)
            comments.append(comment)
            configuration_list.append(newconf)
            energy_list.append(energy)
    return (energy_list, configuration_list, comments)
