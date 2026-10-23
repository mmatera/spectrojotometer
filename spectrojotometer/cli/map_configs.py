"""
map_configs: translate spin configurations from one model to another.
"""
import argparse
import sys

import numpy as np

from spectrojotometer.magnetic_model import (
    compatible_configurations,
    equivalent_model,
    map_configurations,
    site_map,
)
from spectrojotometer.model_io import (
    magnetic_model_from_file,
    read_spin_configurations_file,
)
from spectrojotometer.tools import add_verbosity_argument, eprint, setup_cli_logging


def main():
    parser = argparse.ArgumentParser(
        description="""
Translate the spin configurations of a source model into configurations of
a target model: each site of the target takes the spin of the site of the
source at the same position (up to a translation of the source lattice).
The energies are scaled by the ratio between the number of sites of the
target and of the source cells. Configurations that are not periodic with
the lattice of the target (when it is smaller than the source cell) cannot
be represented: they are written as comments.
"""
    )
    parser.add_argument(
        "modelin", metavar="modelfile1", type=str, help="model file of the source model"
    )
    parser.add_argument(
        "modelout", metavar="modelfile2", type=str, help="model file of the target model"
    )
    parser.add_argument(
        "configin",
        metavar="conffile",
        type=str,
        help="configurations of the source model",
    )
    parser.add_argument(
        "out",
        metavar="outfile",
        type=str,
        help='output file; "-" prints to the standard output',
    )
    parser.add_argument(
        "--tol",
        metavar="tolerance",
        type=float,
        default=0.1,
        help="largest distance between matching sites (default 0.1)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "position", "structure"],
        default="auto",
        help="how to match the sites: comparing positions (both models in the "
        "same frame), aligning the structures with pymatgen's StructureMatcher "
        "(different origin, orientation or cell), or auto: positions if every "
        "site is found, otherwise the alignment (default)",
    )
    parser.add_argument(
        "--model-out",
        metavar="cif file",
        type=str,
        default=None,
        help="also save the source model rewritten in the cell and order of "
        "sites of the target (its own positions and couplings) to this CIF file",
    )
    parser.add_argument(
        "--fill",
        type=int,
        choices=[0, 1],
        default=None,
        help="spin for the sites of the target without counterpart in the source "
        "(by default, such sites are an error)",
    )
    add_verbosity_argument(parser)
    args = vars(parser.parse_args())
    setup_cli_logging(args["verbose"])

    modelin = magnetic_model_from_file(args["modelin"])
    modelout = magnetic_model_from_file(args["modelout"])
    energies, confs, comments = read_spin_configurations_file(args["configin"], modelin)
    # Configurations the target cannot represent (not periodic with its
    # lattice) are not mapped: they are written as comments.
    compatible = compatible_configurations(
        modelin, confs, modelout, args["tol"], method=args["method"]
    )
    try:
        mapped = map_configurations(
            modelin,
            [c for c, ok in zip(confs, compatible) if ok],
            modelout,
            tol=args["tol"],
            fill=args["fill"],
            method=args["method"],
        )
    except ValueError as error:
        eprint(f"map_configs: {error}")
        return 1
    mapped = iter(mapped)
    skipped = compatible.count(False)
    if skipped:
        eprint(
            f"map_configs: {skipped} configurations are not periodic with the "
            "lattice of the target model, and were not mapped (they are "
            "written as comments)."
        )

    if args["model_out"]:
        try:
            equivalent_model(
                modelin, modelout, args["tol"], method=args["method"]
            ).save_cif(args["model_out"])
        except ValueError as error:
            eprint(f"map_configs: cannot write {args['model_out']}: {error}")
            return 1

    size_in = len(modelin.site_properties["coord_atomos"])
    size_out = len(modelout.site_properties["coord_atomos"])
    scale = size_out / size_in
    lines = [
        "# site map (target <- source): "
        + str(site_map(modelin, modelout, args["tol"], method=args["method"]))
    ]
    for energy, conf, comment, ok in zip(energies, confs, comments, compatible):
        if ok:
            lines.append(
                f"{np.float64(energy) * scale}\t{next(mapped)}\t# {comment.strip()}"
            )
        else:
            lines.append(
                f"# not periodic with the target cell: {energy}\t{conf}\t# {comment.strip()}"
            )
    text = "\n".join(lines) + "\n"
    if args["out"] == "-":
        sys.stdout.write(text)
    else:
        with open(args["out"], "w") as fout:
            fout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
