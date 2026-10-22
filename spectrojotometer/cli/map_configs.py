"""
map_configs: translate spin configurations from one model to another.
"""
import argparse
import sys

import numpy as np

from spectrojotometer.magnetic_model import map_configurations, site_map
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
target and of the source cells.
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
    try:
        mapped = map_configurations(
            modelin, confs, modelout, tol=args["tol"], fill=args["fill"]
        )
    except ValueError as error:
        eprint(f"map_configs: {error}")
        return 1

    size_in = len(modelin.site_properties["coord_atomos"])
    size_out = len(modelout.site_properties["coord_atomos"])
    scale = size_out / size_in
    lines = ["# site map (target <- source): " + str(site_map(modelin, modelout, args["tol"]))]
    for energy, conf, comment in zip(energies, mapped, comments):
        lines.append(f"{np.float64(energy) * scale}\t{conf}\t# {comment.strip()}")
    text = "\n".join(lines) + "\n"
    if args["out"] == "-":
        sys.stdout.write(text)
    else:
        with open(args["out"], "w") as fout:
            fout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
