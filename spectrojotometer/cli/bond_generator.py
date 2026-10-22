#!/opt/anaconda/bin/python3
from __future__ import print_function

import argparse
import sys

from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.tools import add_verbosity_argument, eprint, setup_cli_logging


def main():
    parser = argparse.ArgumentParser(
        description="""
From an standard cif file, produces a new cif file containing information about the bonds that defines a Heisenberg magnetic effective model.
If the input file has already information about bonds, the output file contains both the bonds in the input as well as the new generated bonds.
"""
    )

    parser.add_argument(
        "modelfile",
        metavar="modelfile",
        type=str,
        help="load atomic positions from a model file. It can be specified by a cif or a Wien2k struct file",
    )

    parser.add_argument(
        "-d",
        "--discretization",
        metavar="[value]",
        type=float,
        default=1.0,
        help="the tolerance in considering two distances as the same.",
    )

    parser.add_argument(
        "--rmin",
        metavar="[value]",
        type=float,
        default=0.0,
        help="the minimal distance between atoms of the bond",
    )

    parser.add_argument(
        "--rmax",
        metavar="[value]",
        type=float,
        default=1.0,
        help="the maximal distance between atoms of the bond",
    )

    parser.add_argument(
        "out", metavar="outfile", type=str, default=None, help="Name of the outputfile."
    )

    add_verbosity_argument(parser)
    args = vars(parser.parse_args())
    setup_cli_logging(args["verbose"])

    if args["modelfile"] is None:
        eprint(
            "A model file is required. You should provide it by means of the parameter --modelfile"
        )
        return -1
    if args["out"] is None:
        eprint(
            "An output file is required. You should provide it by means of the parameter --out"
        )
        return -1
    if args["discretization"] is None:
        args["discretization"] = 0.001

    if args["rmin"] is None:
        args["rmin"] = 0.0
    if args["rmax"] is None:
        args["rmax"] = 1.0

    model = magnetic_model_from_file(args["modelfile"])
    model.generate_bonds(
        ranges=[[args["rmin"], args["rmax"]]], discretization=args["discretization"]
    )

    for name, bond in model.bonds.items():
        print(f"{name}: distance {bond['distance']}, {len(bond['bonds'])} bonds")
    model.save_cif(args["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
