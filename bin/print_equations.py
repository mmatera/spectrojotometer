#!/usr/bin/env python3

import argparse

import numpy as np
from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import (magnetic_model_from_file,
                                       read_spin_configurations_file)


def main():
    parser = argparse.ArgumentParser(
        description="""
From an standard cif or wien2k struct file, and a list of configurations, prints
the list of equations for the coupling constants
"""
    )

    parser.add_argument(
        "model",
        metavar="modelfile",
        type=str,
        help="load atomic positions from a model file for the source model",
    )

    parser.add_argument(
        "config", metavar="conffile", type=str, help="configs in the src model"
    )

    parser.add_argument(
        "--format",
        metavar="format",
        type=str,
        choices=["plain", "latex", "wolfram"],
        help="output format",
        default="plain",
    )

    args = vars(parser.parse_args())

    if args["model"] is None:
        print(
            "An input model file is required. You should provide it by means of the parameter --model"
        )
        return -1
    if args["config"] is None:
        print(
            "A file with a list of configurations is required. You should provide it by means of the parameter --config"
        )
        return -1
    fmt = args["format"]

    model = magnetic_model_from_file(args["model"])
    ens, confs, comments = read_spin_configurations_file(args["config"], model)
    model.print_equations(
        model.coefficient_matrix(confs, False), comments=comments, format=fmt
    )
    return 0


if __name__ == "__main__":
    main()
