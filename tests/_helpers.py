"""Helpers shared by the tests."""

import os
import pathlib

# Root of the repository (the directory that contains the package).
ROOT = pathlib.Path(__file__).resolve().parent.parent


def repo_env(**extra):
    """
    Environment for subprocesses that must run the code of this repository,
    not another installed copy of spectrojotometer, wherever pytest runs.
    """
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env
