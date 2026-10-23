"""
common
Utilities shared by the CIF and .struct readers, plus a couple of
helpers (spin-configuration files, config indices) that are not tied
to either file format.
"""
import numpy as np

from ..magnetic_model import MagneticModel

DEFAULT_MAGNETIC_ATOMS = tuple(
    (
        "Co", "Cr", "Cu", "Cu", "Dy", "Eu", "Fe", "Mn", "Ni", "Tb", "Ti", "V",
    )
)


def read_bravais_vectors(bravais_params: dict) -> list:
    """
    Build a Cartesian Bravais basis from cell parameters (lengths and
    angles), following the standard crystallographic convention:
    `a` is placed along x, `b` in the xy-plane, and `c` completes the
    basis so that the angle between each pair of vectors matches the
    given values.

    Shared by the CIF reader (`cif.magnetic_model_from_cif`) and the
    WIEN2k `.struct` reader (`struct.magnetic_model_from_wk2_struct`),
    which both build `bravais_params` from the cell they parsed and
    call this function to turn it into actual lattice vectors.

    Parameters
    ----------
    bravais_params : dict
        A dict with up to six keys: `"a"`, `"b"`, `"c"` (cell edge
        lengths, in the same length unit the returned vectors will be
        in) and `"alpha"`, `"beta"`, `"gamma"` (cell angles, in
        radians -- the angle opposite to a/b/c respectively, with
        `"gamma"` the angle between `a` and `b`). A missing angle
        defaults to 90 degrees. `"a"`/`"b"`/`"c"` are each optional
        too: only the vectors for the keys that are present get
        built, so passing e.g. only `"a"` returns a single vector.

        Note: building the `c` vector reuses `"gamma"` (needed to
        project `b` onto the xy-plane), so a `bravais_params` that has
        `"c"` but not `"b"` will raise a `NameError`; in practice both
        callers always provide all six keys together, but this
        function does not defend against that particular partial
        input.

    Returns
    -------
    bravais_vectors : list of numpy.ndarray
        0 to 3 Cartesian vectors (each length-3), in the order
        a, b, c -- only for the keys that were present in
        `bravais_params`.
    """
    def value_or_pi_half(x):
        return 3.1415926*.5 if x is None else x

    bravais_vectors = []
    if bravais_params.get("a") is not None:
        bravais_vectors.append(np.array([bravais_params.get("a"), 0, 0]))

    if bravais_params.get("b") is not None:
        gamma = value_or_pi_half(bravais_params.get("gamma"))
        bravais_vectors.append(
            np.array(
                [
                    bravais_params.get("b") * np.cos(gamma),
                    bravais_params.get("b") * np.sin(gamma),
                    0,
                ]
            )
        )

    if bravais_params.get("c") is not None:
        alpha:float = value_or_pi_half(bravais_params.get("alpha"))
        beta:float = value_or_pi_half(bravais_params.get("beta"))
        x = np.cos(alpha)
        y = np.cos(beta) - x * np.cos(gamma)
        y = y / np.sin(gamma)
        z = bravais_params.get("c") * np.sqrt(1 - x * x - y * y)
        x = bravais_params.get("c") * x
        y = bravais_params.get("c") * y
        bravais_vectors.append(np.array([x, y, z]))
    return bravais_vectors


def confindex(c: list) -> int:
    """
    Encode a spin configuration as a single integer label, treating it
    as a binary number with `c[0]` as the least significant bit.

    Parameters
    ----------
    c : list
        A spin configuration: a sequence of 0/1 values, one per
        magnetic site (as produced by `read_spin_configurations_file`).

    Returns
    -------
    int
        `sum(c[n] * 2**n for n in range(len(c)))`, used as a compact
        key to identify or compare configurations.
    """
    return sum([i * 2**n for n, i in enumerate(c)])


def parse_spin_configurations(text: str, num_sites: int = 0) -> tuple:
    """
    Parse spin configurations from text, in the format of
    read_spin_configurations_file: one configuration per line,
    `<energy> <config>[ #<comment>]`. Everything after the first `#` is
    the comment (with or without a space after it); only the `0` and `1`
    before it make the configuration. Blank lines and lines starting with
    `#` are skipped. Configurations shorter than num_sites are
    right-padded with `0`.

    Returns (energy_list, configuration_list, comments), as
    read_spin_configurations_file.
    """
    configuration_list = []
    energy_list = []
    comments = []
    for line in text.splitlines():
        ls = line.strip()
        if ls == "" or ls[0] == "#":
            continue
        fields = ls.split(maxsplit=1)
        energy = float(fields[0])
        ls = fields[1] if len(fields) > 1 else ""
        newconf = []
        comment = ""
        for pos, c in enumerate(ls):
            if c == "#":
                comment = ls[(pos + 1) :]
                break
            if c == "0":
                newconf.append(0)
            elif c == "1":
                newconf.append(1)
        while len(newconf) < num_sites:
            newconf.append(0)
        comments.append(comment)
        configuration_list.append(newconf)
        energy_list.append(energy)
    return (energy_list, configuration_list, comments)


def read_spin_configurations_file(filename: str, model: MagneticModel) -> tuple:
    """
    Read a set of spin configurations relative to a model
    from a file.

    The expected file format is one configuration per line:
    `<energy> <config>[ #<comment>]`, where `<energy>` is a float and
    `<config>` is a string of `0`/`1` characters, one per magnetic
    site, in the order `model`'s sites are indexed (any character
    other than `0` or `1` before a `#` is ignored, which in practice
    allows separating groups of sites with spaces for readability).
    Configurations shorter than `model.cell_size` are right-padded
    with `0`. Blank lines and lines starting with `#` are skipped.
    See parse_spin_configurations.

    Parameters
    ----------
    filename : str
        the file to read.
    model : MagneticModel
        the reference model.

    Returns
    -------
    energy_list : list of float
        The energy declared for each configuration, in file order.
    configuration_list : list of list of int
        Each configuration as a list of 0/1 values (one per magnetic
        site), padded to `model.cell_size`, in file order and aligned
        with `energy_list`.
    comments : list of str
        The text after `#` on each line (`""` if there was none), in
        file order and aligned with the other two lists.
    """
    with open(filename, "r") as stream:
        return parse_spin_configurations(stream.read(), model.cell_size)
