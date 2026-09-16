"""
Tests unitarios para las funciones de más bajo nivel usadas al
interpretar la geometría de un CIF: parseo de operadores de simetría,
letra de centrado, expansión por centrado, y el empaquetado/
desempaquetado de offsets de celda.
"""
import numpy as np
import pytest

from spectrojotometer.model_io import (
    centering_letter_from_symbol,
    expand_symmetries_with_centering,
    normalize_bond,
    parse_symmetry,
)
from spectrojotometer.tools import pack_offset, unpack_offset


# ---------------------------------------------------------------------
# parse_symmetry
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "strsymm,expected_rot,expected_offset",
    [
        ("x, y, z", np.eye(3), [0.0, 0.0, 0.0]),
        ("-x, -y, z", np.diag([-1, -1, 1]), [0.0, 0.0, 0.0]),
        ("x, y+1/2, z+1/2", np.eye(3), [0.0, 0.5, 0.5]),
        ("x+1/2, y+1/2, z", np.eye(3), [0.5, 0.5, 0.0]),
    ],
)
def test_parse_symmetry(strsymm, expected_rot, expected_offset):
    rot, offset = parse_symmetry(strsymm)
    np.testing.assert_allclose(rot.astype(float), expected_rot)
    np.testing.assert_allclose(offset.astype(float), expected_offset)


# ---------------------------------------------------------------------
# centering_letter_from_symbol
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("F d -3 m", "F"),
        ("I 4/m m m", "I"),
        ("'P 1'", "P"),
        (None, "P"),
        ("", "P"),
        ("Q -1", "P"),  # letra no reconocida -> default P
    ],
)
def test_centering_letter_from_symbol(symbol, expected):
    assert centering_letter_from_symbol(symbol) == expected


# ---------------------------------------------------------------------
# expand_symmetries_with_centering
# ---------------------------------------------------------------------

def test_expand_with_centering_fcc_adds_three_translations():
    identity = (np.eye(3), np.array([0.0, 0.0, 0.0]))
    expanded = expand_symmetries_with_centering([identity], "F d -3 m")

    assert len(expanded) == 4
    translations = sorted(tuple(np.round(t, 4)) for _, t in expanded)
    expected = sorted(
        [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    )
    assert translations == expected


def test_expand_with_centering_is_noop_for_primitive_lattice():
    identity = (np.eye(3), np.array([0.0, 0.0, 0.0]))
    expanded = expand_symmetries_with_centering([identity], "P 1")
    assert expanded == [identity]


def test_expand_with_centering_is_noop_for_empty_symmetries():
    assert expand_symmetries_with_centering([], "F d -3 m") == []


def test_expand_with_centering_deduplicates():
    """
    Si la lista original ya incluye una traslación de centrado, no
    debe duplicarse tras la expansión.
    """
    identity = (np.eye(3), np.array([0.0, 0.0, 0.0]))
    already_centered = (np.eye(3), np.array([0.0, 0.5, 0.5]))
    expanded = expand_symmetries_with_centering(
        [identity, already_centered], "F d -3 m"
    )
    translations = {tuple(np.round(t % 1.0, 4)) for _, t in expanded}
    assert len(translations) == 4


# ---------------------------------------------------------------------
# normalize_bond / pack_offset / unpack_offset
# ---------------------------------------------------------------------

def test_normalize_bond_keeps_order_when_already_sorted():
    assert normalize_bond(0, 1, np.array([0, 0, 0])) == (0, 1, ".")


def test_normalize_bond_swaps_and_negates_offset_when_reversed():
    src, dest, offset = normalize_bond(2, 1, np.array([1, -1, 0]))
    assert (src, dest) == (1, 2)
    # Comparamos contra lo que produce pack_offset directamente (ver
    # más abajo por qué NO usamos unpack_offset(offset) acá: no es su
    # inversa).
    assert offset == pack_offset([-1, 1, 0])


@pytest.mark.parametrize(
    "offset,expected_key",
    [
        ([0, 0, 0], "."),
        ([1, -1, 1], "1_191"),  # ejemplo del propio docstring de pack_offset
        ([1, 0, -1], "1_109"),
    ],
)
def test_pack_offset_matches_its_own_documented_convention(offset, expected_key):
    assert pack_offset(offset) == expected_key


@pytest.mark.parametrize(
    "encoded,expected",
    [
        (".", [0, 0, 0]),
        ("655", [0, 0, 0]),  # sin "_": unpack_offset no lo interpreta, devuelve 0
        ("1_555", [0, 0, 0]),  # convención CIF/SHELX: "555" = sin traslación
        ("1_655", [1, 0, 0]),
        ("1_455", [-1, 0, 0]),
        ("2_555", [0, 0, 0]),  # con índice de simetría antepuesto
    ],
)
def test_unpack_offset_matches_cif_symmetry_code_convention(encoded, expected):
    """
    `unpack_offset` sólo decodifica cadenas con el formato
    "<indice>_<ddd>" (la convención estándar de CIF/SHELX para
    códigos de simetría: dígito 5 = sin desplazamiento, dígito
    d = desplazamiento d-5, como en `_geom_bond_site_symmetry_2`, p.ej.
    "2_655"). Una cadena sin guion bajo se interpreta como offset nulo.
    """
    np.testing.assert_array_equal(unpack_offset(encoded), expected)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pack_offset y unpack_offset NO son funciones inversas entre sí "
        "para offsets distintos de cero: pack_offset codifica con "
        "digit=(coord+10)%10, mientras que unpack_offset decodifica con "
        "coord=digit-5 (la convención estándar de CIF/SHELX). "
        "normalize_bond empaqueta offsets con pack_offset, así que "
        "cualquier código que luego intente recuperar ese offset con "
        "unpack_offset (como se hace, por ejemplo, al reconstruir la "
        "geometría real de un enlace periódico) obtiene un valor "
        "incorrecto. Los bonds con offset '.' (el caso más común, sin "
        "imagen periódica) no se ven afectados porque unpack_offset trata "
        "'.' como caso especial. Este test documenta el bug con un caso "
        "concreto; en cuanto se arregle la codificación, hay que quitar "
        "el xfail."
    ),
)
@pytest.mark.parametrize(
    "offset",
    [
        [1, 0, -1],
        [4, -4, 2],
        [-4, 4, -4],
    ],
)
def test_pack_unpack_offset_are_not_actually_inverses(offset):
    packed = pack_offset(offset)
    recovered = unpack_offset(packed)
    np.testing.assert_array_equal(recovered, offset)
