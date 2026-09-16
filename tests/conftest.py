"""
Fixtures y utilidades compartidas por la batería de tests de
spectrojotometer.model_io.

La suite combina dos tipos de datos:

* Los archivos de ejemplo reales que se distribuyen en ``examples/``
  (``example1.cif``, ``cromita_ortogonal.cif``): sirven como "smoke
  tests" -- si el parser deja de poder leerlos, algo se rompió.
* Fixtures sintéticas en ``tests/fixtures/``: geometrías mínimas,
  diseñadas a mano para que la posición cartesiana y la distancia de
  cada enlace se puedan verificar exactamente, sin depender de que los
  datos "reales" sean físicamente consistentes.
"""
from pathlib import Path

import numpy as np
import pytest

from spectrojotometer.tools import unpack_offset

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def bond_length(model, src: int, dest: int, offset) -> float:
    """
    Distancia cartesiana real entre dos sitios de ``model`` separados
    por ``offset`` celdas de Bravais.

    Sirve para comprobar, de forma independiente de lo que el archivo
    de entrada *dice* que mide un enlace, cuál es la distancia que el
    modelo efectivamente contiene una vez cargado.
    """
    if isinstance(offset, str):
        offset = unpack_offset(offset)
    positions = model.site_properties["coord_atomos"]
    bravais_vectors = model.lattice_properties["bravais_vectors"]
    shift = sum(
        coeff * np.array(vector) for coeff, vector in zip(offset, bravais_vectors)
    )
    return float(np.linalg.norm(positions[dest] + shift - positions[src]))


def all_bond_lengths(model, bond_name: str) -> list:
    """Todas las distancias reales asociadas a los enlaces ``bond_name``."""
    return [
        bond_length(model, src, dest, offset)
        for src, dest, offset in model.bonds[bond_name]["bonds"]
    ]
