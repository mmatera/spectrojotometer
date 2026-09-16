# Tests de carga de modelos (CIF / struct)

Esta batería cubre `spectrojotometer/model_io.py`: las funciones que
leen un archivo CIF o un `.struct` de WIEN2k y arman un
`MagneticModel`.

## Cómo correrlos

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Organización

- `conftest.py` — fixtures de rutas (`examples_dir`, `fixtures_dir`) y
  un helper (`bond_length` / `all_bond_lengths`) que recalcula la
  distancia *real* entre dos átomos de un modelo ya cargado, a partir
  de sus posiciones cartesianas y la red de Bravais, en vez de confiar
  en el valor "declarado" en el archivo de entrada.
- `fixtures/` — geometrías sintéticas mínimas, elegidas para que la
  posición cartesiana y la distancia de cada enlace se puedan
  verificar a mano (a diferencia de los archivos de `examples/`, que
  son datos reales pero no necesariamente autoconsistentes).
- `test_cif_loading.py` — parseo de `.cif`: átomos, especies,
  expansión por simetría vs. `primitive_cell=True`, enlaces.
- `test_struct_loading.py` — parseo de `.struct` (WIEN2k).
- `test_dispatch.py` — `magnetic_model_from_file`: que delegue en la
  función correcta según extensión (insensible a mayúsculas), y las
  inconsistencias de default que se describen abajo.
- `test_symmetry_helpers.py` — funciones puras de más bajo nivel:
  `parse_symmetry`, `centering_letter_from_symbol`,
  `expand_symmetries_with_centering`, `normalize_bond`,
  `pack_offset`/`unpack_offset`.
- `test_geometry_consistency.py` — invariantes cruzados: guardar con
  `save_cif` y releer debe reproducir la geometría; la celda
  convencional y la primitiva de una misma estructura deben describir
  la misma densidad atómica.

## Bugs encontrados mientras se escribían los tests

Quedaron documentados como tests que fallan a propósito
(`pytest.mark.xfail(strict=True)`) o como tests que verifican el
comportamiento *actual* (no necesariamente el deseable). Si en algún
momento se corrigen, esos tests van a empezar a fallar/xpassar y hay
que actualizarlos:

1. **`ValueError` poco informativo cuando el filtro de
   `magnetic_atoms` no deja ningún átomo** (ver
   `test_cromita_without_cr_in_magnetic_atoms_raises` y
   `test_example1_magnetic_atoms_filter_excludes_non_magnetic`): en vez
   de devolver un modelo vacío, `magnetic_positions` queda como lista
   vacía y después se intenta `np.array([]).dot(bravais_vectors)`, que
   explota con `ValueError: shapes (0,) and (3,3) not aligned...` — un
   mensaje que no indica la causa real (ningún átomo coincide con
   `magnetic_atoms`).


