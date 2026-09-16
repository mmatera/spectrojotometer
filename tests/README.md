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

1. **Defaults de `magnetic_atoms` inconsistentes entre las tres
   funciones** (`test_dispatch.py::TestDefaultMagneticAtomsInconsistencies`):
   - `magnetic_model_from_cif`: ... Cu, V (sin Cr, sin Ti)
   - `magnetic_model_from_wk2_struct`: ... V (sin Cu, sin Cr, sin Ti)
   - `magnetic_model_from_file`: ... Cu, V, Ti, Cr (superset)

   Un CIF con átomos de Cr (como `examples/cromita_ortogonal.cif`) sólo
   carga con los valores por default a través del despachador; llamar
   directamente a `magnetic_model_from_cif(path)` sin pasar
   `magnetic_atoms` explícitamente falla.

2. **`ValueError` poco informativo cuando el filtro de
   `magnetic_atoms` no deja ningún átomo** (ver
   `test_cromita_without_cr_in_magnetic_atoms_raises` y
   `test_example1_magnetic_atoms_filter_excludes_non_magnetic`): en vez
   de devolver un modelo vacío, `magnetic_positions` queda como lista
   vacía y después se intenta `np.array([]).dot(bravais_vectors)`, que
   explota con `ValueError: shapes (0,) and (3,3) not aligned...` — un
   mensaje que no indica la causa real (ningún átomo coincide con
   `magnetic_atoms`).

3. **`magnetic_model_from_wk2_struct` arma mal la posición de las
   réplicas cuando `MULT > 1`** (`tests/test_struct_loading.py::
   test_multiplicity_replica_uses_its_own_z_coordinate`, xfail): usa
   `fields[2][3:]` dos veces (para Y *y* Z) en lugar de `fields[2][3:]`
   y `fields[3][3:]`, así que la coordenada Z de cada réplica queda
   igual a la Y.

4. **`pack_offset` y `unpack_offset` no son funciones inversas para
   offsets no nulos** (`test_pack_unpack_offset_are_not_actually_inverses`,
   xfail): `pack_offset` codifica con `digit=(coord+10)%10`, mientras
   que `unpack_offset` decodifica con `coord=digit-5` (la convención
   estándar de CIF/SHELX, correcta para leer `_geom_bond_site_symmetry_2`
   de un archivo externo, p.ej. `"2_655"`). Como `normalize_bond`
   empaqueta los offsets que calcula con `pack_offset`, cualquier
   intento posterior de recuperar ese offset con `unpack_offset` da un
   valor incorrecto. Esto **no** afecta a los enlaces con offset `"."`
   (sin imagen periódica, el caso más común — incluido en todos los
   ejemplos actuales del repo), porque `unpack_offset` trata `"."`
   como caso especial; sólo importa para enlaces periódicos
   (`offset != "."`).

Ninguno de estos se corrigió en este cambio: la consigna era escribir
la batería de tests, no arreglar el código. Quedan documentados acá
para decidir conscientemente qué hacer con cada uno.
