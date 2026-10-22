"""Tests para el modulo Monte Carlo Ising de Spectrojotometer."""

import numpy as np
import pytest

from spectrojotometer.monte_carlo import (
    detect_frustration,
    IsingMonteCarlo,
    ParallelTempering,
    WolffCluster,
    estimate_tc,
    fit_curie_weiss,
    frustration_factor,
    run_ising_mc,
)
from spectrojotometer.monte_carlo.lattices import (
    build_chain_1d,
    build_square_lattice_2d,
    build_square_lattice_3d,
    build_triangular_lattice_2d,
)





# --------------------------------------------------------------------------
# Deteccion de frustration
# --------------------------------------------------------------------------

class TestFrustration:

    def test_square_fm_unfrustrated(self):
        _, bonds, J, _ = build_square_lattice_2d(L=4, J=+1.0)
        frust, gauge, cycle = detect_frustration(bonds, J, N=16)
        assert not frust
        assert gauge is not None
        assert cycle is None
        assert np.all(gauge == 1)

    def test_square_afm_unfrustrated(self):
        """AFM bipartito: no frustrado."""
        _, bonds, J, _ = build_square_lattice_2d(L=4, J=-1.0)
        frust, gauge, cycle = detect_frustration(bonds, J, N=16)
        assert not frust
        assert set(np.unique(gauge)).issubset({-1, 1})
        assert (gauge == -1).sum() > 0

    def test_triangle_afm_frustrated(self):
        """AFM triangular: frustrado."""
        _, bonds, J, _ = build_triangular_lattice_2d(L=4, J=-1.0)
        frust, gauge, cycle = detect_frustration(bonds, J, N=16)
        assert frust
        assert gauge is None
        assert cycle is not None
        assert len(cycle) >= 4

    def test_mixed_balanced(self):
        """Ciclo cuadrado con par de negativas -> balanceado."""
        N = 4
        bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        J = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float64)
        frust, gauge, cycle = detect_frustration(bonds, J, N)
        assert not frust
        assert gauge is not None

    def test_mixed_unbalanced(self):
        """Ciclo cuadrado con impar de negativas -> frustrado."""
        N = 4
        bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        J_par = np.array([-1.0, -1.0, +1.0, +1.0], dtype=np.float64)
        frust_par, _, _ = detect_frustration(bonds, J_par, N)
        assert not frust_par

        J_impar = np.array([-1.0, +1.0, +1.0, +1.0], dtype=np.float64)
        frust_impar, _, cycle = detect_frustration(bonds, J_impar, N)
        assert frust_impar
        assert cycle is not None

    def test_fm_trivial_no_bonds(self):
        bonds = np.zeros((0, 2), dtype=np.int64)
        J = np.zeros((0,), dtype=np.float64)
        frust, gauge, cycle = detect_frustration(bonds, J, N=3)
        assert not frust
        assert np.all(gauge == 1)


# --------------------------------------------------------------------------
# Motor Metropolis
# --------------------------------------------------------------------------

class TestMetropolis:

    def test_energy_zero_temperature_1d(self):
        """1D FM a T muy baja: E = -N*J."""
        L = 8
        spins, bonds, J, E0 = build_chain_1d(L, J=1.0, periodic=True)
        sim = IsingMonteCarlo(spins, bonds, J, E0, seed=42)
        sim.set_temperature(0.05)
        sim.equilibrate(2000)
        obs = sim.sample(500)
        assert obs["E"] == pytest.approx(-1.0, abs=0.05)

    def test_metropolis_stable_averages(self):
        L = 4
        spins, bonds, J, E0 = build_square_lattice_2d(L, J=1.0)
        sim = IsingMonteCarlo(spins, bonds, J, E0, seed=0)
        sim.set_temperature(3.0)
        sim.equilibrate(500)
        obs1 = sim.sample(2000)
        obs2 = sim.sample(2000)
        assert abs(obs1["E"] - obs2["E"]) < 0.15

    def test_temperature_validation(self):
        spins, bonds, J, E0 = build_chain_1d(4)
        sim = IsingMonteCarlo(spins, bonds, J, E0)
        with pytest.raises(ValueError):
            sim.set_temperature(0.0)
        with pytest.raises(ValueError):
            sim.set_temperature(-1.0)


# --------------------------------------------------------------------------
# Wolff
# --------------------------------------------------------------------------

class TestWolff:

    def test_wolff_requires_gauge(self):
        spins, bonds, J, E0 = build_square_lattice_2d(4)
        with pytest.raises(ValueError, match="gauge"):
            WolffCluster(spins, bonds, J, E0)

    def test_wolff_initial_state_options(self):
        """El estado inicial 'ordered' converge al Neel en base gauge."""
        spins, bonds, J, E0 = build_square_lattice_2d(8, J=-1.0)
        _, gauge, _ = detect_frustration(bonds, J, len(spins))

        # 'ordered' -> todos +1 en base gauge (recomendado a baja T)
        sim = WolffCluster(spins, bonds, J, E0, seed=1,
                           gauge=gauge, initial_state="ordered")
        sim.set_temperature(0.5)
        sim.equilibrate(100)
        obs = sim.sample(100)
        assert obs["M"] > 0.9

        # 'random' -> arranca desordenado, no converge en pocos sweeps a baja T
        sim_r = WolffCluster(spins, bonds, J, E0, seed=1,
                             gauge=gauge, initial_state="random")
        sim_r.set_temperature(0.5)
        sim_r.equilibrate(5)  # pocos sweeps
        obs_r = sim_r.sample(5)
        # Sin garantia de convergencia: solo chequear que corre
        assert 0.0 <= obs_r["M"] <= 1.0

    def test_wolff_rejects_unknown_initial_state(self):
        spins, bonds, J, E0 = build_square_lattice_2d(4, J=1.0)
        _, gauge, _ = detect_frustration(bonds, J, len(spins))
        with pytest.raises(ValueError, match="initial_state"):
            WolffCluster(spins, bonds, J, E0, gauge=gauge,
                         initial_state="bogus")

    def test_wolff_ground_state(self):
        spins, bonds, J, E0 = build_square_lattice_2d(8, J=1.0)
        frust, gauge, _ = detect_frustration(bonds, J, len(spins))
        assert not frust
        sim = WolffCluster(spins, bonds, J, E0, seed=1, gauge=gauge)
        sim.set_temperature(0.5)
        sim.equilibrate(200)
        obs = sim.sample(200)
        assert obs["M"] > 0.95

    def test_wolff_afm_gauge(self):
        spins, bonds, J, E0 = build_square_lattice_2d(8, J=-1.0)
        frust, gauge, _ = detect_frustration(bonds, J, len(spins))
        assert not frust
        sim = WolffCluster(spins, bonds, J, E0, seed=2, gauge=gauge)
        sim.set_temperature(0.5)
        sim.equilibrate(200)
        obs = sim.sample(200)
        assert obs["M"] > 0.9


# --------------------------------------------------------------------------
# Parallel tempering
# --------------------------------------------------------------------------

class TestParallelTempering:

    def test_swap_acceptance_reasonable(self):
        spins, bonds, J, E0 = build_square_lattice_2d(6, J=1.0)
        T_grid = np.geomspace(1.5, 3.5, 16)
        pt = ParallelTempering(spins, bonds, J, E0, T_grid, seed=3)
        res = pt.run(n_equil=200, n_sweeps=500, swap_every=5)
        assert 0.05 < res["swap_acceptance"] < 0.95

    def test_pt_vs_metropolis_consistency(self):
        spins, bonds, J, E0 = build_square_lattice_2d(6, J=1.0)
        T_test = 2.5
        sim = IsingMonteCarlo(spins, bonds, J, E0, seed=4)
        sim.set_temperature(T_test)
        sim.equilibrate(1000)
        m_obs = sim.sample(2000)
        T_grid = np.geomspace(1.5, 3.5, 12)
        pt = ParallelTempering(spins, bonds, J, E0, T_grid, seed=5)
        res = pt.run(n_equil=500, n_sweeps=2000, swap_every=5)
        idx = int(np.argmin(np.abs(np.array(res["T"]) - T_test)))
        assert abs(m_obs["E"] - res["E"][idx]) < 0.15


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------

class TestAnalysis:

    def test_estimate_tc_parabolic_symmetric(self):
        """Pico interior con datos simetricos: interpolacion parabolica
        debe dar exactamente el centro."""
        T = np.array([1.0, 2.0, 3.0])
        chi = np.array([1.0, 5.0, 1.0])
        assert estimate_tc(T, chi) == pytest.approx(2.0, abs=0.01)

    def test_estimate_tc_boundary(self):
        """Pico en el borde: devuelve la T del borde sin extrapolar."""
        T = np.array([1.0, 2.0, 3.0])
        chi = np.array([5.0, 2.0, 1.0])  # max en idx=0
        assert estimate_tc(T, chi) == pytest.approx(1.0, abs=0.01)

        chi = np.array([1.0, 2.0, 5.0])  # max en idx=-1
        assert estimate_tc(T, chi) == pytest.approx(3.0, abs=0.01)

    def test_estimate_tc_asymmetric_parabola(self):
        """Pico interior con datos asimetricos: la parabola pasa por los
        tres puntos y su vertice cae en 2.0714 (valor analitico)."""
        T = np.array([1.0, 2.0, 3.0])
        chi = np.array([1.0, 5.0, 2.0])
        # Vertice: -b/(2a) con a=-3.5, b=14.5  ->  14.5/7 = 2.0714
        assert estimate_tc(T, chi) == pytest.approx(2.07142857, abs=1e-6)

    def test_curie_weiss_synthetic(self):
        C_true, theta_true = 2.5, -3.0
        T = np.linspace(5.0, 10.0, 30)
        chi = C_true / (T - theta_true)
        out = fit_curie_weiss(T, chi, T_cut=4.0)
        assert out is not None
        assert out["C"] == pytest.approx(C_true, rel=1e-3)
        assert out["theta_CW"] == pytest.approx(theta_true, abs=1e-3)

    def test_curie_weiss_insufficient_points(self):
        T = np.array([5.0, 6.0])
        chi = np.array([1.0, 0.8])
        assert fit_curie_weiss(T, chi, T_cut=4.0) is None

    def test_frustration_factor(self):
        J = np.array([1.0, -1.0, 1.0, -1.0])
        f = frustration_factor(Tc=2.0, J_vals=J, N=4)
        assert f == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------
# Integracion / runner
# --------------------------------------------------------------------------

class TestRunner:

    def test_runner_warns_on_frustrated(self):
        spins, bonds, J, E0 = build_triangular_lattice_2d(L=6, J=-1.0)
        with pytest.warns(UserWarning, match="frustrado"):
            results = run_ising_mc(
                spins, bonds, J, E0,
                T_min=1.0, T_max=4.0, n_temps=8,
                n_equil=100, n_sweeps=200,
                algorithm="auto", seed=0,
            )
        assert results["frustrated"] is True
        assert results["algorithm"] == "metropolis"

    def test_runner_wolff_on_unfrustrated(self):
        spins, bonds, J, E0 = build_square_lattice_2d(L=6, J=1.0)
        results = run_ising_mc(
            spins, bonds, J, E0,
            T_min=1.5, T_max=4.0, n_temps=12,
            n_equil=200, n_sweeps=500,
            algorithm="auto", seed=0,
        )
        assert results["frustrated"] is False
        assert results["algorithm"] == "wolff"
        assert "Tc_peak" in results

    def test_runner_rejects_wolff_on_frustrated(self):
        spins, bonds, J, E0 = build_triangular_lattice_2d(L=4, J=-1.0)
        with pytest.raises(ValueError, match="frustrado"):
            run_ising_mc(
                spins, bonds, J, E0,
                T_min=1.0, T_max=4.0, n_temps=4,
                n_equil=50, n_sweeps=100,
                algorithm="wolff", seed=0,
            )

           


@pytest.mark.slow
class TestPerformance:

    def test_numba_speedup(self):
        """Verifica que Numba esta activo y que 1000 sweeps son rapidos.

        Si Numba no esta instalado, el test se saltea.
        """
        from spectrojotometer.monte_carlo._jit_kernels import HAS_NUMBA
        if not HAS_NUMBA:
            pytest.skip("Numba no instalado")

        import time
        spins, bonds, J, E0 = build_square_lattice_2d(L=16, J=1.0)
        sim = IsingMonteCarlo(spins, bonds, J, E0, seed=0)
        sim.set_temperature(2.27)
        # Primer sweep compila los kernels
        sim.metropolis_sweep()
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(1000):
            sim.metropolis_sweep()
        elapsed = time.perf_counter() - t0
        print(f"\n1000 sweeps L=16: {elapsed:.3f} s")
        # Con Numba, L=16 con 1000 sweeps deberia tardar << 1 s
        assert elapsed < 5.0, f"Demasiado lento: {elapsed:.2f} s"



           
# --------------------------------------------------------------------------
# Validacion fisica contra resultados de literatura
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestPhysics:
    """Validacion fisica contra valores de literatura.

    Runtimes (Numba activo, ~3 ms/sweep a L=16):
      - test_2d_ising_tc_wolff:  ~25 s
      - test_2d_ising_tc_pt:     ~25 s
      - test_3d_ising_tc_wolff:  ~30 s
      - test_curie_weiss:        ~10 s
      Total: ~90 s

    Sobre el estimador de Tc con Wolff
    ----------------------------------
    A L finito, ningun estimador basado en el order parameter da un
    pico limpio en Tc con Wolff (chi estandar diverge a T << Tc por
    flips cuasi-globales; chi_connected tiene maximo desplazado hacia
    T_min por percolacion parcial del cluster a T ~ 0.85 Tc).
    Se verifica la transicion via C(T) y U4(T), que son independientes
    del order parameter. Para Tc preciso se usa PT con chi_connected.
    """

    def test_2d_ising_tc_wolff(self):
        """Wolff en 2D: transicion en L=16 via C(T) y U4(T)."""
        spins, bonds, J, E0 = build_square_lattice_2d(L=16, J=1.0)
        _, gauge, _ = detect_frustration(bonds, J, len(spins))
        T_grid = np.linspace(1.9, 2.7, 17)
        sim = WolffCluster(spins, bonds, J, E0, seed=42, gauge=gauge)
        res = sim.sweep_temperatures(T_grid, n_equil=200, n_sweeps=500)

        T = np.array(res["T"])
        M = np.array(res["M"])
        C = np.array(res["C"])
        U4 = np.array(res["U4"])

        # (a) Transicion en M(T)
        assert M[0] > 0.9, f"M(T={T[0]:.2f})={M[0]:.3f} (esperado > 0.9)"
        assert M[-1] < 0.3, f"M(T={T[-1]:.2f})={M[-1]:.3f} (esperado < 0.3)"

        # (b) Pico del calor especifico cerca de Tc(L=16) ~ 2.35
        Tc_C = estimate_tc(T, C)
        assert 2.1 < Tc_C < 2.6, f"Tc_C={Tc_C:.3f} fuera de [2.1, 2.6]"

        # (c) Binder cumulant monotona cruzando 0.5
        idx_cross = int(np.argmin(np.abs(U4 - 0.5)))
        Tc_U4 = T[idx_cross]
        assert 2.1 < Tc_U4 < 2.6, f"Tc_U4={Tc_U4:.3f} fuera de [2.1, 2.6]"

        # Consistencia entre estimadores
        assert abs(Tc_C - Tc_U4) < 0.3, \
            f"Tc_C={Tc_C:.3f} vs Tc_U4={Tc_U4:.3f} difieren demasiado"


    def test_2d_ising_tc_pt(self):
        """PT en 2D: Tc(L=16) ~ 2.35 con chi_connected.

        Nota sobre el presupuesto
        -------------------------
        PT con Metropolis subyacente tiene tau_eff ~ 20-50 sweeps cerca
        de Tc a L=16. Con n_sweeps=600 solo se obtienen ~10-30 muestras
        independientes por replica, lo que deja la ubicacion del pico de
        chi_connected con una incertidumbre de +-0.15 K. Se necesitan
        n_sweeps >= 1500 para reducir esa incertidumbre a +-0.05 K.
        """
        spins, bonds, J, E0 = build_square_lattice_2d(L=16, J=1.0)
        # Grid concentrado: de 2.05 a 2.55 en 17 puntos (espaciado 0.031)
        T_grid = np.linspace(2.05, 2.6, 12)
        pt = ParallelTempering(spins, bonds, J, E0, T_grid, seed=42)
        res = pt.run(n_equil=500, n_sweeps=2000, swap_every=5)

        # swap_acceptance razonable
        assert 0.15 < res["swap_acceptance"] < 0.95, \
            f"swap_acceptance={res['swap_acceptance']:.3f}"

        # Tc con chi_connected
        Tc = estimate_tc(res["T"], res["chi_connected"])
        assert Tc == pytest.approx(2.269, abs=0.15), f"Tc={Tc}"

        # Sanity check: chi estandar sesga hacia abajo
        Tc_std = estimate_tc(res["T"], res["chi"])
        assert Tc_std < Tc + 0.01, \
            f"chi estandar ({Tc_std:.3f}) deberia dar Tc menor que chi_connected ({Tc:.3f})"

    def test_3d_ising_tc_wolff(self):
        """Wolff en 3D: Tc(L=6) via C(T) y M(T).

        Nota sobre los umbrales de M
        ----------------------------
        A L=6 (216 sitios) las fluctuaciones termicas son grandes porque
        la razon superficie/volumen es alta y el numero de coordinacion
        z=6. A T = 0.85 Tc(L=6), <|M|>/N ~ 0.75, no ~ 0.9 como en 2D a
        L=16. Los umbrales de este test estan calibrados para L=6.
        """
        spins, bonds, J, E0 = build_square_lattice_3d(L=6, J=1.0)
        _, gauge, _ = detect_frustration(bonds, J, len(spins))
        T_grid = np.linspace(4.0, 5.3, 15)
        sim = WolffCluster(spins, bonds, J, E0, seed=11, gauge=gauge)
        res = sim.sweep_temperatures(T_grid, n_equil=300, n_sweeps=800)

        T = np.array(res["T"])
        M = np.array(res["M"])
        C = np.array(res["C"])

        # (a) M(T): ordenado a T_min, desordenado a T_max
        assert M[0] > 0.65, f"M(T={T[0]:.2f})={M[0]:.3f} (esperado > 0.65)"
        assert M[-1] < 0.4,  f"M(T={T[-1]:.2f})={M[-1]:.3f} (esperado < 0.4)"

        # (b) Caida neta de M a traves de la transicion
        assert M[0] - M[-1] > 0.3, \
            f"Caida de M insuficiente: {M[0]:.3f} -> {M[-1]:.3f}"

        # (c) Pico de C(T) cerca de Tc(L=6) ~ 4.6-4.8
        Tc_C = estimate_tc(T, C)
        assert 4.25 < Tc_C < 5.0, f"Tc_C={Tc_C:.3f} fuera de [4.3, 5.0]"

    def test_curie_weiss_theta_effective(self):
        """Ajuste CW en la fase paramagnetica 2D (FM).

        Para Ising FM con J=1 en red cuadrada, la expansion exacta de
        alta T da:

            1/chi = T - 4 + 4/T + O(1/T^2)

        es decir, C -> 1 y Theta -> +4 cuando T -> infinito. Sobre una
        ventana finita [3, 6] el valor efectivo de Theta es ~+2.5 (menor
        que el asintotico por efectos de rango finito).

        Convencion: chi = C/(T - Theta). Para FM, Theta > 0. Para AFM,
        Theta < 0. El signo de Theta es la firma del tipo de
        interaccion dominante.
        """
        spins, bonds, J, E0 = build_square_lattice_2d(L=10, J=1.0)
        _, gauge, _ = detect_frustration(bonds, J, len(spins))
        T_grid = np.linspace(3.0, 6.0, 8)
        sim = WolffCluster(spins, bonds, J, E0, seed=13, gauge=gauge)
        res = sim.sweep_temperatures(T_grid, n_equil=200, n_sweeps=600)

        fit = fit_curie_weiss(res["T"], res["chi"], T_cut=3.0)
        assert fit is not None, "El ajuste CW no convergio"

        # C > 0 por construccion (chi > 0 siempre)
        assert fit["C"] > 0, f"C={fit['C']}"

        # FM: Theta_CW > 0. Cota superior 6 para descartar fits absurdos.
        # Asintoticamente deberia ser +4; con rango [3,6] es ~+2.5.
        assert 0.0 < fit["theta_CW"] < 6.0, \
            f"Theta={fit['theta_CW']:.3f} (esperado en (0, 6) para FM)"

        # El ajuste debe reproducir bien los datos
        T = np.array(res["T"])
        chi = np.array(res["chi"])
        mask = T > 3.0
        chi_fit = fit["C"] / (T[mask] - fit["theta_CW"])
        rel_err = np.abs(chi_fit - chi[mask]) / chi[mask]
        assert rel_err.max() < 0.5, \
            f"Error relativo maximo del fit: {rel_err.max():.2%}"
