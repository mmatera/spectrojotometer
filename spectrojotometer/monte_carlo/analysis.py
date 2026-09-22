"""Analisis de resultados de Monte Carlo Ising."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def estimate_tc(T, chi):
    """Estima Tc como el maximo de chi con interpolacion parabolica."""
    T = np.asarray(T, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    if T.size == 0:
        raise ValueError("T vacio")
    idx = int(np.argmax(chi))
    if 0 < idx < len(T) - 1:
        c = np.polyfit(T[idx - 1: idx + 2], chi[idx - 1: idx + 2], 2)
        if c[0] < 0:
            return float(-c[1] / (2 * c[0]))
    return float(T[idx])


def fit_curie_weiss(T, chi, T_cut):
    """Ajusta chi = C / (T - Theta_CW) para T > T_cut.

    Returns None si no hay suficientes puntos validos o el fit falla.
    """
    T = np.asarray(T, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    mask = (T > T_cut) & (chi > 1e-12)
    if mask.sum() < 3:
        return None

    T_m = T[mask]
    chi_m = chi[mask]

    # Semilla por ajuste lineal de 1/chi vs T
    inv_chi = 1.0 / chi_m
    slope, intercept = np.polyfit(T_m, inv_chi, 1)
    if slope == 0:
        return None
    C0 = 1.0 / slope
    theta0 = -intercept * C0

    def cw(T, C, theta):
        return C / (T - theta)

    try:
        popt, pcov = curve_fit(cw, T_m, chi_m, p0=[C0, theta0], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except Exception:
        return None

    return {
        "C": float(popt[0]),
        "C_err": float(perr[0]),
        "theta_CW": float(popt[1]),
        "theta_err": float(perr[1]),
        "T_range": (float(T_m.min()), float(T_m.max())),
        "n_points": int(mask.sum()),
    }


def binder_crossing(T_list, U4_list, L_list):
    """Estima Tc por cruce de cumulantes de Binder entre los dos L mayores.

    Devuelve None si las curvas no se cruzan en el rango solapado.
    """
    order = np.argsort(L_list)
    i1, i2 = int(order[-2]), int(order[-1])
    T1, T2 = np.asarray(T_list[i1]), np.asarray(T_list[i2])
    U1, U2 = np.asarray(U4_list[i1]), np.asarray(U4_list[i2])

    T_lo = max(T1.min(), T2.min())
    T_hi = min(T1.max(), T2.max())
    if T_hi <= T_lo:
        return None

    f1 = interp1d(T1, U1, kind="cubic", bounds_error=False, fill_value="extrapolate")
    f2 = interp1d(T2, U2, kind="cubic", bounds_error=False, fill_value="extrapolate")

    def diff(T):
        return float(f1(T) - f2(T))

    # Buscar cambio de signo en el rango de solapamiento
    grid = np.linspace(T_lo, T_hi, 200)
    vals = np.array([diff(t) for t in grid])
    sign = np.sign(vals)
    changes = np.where(np.diff(sign) != 0)[0]
    if changes.size == 0:
        return None
    k = int(changes[0])
    try:
        return float(brentq(diff, grid[k], grid[k + 1]))
    except ValueError:
        return None


def frustration_factor(Tc, J_vals, N):
    """f = Tc / (<sum_bonds |J|>/N). Bajo si hay frustration."""
    J_vals = np.asarray(J_vals, dtype=np.float64)
    if J_vals.size == 0 or N == 0:
        return None
    scale = float(np.sum(np.abs(J_vals))) / N
    if scale <= 0:
        return None
    return float(Tc / scale)
