from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drift_estimation import center_across_runs, trapezoid_matrix, two_fold_indices, validate_panel
from numba_kernels import basis_moment_kernel, crossfit_B2_kernel


@dataclass
class MartingaleBasisEstimate:
    G_hat: np.ndarray
    G_hat_cf: np.ndarray
    B_hat: np.ndarray
    B_dot_hat: np.ndarray
    coefficients: np.ndarray
    B2_crossfit: np.ndarray
    drift_times: np.ndarray
    degree: int
    design_condition: float
    fold_sizes: tuple[int, int]
    residual_rms: float


def _legendre_basis(times: np.ndarray, degree: int, derivative: bool = False) -> np.ndarray:
    if degree < 0 or degree > 3:
        raise ValueError("Current compact implementation supports Legendre degree 0 through 3.")
    t = np.asarray(times, dtype=float)
    H = float(t[-1] - t[0])
    if H <= 0.0:
        raise ValueError("Basis interval has zero width.")
    x = 2.0 * (t - t[0]) / H - 1.0
    if derivative:
        cols = [np.zeros_like(x)]
        if degree >= 1:
            cols.append(np.full_like(x, 2.0 / H))
        if degree >= 2:
            cols.append(6.0 * x / H)
        if degree >= 3:
            cols.append((15.0 * x * x - 3.0) / H)
    else:
        cols = [np.ones_like(x)]
        if degree >= 1:
            cols.append(x)
        if degree >= 2:
            cols.append(0.5 * (3.0 * x * x - 1.0))
        if degree >= 3:
            cols.append(0.5 * (5.0 * x * x * x - 3.0 * x))
    return np.column_stack(cols)


def _fit_basis_drift(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    obs, times = validate_panel(observations, times, dt)
    Z = np.ascontiguousarray(center_across_runs(obs), dtype=float)
    drift_times = times[:-1]
    phi = np.ascontiguousarray(_legendre_basis(drift_times, degree), dtype=float)
    gram, cross, yss = basis_moment_kernel(Z, phi, float(dt))

    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > 1e14:
        raise RuntimeError(f"Martingale basis regression is ill-conditioned: {cond:.2e}.")

    coef = np.linalg.solve(gram, cross)
    Kb, N = degree + 1, Z.shape[2]
    theta = coef.reshape(Kb, N, N).transpose(0, 2, 1)
    B = np.einsum("tk,kij->tij", phi, theta)
    Bdot = np.einsum("tk,kij->tij", _legendre_basis(drift_times, degree, True), theta)

    sse = np.empty(N, dtype=float)
    for i in range(N):
        c = coef[:, i]
        sse[i] = max(float(yss[i] - 2.0 * c @ cross[:, i] + c @ gram @ c), 0.0)
    rms = float(np.sqrt(np.mean(sse) / (Z.shape[0] * (Z.shape[1] - 1))) / dt)
    return B, Bdot, theta, cond, rms


def _direct_identity(Bdot: np.ndarray, B2: np.ndarray, times: np.ndarray) -> np.ndarray:
    H = float(times[-1] - times[0])
    G = trapezoid_matrix(-Bdot - B2, times) / H
    np.fill_diagonal(G, 0.0)
    return G


def estimate_G_martingale_basis(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
    *,
    degree: int = 3,
    split_seed: int = 1729,
) -> MartingaleBasisEstimate:
    """Degree-3 martingale drift fit with plain and cross-fitted network recovery."""
    obs, times = validate_panel(observations, times, dt)
    R = obs.shape[0]
    fold_a, fold_b = two_fold_indices(R, seed=split_seed)

    B, Bdot, theta, cond, rms = _fit_basis_drift(obs, times, dt, degree)
    B_a, _, _, _, _ = _fit_basis_drift(obs[fold_a], times, dt, degree)
    B_b, _, _, _, _ = _fit_basis_drift(obs[fold_b], times, dt, degree)
    B2_cf = crossfit_B2_kernel(
        np.ascontiguousarray(B_a, dtype=float),
        np.ascontiguousarray(B_b, dtype=float),
    )
    B2_same = np.matmul(B, B)
    drift_times = times[:-1]

    return MartingaleBasisEstimate(
        G_hat=_direct_identity(Bdot, B2_same, drift_times),
        G_hat_cf=_direct_identity(Bdot, B2_cf, drift_times),
        B_hat=B,
        B_dot_hat=Bdot,
        coefficients=theta,
        B2_crossfit=B2_cf,
        drift_times=drift_times.copy(),
        degree=degree,
        design_condition=cond,
        fold_sizes=(len(fold_a), len(fold_b)),
        residual_rms=rms,
    )
