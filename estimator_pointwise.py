from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drift_estimation import PointwiseDriftEstimate, estimate_pointwise_drift, trapezoid_matrix, two_fold_indices, validate_panel
from numba_kernels import crossfit_B2_kernel


@dataclass
class PointwiseOLSEstimate:
    G_hat: np.ndarray
    G_hat_cf: np.ndarray
    B_hat: np.ndarray
    B2_crossfit: np.ndarray
    direct_times: np.ndarray
    fold_sizes: tuple[int, int]


def _direct_average(B: np.ndarray, B2: np.ndarray, times: np.ndarray, dt: float) -> np.ndarray:
    if len(times) < 3:
        raise ValueError("Pointwise direct recovery needs at least three drift time points.")
    Bdot = (B[2:] - B[:-2]) / (2.0 * dt)
    t = times[1:-1]
    G = trapezoid_matrix(-Bdot - B2[1:-1], t) / float(t[-1] - t[0])
    np.fill_diagonal(G, 0.0)
    return G


def estimate_G_pointwise_ols(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
    *,
    split_seed: int = 1729,
    pointwise_full: PointwiseDriftEstimate | None = None,
) -> PointwiseOLSEstimate:
    """Pointwise OLS recovery with same-sample and cross-fitted B^2."""
    obs, times = validate_panel(observations, times, dt)
    R, _, N = obs.shape
    full = estimate_pointwise_drift(obs, times, dt) if pointwise_full is None else pointwise_full

    a, b = two_fold_indices(R, seed=split_seed)
    if min(len(a), len(b)) <= N + 1:
        raise ValueError(f"Each cross-fitting fold must contain more runs than populations; sizes={len(a)}, {len(b)}, N={N}.")
    B_a = estimate_pointwise_drift(obs[a], times, dt).B_hat
    B_b = estimate_pointwise_drift(obs[b], times, dt).B_hat

    B = full.B_hat
    B2 = np.matmul(B, B)
    B2_cf = crossfit_B2_kernel(np.ascontiguousarray(B_a), np.ascontiguousarray(B_b))
    return PointwiseOLSEstimate(
        G_hat=_direct_average(B, B2, full.times, dt),
        G_hat_cf=_direct_average(B, B2_cf, full.times, dt),
        B_hat=B.copy(),
        B2_crossfit=B2_cf,
        direct_times=full.times[1:-1].copy(),
        fold_sizes=(len(a), len(b)),
    )
