from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numba_kernels import pointwise_ols_kernel


@dataclass
class PointwiseDriftEstimate:
    """Cross-run OLS estimate of B(t_n) at every usable time."""

    B_hat: np.ndarray
    times: np.ndarray
    centered_observations: np.ndarray
    design_ranks: np.ndarray
    condition_numbers: np.ndarray
    residual_rms: np.ndarray
    n_runs: int
    n_time_points: int
    n_populations: int


def validate_panel(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(observations, dtype=float)
    t = np.asarray(times, dtype=float)

    if X.ndim != 3:
        raise ValueError("observations must have shape (runs, time, population).")
    if t.ndim != 1 or X.shape[1] != len(t):
        raise ValueError("times must be one-dimensional and match observations.shape[1].")
    if len(t) < 3:
        raise ValueError("Need at least three observation times.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(t)):
        raise ValueError("Inputs contain non-finite values.")

    diffs = np.diff(t)
    if not np.allclose(diffs, dt, rtol=1e-7, atol=1e-10):
        raise ValueError("Current estimators require an equally spaced grid matching dt.")
    return X, t


def center_across_runs(observations: np.ndarray) -> np.ndarray:
    X = np.asarray(observations, dtype=float)
    return X - X.mean(axis=0, keepdims=True)


def estimate_pointwise_drift(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
) -> PointwiseDriftEstimate:
    """Estimate B(t_n) from repeated trajectories using cross-sectional OLS.

    At each t_n, after centering across independent runs,
        (X_{n+1}^{(r)} - X_n^{(r)}) / dt = B(t_n) X_n^{(r)} + noise.

    The repeated small least-squares problems are executed by a Numba-compiled
    kernel using LAPACK through ``np.linalg.lstsq``. The estimator receives no
    game parameters and no continuum mean path.
    """
    obs, times = validate_panel(observations, times, dt)
    centered = np.ascontiguousarray(center_across_runs(obs), dtype=float)
    R, L, N = centered.shape
    n_steps = L - 1

    if R <= N + 1:
        raise ValueError(
            f"Pointwise drift regression needs more runs than populations; R={R}, N={N}."
        )

    B_hat, ranks, conds, rms = pointwise_ols_kernel(centered, float(dt))

    return PointwiseDriftEstimate(
        B_hat=B_hat,
        times=times[:-1].copy(),
        centered_observations=centered,
        design_ranks=ranks.astype(int, copy=False),
        condition_numbers=conds,
        residual_rms=rms,
        n_runs=R,
        n_time_points=n_steps,
        n_populations=N,
    )


def two_fold_indices(n_runs: int, seed: int = 1729) -> tuple[np.ndarray, np.ndarray]:
    if n_runs < 2:
        raise ValueError("Need at least two runs for sample splitting.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_runs)
    cut = n_runs // 2
    a, b = np.sort(perm[:cut]), np.sort(perm[cut:])
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both sample-split folds must be nonempty.")
    return a, b


def trapezoid_matrix(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape[0] != len(times):
        raise ValueError("values and times disagree along the integration axis.")
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=times, axis=0)
    return np.trapz(values, x=times, axis=0)
