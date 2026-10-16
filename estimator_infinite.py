from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import logm

from drift_estimation import center_across_runs, validate_panel
from numba_kernels import pooled_linear_moments_kernel


@dataclass
class InfiniteHorizonEstimate:
    G_hat: np.ndarray
    B_hat: np.ndarray
    design_condition: float
    residual_rms: float
    n_transitions: int
    method: str
    transition_hat: np.ndarray | None = None


def _panel_moments(observations: np.ndarray, times: np.ndarray, dt: float):
    obs, _ = validate_panel(observations, times, dt)
    Z = np.ascontiguousarray(obs, dtype=float)
    gram, increment_cross = pooled_linear_moments_kernel(Z)
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > 1e14:
        raise RuntimeError(f"Infinite-horizon regression is ill-conditioned: {cond:.2e}.")
    return Z, gram, increment_cross, cond


def _network_from_B(B: np.ndarray) -> np.ndarray:
    G = -(B @ B)
    np.fill_diagonal(G, 0.0)
    return G


def estimate_G_infinite_exact(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
) -> InfiniteHorizonEstimate:
    """Exact discrete-OU transition regression: A=exp(B dt), then B=log(A)/dt."""
    Z, gram, inc_cross, cond = _panel_moments(observations, times, dt)
    A = np.linalg.solve(gram.T, (gram + inc_cross).T).T
    Bc = logm(A) / float(dt)
    imag = float(np.max(np.abs(np.imag(Bc))))
    if imag > 1e-7:
        raise RuntimeError(f"Estimated transition matrix has no numerically real logarithm (imag={imag:.2e}).")
    B = np.real(Bc)

    residual = Z[:, 1:] - np.einsum("ij,rtj->rti", A, Z[:, :-1])
    return InfiniteHorizonEstimate(
        G_hat=_network_from_B(B),
        B_hat=B,
        transition_hat=A,
        design_condition=cond,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        n_transitions=int(Z.shape[0] * (Z.shape[1] - 1)),
        method="exact OU transition",
    )


def estimate_G_infinite_kutoyants(
    observations: np.ndarray,
    times: np.ndarray,
    dt: float,
) -> InfiniteHorizonEstimate:
    """Discrete continuous-time MLE/estimating-equation approximation from ergodic diffusion theory."""
    Z, gram, inc_cross, cond = _panel_moments(observations, times, dt)
    B = np.linalg.solve(gram.T, inc_cross.T).T / float(dt)
    increment = Z[:, 1:] - Z[:, :-1]
    residual = increment - float(dt) * np.einsum("ij,rtj->rti", B, Z[:, :-1])

    return InfiniteHorizonEstimate(
        G_hat=_network_from_B(B),
        B_hat=B,
        transition_hat=None,
        design_condition=cond,
        residual_rms=float(np.sqrt(np.mean(residual**2)) / dt),
        n_transitions=int(Z.shape[0] * (Z.shape[1] - 1)),
        method="Kutoyants continuous-time MLE",
    )


# Backward-compatible name used by earlier project versions.
def estimate_G_infinite(observations: np.ndarray, times: np.ndarray, dt: float) -> InfiniteHorizonEstimate:
    return estimate_G_infinite_exact(observations, times, dt)
