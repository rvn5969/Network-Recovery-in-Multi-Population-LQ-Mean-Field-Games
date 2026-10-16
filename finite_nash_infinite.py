from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import root

from config import ExperimentConfig
from finite_nash import _as_K_vector, _make_reduced_rhs
from numba_kernels import finite_nash_B_path_kernel, simulate_transition_panel_kernel


@dataclass
class InfiniteFiniteNashSolution:
    K_by_population: np.ndarray
    reduced_P: np.ndarray
    B_eff: np.ndarray
    algebraic_residual: float
    spectral_abscissa: float
    nfev: int


@dataclass
class InfiniteFiniteNashPanel:
    times: np.ndarray
    empirical_means: np.ndarray
    B_eff: np.ndarray
    K_by_population: np.ndarray
    seed: int


def solve_infinite_finite_nash(
    cfg: ExperimentConfig,
    *,
    K: int | np.ndarray | None = None,
    initial_P: np.ndarray | None = None,
    tol: float = 1e-9,
) -> InfiniteFiniteNashSolution:
    """Solve the stabilizing algebraic finite-player Nash Riccati system."""
    cfg.validate()
    N = cfg.n_pop
    Kvec = _as_K_vector(cfg.K_vector if K is None else K, N)
    d = N + 1
    rhs, own, Abar, Qbar, _ = _make_reduced_rhs(cfg, Kvec)
    tri = np.triu_indices(d)

    def pack(P: np.ndarray) -> np.ndarray:
        return P[:, tri[0], tri[1]].ravel()

    def unpack(x: np.ndarray) -> np.ndarray:
        vals = x.reshape(N, -1)
        P = np.zeros((N, d, d), dtype=float)
        P[:, tri[0], tri[1]] = vals
        P[:, tri[1], tri[0]] = vals
        return P

    def residual(x: np.ndarray) -> np.ndarray:
        R = rhs(0.0, unpack(x).ravel()).reshape(N, d, d)
        return R[:, tri[0], tri[1]].ravel()

    guesses = [np.zeros(N * len(tri[0]), dtype=float)]
    if initial_P is not None and np.asarray(initial_P).shape == (N, d, d):
        guesses.insert(0, pack(np.asarray(initial_P, dtype=float)))

    best = None
    for guess in guesses:
        fit = root(residual, guess, method="hybr", options={"xtol": tol})
        P = unpack(fit.x)
        B, Bdot = finite_nash_B_path_kernel(
            P[None, ...],
            np.ascontiguousarray(cfg.b, dtype=float),
            np.ascontiguousarray(cfg.r_control, dtype=float),
            np.ascontiguousarray(cfg.s_state_control, dtype=float),
            np.ascontiguousarray(cfg.a, dtype=float),
            Kvec,
            own,
            Abar,
            Qbar,
        )
        err = float(max(np.max(np.abs(residual(fit.x))), np.max(np.abs(Bdot))))
        alpha = float(np.max(np.real(np.linalg.eigvals(B[0]))))
        candidate = (err, alpha, fit, P, B[0])
        if best is None or err < best[0]:
            best = candidate
        if err <= 100.0 * tol and alpha < -1e-8:
            best = candidate
            break

    assert best is not None
    err, alpha, fit, P, B = best
    if err > 100.0 * tol or alpha >= -1e-8:
        raise RuntimeError(
            "No stabilizing infinite-horizon finite-Nash solution was found for this config "
            f"(algebraic residual={err:.2e}, spectral abscissa={alpha:.2e})."
        )

    return InfiniteFiniteNashSolution(
        K_by_population=Kvec.copy(),
        reduced_P=P,
        B_eff=B.copy(),
        algebraic_residual=err,
        spectral_abscissa=alpha,
        nfev=int(fit.nfev),
    )


def simulate_infinite_nash_centered_means(
    cfg: ExperimentConfig,
    sol: InfiniteFiniteNashSolution,
    *,
    R: int | None = None,
    seed: int | None = None,
) -> InfiniteFiniteNashPanel:
    """Exact-grid simulation of the stationary closed-loop aggregate OU process."""
    R = 1 if R is None else int(R)
    K = sol.K_by_population
    rng_seed = cfg.seed + 37001 if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)

    B = sol.B_eff
    A = expm(B * cfg.dt)
    Qc = np.diag(cfg.sigma**2 / K)
    Sigma = solve_continuous_lyapunov(B, -Qc)
    Qd = 0.5 * (Sigma - A @ Sigma @ A.T + (Sigma - A @ Sigma @ A.T).T)
    vals, vecs = np.linalg.eigh(Qd)
    L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))

    initial = rng.normal(0.0, cfg.initial_std / np.sqrt(K), size=(R, cfg.n_pop))
    noise = rng.standard_normal((R, cfg.n_steps, cfg.n_pop))
    X = simulate_transition_panel_kernel(
        np.ascontiguousarray(initial),
        np.ascontiguousarray(noise),
        np.ascontiguousarray(A),
        np.ascontiguousarray(L),
    )

    return InfiniteFiniteNashPanel(
        times=cfg.times.copy(),
        empirical_means=X,
        B_eff=B.copy(),
        K_by_population=K.copy(),
        seed=rng_seed,
    )
