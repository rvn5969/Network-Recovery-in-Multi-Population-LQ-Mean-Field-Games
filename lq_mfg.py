from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from config import ExperimentConfig
from numba_kernels import build_continuum_B_kernel, sensitivity_rhs_kernel


@dataclass
class LQMFGSolution:
    """Continuum equilibrium for the generalized scalar LQ network model.

    This solver is retained as an oracle/theory check. Synthetic observations in
    the project are generated only from the exact finite-player Nash game.
    """

    times: np.ndarray
    P: np.ndarray
    D: np.ndarray
    ell: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    S: np.ndarray
    B_eff: np.ndarray


def _interp_vector(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    j = int(np.searchsorted(times, t))
    t0, t1 = times[j - 1], times[j]
    w = (t - t0) / (t1 - t0)
    return (1.0 - w) * values[j - 1] + w * values[j]


def _solve_scalar_riccati(cfg: ExperimentConfig) -> np.ndarray:
    """Pdot=-2aP-q+(bP+s)^2/r, P(T)=qbar."""

    def rhs(_t: float, p: np.ndarray) -> np.ndarray:
        return (
            -2.0 * cfg.a * p
            - cfg.q_state
            + (cfg.b * p + cfg.s_state_control) ** 2 / cfg.r_control
        )

    sol = solve_ivp(
        rhs,
        (cfg.T, 0.0),
        cfg.q_terminal.copy(),
        t_eval=cfg.times[::-1],
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"Scalar Riccati solve failed: {sol.message}")
    return sol.y[:, ::-1].T


def _solve_mean_bvp(cfg: ExperimentConfig, P: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve mdot=-Dm-Mell+delta, elldot=Dell+Cm+eta."""
    times = cfg.times
    N = cfg.n_pop
    M = cfg.M_control
    delta = cfg.drift_intercept - cfg.b * cfg.rho_control / cfg.r_control

    # Eta depends on P(t).
    eta = (
        (cfg.b[None, :] * P + cfg.s_state_control[None, :])
        * cfg.rho_control[None, :]
        / cfg.r_control[None, :]
        - cfg.drift_intercept[None, :] * P
        - cfg.beta_state[None, :]
    )

    def ode(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        D_t = np.vstack([np.interp(t, times, D[:, i]) for i in range(N)])
        eta_t = np.vstack([np.interp(t, times, eta[:, i]) for i in range(N)])
        m = y[:N]
        ell = y[N:]
        m_dot = -D_t * m - M[:, None] * ell + delta[:, None]
        ell_dot = D_t * ell + cfg.C @ m + eta_t
        return np.vstack([m_dot, ell_dot])

    # ell(T)=beta_bar-H m(T).
    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        terminal_residual = yb[N:] - cfg.beta_terminal + cfg.H_terminal @ yb[:N]
        return np.r_[ya[:N] - cfg.initial_mean, terminal_residual]

    guess = np.zeros((2 * N, len(times)), dtype=float)
    guess[:N] = cfg.initial_mean[:, None]
    guess[N:] = cfg.beta_terminal[:, None]

    sol = solve_bvp(
        ode,
        bc,
        times,
        guess,
        tol=1e-8,
        max_nodes=max(2000, 10 * len(times)),
        verbose=0,
    )
    if not sol.success:
        raise RuntimeError(f"Mean-field boundary value solve failed: {sol.message}")
    vals = sol.sol(times)
    return vals[:N].T, vals[N:].T


def _solve_variance(cfg: ExperimentConfig, D: np.ndarray) -> np.ndarray:
    times = cfg.times

    def rhs(t: float, v: np.ndarray) -> np.ndarray:
        Dt = _interp_vector(times, D, t)
        return -2.0 * Dt * v + cfg.sigma**2

    sol = solve_ivp(
        rhs,
        (0.0, cfg.T),
        cfg.initial_std**2,
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"Variance solve failed: {sol.message}")
    return sol.y.T


def _solve_sensitivity(cfg: ExperimentConfig, D: np.ndarray) -> np.ndarray:
    """Solve Sdot=DS+SD+SMS+C, S(T)=-H."""
    times = cfg.times
    N = cfg.n_pop
    M = cfg.M_control

    def rhs(t: float, s_flat: np.ndarray) -> np.ndarray:
        Dt = _interp_vector(times, D, t)
        return sensitivity_rhs_kernel(
            np.asarray(s_flat, dtype=float),
            np.asarray(Dt, dtype=float),
            np.asarray(M, dtype=float),
            np.asarray(cfg.C, dtype=float),
        )

    sol = solve_ivp(
        rhs,
        (cfg.T, 0.0),
        (-cfg.H_terminal).ravel(),
        t_eval=times[::-1],
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"Sensitivity Riccati solve failed: {sol.message}")
    return sol.y[:, ::-1].T.reshape(len(times), N, N)


def solve_lq_mfg(cfg: ExperimentConfig) -> LQMFGSolution:
    """Solve the generalized continuum LQ MFG; not used to generate data."""
    cfg.validate()
    P = _solve_scalar_riccati(cfg)
    D = (
        cfg.b[None, :] * (cfg.b[None, :] * P + cfg.s_state_control[None, :])
        / cfg.r_control[None, :]
        - cfg.a[None, :]
    )
    mean, ell = _solve_mean_bvp(cfg, P, D)
    variance = _solve_variance(cfg, D)
    S = _solve_sensitivity(cfg, D)
    B = build_continuum_B_kernel(
        np.ascontiguousarray(D, dtype=float),
        np.ascontiguousarray(S, dtype=float),
        np.ascontiguousarray(cfg.M_control, dtype=float),
    )
    return LQMFGSolution(
        times=cfg.times.copy(),
        P=P,
        D=D,
        ell=ell,
        mean=mean,
        variance=variance,
        S=S,
        B_eff=B,
    )
