from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from config import ExperimentConfig
from numba_kernels import (
    finite_nash_B_path_kernel,
    finite_nash_rhs_kernel,
    simulate_linear_panel_kernel,
)


@dataclass
class ExactFiniteNashSolution:
    """Exact symmetry-reduced finite-player Nash quadratic solution."""

    times: np.ndarray
    K_by_population: np.ndarray
    reduced_P: np.ndarray          # (time, N, N+1, N+1)
    B_eff: np.ndarray              # exact centered empirical-mean drift
    B_dot: np.ndarray
    nfev: int
    success: bool
    message: str
    rtol: float
    atol: float


@dataclass
class ExactFiniteNashPanel:
    """Repeated centered empirical-mean trajectories under exact finite Nash."""

    times: np.ndarray
    empirical_means: np.ndarray    # (R, time, N), centered-equivalent aggregate data
    B_eff: np.ndarray
    K_by_population: np.ndarray
    seed: int


def _as_K_vector(K: int | np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(K, dtype=int)
    if arr.ndim == 0:
        arr = np.full(n, int(arr), dtype=int)
    else:
        arr = arr.reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"K must be scalar or length {n}; got {arr.shape}.")
    if np.any(arr < 2):
        raise ValueError("Every population size must be at least 2.")
    return np.ascontiguousarray(arr)


def _reduced_geometry(
    cfg: ExperimentConfig,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact reduced quadratic data in basis H_i=[e_p,mu_1,...,mu_N]."""
    N = cfg.n_pop
    d = N + 1
    own_eval = np.zeros((N, d), dtype=float)
    Abar_diag = np.zeros((N, d), dtype=float)
    running_Qbar = np.zeros((N, d, d), dtype=float)
    terminal_Qbar = np.zeros((N, d, d), dtype=float)

    for i in range(N):
        # H_i^T e_p.
        own_eval[i, 0] = 1.0
        own_eval[i, i + 1] = 1.0 / K[i]
        Abar_diag[i] = np.r_[cfg.a[i], cfg.a]

        # 0.5 [x_p,m]^T Qbar [x_p,m]
        # = 0.5 q x_p^2 - x_p C_i^T m + 0.5 m^T R_i m.
        running_Qbar[i, 0, 0] = cfg.q_state[i]
        running_Qbar[i, 0, 1:] = -cfg.C[i]
        running_Qbar[i, 1:, 0] = -cfg.C[i]
        running_Qbar[i, 1:, 1:] = cfg.mean_quadratic[i]

        terminal_Qbar[i, 0, 0] = cfg.q_terminal[i]
        terminal_Qbar[i, 0, 1:] = -cfg.H_terminal[i]
        terminal_Qbar[i, 1:, 0] = -cfg.H_terminal[i]
        terminal_Qbar[i, 1:, 1:] = cfg.terminal_mean_quadratic[i]

    return (
        np.ascontiguousarray(own_eval),
        np.ascontiguousarray(Abar_diag),
        np.ascontiguousarray(running_Qbar),
        np.ascontiguousarray(terminal_Qbar),
    )


def _make_reduced_rhs(cfg: ExperimentConfig, K: np.ndarray):
    N = cfg.n_pop
    d = N + 1
    own_eval, Abar_diag, running_Qbar, terminal_Qbar = _reduced_geometry(cfg, K)

    b = np.ascontiguousarray(cfg.b, dtype=float)
    r = np.ascontiguousarray(cfg.r_control, dtype=float)
    s = np.ascontiguousarray(cfg.s_state_control, dtype=float)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return finite_nash_rhs_kernel(
            np.asarray(y, dtype=float),
            N,
            d,
            b,
            r,
            s,
            K,
            own_eval,
            Abar_diag,
            running_Qbar,
        )

    return rhs, own_eval, Abar_diag, running_Qbar, terminal_Qbar


def solve_exact_finite_nash_mean_drift(
    cfg: ExperimentConfig,
    *,
    K: int | np.ndarray | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    method: str = "DOP853",
) -> ExactFiniteNashSolution:
    """Solve the exact generalized finite-player Nash Hessian system.

    Affine primitive terms (d, rho, beta, gamma, constants) change only affine
    feedback/intercept terms.  Cross-run centering removes those terms, so the
    exact centered aggregate drift B^K depends only on the quadratic Hessian
    system solved here.
    """
    cfg.validate()
    N = cfg.n_pop
    Kvec = _as_K_vector(cfg.K_vector if K is None else K, N)
    d = N + 1

    rhs, own_eval, Abar_diag, running_Qbar, terminal_Qbar = _make_reduced_rhs(cfg, Kvec)
    terminal = terminal_Qbar.copy()

    # Trigger JIT before repeated SciPy callbacks.
    _ = rhs(cfg.T, terminal.ravel())

    sol = solve_ivp(
        rhs,
        (cfg.T, 0.0),
        terminal.ravel(),
        t_eval=cfg.times[::-1],
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if not sol.success:
        raise RuntimeError(
            "Exact finite-player Nash Riccati solve failed on [0,T]. "
            f"K={Kvec.tolist()}, message={sol.message}"
        )

    reduced_P = np.ascontiguousarray(
        sol.y[:, ::-1].T.reshape(len(cfg.times), N, d, d), dtype=float
    )
    B, Bdot = finite_nash_B_path_kernel(
        reduced_P,
        np.ascontiguousarray(cfg.b, dtype=float),
        np.ascontiguousarray(cfg.r_control, dtype=float),
        np.ascontiguousarray(cfg.s_state_control, dtype=float),
        np.ascontiguousarray(cfg.a, dtype=float),
        Kvec,
        own_eval,
        Abar_diag,
        running_Qbar,
    )

    return ExactFiniteNashSolution(
        times=cfg.times.copy(),
        K_by_population=Kvec.copy(),
        reduced_P=reduced_P,
        B_eff=B,
        B_dot=Bdot,
        nfev=int(sol.nfev),
        success=bool(sol.success),
        message=str(sol.message),
        rtol=float(rtol),
        atol=float(atol),
    )


def simulate_exact_nash_centered_means(
    cfg: ExperimentConfig,
    sol: ExactFiniteNashSolution,
    *,
    R: int | None = None,
    seed: int | None = None,
) -> ExactFiniteNashPanel:
    """Simulate exact centered empirical means under the finite Nash feedback.

    Population symmetry gives an exact affine aggregate SDE.  Subtracting the
    deterministic affine mean path leaves

        dZ = B^K(t) Z dt + diag(sigma_i/sqrt(K_i)) dWbar.

    Thus no continuum fluctuation approximation is used here.
    """
    R = cfg.R_runs if R is None else int(R)
    if R < 2:
        raise ValueError("Need at least two repeated runs.")
    K = sol.K_by_population
    if len(K) != cfg.n_pop:
        raise ValueError("Finite-Nash solution dimension does not match cfg.")

    rng_seed = cfg.seed + 7001 if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)
    times = sol.times
    dt = cfg.dt
    N = cfg.n_pop

    initial = rng.normal(
        loc=0.0,
        scale=cfg.initial_std / np.sqrt(K),
        size=(R, N),
    )
    noise_t = rng.standard_normal((len(times) - 1, R, N))
    noise = np.ascontiguousarray(np.transpose(noise_t, (1, 0, 2)))
    noise_scale = np.sqrt(dt) * cfg.sigma / np.sqrt(K)

    means = simulate_linear_panel_kernel(
        np.ascontiguousarray(initial, dtype=float),
        noise,
        np.ascontiguousarray(sol.B_eff, dtype=float),
        float(dt),
        np.ascontiguousarray(noise_scale, dtype=float),
    )
    return ExactFiniteNashPanel(
        times=times.copy(),
        empirical_means=means,
        B_eff=sol.B_eff.copy(),
        K_by_population=K.copy(),
        seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# Optional brute-force validation for tiny K.
# ---------------------------------------------------------------------------

def validate_reduction_against_bruteforce(
    cfg: ExperimentConfig,
    *,
    K: int = 2,
    check_times: int = 7,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> float:
    """Verify the generalized reduced Riccati system against the literal game."""
    if K > 3:
        raise ValueError("Brute-force validation is intentionally restricted to K<=3.")

    N = cfg.n_pop
    Kvec = np.full(N, int(K), dtype=int)
    L = int(Kvec.sum())
    starts = np.r_[0, np.cumsum(Kvec)]
    pops = np.concatenate([np.full(Kvec[i], i, dtype=int) for i in range(N)])

    A = np.diag(cfg.a[pops])
    mu = np.zeros((L, N), dtype=float)
    for j in range(N):
        mu[starts[j] : starts[j + 1], j] = 1.0 / Kvec[j]

    running_Q = np.zeros((L, L, L), dtype=float)
    terminal_Q = np.zeros((L, L, L), dtype=float)
    for p in range(L):
        i = pops[p]
        ep = np.zeros(L, dtype=float)
        ep[p] = 1.0
        cvec = mu @ cfg.C[i]
        hvec = mu @ cfg.H_terminal[i]
        running_Q[p] = (
            cfg.q_state[i] * np.outer(ep, ep)
            - np.outer(ep, cvec)
            - np.outer(cvec, ep)
            + mu @ cfg.mean_quadratic[i] @ mu.T
        )
        terminal_Q[p] = (
            cfg.q_terminal[i] * np.outer(ep, ep)
            - np.outer(ep, hvec)
            - np.outer(hvec, ep)
            + mu @ cfg.terminal_mean_quadratic[i] @ mu.T
        )

    def full_rhs(_t: float, y: np.ndarray) -> np.ndarray:
        Ps = y.reshape(L, L, L)
        out = np.empty_like(Ps)
        feedback = np.empty((L, L), dtype=float)
        for qidx in range(L):
            j = pops[qidx]
            feedback[qidx] = cfg.b[j] * Ps[qidx, :, qidx]
            feedback[qidx, qidx] += cfg.s_state_control[j]

        for p in range(L):
            P = Ps[p]
            i = pops[p]
            val = -(P @ A + A @ P + running_Q[p])

            kp = feedback[p]
            val += np.outer(kp, kp) / cfg.r_control[i]

            for qidx in range(L):
                if qidx == p:
                    continue
                j = pops[qidx]
                u = P[:, qidx]
                kq = feedback[qidx]
                val += (cfg.b[j] / cfg.r_control[j]) * (
                    np.outer(u, kq) + np.outer(kq, u)
                )

            out[p] = 0.5 * (val + val.T)
        return out.ravel()

    idx = np.unique(np.linspace(0, cfg.n_steps, check_times, dtype=int))
    t_check = cfg.times[idx]
    full = solve_ivp(
        full_rhs,
        (cfg.T, 0.0),
        terminal_Q.ravel(),
        t_eval=t_check[::-1],
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not full.success:
        raise RuntimeError(full.message)
    full_P = full.y[:, ::-1].T.reshape(len(t_check), L, L, L)

    reduced = solve_exact_finite_nash_mean_drift(cfg, K=Kvec, rtol=rtol, atol=atol)
    reduced_P = reduced.reduced_P[idx]

    max_err = 0.0
    for n in range(len(t_check)):
        for i in range(N):
            p = int(starts[i])
            Hbasis = np.zeros((L, N + 1), dtype=float)
            Hbasis[p, 0] = 1.0
            for j in range(N):
                Hbasis[starts[j] : starts[j + 1], j + 1] = 1.0 / Kvec[j]
            reconstructed = Hbasis @ reduced_P[n, i] @ Hbasis.T
            err = float(np.max(np.abs(reconstructed - full_P[n, p])))
            max_err = max(max_err, err)

    return max_err
