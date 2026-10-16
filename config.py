from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np



# USER EDITABLE CONSTANT COEFFICIENT MODEL SETTINGS

N_POPULATIONS = 5

C_MATRIX = np.array(
    [
        [ 0.00,  1.20,  0.00,  0.00,  0.00],
        [ 0.90,  0.00, -1.10,  0.00,  0.00],
        [ 0.00,  1.30,  0.00,  1.00,  0.00],
        [ 0.00,  0.00, -0.85,  0.00,  1.25],
        [ 0.00,  0.00,  0.00,  1.10,  0.00],
    ],
    dtype=float,
)

K_AGENTS_PER_POPULATION: int | Sequence[int] = 500
R_RUNS = 1000
T = 100.0
DT = 0.01

A = np.array([-0.45, -0.40, -0.50, -0.42, -0.47], dtype=float)
B_CONTROL = np.array([0.90, 1.05, 0.95, 1.10, 1.00], dtype=float)
DRIFT_INTERCEPT = np.array([0.04, -0.03, 0.02, -0.04, 0.03], dtype=float)
SIGMA = np.array([0.60, 0.55, 0.65, 0.58, 0.62], dtype=float)

R_CONTROL = B_CONTROL**2
S_STATE_CONTROL = np.array([0.03, -0.02, 0.025, -0.015, 0.02], dtype=float)
RHO_CONTROL = np.array([0.03, -0.02, 0.01, -0.025, 0.02], dtype=float)
Q_STATE = np.array([1.30, 1.15, 1.40, 1.20, 1.35], dtype=float)
BETA_STATE = np.array([-0.10, 0.08, -0.06, 0.07, -0.09], dtype=float)

MEAN_QUADRATIC = 0.02 * np.eye(N_POPULATIONS)
GAMMA_MEAN = np.zeros(N_POPULATIONS)
RUNNING_CONSTANT = 0.00

Q_TERMINAL = np.array([1.0, 1.1, 0.9, 1.05, 0.95], dtype=float)
BETA_TERMINAL = np.array([0.05, -0.04, 0.03, -0.05, 0.04], dtype=float)
H_TERMINAL = 0.10 * C_MATRIX
TERMINAL_MEAN_QUADRATIC = 0.01 * np.eye(N_POPULATIONS)
GAMMA_TERMINAL = np.zeros(N_POPULATIONS)
TERMINAL_CONSTANT = 0.00

INITIAL_MEAN = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=float)
INITIAL_STD = np.array([0.75, 0.70, 0.80, 0.72, 0.78], dtype=float)
SEED = 12345


def _broadcast_vector(value, n: int, name: str) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    if x.ndim == 0:
        return np.full(n, float(x), dtype=float)
    x = x.reshape(-1)
    if x.shape != (n,):
        raise ValueError(f"{name} must be scalar or length {n}; got shape {x.shape}.")
    return x.copy()


def _broadcast_row_matrix(value, n: int, name: str) -> np.ndarray:
    """Return an (N,N) stack of row vectors, allowing a single length-N row."""
    x = np.asarray(value, dtype=float)
    if x.ndim == 0:
        return np.full((n, n), float(x), dtype=float)
    if x.shape == (n,):
        return np.tile(x[None, :], (n, 1))
    if x.shape == (n, n):
        return x.copy()
    raise ValueError(f"{name} must be scalar, length {n}, or shape ({n},{n}); got {x.shape}.")


def _broadcast_matrix_stack(value, n: int, name: str) -> np.ndarray:
    """Return an (N,N,N) stack, one symmetric mean-quadratic matrix per target."""
    x = np.asarray(value, dtype=float)
    if x.ndim == 0:
        base = float(x) * np.eye(n)
        return np.tile(base[None, :, :], (n, 1, 1))
    if x.shape == (n, n):
        return np.tile(x[None, :, :], (n, 1, 1))
    if x.shape == (n, n, n):
        return x.copy()
    raise ValueError(
        f"{name} must be scalar, shape ({n},{n}), or shape ({n},{n},{n}); got {x.shape}."
    )


def _normalize_K(value: int | Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(value, dtype=int)
    if x.ndim == 0:
        return np.full(n, int(x), dtype=int)
    x = x.reshape(-1)
    if x.shape != (n,):
        raise ValueError(f"K must be scalar or length {n}; got shape {x.shape}.")
    return x.copy()


@dataclass
class ExperimentConfig:
    n_populations: int
    K_agents_per_population: int | Sequence[int] | np.ndarray
    R_runs: int
    T: float
    dt: float

    # Dynamics.
    a: np.ndarray
    b: np.ndarray
    drift_intercept: np.ndarray
    sigma: np.ndarray

    # Running cost.
    r_control: np.ndarray
    s_state_control: np.ndarray
    rho_control: np.ndarray
    q_state: np.ndarray
    beta_state: np.ndarray
    C: np.ndarray
    mean_quadratic: np.ndarray
    gamma_mean: np.ndarray
    running_constant: np.ndarray

    # Terminal cost.
    q_terminal: np.ndarray
    beta_terminal: np.ndarray
    H_terminal: np.ndarray
    terminal_mean_quadratic: np.ndarray
    gamma_terminal: np.ndarray
    terminal_constant: np.ndarray

    # Initial distribution.
    initial_mean: np.ndarray
    initial_std: np.ndarray
    seed: int

    @property
    def n_pop(self) -> int:
        return int(self.n_populations)

    @property
    def K_vector(self) -> np.ndarray:
        return _normalize_K(self.K_agents_per_population, self.n_pop)

    @property
    def K_agents(self) -> int:
        k = self.K_vector
        if not np.all(k == k[0]):
            raise ValueError("Population sizes are heterogeneous; use cfg.K_vector instead.")
        return int(k[0])

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.n_steps + 1)

    @property
    def M_control(self) -> np.ndarray:
        # Constant-coefficient experiments automatically satisfy the blind
        # identifiability requirement M_i=b_i^2/r_i constant in time.
        return self.b**2 / self.r_control

    @property
    def G_true(self) -> np.ndarray:
        # Blindly identifiable scaled strategic network.
        return self.M_control[:, None] * self.C

    def validate(self) -> None:
        n = self.n_pop
        if n < 2:
            raise ValueError("Need at least two populations.")

        vector_names = (
            "a",
            "b",
            "drift_intercept",
            "sigma",
            "r_control",
            "s_state_control",
            "rho_control",
            "q_state",
            "beta_state",
            "running_constant",
            "q_terminal",
            "beta_terminal",
            "terminal_constant",
            "initial_mean",
            "initial_std",
        )
        for name in vector_names:
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},).")

        for name in ("C", "gamma_mean", "H_terminal", "gamma_terminal"):
            if np.asarray(getattr(self, name)).shape != (n, n):
                raise ValueError(f"{name} must have shape ({n},{n}).")

        for name in ("mean_quadratic", "terminal_mean_quadratic"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n, n, n):
                raise ValueError(f"{name} must have shape ({n},{n},{n}).")
            if not np.allclose(arr, np.swapaxes(arr, 1, 2), atol=1e-12, rtol=1e-12):
                raise ValueError(f"Every matrix in {name} must be symmetric.")

        if np.any(self.K_vector < 2):
            raise ValueError("Every finite population size K_i must be at least 2.")
        if self.T <= 0 or self.dt <= 0:
            raise ValueError("T and dt must be positive.")
        if abs(self.n_steps * self.dt - self.T) > 1e-10:
            raise ValueError("T / dt must be an integer for the current implementation.")
        if np.any(self.r_control <= 0):
            raise ValueError("r_control must be strictly positive.")
        if np.any(self.sigma < 0):
            raise ValueError("sigma must be nonnegative.")
        if np.any(self.initial_std <= 0):
            raise ValueError("initial_std must be strictly positive.")

        arrays = [getattr(self, name) for name in vector_names]
        arrays += [self.C, self.gamma_mean, self.H_terminal, self.gamma_terminal]
        arrays += [self.mean_quadratic, self.terminal_mean_quadratic]
        if not all(np.all(np.isfinite(np.asarray(x))) for x in arrays):
            raise ValueError("Model parameters contain non-finite values.")


def make_config(quick: bool = False) -> ExperimentConfig:
    n = int(N_POPULATIONS)
    cfg = ExperimentConfig(
        n_populations=n,
        K_agents_per_population=K_AGENTS_PER_POPULATION,
        R_runs=int(R_RUNS),
        T=float(T),
        dt=float(DT),
        a=_broadcast_vector(A, n, "A"),
        b=_broadcast_vector(B_CONTROL, n, "B_CONTROL"),
        drift_intercept=_broadcast_vector(DRIFT_INTERCEPT, n, "DRIFT_INTERCEPT"),
        sigma=_broadcast_vector(SIGMA, n, "SIGMA"),
        r_control=_broadcast_vector(R_CONTROL, n, "R_CONTROL"),
        s_state_control=_broadcast_vector(S_STATE_CONTROL, n, "S_STATE_CONTROL"),
        rho_control=_broadcast_vector(RHO_CONTROL, n, "RHO_CONTROL"),
        q_state=_broadcast_vector(Q_STATE, n, "Q_STATE"),
        beta_state=_broadcast_vector(BETA_STATE, n, "BETA_STATE"),
        C=np.asarray(C_MATRIX, dtype=float).copy(),
        mean_quadratic=_broadcast_matrix_stack(MEAN_QUADRATIC, n, "MEAN_QUADRATIC"),
        gamma_mean=_broadcast_row_matrix(GAMMA_MEAN, n, "GAMMA_MEAN"),
        running_constant=_broadcast_vector(RUNNING_CONSTANT, n, "RUNNING_CONSTANT"),
        q_terminal=_broadcast_vector(Q_TERMINAL, n, "Q_TERMINAL"),
        beta_terminal=_broadcast_vector(BETA_TERMINAL, n, "BETA_TERMINAL"),
        H_terminal=np.asarray(H_TERMINAL, dtype=float).copy(),
        terminal_mean_quadratic=_broadcast_matrix_stack(
            TERMINAL_MEAN_QUADRATIC, n, "TERMINAL_MEAN_QUADRATIC"
        ),
        gamma_terminal=_broadcast_row_matrix(GAMMA_TERMINAL, n, "GAMMA_TERMINAL"),
        terminal_constant=_broadcast_vector(TERMINAL_CONSTANT, n, "TERMINAL_CONSTANT"),
        initial_mean=_broadcast_vector(INITIAL_MEAN, n, "INITIAL_MEAN"),
        initial_std=_broadcast_vector(INITIAL_STD, n, "INITIAL_STD"),
        seed=int(SEED),
    )

    if cfg.C.shape != (n, n):
        raise ValueError(f"C_MATRIX must have shape ({n},{n}).")
    if cfg.H_terminal.shape != (n, n):
        raise ValueError(f"H_TERMINAL must have shape ({n},{n}).")

    if quick:
        cfg = replace(
            cfg,
            K_agents_per_population=np.minimum(cfg.K_vector, 30),
            R_runs=min(cfg.R_runs, 120),
        )

    cfg.validate()
    return cfg
