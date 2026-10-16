from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
from numba import get_num_threads

from config import ExperimentConfig, make_config
from drift_estimation import estimate_pointwise_drift
from estimator_infinite import estimate_G_infinite_exact, estimate_G_infinite_kutoyants
from estimator_martingale import estimate_G_martingale_basis
from estimator_pointwise import estimate_G_pointwise_ols
from finite_nash import (
    simulate_exact_nash_centered_means,
    solve_exact_finite_nash_mean_drift,
    validate_reduction_against_bruteforce,
)
from finite_nash_infinite import simulate_infinite_nash_centered_means, solve_infinite_finite_nash
from metrics import format_matrix, offdiag_correlation, relative_offdiag_frobenius_error

BASIS_DEGREE = 3


def _parse_K(values: list[int] | None, n: int):
    if values is None:
        return None
    if len(values) == 1:
        return int(values[0])
    if len(values) == n:
        return np.asarray(values, dtype=int)
    raise ValueError(f"--K accepts either one integer or exactly N={n} integers.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact finite-Nash strategic-network recovery.")
    p.add_argument("--horizon", choices=("finite", "infinite"), default="finite")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--K", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--verify-reduction", action="store_true")
    return p.parse_args()


def _config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = make_config(quick=args.quick)
    updates = {}
    if args.runs is not None:
        updates["R_runs"] = int(args.runs)
    if args.seed is not None:
        updates["seed"] = int(args.seed)
    K = _parse_K(args.K, cfg.n_pop)
    if K is not None:
        updates["K_agents_per_population"] = K
    if updates:
        cfg = replace(cfg, **updates)
        cfg.validate()
    return cfg


def _score(name: str, G_hat: np.ndarray, G_true: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"relative offdiag err: {relative_offdiag_frobenius_error(G_hat, G_true):.6f}")
    print(f"offdiag correlation : {offdiag_correlation(G_hat, G_true):.6f}")
    print("G_hat:")
    print(format_matrix(G_hat, 5))


def _header(cfg: ExperimentConfig, horizon: str) -> None:
    print("=" * 88)
    print(f"{horizon.upper()} HORIZON STRATEGIC NETWORK RECOVERY")
    print("=" * 88)
    print(f"N populations   : {cfg.n_pop}")
    print(f"R runs          : {cfg.R_runs}")
    print(f"T, dt           : {cfg.T}, {cfg.dt}")
    print(f"finite K vector : {cfg.K_vector.tolist()}")
    print(f"Numba threads   : {get_num_threads()}")
    print("\nBlind target G = diag(b_i^2/r_i) C:")
    print(format_matrix(cfg.G_true, 4))


def run_finite(cfg: ExperimentConfig, verify: bool) -> None:
    if verify:
        err = validate_reduction_against_bruteforce(cfg, K=2, check_times=5)
        print(f"\n[verification] reduced-vs-literal Hessian discrepancy: {err:.3e}")
        if err > 1e-6:
            raise RuntimeError("Finite-Nash symmetry reduction verification failed.")

    started = time.perf_counter()
    sol = solve_exact_finite_nash_mean_drift(cfg, K=cfg.K_vector)
    panel = simulate_exact_nash_centered_means(cfg, sol, R=cfg.R_runs, seed=cfg.seed + 22001)
    print(f"\nfinite-Nash solve + simulation: {time.perf_counter() - started:.2f}s")

    point = estimate_pointwise_drift(panel.empirical_means, panel.times, cfg.dt)
    point_fit = estimate_G_pointwise_ols(
        panel.empirical_means, panel.times, cfg.dt,
        split_seed=cfg.seed + 20017, pointwise_full=point,
    )
    print(f"max pointwise cond  : {np.max(point.condition_numbers):.2e}")
    _score("Pointwise OLS", point_fit.G_hat, cfg.G_true)
    _score("CF pointwise OLS", point_fit.G_hat_cf, cfg.G_true)

    mart = estimate_G_martingale_basis(
        panel.empirical_means, panel.times, cfg.dt,
        degree=BASIS_DEGREE, split_seed=cfg.seed + 20017,
    )
    print(f"\nmartingale basis cond: {mart.design_condition:.2e}")
    _score(f"Martingale (degree {BASIS_DEGREE})", mart.G_hat, cfg.G_true)
    _score(f"CF martingale (degree {BASIS_DEGREE})", mart.G_hat_cf, cfg.G_true)


def run_infinite(cfg: ExperimentConfig) -> None:
    started = time.perf_counter()
    sol = solve_infinite_finite_nash(cfg, K=cfg.K_vector)
    panel = simulate_infinite_nash_centered_means(cfg, sol, R=cfg.R_runs, seed=cfg.seed + 42001)
    print(f"\ninfinite-Nash solve + simulation: {time.perf_counter() - started:.2f}s")
    print(f"ARE residual        : {sol.algebraic_residual:.2e}")
    print(f"spectral abscissa   : {sol.spectral_abscissa:.3e}")

    exact = estimate_G_infinite_exact(panel.empirical_means, panel.times, cfg.dt)
    print(f"transition cond     : {exact.design_condition:.2e}")
    _score("Exact OU transition regression", exact.G_hat, cfg.G_true)

    kut = estimate_G_infinite_kutoyants(panel.empirical_means, panel.times, cfg.dt)
    print(f"\nKutoyants design cond: {kut.design_condition:.2e}")
    _score("Kutoyants continuous-time MLE regression", kut.G_hat, cfg.G_true)


def main() -> None:
    args = parse_args()
    cfg = _config(args)
    _header(cfg, args.horizon)
    try:
        run_finite(cfg, args.verify_reduction) if args.horizon == "finite" else run_infinite(cfg)
    except RuntimeError as exc:
        print(f"\n{args.horizon.capitalize()}-horizon solve unavailable: {exc}")


if __name__ == "__main__":
    main()
