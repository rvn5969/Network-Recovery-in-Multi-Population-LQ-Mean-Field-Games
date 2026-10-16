from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import make_config
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
from lq_mfg import solve_lq_mfg


def main() -> None:
    cfg = replace(make_config(quick=True), R_runs=100, seed=12345)
    cfg.validate()

    mfg = solve_lq_mfg(cfg)
    assert np.all(np.isfinite(mfg.B_eff))
    reduction_error = validate_reduction_against_bruteforce(cfg, K=2, check_times=4)
    assert reduction_error < 1e-6, reduction_error

    finite = solve_exact_finite_nash_mean_drift(cfg, K=np.array([3, 4, 5, 3, 4]))
    panel = simulate_exact_nash_centered_means(cfg, finite, R=cfg.R_runs, seed=202)
    point = estimate_pointwise_drift(panel.empirical_means, panel.times, cfg.dt)
    point_fit = estimate_G_pointwise_ols(
        panel.empirical_means,
        panel.times,
        cfg.dt,
        pointwise_full=point,
        split_seed=88,
    )
    mart = estimate_G_martingale_basis(
        panel.empirical_means, panel.times, cfg.dt, degree=3, split_seed=88
    )
    assert np.all(np.isfinite(point_fit.G_hat))
    assert np.all(np.isfinite(point_fit.G_hat_cf))
    assert np.all(np.isfinite(mart.G_hat))
    assert np.all(np.isfinite(mart.G_hat_cf))

    # Mild network: checks stationary code without requiring the research config
    # itself to admit a stabilizing infinite-horizon Nash equilibrium.
    inf_cfg = replace(cfg, C=0.10 * cfg.C, R_runs=80)
    inf_cfg.validate()
    stationary = solve_infinite_finite_nash(inf_cfg, K=inf_cfg.K_vector)
    assert stationary.spectral_abscissa < 0.0
    panel_inf = simulate_infinite_nash_centered_means(inf_cfg, stationary, seed=303)
    exact = estimate_G_infinite_exact(panel_inf.empirical_means, panel_inf.times, inf_cfg.dt)
    kut = estimate_G_infinite_kutoyants(panel_inf.empirical_means, panel_inf.times, inf_cfg.dt)
    assert np.all(np.isfinite(exact.G_hat))
    assert np.all(np.isfinite(kut.G_hat))
    assert np.allclose(np.diag(exact.G_hat), 0.0)
    assert np.allclose(np.diag(kut.G_hat), 0.0)

    print("smoke_test.py: PASS")
    print(f"finite-Nash reduction max discrepancy = {reduction_error:.3e}")


if __name__ == "__main__":
    main()
