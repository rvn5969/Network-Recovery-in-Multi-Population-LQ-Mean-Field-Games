from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import make_config
from drift_estimation import estimate_pointwise_drift
from estimator_infinite import estimate_G_infinite_exact, estimate_G_infinite_kutoyants
from estimator_martingale import estimate_G_martingale_basis
from estimator_pointwise import estimate_G_pointwise_ols
from finite_nash import simulate_exact_nash_centered_means, solve_exact_finite_nash_mean_drift
from finite_nash_infinite import simulate_infinite_nash_centered_means, solve_infinite_finite_nash
from metrics import relative_offdiag_frobenius_error

BASIS_DEGREE = 3
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction heatmaps for finite and infinite horizon estimators.")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _cell_text(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 1e-3 and value != 0.0:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _draw_matrix(ax: plt.Axes, matrix: np.ndarray, title: str, vmax: float) -> None:
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    n = matrix.shape[0]
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)])
    ax.set_yticklabels([str(i + 1) for i in range(n)])
    ax.set_xlabel("source population j")
    ax.set_ylabel("target population i")
    threshold = 0.55 * vmax if vmax > 0 else 0.0
    for i in range(n):
        for j in range(n):
            value = float(matrix[i, j])
            color = "white" if abs(value) >= threshold else "black"
            ax.text(j, i, _cell_text(value), ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    return image


def save_pair_figure(true_matrix: np.ndarray, estimate_matrix: np.ndarray, method_name: str, horizon: str, error_value: float) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{horizon.lower()}_{method_name.lower().replace(' ', '_').replace('-', '_')}"
    output_path = FIGURES_DIR / f"prediction_heatmap_{slug}.png"
    vmax = float(max(np.max(np.abs(true_matrix)), np.max(np.abs(estimate_matrix)), 1e-12))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    image = _draw_matrix(axes[0], true_matrix, f"True network G", vmax)
    _draw_matrix(axes[1], estimate_matrix, f"{method_name}\nrelative error = {error_value:.6f}", vmax)
    fig.suptitle(f"{horizon} horizon — {method_name}", fontsize=15)
    cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.90, pad=0.02)
    cbar.set_label("matrix entry value")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_finite_results(quick: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cfg = make_config(quick=quick)
    sol = solve_exact_finite_nash_mean_drift(cfg, K=cfg.K_vector)
    panel = simulate_exact_nash_centered_means(cfg, sol, R=cfg.R_runs, seed=cfg.seed + 22001)
    point = estimate_pointwise_drift(panel.empirical_means, panel.times, cfg.dt)
    point_fit = estimate_G_pointwise_ols(
        panel.empirical_means,
        panel.times,
        cfg.dt,
        split_seed=cfg.seed + 20017,
        pointwise_full=point,
    )
    mart = estimate_G_martingale_basis(
        panel.empirical_means,
        panel.times,
        cfg.dt,
        degree=BASIS_DEGREE,
        split_seed=cfg.seed + 20017,
    )
    results = {
        "Pointwise OLS": point_fit.G_hat,
        "CF Pointwise OLS": point_fit.G_hat_cf,
        "Martingale": mart.G_hat,
        "CF Martingale": mart.G_hat_cf,
    }
    return cfg.G_true.copy(), results


def compute_infinite_results(quick: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cfg = make_config(quick=quick)
    sol = solve_infinite_finite_nash(cfg, K=cfg.K_vector)
    panel = simulate_infinite_nash_centered_means(cfg, sol, R=cfg.R_runs, seed=cfg.seed + 42001)
    exact = estimate_G_infinite_exact(panel.empirical_means, panel.times, cfg.dt)
    kut = estimate_G_infinite_kutoyants(panel.empirical_means, panel.times, cfg.dt)
    results = {
        "Exact OU Transition": exact.G_hat,
        "Kutoyants Continuous-Time": kut.G_hat,
    }
    return cfg.G_true.copy(), results


def save_summary_csv(rows: list[tuple[str, str, float]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "prediction_heatmap_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon", "method", "relative_offdiag_frobenius_error"])
        for horizon, method, error in rows:
            writer.writerow([horizon, method, error])
    return path


def save_matrix_csv(records: list[tuple[str, str, str, int, int, float]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "prediction_heatmap_matrices.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["horizon", "method", "matrix_type", "row", "col", "value"])
        for horizon, method, matrix_type, row, col, value in records:
            writer.writerow([horizon, method, matrix_type, row, col, value])
    return path


def matrix_records(horizon: str, method: str, true_matrix: np.ndarray, estimate_matrix: np.ndarray) -> list[tuple[str, str, str, int, int, float]]:
    records: list[tuple[str, str, str, int, int, float]] = []
    n = true_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            records.append((horizon, method, "truth", i, j, float(true_matrix[i, j])))
            records.append((horizon, method, "estimate", i, j, float(estimate_matrix[i, j])))
    return records


def main() -> None:
    args = parse_args()
    summary_rows: list[tuple[str, str, float]] = []
    matrix_rows: list[tuple[str, str, str, int, int, float]] = []

    finite_truth, finite_results = compute_finite_results(args.quick)
    for method, estimate in finite_results.items():
        error = relative_offdiag_frobenius_error(estimate, finite_truth)
        figure_path = save_pair_figure(finite_truth, estimate, method, "Finite", error)
        summary_rows.append(("finite", method, float(error)))
        matrix_rows.extend(matrix_records("finite", method, finite_truth, estimate))
        print(f"Saved figure to: {figure_path.resolve()}")

    infinite_truth, infinite_results = compute_infinite_results(args.quick)
    for method, estimate in infinite_results.items():
        error = relative_offdiag_frobenius_error(estimate, infinite_truth)
        figure_path = save_pair_figure(infinite_truth, estimate, method, "Infinite", error)
        summary_rows.append(("infinite", method, float(error)))
        matrix_rows.extend(matrix_records("infinite", method, infinite_truth, estimate))
        print(f"Saved figure to: {figure_path.resolve()}")

    summary_path = save_summary_csv(summary_rows)
    matrices_path = save_matrix_csv(matrix_rows)
    print(f"Saved CSV to: {summary_path.resolve()}")
    print(f"Saved CSV to: {matrices_path.resolve()}")


if __name__ == "__main__":
    main()
