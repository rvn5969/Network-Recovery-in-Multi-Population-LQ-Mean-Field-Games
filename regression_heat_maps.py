from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


R_VALUES = np.array([20, 40, 80, 160, 320, 640, 1000], dtype=int)
T_VALUES = np.array([5.0, 10.0, 20.0, 40.0, 70.0, 100.0], dtype=float)

BASIS_DEGREE = 3
BASE_SEED = 12345
N_REPETITIONS = 1
COMMON_COLOR_SCALE = True

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FINITE_OUTPUT_FIGURE = FIGURES_DIR / "rt_relative_error_heatmaps_finite.png"
FINITE_OUTPUT_NUMBERS = RESULTS_DIR / "rt_relative_error_values_finite.csv"
INFINITE_OUTPUT_FIGURE = FIGURES_DIR / "rt_relative_error_heatmaps_infinite.png"
INFINITE_OUTPUT_NUMBERS = RESULTS_DIR / "rt_relative_error_values_infinite.csv"

FINITE_METHODS = (
    "Pointwise OLS",
    "CF Pointwise OLS",
    "Martingale",
    "CF Martingale",
)

INFINITE_METHODS = (
    "Exact OU Transition",
    "Kutoyants Continuous-Time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=("finite", "infinite", "both"), default="both")
    return parser.parse_args()


def _validate_grid() -> None:
    if np.any(R_VALUES <= 0):
        raise ValueError("Every R value must be positive.")
    if np.any(T_VALUES <= 0):
        raise ValueError("Every T value must be positive.")
    if N_REPETITIONS < 1:
        raise ValueError("N_REPETITIONS must be at least 1.")


def _cell_text(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 1e-3 and value != 0.0:
        return f"{value:.1e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _empty_errors(methods: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        name: np.full(
            (N_REPETITIONS, len(T_VALUES), len(R_VALUES)),
            np.nan,
            dtype=float,
        )
        for name in methods
    }


def run_finite_rt_sweep() -> dict[str, np.ndarray]:
    _validate_grid()
    base_cfg = replace(make_config(quick=False), seed=BASE_SEED)
    base_cfg.validate()
    n = base_cfg.n_pop

    for R in R_VALUES:
        cut = int(R) // 2
        if min(cut, int(R) - cut) <= n + 1:
            raise ValueError(
                f"R={R} is too small for pointwise cross-fitting with N={n}. "
                f"Each fold must contain more than N+1={n+1} runs."
            )

    errors = _empty_errors(FINITE_METHODS)
    max_R = int(np.max(R_VALUES))

    for rep in range(N_REPETITIONS):
        print(f"\n=== finite repetition {rep + 1}/{N_REPETITIONS} ===")

        for ti, T in enumerate(T_VALUES):
            cfg = replace(
                base_cfg,
                T=float(T),
                R_runs=max_R,
                seed=BASE_SEED + 1_000_000 * rep + 10_000 * ti,
            )
            cfg.validate()

            print(f"\nT={T:g}: solving finite-Nash system ...")
            sol = solve_exact_finite_nash_mean_drift(cfg, K=cfg.K_vector)
            panel = simulate_exact_nash_centered_means(
                cfg,
                sol,
                R=max_R,
                seed=cfg.seed + 22001,
            )

            for ri, R in enumerate(R_VALUES):
                R = int(R)
                obs = np.ascontiguousarray(panel.empirical_means[:R], dtype=float)
                point = estimate_pointwise_drift(obs, panel.times, cfg.dt)
                split_seed = BASE_SEED + 20017 + rep * 1000 + ti * 100 + ri
                point_fit = estimate_G_pointwise_ols(
                    obs,
                    panel.times,
                    cfg.dt,
                    split_seed=split_seed,
                    pointwise_full=point,
                )
                mart = estimate_G_martingale_basis(
                    obs,
                    panel.times,
                    cfg.dt,
                    degree=BASIS_DEGREE,
                    split_seed=split_seed,
                )

                errors["Pointwise OLS"][rep, ti, ri] = relative_offdiag_frobenius_error(
                    point_fit.G_hat,
                    cfg.G_true,
                )
                errors["CF Pointwise OLS"][rep, ti, ri] = relative_offdiag_frobenius_error(
                    point_fit.G_hat_cf,
                    cfg.G_true,
                )
                errors["Martingale"][rep, ti, ri] = relative_offdiag_frobenius_error(
                    mart.G_hat,
                    cfg.G_true,
                )
                errors["CF Martingale"][rep, ti, ri] = relative_offdiag_frobenius_error(
                    mart.G_hat_cf,
                    cfg.G_true,
                )

                print(
                    f"  R={R:4d} | "
                    f"PW={errors['Pointwise OLS'][rep, ti, ri]:.4f}  "
                    f"CF-PW={errors['CF Pointwise OLS'][rep, ti, ri]:.4f}  "
                    f"M={errors['Martingale'][rep, ti, ri]:.4f}  "
                    f"CF-M={errors['CF Martingale'][rep, ti, ri]:.4f}"
                )

    return {name: np.mean(values, axis=0) for name, values in errors.items()}


def run_infinite_rt_sweep() -> dict[str, np.ndarray]:
    _validate_grid()
    base_cfg = replace(make_config(quick=False), seed=BASE_SEED)
    base_cfg.validate()
    errors = _empty_errors(INFINITE_METHODS)
    max_R = int(np.max(R_VALUES))

    for rep in range(N_REPETITIONS):
        solve_cfg = replace(
            base_cfg,
            R_runs=max_R,
            seed=BASE_SEED + 2_000_000 * rep,
        )
        solve_cfg.validate()

        print(f"\n=== infinite repetition {rep + 1}/{N_REPETITIONS} ===")
        print("Solving stationary finite-Nash system ...")
        sol = solve_infinite_finite_nash(solve_cfg, K=solve_cfg.K_vector)

        for ti, T in enumerate(T_VALUES):
            cfg = replace(
                base_cfg,
                T=float(T),
                R_runs=max_R,
                seed=BASE_SEED + 2_000_000 * rep + 10_000 * ti,
            )
            cfg.validate()

            print(f"\nT_obs={T:g}: simulating stationary OU panel ...")
            panel = simulate_infinite_nash_centered_means(
                cfg,
                sol,
                R=max_R,
                seed=cfg.seed + 42001,
            )

            for ri, R in enumerate(R_VALUES):
                R = int(R)
                obs = np.ascontiguousarray(panel.empirical_means[:R], dtype=float)

                try:
                    exact = estimate_G_infinite_exact(obs, panel.times, cfg.dt)
                    exact_error = relative_offdiag_frobenius_error(exact.G_hat, cfg.G_true)
                except RuntimeError:
                    exact_error = np.nan

                try:
                    kut = estimate_G_infinite_kutoyants(obs, panel.times, cfg.dt)
                    kut_error = relative_offdiag_frobenius_error(kut.G_hat, cfg.G_true)
                except RuntimeError:
                    kut_error = np.nan

                errors["Exact OU Transition"][rep, ti, ri] = exact_error
                errors["Kutoyants Continuous-Time"][rep, ti, ri] = kut_error

                print(
                    f"  R={R:4d} | "
                    f"OU={exact_error:.4f}  "
                    f"KUT={kut_error:.4f}"
                )

    return {name: np.nanmean(values, axis=0) for name, values in errors.items()}


def save_numeric_results(
    results: dict[str, np.ndarray],
    methods: tuple[str, ...],
    output_numbers: Path,
) -> None:
    output_numbers.parent.mkdir(parents=True, exist_ok=True)
    with output_numbers.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["T", "R", "method", "relative_offdiag_frobenius_error"])
        for ti, T in enumerate(T_VALUES):
            for ri, R in enumerate(R_VALUES):
                for method in methods:
                    writer.writerow([
                        float(T),
                        int(R),
                        method,
                        float(results[method][ti, ri]),
                    ])
    print(f"\nSaved numeric results to: {output_numbers.resolve()}")


def plot_heatmaps(
    results: dict[str, np.ndarray],
    methods: tuple[str, ...],
    title: str,
    y_label: str,
    output_figure: Path,
) -> None:
    if len(methods) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        axes = np.asarray(axes).ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        axes = np.asarray(axes).ravel()

    all_values = np.concatenate([results[name].ravel() for name in methods])
    finite_values = all_values[np.isfinite(all_values)]

    if finite_values.size == 0:
        raise RuntimeError("No finite error values were produced.")

    global_vmin = 0.0
    global_vmax = float(np.max(finite_values))
    images = []

    for ax, name in zip(axes, methods):
        matrix = results[name]

        if COMMON_COLOR_SCALE:
            vmin, vmax = global_vmin, global_vmax
        else:
            finite = matrix[np.isfinite(matrix)]
            vmin = 0.0
            vmax = float(np.max(finite))

        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)

        ax.set_title(name)
        ax.set_xlabel("Number of independent runs R")
        ax.set_ylabel(y_label)
        ax.set_xticks(np.arange(len(R_VALUES)))
        ax.set_xticklabels([str(int(r)) for r in R_VALUES])
        ax.set_yticks(np.arange(len(T_VALUES)))
        ax.set_yticklabels([f"{t:g}" for t in T_VALUES])

        threshold = vmin + 0.55 * (vmax - vmin) if vmax > vmin else vmax

        for ti in range(len(T_VALUES)):
            for ri in range(len(R_VALUES)):
                value = matrix[ti, ri]
                text_color = "white" if np.isfinite(value) and value >= threshold else "black"
                ax.text(
                    ri,
                    ti,
                    _cell_text(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )

    if COMMON_COLOR_SCALE:
        cbar = fig.colorbar(images[0], ax=axes.tolist(), shrink=0.90, pad=0.02)
        cbar.set_label("Relative off-diagonal Frobenius error")
    else:
        for ax, image in zip(axes, images):
            cbar = fig.colorbar(image, ax=ax, shrink=0.85)
            cbar.set_label("Relative off-diagonal Frobenius error")

    final_title = title
    if N_REPETITIONS > 1:
        final_title += f" — mean of {N_REPETITIONS} repetitions"
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(final_title, fontsize=15)
    fig.savefig(output_figure, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap figure to: {output_figure.resolve()}")


def print_error_tables(
    results: dict[str, np.ndarray],
    methods: tuple[str, ...],
    horizon_label: str,
) -> None:
    for name in methods:
        print("\n" + "=" * 80)
        print(f"{horizon_label}: {name}")
        print("rows = T_VALUES, columns = R_VALUES")
        print(f"R_VALUES = {R_VALUES.tolist()}")
        print(f"T_VALUES = {T_VALUES.tolist()}")
        print(np.array2string(results[name], precision=4, suppress_small=True))


def run_finite() -> None:
    results = run_finite_rt_sweep()
    print_error_tables(results, FINITE_METHODS, "Finite horizon")
    save_numeric_results(results, FINITE_METHODS, FINITE_OUTPUT_NUMBERS)
    plot_heatmaps(
        results,
        FINITE_METHODS,
        "Finite-Horizon Network Recovery Error over (R, T)",
        "Observation horizon T",
        FINITE_OUTPUT_FIGURE,
    )


def run_infinite() -> None:
    results = run_infinite_rt_sweep()
    print_error_tables(results, INFINITE_METHODS, "Infinite horizon")
    save_numeric_results(results, INFINITE_METHODS, INFINITE_OUTPUT_NUMBERS)
    plot_heatmaps(
        results,
        INFINITE_METHODS,
        "Infinite-Horizon Network Recovery Error over (R, T_obs)",
        "Observation duration T_obs",
        INFINITE_OUTPUT_FIGURE,
    )


def main() -> None:
    args = parse_args()
    if args.horizon in ("finite", "both"):
        run_finite()
    if args.horizon in ("infinite", "both"):
        run_infinite()


if __name__ == "__main__":
    main()
