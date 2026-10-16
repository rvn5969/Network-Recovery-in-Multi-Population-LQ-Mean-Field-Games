# Recovering Strategic Interaction Networks from Passive Fluctuations in Multi-Population LQ Mean Field Games

Python project for recovering a hidden directed strategic-interaction network in a multi-population linear-quadratic mean field game from passive finite-population fluctuations.

The interaction network enters the agents' costs rather than the physical state dynamics. Strategic coupling changes the equilibrium feedback law, which induces an observable linear drift in the centered empirical population means. The project estimates that drift from repeated trajectories and reconstructs the off-diagonal strategic network.

The finite-horizon study compares pointwise OLS and globally pooled martingale regression, with and without cross-fitting of the nonlinear $B^2$ term. The infinite-horizon study compares exact Ornstein-Uhlenbeck transition regression against a continuous-time Kutoyants-style regression.

## Architecture

```text
multi-population finite-player LQ game
        ↓
exact symmetry-reduced Nash solve
        ↓
centered empirical population means
        ↓
linear fluctuation dynamics
        ↓
estimate equilibrium drift B
        ↓
finite horizon:
G_ij = -Bdot_ij - (B^2)_ij

infinite horizon:
G_ij = -(B^2)_ij
        ↓
recover directed strategic network
```

For the default configuration,

$$
G = \operatorname{diag}(b_i^2 / r_i)\, C
$$

and

$$
r_i = b_i^2
$$

so that

$$
G = C.
$$

Main components:

```text
config.py                    central model and experiment configuration
finite_nash.py               exact finite-horizon finite-player Nash solver
finite_nash_infinite.py      stationary infinite-horizon Nash solver
drift_estimation.py          pointwise cross-run drift estimation
estimator_pointwise.py       pointwise and CF-pointwise recovery
estimator_martingale.py      martingale and CF-martingale recovery
estimator_infinite.py        stationary infinite-horizon recovery
lq_mfg.py                    continuum LQ-MFG oracle / theory check
numba_kernels.py             compiled simulation and regression kernels
metrics.py                   network-recovery metrics
main.py                      finite / infinite experiment runner
heat_map.py                  R-T relative-error heatmaps
prediction_heatmap.py        true-vs-estimated network heatmaps
smoke_test.py                end-to-end validation
requirements.txt             Python dependencies
figures/                     generated figures
results/                     generated CSV results
```

For the mathematical derivation, see [`theorynote.md`](theorynote.md).

## Installation

From the repository root:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project requires:

```text
numpy
scipy
numba
matplotlib
```

## Build and test

Run:

```bash
python smoke_test.py
```

The smoke test checks the main finite- and infinite-horizon code paths, including the reduced finite-player Nash solve and the network estimators.

## Finite-horizon experiment

Run:

```bash
python main.py --horizon finite
```

The finite-horizon branch solves the exact time-varying finite-player Nash game and simulates repeated centered empirical-population trajectories.

The same data are passed to four estimators:

```text
Pointwise OLS
CF Pointwise OLS
Martingale
CF Martingale
```

Useful overrides:

```bash
python main.py --horizon finite --runs 1000
python main.py --horizon finite --K 500
python main.py --horizon finite --seed 12345
python main.py --horizon finite --quick
python main.py --horizon finite --quick --verify-reduction
```

## Finite-horizon recovery methods

### Pointwise OLS

At each observation time $t_n$, the drift is estimated across repeated runs:

$$
\frac{Z_{n+1} - Z_n}{dt} = B(t_n)\, Z_n + \text{noise}.
$$

The estimated drift path is differentiated numerically and inserted into

$$
G_{ij} = -\dot B_{ij} - (B^2)_{ij}, \qquad i \neq j.
$$

Because each local regression uses only the cross-sectional sample of repeated runs, pointwise OLS can have high finite-sample error when $R$ is small.

### CF Pointwise OLS

The repeated trajectories are split into two independent folds and the nonlinear term is estimated using

$$
\frac{1}{2}\left(B_A B_B + B_B B_A\right).
$$

The purpose of cross-fitting is to reduce the same-sample quadratic error contribution that appears when $\widehat B^2$ is formed directly from a noisy drift estimate.

### Martingale regression

The time-varying drift is represented by a cubic Legendre expansion:

$$
B(t) = \Theta_0 \phi_0(t)
     + \Theta_1 \phi_1(t)
     + \Theta_2 \phi_2(t)
     + \Theta_3 \phi_3(t).
$$

All runs and all time increments are pooled into one global regression. The fitted drift is smooth by construction and its derivative is computed analytically from the basis expansion.

This avoids estimating a separate drift matrix at every time point and reduces the finite-sample instability of local pointwise regression.

### CF Martingale

The martingale drift model is fit independently on two folds and the same cross-fitted quadratic product is used:

$$
\frac{1}{2}\left(B_A B_B + B_B B_A\right).
$$

In the numerical experiments, the ordinary martingale estimator is already strongly stabilized by global pooling, so cross-fitting typically gives a smaller additional improvement than it does for pointwise OLS.

## Infinite-horizon experiment

Run:

```bash
python main.py --horizon infinite
```

At stationary infinite-horizon equilibrium,

$$
dZ_t = B_{\infty} Z_t\, dt + \Gamma\, dW_t
$$

with constant drift $B_{\infty}$.

The network identity becomes

$$
G_{ij} = -(B_{\infty}^2)_{ij}, \qquad i \neq j.
$$

Two stationary estimators are compared.

### Exact OU transition regression

The exact discrete-time transition is

$$
Z_{n+1} = A Z_n + \text{noise}, \qquad A = \exp(B_{\infty} dt).
$$

The transition matrix is estimated by multivariate regression and the continuous-time drift is recovered from

$$
\widehat B = \log(\widehat A) / dt.
$$

### Kutoyants continuous-time regression

The second estimator uses the pooled continuous-time estimating equation

$$
\widehat B
=
\left[\sum dZ_n Z_n^\top\right]
\left[dt \sum Z_n Z_n^\top\right]^{-1}.
$$

The exact OU transition regression avoids the Euler conditional-mean approximation used by the continuous-time regression.

## R-T error heatmaps

Generate relative-error heatmaps with:

```bash
python heat_map.py --horizon finite
```

or:

```bash
python heat_map.py --horizon infinite
```

To generate both:

```bash
python heat_map.py --horizon both
```

The heatmaps use:

```text
x-axis = number of independent runs R
y-axis = observation horizon T
color  = relative off-diagonal Frobenius error
```

For infinite horizon, $T$ is the observation duration rather than a terminal game horizon.

Generated figures are saved to:

```text
figures/
```

and the numerical values are saved as CSV files to:

```text
results/
```

No interactive Matplotlib window is opened.

## Network prediction heatmaps

Generate true-vs-estimated network matrices with:

```bash
python prediction_heatmap.py
```

For a smaller test run:

```bash
python prediction_heatmap.py --quick
```

The script generates six separate figures:

```text
finite horizon:
    Pointwise OLS
    CF Pointwise OLS
    Martingale
    CF Martingale

infinite horizon:
    Exact OU Transition
    Kutoyants Continuous-Time
```

Each figure places the true strategic network $G$ beside the estimated network $\widehat G$ using a common heatmap scale.

Figures are written to:

```text
figures/
```

The corresponding numerical results are written to:

```text
results/prediction_heatmap_summary.csv
results/prediction_heatmap_matrices.csv
```

The summary CSV reports the relative off-diagonal Frobenius error for each estimator. The matrix CSV contains the individual true and estimated matrix entries.

## Recovery metric

The primary error metric is the relative off-diagonal Frobenius error:

$$
\text{relative error}
=
\frac{\lVert \widehat G - G_{\text{true}} \rVert_{F,\mathrm{off}}}
{\lVert G_{\text{true}} \rVert_{F,\mathrm{off}}}.
$$

Only off-diagonal entries are scored because the network identification formulas used in the project recover $G_{ij}$ for $i \neq j$.

## Empirical scaling

The finite-horizon experiments are designed to study the approximate scaling

$$
\text{error} \propto \frac{1}{\sqrt{RT}}
$$

as the number of repeated trajectories $R$ and the observation horizon $T$ increase.

The main finite-sample pattern is:

- pointwise OLS can remain noisy when $R$ is small
- cross-fitting substantially improves pointwise recovery in that regime
- martingale regression benefits from pooling information across the full horizon
- CF martingale is usually close to ordinary martingale
- at sufficiently large $R$ and $T$, the estimators approach the same first-order scaling regime

The $1/\sqrt{RT}$ behavior is treated as an empirical scaling law unless additional asymptotic assumptions are imposed.

## Exact finite-player simulation

Synthetic finite-horizon data are generated from the symmetry-reduced finite-player Nash system rather than directly from the continuum MFG.

After removing the deterministic affine component, the centered aggregate dynamics take the form

$$
dZ_t = B^K(t) Z_t\, dt
     + \operatorname{diag}\!\left(\sigma_i / \sqrt{K_i}\right) d\overline W_t.
$$

The reduced Nash system can be checked against a literal small-$K$ finite-player Riccati solve with:

```bash
python main.py --horizon finite --quick --verify-reduction
```

The continuum solver in `lq_mfg.py` is retained as a theory/oracle check.

## Configuration

The main model and experiment parameters are defined in:

```text
config.py
```

including:

```text
number of populations
directed interaction matrix C
finite population sizes K_i
number of repeated runs R
observation horizon T
time step dt
state dynamics
control costs
running costs
terminal costs
diffusion coefficients
initial distributions
random seed
```

The convention is:

```text
C[i,j] != 0
```

means population $j$ enters the running cost of population $i$.

## Generated outputs

The scripts create two output directories in the repository root:

```text
figures/
results/
```

These directories are ignored by Git through `.gitignore` so generated outputs do not need to be committed to the repository.

## Scope and limitations

The current implementation studies:

- scalar state dynamics within each population
- linear-quadratic strategic interactions
- directed cost-interaction networks
- passive observation of empirical population means
- repeated independent finite-horizon trajectories
- off-diagonal network recovery
- cubic Legendre drift approximation for the finite-horizon martingale estimator
- stationary Ornstein-Uhlenbeck aggregate dynamics in the infinite-horizon branch
- simulation-based estimator comparison and reproducibility

The finite-horizon martingale method is parametric in time through the cubic Legendre representation and is not intended as unrestricted nonparametric drift recovery.

The infinite-horizon branch requires a stabilizing stationary finite-player Nash solution. If the configured model does not admit one, the stationary experiment is not run.

This repository is intended as a research implementation accompanying work on inverse identification of strategic interaction networks from passive equilibrium fluctuations.
