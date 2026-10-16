"""Numba kernels for exact finite-Nash network-recovery experiments."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def simulate_linear_panel_kernel(
    initial: np.ndarray,
    standard_normal_noise: np.ndarray,
    B_path: np.ndarray,
    dt: float,
    noise_scale: np.ndarray,
) -> np.ndarray:
    """Euler simulate independent centered aggregate paths in parallel."""
    R, N = initial.shape
    L = B_path.shape[0]
    out = np.empty((R, L, N), dtype=np.float64)

    for r in prange(R):
        for i in range(N):
            out[r, 0, i] = initial[r, i]
        for n in range(L - 1):
            for i in range(N):
                drift = 0.0
                for j in range(N):
                    drift += B_path[n, i, j] * out[r, n, j]
                out[r, n + 1, i] = (
                    out[r, n, i]
                    + dt * drift
                    + standard_normal_noise[r, n, i] * noise_scale[i]
                )
    return out


@njit(cache=True, fastmath=True)
def pointwise_ols_kernel(
    centered: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-run OLS estimate of B(t_n) at every time point."""
    R, L, N = centered.shape
    n_steps = L - 1
    B_hat = np.empty((n_steps, N, N), dtype=np.float64)
    ranks = np.zeros(n_steps, dtype=np.int64)
    conds = np.zeros(n_steps, dtype=np.float64)
    rms = np.zeros((n_steps, N), dtype=np.float64)
    rcond = np.finfo(np.float64).eps * max(R, N)

    for n in range(n_steps):
        Xs = np.empty((R, N), dtype=np.float64)
        Y = np.empty((R, N), dtype=np.float64)
        scales = np.empty(N, dtype=np.float64)

        for j in range(N):
            ss = 0.0
            for r in range(R):
                x = centered[r, n, j]
                ss += x * x
            scale = np.sqrt(ss / R)
            if scale <= 1.0e-14:
                scale = 1.0
            scales[j] = scale

        for r in range(R):
            for j in range(N):
                Xs[r, j] = centered[r, n, j] / scales[j]
                Y[r, j] = (centered[r, n + 1, j] - centered[r, n, j]) / dt

        coef_scaled, _resids, rank, singular_values = np.linalg.lstsq(Xs, Y, rcond=rcond)
        ranks[n] = rank
        if singular_values.size > 0 and singular_values[-1] > 0.0:
            conds[n] = singular_values[0] / singular_values[-1]
        else:
            conds[n] = np.inf

        for i in range(N):
            for j in range(N):
                B_hat[n, i, j] = coef_scaled[j, i] / scales[j]

        for j in range(N):
            ss = 0.0
            for r in range(R):
                pred = 0.0
                for k in range(N):
                    pred += Xs[r, k] * coef_scaled[k, j]
                e = Y[r, j] - pred
                ss += e * e
            rms[n, j] = np.sqrt(ss / R)

    return B_hat, ranks, conds, rms


@njit(cache=True, fastmath=True, parallel=True)
def crossfit_B2_kernel(B_a: np.ndarray, B_b: np.ndarray) -> np.ndarray:
    """Compute 0.5*(B_a B_b + B_b B_a) at every time."""
    L, N, _ = B_a.shape
    out = np.empty_like(B_a)
    for n in prange(L):
        for i in range(N):
            for j in range(N):
                ab = 0.0
                ba = 0.0
                for k in range(N):
                    ab += B_a[n, i, k] * B_b[n, k, j]
                    ba += B_b[n, i, k] * B_a[n, k, j]
                out[n, i, j] = 0.5 * (ab + ba)
    return out


# ---------------------------------------------------------------------------
# Continuum equilibrium helpers (used only for oracle/unit checks, not data).
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def build_continuum_B_kernel(D: np.ndarray, S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Build B(t)=-diag(D(t))-diag(M)S(t)."""
    L, N = D.shape
    B = np.empty((L, N, N), dtype=np.float64)
    for n in range(L):
        for i in range(N):
            for j in range(N):
                val = -M[i] * S[n, i, j]
                if i == j:
                    val -= D[n, i]
                B[n, i, j] = val
    return B


@njit(cache=True, fastmath=True)
def sensitivity_rhs_kernel(
    s_flat: np.ndarray,
    D_t: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """RHS of Sdot=DS+SD+S diag(M) S+C."""
    N = D_t.size
    S = s_flat.reshape((N, N))
    out = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            quad = 0.0
            for k in range(N):
                quad += S[i, k] * M[k] * S[k, j]
            out[i, j] = D_t[i] * S[i, j] + S[i, j] * D_t[j] + quad + C[i, j]
    return out.ravel()


# ---------------------------------------------------------------------------
# Exact finite-player Nash reduced Riccati algebra.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _finite_nash_rhs_pis(
    Pis: np.ndarray,
    b: np.ndarray,
    r_control: np.ndarray,
    s_state_control: np.ndarray,
    K: np.ndarray,
    own_eval: np.ndarray,
    Abar_diag: np.ndarray,
    running_Qbar: np.ndarray,
) -> np.ndarray:
    """Exact reduced Hessian ODE for the generalized finite-player Nash game."""
    N, d, _ = Pis.shape
    out = np.empty_like(Pis)

    # k_j = b_j P^j e_j + s_j e_j, represented in player-j's reduced basis.
    feedback_vec = np.empty((N, d), dtype=np.float64)
    for j in range(N):
        for a in range(d):
            total = 0.0
            for bb in range(d):
                total += Pis[j, a, bb] * own_eval[j, bb]
            feedback_vec[j, a] = b[j] * total
        feedback_vec[j, 0] += s_state_control[j]

    for i in range(N):
        val = np.empty((d, d), dtype=np.float64)

        # -[Pi Abar + Abar Pi + Qbar_i]
        for a in range(d):
            for bb in range(d):
                val[a, bb] = -(
                    Pis[i, a, bb] * (Abar_diag[i, a] + Abar_diag[i, bb])
                    + running_Qbar[i, a, bb]
                )

        # Own optimized-control term +(1/r_i) k_i k_i^T.
        inv_r = 1.0 / r_control[i]
        for a in range(d):
            for bb in range(d):
                val[a, bb] += inv_r * feedback_vec[i, a] * feedback_vec[i, bb]

        # Other players' Nash feedback enters player i's HJB through the drift.
        for j in range(N):
            n_sources = K[j] - (1 if j == i else 0)
            if n_sources <= 0:
                continue

            # For any source q in population j distinct from the target player,
            # H_i^T e_q = e_{j+1}/K_j.
            u = np.empty(d, dtype=np.float64)
            invK = 1.0 / K[j]
            for a in range(d):
                u[a] = Pis[i, a, j + 1] * invK

            # Sum of all source feedback vectors k_q over q in population j,
            # excluding the target player when j=i, represented in H_i's basis.
            summed_k = np.zeros(d, dtype=np.float64)
            z = feedback_vec[j]
            if j != i:
                for a in range(1, d):
                    summed_k[a] = K[j] * z[a]
                summed_k[j + 1] += K[j] * z[0]
            else:
                summed_k[0] = -z[0]
                for a in range(1, d):
                    summed_k[a] = (K[i] - 1) * z[a]
                summed_k[i + 1] += K[i] * z[0]

            coeff = b[j] / r_control[j]
            for a in range(d):
                for bb in range(d):
                    val[a, bb] += coeff * (
                        u[a] * summed_k[bb] + summed_k[a] * u[bb]
                    )

        # Preserve symmetry against roundoff.
        for a in range(d):
            for bb in range(d):
                out[i, a, bb] = 0.5 * (val[a, bb] + val[bb, a])

    return out


@njit(cache=True, fastmath=True)
def finite_nash_rhs_kernel(
    y: np.ndarray,
    N: int,
    d: int,
    b: np.ndarray,
    r_control: np.ndarray,
    s_state_control: np.ndarray,
    K: np.ndarray,
    own_eval: np.ndarray,
    Abar_diag: np.ndarray,
    running_Qbar: np.ndarray,
) -> np.ndarray:
    Pis = y.reshape((N, d, d))
    return _finite_nash_rhs_pis(
        Pis,
        b,
        r_control,
        s_state_control,
        K,
        own_eval,
        Abar_diag,
        running_Qbar,
    ).ravel()


@njit(cache=True, fastmath=True)
def finite_nash_B_path_kernel(
    reduced_P: np.ndarray,
    b: np.ndarray,
    r_control: np.ndarray,
    s_state_control: np.ndarray,
    a_local: np.ndarray,
    K: np.ndarray,
    own_eval: np.ndarray,
    Abar_diag: np.ndarray,
    running_Qbar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the reduced Hessian path to the exact centered aggregate drift B^K."""
    L, N, d, _ = reduced_P.shape
    B = np.empty((L, N, N), dtype=np.float64)
    Bdot = np.empty((L, N, N), dtype=np.float64)

    for n in range(L):
        Pis = reduced_P[n]
        dPis = _finite_nash_rhs_pis(
            Pis,
            b,
            r_control,
            s_state_control,
            K,
            own_eval,
            Abar_diag,
            running_Qbar,
        )

        for i in range(N):
            z = np.empty(d, dtype=np.float64)
            zdot = np.empty(d, dtype=np.float64)
            for a in range(d):
                total = 0.0
                total_dot = 0.0
                for bb in range(d):
                    total += Pis[i, a, bb] * own_eval[i, bb]
                    total_dot += dPis[i, a, bb] * own_eval[i, bb]
                z[a] = total
                zdot[a] = total_dot

            for j in range(N):
                delta = 1.0 if i == j else 0.0
                # Average over target players: coefficient of m_j in
                # b_i P^p e_p + s_i e_p is
                # b_i(z_{j+1}+delta*z_0)+delta*s_i.
                feedback_coeff = b[i] * (z[j + 1] + delta * z[0]) + delta * s_state_control[i]
                feedback_coeff_dot = b[i] * (zdot[j + 1] + delta * zdot[0])

                B[n, i, j] = delta * a_local[i] - (b[i] / r_control[i]) * feedback_coeff
                Bdot[n, i, j] = -(b[i] / r_control[i]) * feedback_coeff_dot

    return B, Bdot


# ---------------------------------------------------------------------------
# Compact kernels for the textbook regression extensions.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def simulate_transition_panel_kernel(
    initial: np.ndarray,
    standard_normal_noise: np.ndarray,
    transition: np.ndarray,
    noise_factor: np.ndarray,
) -> np.ndarray:
    """Exact-grid simulation X_{n+1}=A X_n+L eps_n for a constant linear SDE."""
    R, N = initial.shape
    steps = standard_normal_noise.shape[1]
    out = np.empty((R, steps + 1, N), dtype=np.float64)
    for r in prange(R):
        for i in range(N):
            out[r, 0, i] = initial[r, i]
        for n in range(steps):
            for i in range(N):
                value = 0.0
                noise = 0.0
                for j in range(N):
                    value += transition[i, j] * out[r, n, j]
                    noise += noise_factor[i, j] * standard_normal_noise[r, n, j]
                out[r, n + 1, i] = value + noise
    return out


@njit(cache=True, fastmath=True, parallel=True)
def basis_moment_kernel(
    centered: np.ndarray,
    basis: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal equations for sum phi_k Z (dZ-B_theta Z dt)^T = 0."""
    R, L, N = centered.shape
    steps = L - 1
    Kb = basis.shape[1]

    cov = np.zeros((steps, N, N), dtype=np.float64)
    cross_t = np.zeros((steps, N, N), dtype=np.float64)
    yss_t = np.zeros((steps, N), dtype=np.float64)

    for n in prange(steps):
        for r in range(R):
            for j in range(N):
                xj = centered[r, n, j]
                for m in range(N):
                    cov[n, j, m] += xj * centered[r, n, m]
                for i in range(N):
                    dz = centered[r, n + 1, i] - centered[r, n, i]
                    cross_t[n, j, i] += xj * dz
            for i in range(N):
                dz = centered[r, n + 1, i] - centered[r, n, i]
                yss_t[n, i] += dz * dz

    P = Kb * N
    gram = np.zeros((P, P), dtype=np.float64)
    cross = np.zeros((P, N), dtype=np.float64)
    yss = np.zeros(N, dtype=np.float64)

    for n in range(steps):
        for i in range(N):
            yss[i] += yss_t[n, i]
        for k in range(Kb):
            pk = dt * basis[n, k]
            for j in range(N):
                a = k * N + j
                for i in range(N):
                    cross[a, i] += pk * cross_t[n, j, i]
                for ell in range(Kb):
                    ql = dt * basis[n, ell]
                    for m in range(N):
                        b = ell * N + m
                        gram[a, b] += pk * ql * cov[n, j, m]
    return gram, cross, yss


@njit(cache=True, fastmath=True)
def pooled_linear_moments_kernel(
    centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sum Z_n Z_n^T and sum dZ_n Z_n^T over runs and time."""
    R, L, N = centered.shape
    gram = np.zeros((N, N), dtype=np.float64)
    cross = np.zeros((N, N), dtype=np.float64)
    for r in range(R):
        for n in range(L - 1):
            for j in range(N):
                xj = centered[r, n, j]
                for m in range(N):
                    gram[j, m] += xj * centered[r, n, m]
                for i in range(N):
                    dz = centered[r, n + 1, i] - centered[r, n, i]
                    cross[i, j] += dz * xj
    return gram, cross
