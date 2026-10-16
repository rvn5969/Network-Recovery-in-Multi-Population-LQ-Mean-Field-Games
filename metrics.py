from __future__ import annotations

import numpy as np


def offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def relative_offdiag_frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    A = np.asarray(estimate, dtype=float)
    B = np.asarray(truth, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("estimate and truth must be square matrices with the same shape.")
    mask = offdiag_mask(A.shape[0])
    denom = float(np.linalg.norm(B[mask]))
    if denom <= 1e-14:
        return float("nan")
    return float(np.linalg.norm((A - B)[mask]) / denom)


def offdiag_correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    A = np.asarray(estimate, dtype=float)
    B = np.asarray(truth, dtype=float)
    mask = offdiag_mask(A.shape[0])
    x, y = A[mask], B[mask]
    if np.std(x) <= 1e-14 or np.std(y) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def format_matrix(M: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(M, dtype=float),
        precision=precision,
        suppress_small=True,
        floatmode="fixed",
    )
