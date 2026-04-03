"""
dct.py – Discrete Cosine Transform (DCT) utilities
====================================================
A NumPy implementation of the 1-D and 2-D DCT-II and its inverse (IDCT-II),
designed for learning purposes.

Key concepts illustrated
------------------------
* DCT-II formula       – how cosine basis functions represent a signal
* Energy compaction    – why most energy ends up in low-frequency coefficients
* Reconstruction       – how discarding high-frequency coefficients causes blur
* 2-D extension        – applying 1-D DCT along rows then columns
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1-D DCT-II  (the "standard" DCT used in JPEG, MP3, …)
# ---------------------------------------------------------------------------

def dct1d(x: np.ndarray) -> np.ndarray:
    """Compute the 1-D DCT-II of a real sequence *x*.

    The transform is defined as::

        X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n+1) / (2N))

    for k = 0, 1, …, N-1.

    Parameters
    ----------
    x : 1-D array of length N

    Returns
    -------
    X : 1-D array of length N  (DCT coefficients)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n[:, np.newaxis]          # column vector
    cos_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return 2.0 * cos_matrix @ x


def idct1d(X: np.ndarray) -> np.ndarray:
    """Compute the inverse 1-D DCT-II (i.e. DCT-III / N).

    The inverse is::

        x[n] = (X[0] + 2 * sum_{k=1}^{N-1} X[k] * cos(pi*k*(2n+1)/(2N))) / (2N)

    Parameters
    ----------
    X : 1-D array of N DCT coefficients

    Returns
    -------
    x : 1-D array of length N (reconstructed signal)
    """
    X = np.asarray(X, dtype=float)
    N = len(X)
    n = np.arange(N)
    k = np.arange(1, N)
    result = np.empty(N)
    for i in range(N):
        cos_sum = np.sum(X[k] * np.cos(np.pi * k * (2 * i + 1) / (2 * N)))
        result[i] = (X[0] + 2.0 * cos_sum) / (2.0 * N)
    return result


# ---------------------------------------------------------------------------
# 2-D DCT-II  (used for image compression)
# ---------------------------------------------------------------------------

def dct2d(block: np.ndarray) -> np.ndarray:
    """Compute the 2-D DCT-II by applying 1-D DCT along rows then columns.

    Parameters
    ----------
    block : 2-D array of shape (M, N)

    Returns
    -------
    B : 2-D array of shape (M, N) – DCT coefficient matrix
    """
    block = np.asarray(block, dtype=float)
    # Apply 1-D DCT to each row
    row_transformed = np.apply_along_axis(dct1d, axis=1, arr=block)
    # Apply 1-D DCT to each column
    return np.apply_along_axis(dct1d, axis=0, arr=row_transformed)


def idct2d(B: np.ndarray) -> np.ndarray:
    """Compute the inverse 2-D DCT-II.

    Parameters
    ----------
    B : 2-D array of shape (M, N) – DCT coefficients

    Returns
    -------
    block : 2-D array of shape (M, N) – reconstructed values
    """
    B = np.asarray(B, dtype=float)
    # Inverse along columns first, then rows
    col_restored = np.apply_along_axis(idct1d, axis=0, arr=B)
    return np.apply_along_axis(idct1d, axis=1, arr=col_restored)


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def compress_1d(x: np.ndarray, keep: int) -> np.ndarray:
    """Zero out all but the first *keep* DCT coefficients, then reconstruct.

    This illustrates *energy compaction*: most of the signal information is
    stored in the low-frequency (low-index) coefficients.

    Parameters
    ----------
    x    : original 1-D signal
    keep : number of leading coefficients to retain (1 … N)

    Returns
    -------
    x_reconstructed : 1-D array (same length as *x*)
    """
    X = dct1d(x)
    X_truncated = np.zeros_like(X)
    X_truncated[:keep] = X[:keep]
    return idct1d(X_truncated)


def compress_2d(block: np.ndarray, keep_rows: int, keep_cols: int) -> np.ndarray:
    """Zero out DCT coefficients outside the top-left *keep* corner.

    Parameters
    ----------
    block     : 2-D array (e.g. an 8×8 image patch)
    keep_rows : how many rows of coefficients to retain
    keep_cols : how many columns of coefficients to retain

    Returns
    -------
    reconstructed : 2-D array (same shape as *block*)
    """
    B = dct2d(block)
    B_truncated = np.zeros_like(B)
    B_truncated[:keep_rows, :keep_cols] = B[:keep_rows, :keep_cols]
    return idct2d(B_truncated)


# ---------------------------------------------------------------------------
# Demo signal / image generators
# ---------------------------------------------------------------------------

def make_demo_signal(N: int = 64) -> np.ndarray:
    """Return a smooth test signal (sum of a few sinusoids)."""
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return (
        3.0 * np.sin(t)
        + 1.5 * np.sin(3 * t)
        + 0.5 * np.sin(7 * t)
        + 0.2 * np.random.default_rng(42).normal(size=N)
    )


def make_demo_image(size: int = 32) -> np.ndarray:
    """Return a simple gradient + checkerboard test image (values 0–255)."""
    x = np.linspace(0, 1, size)
    gradient = np.outer(x, x) * 200
    checker = np.indices((size, size)).sum(axis=0) % 2 * 30
    return np.clip(gradient + checker, 0, 255)
