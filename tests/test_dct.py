"""
tests/test_dct.py – Unit tests for the DCT utilities
"""

import numpy as np
import pytest

from dct import (
    dct1d,
    idct1d,
    dct2d,
    idct2d,
    compress_1d,
    compress_2d,
    make_demo_signal,
    make_demo_image,
)


# ---------------------------------------------------------------------------
# Round-trip: DCT then IDCT should recover the original
# ---------------------------------------------------------------------------

def test_dct1d_roundtrip():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(idct1d(dct1d(x)), x, atol=1e-9)


def test_dct2d_roundtrip():
    block = np.arange(16, dtype=float).reshape(4, 4)
    assert np.allclose(idct2d(dct2d(block)), block, atol=1e-9)


# ---------------------------------------------------------------------------
# DCT of a constant signal should pack all energy in X[0]
# ---------------------------------------------------------------------------

def test_dct1d_constant_signal():
    """For a constant signal all energy is in the DC component X[0]."""
    x = np.ones(8)
    X = dct1d(x)
    # X[0] should be 2*N (each term is 2 * cos(0) = 2), others ~0
    assert abs(X[0]) == pytest.approx(2 * 8, rel=1e-9)
    assert np.allclose(X[1:], 0, atol=1e-9)


# ---------------------------------------------------------------------------
# compress_1d: keeping all coefficients → perfect reconstruction
# ---------------------------------------------------------------------------

def test_compress_1d_keep_all():
    x = make_demo_signal(N=32)
    x_rec = compress_1d(x, keep=len(x))
    assert np.allclose(x_rec, x, atol=1e-9)


def test_compress_1d_keep_one():
    """Keeping only 1 coefficient should still return an array of the right size."""
    x = make_demo_signal(N=16)
    x_rec = compress_1d(x, keep=1)
    assert len(x_rec) == len(x)


# ---------------------------------------------------------------------------
# compress_2d: keeping all → perfect reconstruction
# ---------------------------------------------------------------------------

def test_compress_2d_keep_all():
    img = make_demo_image(size=8)
    M, N = img.shape
    img_rec = compress_2d(img, keep_rows=M, keep_cols=N)
    assert np.allclose(img_rec, img, atol=1e-9)


# ---------------------------------------------------------------------------
# Energy compaction: most energy in first half of coefficients
# ---------------------------------------------------------------------------

def test_energy_compaction():
    x = make_demo_signal(N=64)
    X = dct1d(x)
    N = len(X)
    energy_low = np.sum(X[: N // 2] ** 2)
    energy_total = np.sum(X ** 2)
    # At least 90 % of energy in the low half of coefficients
    assert energy_low / energy_total >= 0.90


# ---------------------------------------------------------------------------
# Generators return correct shapes
# ---------------------------------------------------------------------------

def test_make_demo_signal_shape():
    assert make_demo_signal(N=64).shape == (64,)


def test_make_demo_image_shape():
    img = make_demo_image(size=32)
    assert img.shape == (32, 32)
    assert img.min() >= 0
    assert img.max() <= 255
