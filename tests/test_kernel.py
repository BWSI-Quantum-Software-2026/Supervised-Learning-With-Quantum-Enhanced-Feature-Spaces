# this should be the testing command python -m pytest tests/test_kernel.py

import numpy as np
import pytest
from src.kernel.overlap_circuit import estimate_kernel_entry
from src.kernel.kernel_matrix import compute_kernel_matrix

TOL = 0.05  # shot-noise tolerance


def test_self_overlap_is_one():
    # compare a point with itself, should be basically 1
    x = np.array([0.5, 1.2])
    k = estimate_kernel_entry(x, x, shots=8192, seed = 42)
    assert abs(k - 1.0) < TOL


def test_kernel_in_unit_interval():
    # try a few random pairs and make sure none go out of bounds
    for _ in range(3):
        x = np.random.uniform(0, 2 * np.pi, 2)
        z = np.random.uniform(0, 2 * np.pi, 2)
        k = estimate_kernel_entry(x, z, shots = 8192, seed =42)
        assert 0 <= k <= 1


def test_kernel_symmetry():
    # flipping the order shouldnt change the answer
    x = np.array([0.5, 1.2])
    z = np.array([2.0, 0.3])
    assert abs(estimate_kernel_entry(x, z, shots=8192, seed=42) - estimate_kernel_entry(z, x, shots=8192, seed=42)) < TOL


def test_matrix_is_symmetric_with_unit_diagonal():
    # build a lil matrix and check the diagonal is 1 and its mirrored
    A = [np.array([0.5, 1.2]), np.array([2.0, 0.3]), np.array([1.0, 1.0])]
    K = compute_kernel_matrix(A, None, shots=8192, seed=42)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T, atol=TOL)


def test_reproducible_with_fixed_seed():
    # same seed = same exact answer twice
    x = np.array([0.5, 1.2])
    z = np.array([2.0, 0.3])
    assert estimate_kernel_entry(x, z, shots=8192, seed=42) == estimate_kernel_entry(x, z, shots=8192, seed=42)