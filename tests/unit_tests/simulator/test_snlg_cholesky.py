import numpy as np
import pytest
from simulator.simulator_sensor_network_linear_gaussian import cholesky_with_jitter


def test_cholesky_spd_matrix():
    """Test that a well-conditioned SPD matrix factors without jitter."""
    # Simple SPD matrix
    A = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])
    
    L = cholesky_with_jitter(A)
    
    # Verify L @ L.T ≈ A
    assert np.allclose(L @ L.T, A)
    
    # Verify L is lower triangular
    assert np.allclose(L, np.tril(L))


def test_cholesky_diagonal_matrix():
    """Test Cholesky on a diagonal matrix."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    L = cholesky_with_jitter(A)
    
    assert np.allclose(L @ L.T, A)
    # For diagonal matrix, L should also be diagonal
    assert np.allclose(L, np.diag(np.sqrt([1.0, 2.0, 3.0, 4.0])))


def test_cholesky_identity_matrix():
    """Test Cholesky on identity matrix."""
    A = np.eye(5)
    L = cholesky_with_jitter(A)
    
    assert np.allclose(L, np.eye(5))


def test_cholesky_larger_spd_matrix():
    """Test Cholesky on a larger well-conditioned SPD matrix."""
    # Create a random SPD matrix
    rng = np.random.default_rng(42)
    M = rng.normal(size=(10, 10))
    A = M @ M.T + np.eye(10)  # Guaranteed SPD
    
    L = cholesky_with_jitter(A)
    
    assert np.allclose(L @ L.T, A)
    assert np.allclose(L, np.tril(L))


def test_cholesky_nearly_singular_matrix():
    """Test that jitter is added for nearly singular matrices."""
    # Create a nearly singular matrix (very small eigenvalue)
    A = np.array([
        [1.0, 0.999],
        [0.999, 1.0],
    ])
    
    # This might fail without jitter, but should succeed with it
    L = cholesky_with_jitter(A)
    
    # The result should be approximately correct (with some jitter added)
    reconstructed = L @ L.T
    # Allow some tolerance since jitter was added
    assert np.allclose(reconstructed, A, atol=1e-6)


def test_cholesky_ill_conditioned_matrix():
    """Test Cholesky on an ill-conditioned but SPD matrix."""
    # Create ill-conditioned SPD matrix
    A = np.array([
        [1.0, 1.0 - 1e-10],
        [1.0 - 1e-10, 1.0],
    ])
    
    L = cholesky_with_jitter(A, max_tries=5, base_jitter=1e-10)
    
    # Should complete without error
    assert L.shape == (2, 2)
    assert np.allclose(L, np.tril(L))


def test_cholesky_lower_triangular():
    """Test that the result is always lower triangular."""
    matrices = [
        np.eye(3),
        np.diag([1.0, 2.0, 3.0]),
        np.array([[2.0, 1.0], [1.0, 2.0]]),
    ]
    
    for A in matrices:
        L = cholesky_with_jitter(A)
        # Upper triangle (excluding diagonal) should be zero
        assert np.allclose(L - np.tril(L), 0.0)


def test_cholesky_reconstruction_accuracy():
    """Test that L @ L.T accurately reconstructs the original matrix."""
    # Create various SPD matrices
    rng = np.random.default_rng(123)
    
    for size in [2, 5, 10]:
        M = rng.normal(size=(size, size))
        A = M @ M.T + 0.1 * np.eye(size)
        
        L = cholesky_with_jitter(A)
        reconstructed = L @ L.T
        
        rel_error = np.linalg.norm(A - reconstructed) / np.linalg.norm(A)
        assert rel_error < 1e-10


def test_cholesky_determinism():
    """Test that the function is deterministic for the same input."""
    A = np.array([
        [2.0, 1.0, 0.5],
        [1.0, 3.0, 0.8],
        [0.5, 0.8, 2.5],
    ])
    
    L1 = cholesky_with_jitter(A)
    L2 = cholesky_with_jitter(A)
    
    assert np.array_equal(L1, L2)


def test_cholesky_max_tries_parameter():
    """Test that max_tries parameter is respected."""
    A = np.array([
        [1.0, 0.9999],
        [0.9999, 1.0],
    ])
    
    # Should work with different max_tries values
    L1 = cholesky_with_jitter(A, max_tries=1)
    L2 = cholesky_with_jitter(A, max_tries=5)
    
    # Both should succeed 
    assert L1.shape == (2, 2)
    assert L2.shape == (2, 2)


def test_cholesky_base_jitter_parameter():
    """Test that base_jitter parameter affects the result."""
    # Matrix that might need jitter
    A = np.array([
        [1.0, 0.99999],
        [0.99999, 1.0],
    ])
    
    # Try with different base jitter values
    L_small = cholesky_with_jitter(A, base_jitter=1e-12)
    L_large = cholesky_with_jitter(A, base_jitter=1e-8)
    
    # Both should succeed
    assert L_small.shape == (2, 2)
    assert L_large.shape == (2, 2)


def test_cholesky_non_spd_eventually_fails():
    """Test that a truly non-SPD matrix eventually fails."""
    # Create a non-SPD matrix (negative eigenvalue)
    A = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    # This has eigenvalues 3 and -1, so it's not SPD
    
    # Even with jitter, this should eventually fail if max_tries is reasonable
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_with_jitter(A, max_tries=2, base_jitter=1e-12)


def test_cholesky_preserves_spd_property():
    """Test that the result produces an SPD matrix when reconstructed."""
    rng = np.random.default_rng(99)
    M = rng.normal(size=(8, 8))
    A = M @ M.T + 0.01 * np.eye(8)
    
    L = cholesky_with_jitter(A)
    reconstructed = L @ L.T
    
    # Check that reconstructed is SPD (all eigenvalues positive)
    eigvals = np.linalg.eigvalsh(reconstructed)
    assert np.all(eigvals > 0)
