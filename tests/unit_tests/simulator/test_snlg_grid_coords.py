import numpy as np
import pytest
from simulator.simulator_sensor_network_linear_gaussian import make_grid_coords


def test_grid_coords_shape_4x4():
    """Test that grid coordinates have the correct shape for a 4x4 grid."""
    d = 16  # 4x4 grid
    coords = make_grid_coords(d)
    assert coords.shape == (16, 2)


def test_grid_coords_shape_8x8():
    """Test that grid coordinates have the correct shape for an 8x8 grid."""
    d = 64  # 8x8 grid
    coords = make_grid_coords(d)
    assert coords.shape == (64, 2)


def test_grid_coords_dtype():
    """Test that coordinates are float type."""
    coords = make_grid_coords(16)
    assert coords.dtype == np.float64 or coords.dtype == float


def test_grid_coords_range_2x2():
    """Test that coordinates are in the expected range for a 2x2 grid."""
    d = 4  # 2x2 grid
    coords = make_grid_coords(d)
    # Coordinates should be in [0, 1] for a 2x2 grid
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


def test_grid_coords_values_2x2():
    """Test exact coordinate values for a small 2x2 grid."""
    d = 4  # 2x2 grid
    coords = make_grid_coords(d)
    
    # Expected coordinates in row-major order:
    # (0,0), (0,1), (1,0), (1,1)
    expected = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    
    assert np.allclose(coords, expected)


def test_grid_coords_values_3x3():
    """Test exact coordinate values for a 3x3 grid."""
    d = 9  # 3x3 grid
    coords = make_grid_coords(d)
    
    # Expected coordinates in row-major order:
    expected = np.array([
        [0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
        [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
        [2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
    ])
    
    assert np.allclose(coords, expected)


def test_grid_coords_unique_points():
    """Test that all coordinate points are unique."""
    d = 64  # 8x8 grid
    coords = make_grid_coords(d)
    
    # Convert to tuples for set uniqueness check
    unique_coords = set(map(tuple, coords))
    assert len(unique_coords) == d


def test_grid_coords_row_major_ordering():
    """Test that coordinates follow row-major ordering."""
    d = 16  # 4x4 grid
    coords = make_grid_coords(d)
    
    # First row should have x=0 and varying y
    assert np.all(coords[0:4, 0] == 0.0)
    assert np.allclose(coords[0:4, 1], [0, 1, 2, 3])
    
    # Second row should have x=1
    assert np.all(coords[4:8, 0] == 1.0)
    assert np.allclose(coords[4:8, 1], [0, 1, 2, 3])


def test_grid_coords_perfect_square_1():
    """Test that d=1 (1x1 grid) works."""
    coords = make_grid_coords(1)
    assert coords.shape == (1, 2)
    assert np.allclose(coords, [[0.0, 0.0]])


def test_grid_coords_large_grid():
    """Test that large grids work correctly."""
    d = 400  # 20x20 grid
    coords = make_grid_coords(d)
    assert coords.shape == (400, 2)
    assert coords.min() >= 0.0
    assert coords.max() <= 19.0


def test_grid_coords_integer_valued():
    """Test that all coordinates are integer-valued (as floats)."""
    d = 25  # 5x5 grid
    coords = make_grid_coords(d)
    
    # Check that coordinates are close to their rounded values
    assert np.allclose(coords, np.round(coords))


def test_grid_coords_deterministic():
    """Test that the function is deterministic (same output for same input)."""
    d = 64
    coords1 = make_grid_coords(d)
    coords2 = make_grid_coords(d)
    assert np.array_equal(coords1, coords2)


def test_grid_coords_corner_positions():
    """Test that the four corners are correctly positioned."""
    d = 16  # 4x4 grid
    coords = make_grid_coords(d)
    
    # Bottom-left corner (0,0) should be first
    assert np.allclose(coords[0], [0.0, 0.0])
    
    # Bottom-right corner (0,3) should be at index 3
    assert np.allclose(coords[3], [0.0, 3.0])
    
    # Top-left corner (3,0) should be at index 12
    assert np.allclose(coords[12], [3.0, 0.0])
    
    # Top-right corner (3,3) should be last
    assert np.allclose(coords[15], [3.0, 3.0])
