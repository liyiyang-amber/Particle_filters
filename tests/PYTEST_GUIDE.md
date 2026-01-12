# Pytest Configuration Guide

This document explains the pytest configuration for the State-Space Models & Filters project.

## Configuration File

The project uses `pytest.ini` for pytest configuration, located in the project root.

## Custom Markers

We use custom markers to categorize tests:

### `@pytest.mark.integration`
Marks tests as integration tests. These tests verify that multiple components work together correctly.

**Usage:**
```python
import pytest

@pytest.mark.integration
def test_ekf_with_simulator():
    # Integration test code
    pass
```

**Running:**
```bash
# Run only integration tests
pytest -m integration

# Run all except integration tests
pytest -m "not integration"
```

### `@pytest.mark.slow`
Marks tests that take longer to run (typically > 1 second).

**Usage:**
```python
import pytest

@pytest.mark.slow
def test_large_scale_simulation():
    # Slow test code
    pass
```

**Running:**
```bash
# Run only slow tests
pytest -m slow

# Run all except slow tests
pytest -m "not slow"

# Run all except integration and slow tests
pytest -m "not integration and not slow"
```

### `@pytest.mark.tensorflow`
Marks tests that require TensorFlow (used by Differentiable Particle Filter / DPF components).

**Running:**
```bash
# Run only TensorFlow-dependent tests
pytest -m tensorflow

# Run everything except TensorFlow-dependent tests
pytest -m "not tensorflow"
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests in a specific directory
pytest tests/unit_tests/

# Run tests in a specific file
pytest tests/unit_tests/simulator/test_sv_basic_api.py

# Run a specific test
pytest tests/unit_tests/simulator/test_sv_basic_api.py::test_shapes_and_params
```

### With Markers

```bash
# Run only integration tests
pytest -m integration

# Run only unit tests (exclude integration)
pytest -m "not integration"

# Run fast tests only (exclude slow and integration)
pytest -m "not slow and not integration"
```

### Output Options

```bash
# Show detailed output
pytest -v

# Show test durations
pytest --durations=10

# Show all test results (not just failures)
pytest -ra

# Stop on first failure
pytest -x

# Run last failed tests only
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Coverage

```bash
# Run with coverage
pytest --cov=models --cov=simulator

# Generate HTML coverage report
pytest --cov=models --cov=simulator --cov-report=html
```

## Configuration Details

The `pytest.ini` file configures:

- **Test Discovery**: Automatically finds files matching `test_*.py` pattern
- **Test Paths**: Looks in the `tests/` directory
- **Markers**: Registers `integration`, `slow`, and `tensorflow` markers
- **Output**: Uses short traceback format and progress-style output
- **Strict Markers**: Raises errors for unregistered markers (prevents typos)

## Manual Test Runner

For a more user-friendly experience, use the manual test runner:

```bash
# Show all available options
python tests/manual_run.py --help

# Show test statistics
python tests/manual_run.py --summary

# Run specific test phases
python tests/manual_run.py --phase sv         # SV simulator only
python tests/manual_run.py --phase simulator  # All simulators
python tests/manual_run.py --phase filters    # All filters
python tests/manual_run.py --phase integration # Integration tests
python tests/manual_run.py --phase all        # Everything

# Run with verbose output
python tests/manual_run.py --phase sv --verbose
```

## Troubleshooting

### TensorFlow / DPF tests

Some Differentiable Particle Filter (DPF) tests depend on TensorFlow.

- If you see `ModuleNotFoundError: No module named 'tensorflow'`, install dependencies from the repo root:

```bash
python -m pip install -r requirements.txt
```

- To run only DPF-related unit tests:

```bash
pytest tests/unit_tests/models -k dpf
```

### Warning: Unknown pytest.mark.X

If you see warnings about unknown markers, make sure:
1. The marker is registered in `pytest.ini`
2. You're running pytest from the project root
3. The marker name matches exactly (case-sensitive)

### Tests Not Found

If tests aren't being discovered:
1. Make sure test files match the `test_*.py` pattern
2. Test functions should start with `test_`
3. Run from the project root directory
4. Check that `__init__.py` exists if needed

### Import Errors

If you see import errors:
1. Make sure you're in the project root
2. Check that modules are installed in your virtual environment
3. Activate the virtual environment: `source .venv/bin/activate`

## Best Practices

1. **Mark appropriately**: Use `@pytest.mark.integration` for integration tests and `@pytest.mark.slow` for tests that take > 1 second
2. **Test isolation**: Each test should be independent and not rely on execution order
3. **Fixtures**: Use pytest fixtures for common setup/teardown
4. **Assertions**: Use descriptive assertion messages
5. **Fast tests**: Keep unit tests fast (< 0.1s when possible) for quick feedback

## Example Test Structure

```python
import pytest
import numpy as np
from simulator.simulator_sto_volatility_model import simulate_sv_1d


@pytest.fixture
def sv_params():
    """Standard parameters for 1D SV model."""
    return {
        "alpha": 0.9,
        "sigma": 0.2,
        "beta": 1.0,
        "n": 500,
        "seed": 42
    }


def test_basic_simulation(sv_params):
    """Unit test - should be fast."""
    result = simulate_sv_1d(**sv_params)
    assert result.X.shape == (500,)
    assert result.Y.shape == (500,)


@pytest.mark.slow
def test_large_scale_simulation(sv_params):
    """Slow test - takes more time."""
    sv_params["n"] = 100000
    result = simulate_sv_1d(**sv_params)
    # Statistical checks requiring large n
    assert np.abs(result.X.mean()) < 0.1


@pytest.mark.integration
def test_filter_with_simulator(sv_params):
    """Integration test - tests multiple components."""
    from models.extended_kalman_filter import ExtendedKalmanFilter
    
    # Simulate data
    result = simulate_sv_1d(**sv_params)
    
    # Set up filter
    ekf = ExtendedKalmanFilter(...)
    
    # Run filtering and verify
    # ...
```
