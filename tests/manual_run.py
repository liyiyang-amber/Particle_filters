#!/usr/bin/env python3
"""
Ma    # Run only SV tests
    python tests/manual_run.py --phase sv
    
    # Run only MAT tests
    python tests/manual_run.py --phase mat
    
    # Run only Particle Filter tests Test Runner for State-Space Models and Filters

This script provides a convenient way to run tests in phases or all at once,
with nice formatting and summary statistics.

Usage:
    python tests/manual_run.py                    # Run all tests
    python tests/manual_run.py --phase all        # Run all tests (explicit)
    python tests/manual_run.py --phase simulator  # Run all simulator tests (LGSSM + SV + MAT + SNLG + SNLG Skew-t)
    python tests/manual_run.py --phase lgssm      # Run only LGSSM simulator tests
    python tests/manual_run.py --phase sv         # Run only SV simulator tests
    python tests/manual_run.py --phase mat        # Run only MAT simulator tests
    python tests/manual_run.py --phase snlg       # Run only SNLG simulator tests
    python tests/manual_run.py --phase snlg-skewt # Run only SNLG Skew-t simulator tests
    python tests/manual_run.py --phase kf         # Run only Kalman filter tests
    python tests/manual_run.py --phase ekf        # Run only Extended Kalman filter tests
    python tests/manual_run.py --phase ukf        # Run only Unscented Kalman filter tests
    python tests/manual_run.py --phase pf         # Run only Particle filter tests
    python tests/manual_run.py --phase edh        # Run only EDH particle filter tests
    python tests/manual_run.py --phase ledh       # Run only LEDH particle filter tests
    python tests/manual_run.py --phase filters    # Run all filter tests (KF + EKF + UKF + PF + EDH + LEDH)
    python tests/manual_run.py --phase integration # Run only integration tests
    python tests/manual_run.py --phase mat-filters # Run MAT filter integration tests
    python tests/manual_run.py --phase snlg-filters # Run SNLG filter integration tests
    python tests/manual_run.py --phase snlg-skewt-filters # Run SNLG Skew-t filter integration tests
    python tests/manual_run.py --verbose          # Show verbose output
    python tests/manual_run.py --summary          # Show test statistics summary

Examples:
    # Run all tests with verbose output
    python tests/manual_run.py --verbose
    
    # Run only EKF tests
    python tests/manual_run.py --phase ekf
    
    # Run only Particle Filter tests
    python tests/manual_run.py --phase pf
    
    # Run only EDH Particle Filter tests
    python tests/manual_run.py --phase edh
    
    # Run only LEDH Particle Filter tests
    python tests/manual_run.py --phase ledh
    
    # Run all filter tests (KF + EKF + UKF + PF + EDH + LEDH)
    python tests/manual_run.py --phase filters
    
    # Show test statistics without running tests
    python tests/manual_run.py --summary
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str, char: str = "="):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * 80}{Colors.END}\n")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.END}\n")


def run_pytest(test_paths: List[str], description: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Run pytest on specified paths and return results.
    
    Returns:
        Tuple of (success: bool, passed: int, failed: int)
    """
    print_section(f"Running: {description}")
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"] + test_paths
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-v", "--tb=short"])
    
    # Add summary options
    cmd.extend(["--no-header", "-q"])
    
    print(f"{Colors.YELLOW}Command: {' '.join(cmd)}{Colors.END}\n")
    
    # Run pytest
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(f"{Colors.RED}{result.stderr}{Colors.END}")
    
    # Parse results from output
    passed = failed = 0
    for line in result.stdout.split('\n'):
        if 'passed' in line or 'failed' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'passed' in part and i > 0:
                    try:
                        passed = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                if 'failed' in part and i > 0:
                    try:
                        failed = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
    
    success = result.returncode == 0
    
    # Print result
    if success:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.END}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.END}\n")
    
    return success, passed, failed


def run_lgssm_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run LGSSM simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_lgssm_shapes_and_seed.py",
        "tests/unit_tests/simulator/test_lgssm_burnin_and_stats.py",
        "tests/unit_tests/simulator/test_lgssm_io_roundtrip.py"
    ]
    return run_pytest(test_paths, "LGSSM Simulator Unit Tests", verbose)


def run_sv_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Stochastic Volatility simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_sv_basic_api.py",
        "tests/unit_tests/simulator/test_sv_observations.py",
        "tests/unit_tests/simulator/test_sv_statistics.py",
        "tests/unit_tests/simulator/test_sv_io_and_stability.py"
    ]
    return run_pytest(test_paths, "Stochastic Volatility (SV) Simulator Unit Tests", verbose)


def run_mat_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Multi-Target Acoustic Tracking (MAT) simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_mat_shapes_and_seed.py",
        "tests/unit_tests/simulator/test_mat_cv_dynamics.py",
        "tests/unit_tests/simulator/test_mat_measurement.py",
        "tests/unit_tests/simulator/test_mat_end2end.py"
    ]
    return run_pytest(test_paths, "Multi-Target Acoustic Tracking (MAT) Simulator Unit Tests", verbose)


def run_snlg_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Sensor Network Linear Gaussian (SNLG) simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_snlg_config_validation.py",
        "tests/unit_tests/simulator/test_snlg_grid_coords.py",
        "tests/unit_tests/simulator/test_snlg_kernel.py",
        "tests/unit_tests/simulator/test_snlg_cholesky.py",
        "tests/unit_tests/simulator/test_snlg_simulation.py",
        "tests/unit_tests/simulator/test_snlg_io.py"
    ]
    return run_pytest(test_paths, "Sensor Network Linear Gaussian (SNLG) Simulator Unit Tests", verbose)


def run_sn_skewt_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Sensor Network GH Skew-t (SNLG Skew-t) simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_sn_skewt_utility_funcs.py",
        "tests/unit_tests/simulator/test_sn_skewt_config.py",
        "tests/unit_tests/simulator/test_sn_skewt_simulation.py",
        "tests/unit_tests/simulator/test_sn_skewt_io.py"
    ]
    return run_pytest(test_paths, "Sensor Network GH Skew-t (SNLG Skew-t) Simulator Unit Tests", verbose)


def run_lorenz96_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Lorenz 96 simulator unit tests"""
    test_paths = [
        "tests/unit_tests/simulator/test_lorenz96_shapes_and_seed.py",
        "tests/unit_tests/simulator/test_lorenz96_dynamics.py",
        "tests/unit_tests/simulator/test_lorenz96_io.py"
    ]
    return run_pytest(test_paths, "Lorenz 96 Simulator Unit Tests", verbose)


def run_simulator_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run all simulator tests (LGSSM + SV + MAT + SNLG + SNLG Skew-t + Lorenz96)"""
    test_paths = ["tests/unit_tests/simulator/"]
    return run_pytest(test_paths, "All Simulator Unit Tests (LGSSM + SV + MAT + SNLG + SNLG Skew-t + Lorenz96)", verbose)


def run_kf_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Kalman filter unit tests"""
    test_paths = [
        "tests/unit_tests/models/test_kf_shapes.py",
        "tests/unit_tests/models/test_kf_joseph_and_psd.py",
        "tests/unit_tests/models/test_kf_controls_and_errors.py",
        "tests/unit_tests/models/test_kf_timevarying_equivalence.py"
    ]
    return run_pytest(test_paths, "Kalman Filter (KF) Unit Tests", verbose)


def run_ekf_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Extended Kalman filter unit tests"""
    test_paths = [
        "tests/unit_tests/models/test_ekf_shapes_and_api.py",
        "tests/unit_tests/models/test_ekf_innovation_and_gains.py"
    ]
    return run_pytest(test_paths, "Extended Kalman Filter (EKF) Unit Tests", verbose)


def run_ukf_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Unscented Kalman filter unit tests"""
    test_paths = [
        "tests/unit_tests/models/test_ukf_shapes_and_api.py",
        "tests/unit_tests/models/test_ukf_sigma_points_and_weights.py"
    ]
    return run_pytest(test_paths, "Unscented Kalman Filter (UKF) Unit Tests", verbose)


def run_pf_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Particle filter unit tests"""
    test_paths = [
        "tests/unit_tests/models/test_pf_shapes_and_api.py",
        "tests/unit_tests/models/test_pf_resampling.py"
    ]
    return run_pytest(test_paths, "Particle Filter (PF) Unit Tests", verbose)


def run_edh_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run EDH Particle filter tests (unit + integration)"""
    test_paths = [
        "tests/unit_tests/models/test_ekf_tracker_wrapper.py",
        "tests/unit_tests/models/test_ukf_tracker_wrapper.py",
        "tests/unit_tests/models/test_edh_flow_pf.py",
        "tests/integration_tests/test_edh_ekf_vs_simulator_sv.py",
        "tests/integration_tests/test_edh_ukf_vs_simulator_sv.py"
    ]
    return run_pytest(test_paths, "EDH Particle Filter (EDH-PF) Tests - EKF/UKF Trackers + Integration", verbose)


def run_ledh_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run LEDH Particle filter tests (unit + integration)"""
    test_paths = [
        "tests/unit_tests/models/test_ledh_flow_pf.py",
        "tests/integration_tests/test_ledh_ekf_vs_simulator_sv.py",
        "tests/integration_tests/test_ledh_ukf_vs_simulator_sv.py"
    ]
    return run_pytest(test_paths, "LEDH Particle Filter (LEDH-PF) Tests - Per-Particle Linearization", verbose)


def run_kernel_pf_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Kernel Particle filter unit tests"""
    test_paths = [
        "tests/unit_tests/models/test_kernel_pf_shapes_and_api.py",
        "tests/unit_tests/models/test_kernel_pf_kernels.py"
    ]
    return run_pytest(test_paths, "Kernel Particle Filter (KPF) Unit Tests", verbose)


def run_all_filter_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run all filter tests (KF + EKF + UKF + PF + EDH + LEDH)"""
    test_paths = ["tests/unit_tests/models/"]
    return run_pytest(test_paths, "All Filter Unit Tests (KF + EKF + UKF + PF + EDH + LEDH)", verbose)


def run_integration_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run integration tests"""
    test_paths = ["tests/integration_tests/"]
    return run_pytest(test_paths, "Integration Tests (Including MAT Filter Integration)", verbose)


def run_mat_filter_integration_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run MAT filter integration tests (EKF, UKF, EDH, LEDH)"""
    test_paths = ["tests/integration_tests/test_filters_mat_simulator.py"]
    return run_pytest(test_paths, "MAT Filter Integration Tests (EKF + UKF + EDH + LEDH)", verbose)


def run_snlg_filter_integration_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run SNLG filter integration tests (consolidated: EKF, UKF, EDH, LEDH)"""
    test_paths = [
        "tests/integration_tests/test_filters_snlg_simulator.py"
    ]
    return run_pytest(test_paths, "SNLG Filter Integration Tests (EKF + UKF + EDH + LEDH)", verbose)


def run_sn_skewt_filter_integration_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Skew-t filter integration tests (consolidated: EKF, UKF, EDH, LEDH)"""
    test_paths = [
        "tests/integration_tests/test_filters_skewt_simulator.py"
    ]
    return run_pytest(test_paths, "Skew-t Filter Integration Tests (EKF + UKF + EDH + LEDH)", verbose)


def run_kernel_pf_integration_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run Kernel Particle Filter integration tests (LGSSM + Lorenz96)"""
    test_paths = [
        "tests/integration_tests/test_kpf_vs_simulator_lgssm.py",
        "tests/integration_tests/test_kpf_vs_simulator_lorenz96.py"
    ]
    return run_pytest(test_paths, "Kernel Particle Filter (KPF) Integration Tests (LGSSM + Lorenz96)", verbose)


def run_all_tests(verbose: bool = False) -> Tuple[bool, int, int]:
    """Run all tests in sequence"""
    all_passed = []
    all_failed = []
    
    # Phase 1: Simulator tests
    print_section("PHASE 1: Simulators (LGSSM + SV + MAT + SNLG + SNLG Skew-t + Lorenz96)")
    success, passed, failed = run_simulator_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 2: Kalman filter tests
    print_section("PHASE 2: Kalman Filter")
    success, passed, failed = run_kf_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 3: Extended Kalman filter tests
    print_section("PHASE 3: Extended Kalman Filter")
    success, passed, failed = run_ekf_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 4: Unscented Kalman filter tests
    print_section("PHASE 4: Unscented Kalman Filter")
    success, passed, failed = run_ukf_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 5: Particle filter tests
    print_section("PHASE 5: Particle Filter")
    success, passed, failed = run_pf_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 6: EDH Particle filter tests
    print_section("PHASE 6: EDH Particle Filter (EKF/UKF Trackers)")
    success, passed, failed = run_edh_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 7: LEDH Particle filter tests
    print_section("PHASE 7: LEDH Particle Filter (Per-Particle Linearization)")
    success, passed, failed = run_ledh_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 8: Kernel Particle filter tests
    print_section("PHASE 8: Kernel Particle Filter (Matrix-valued Kernels)")
    success, passed, failed = run_kernel_pf_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 9: Integration tests
    print_section("PHASE 8: Integration Tests")
    success, passed, failed = run_integration_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 9: MAT Filter Integration (EKF, UKF, EDH, LEDH)
    print_section("PHASE 9: MAT Filter Integration (EKF + UKF + EDH + LEDH)")
    success, passed, failed = run_mat_filter_integration_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 10: SNLG Filter Integration (KF, UKF, EKF, EDH, LEDH)
    print_section("PHASE 10: SNLG Filter Integration (KF + UKF + EKF + EDH + LEDH)")
    success, passed, failed = run_snlg_filter_integration_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    # Phase 11: SNLG Skew-t Filter Integration (EKF, UKF, PF, EDH, LEDH)
    print_section("PHASE 11: SNLG Skew-t Filter Integration (EKF + UKF + PF + EDH + LEDH)")
    success, passed, failed = run_sn_skewt_filter_integration_tests(verbose)
    all_passed.append(passed)
    all_failed.append(failed)
    
    total_passed = sum(all_passed)
    total_failed = sum(all_failed)
    total_success = total_failed == 0
    
    return total_success, total_passed, total_failed


def print_summary(passed: int, failed: int, total_time: float = None):
    """Print final summary"""
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print_header("TEST SUMMARY", "=")
    
    print(f"{Colors.BOLD}Total Tests:{Colors.END} {total}")
    print(f"{Colors.GREEN}{Colors.BOLD}Passed:{Colors.END} {passed}")
    print(f"{Colors.RED}{Colors.BOLD}Failed:{Colors.END} {failed}")
    print(f"{Colors.BOLD}Pass Rate:{Colors.END} {pass_rate:.1f}%")
    
    if total_time:
        print(f"{Colors.BOLD}Time:{Colors.END} {total_time:.2f}s")
    
    print()
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}{'🎉 ALL TESTS PASSED! 🎉'.center(80)}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}{'⚠ Some tests need attention'.center(80)}{Colors.END}")
    
    print()


def print_test_statistics():
    """Print detailed test statistics"""
    print_header("TEST SUITE STATISTICS", "=")
    
    stats = {
        "Simulators": {
            "LGSSM": 7,
            "Stochastic Volatility (1D)": 14,
            "Multi-Target Acoustic Tracking (MAT)": 25,
            "Sensor Network Linear Gaussian (SNLG)": 90,
            "Sensor Network GH Skew-t (SNLG Skew-t)": 97,
            "Lorenz 96 (Chaotic System)": 38,
            "Subtotal": 271
        },
        "Filters": {
            "Kalman Filter (KF)": 6,
            "Extended Kalman Filter (EKF)": 32,
            "Unscented Kalman Filter (UKF)": 42,
            "Particle Filter (PF)": 27,
            "EDH Particle Filter - Unit Tests": 30,
            "EDH Particle Filter - EKF Tracker": 24,
            "EDH Particle Filter - UKF Tracker": 25,
            "LEDH Particle Filter - Unit Tests": 30,
            "LEDH Particle Filter - EKF Integration": 7,
            "LEDH Particle Filter - UKF Integration": 7,
            "Kernel Particle Filter (KPF) - Unit Tests": 47,
            "Subtotal": 277
        },
        "Integration Tests": {
            "KF Integration": 2,
            "EKF Integration (1D SV)": 8,
            "UKF Integration (1D SV)": 5,
            "PF Integration (1D SV)": 7,
            "EDH-EKF Integration (1D SV)": 5,
            "EDH-UKF Integration (1D SV)": 5,
            "MAT Filter Integration (EKF/UKF/EDH/LEDH)": 6,
            "SNLG-KF Integration": 5,
            "SNLG-UKF Integration": 4,
            "SNLG-EKF Integration": 5,
            "SNLG-EDH Integration": 4,
            "SNLG-LEDH Integration": 5,
            "SNLG Skew-t - EKF Integration": 7,
            "SNLG Skew-t - UKF Integration": 6,
            "SNLG Skew-t - PF Integration": 5,
            "SNLG Skew-t - EDH-EKF Integration": 4,
            "SNLG Skew-t - EDH-UKF Integration": 3,
            "SNLG Skew-t - LEDH-EKF Integration": 3,
            "SNLG Skew-t - LEDH-UKF Integration": 3,
            "KPF-LGSSM Integration": 5,
            "KPF-Lorenz96 Integration": 6,
            "Subtotal": 103
        }
    }
    
    print(f"{Colors.BOLD}Test Breakdown:{Colors.END}\n")
    
    total = 0
    for category, tests in stats.items():
        print(f"{Colors.CYAN}{Colors.BOLD}{category}:{Colors.END}")
        subtotal = 0
        for test_name, count in tests.items():
            if test_name != "Subtotal":
                print(f"  • {test_name}: {Colors.GREEN}{count}{Colors.END} tests")
                subtotal += count
            else:
                print(f"  {Colors.BOLD}Subtotal: {count} tests{Colors.END}")
                total += count
        print()
    
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'─' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}TOTAL: {total} tests{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'─' * 80}{Colors.END}\n")
    
    print(f"{Colors.BOLD}Test Coverage:{Colors.END}")
    print(f"  • Unit Tests: {Colors.GREEN}532{Colors.END} tests (84%)")
    print(f"  • Integration Tests: {Colors.GREEN}114{Colors.END} tests (16%)")
    print()
    
    print(f"{Colors.BOLD}Features Tested:{Colors.END}")
    print(f"  ✓ State estimation (predict/update)")
    print(f"  ✓ Covariance symmetry and positive-definiteness")
    print(f"  ✓ Numerical stability (jitter, Joseph form)")
    print(f"  ✓ Jacobian computation (analytic vs numerical)")
    print(f"  ✓ Sigma point generation and propagation")
    print(f"  ✓ Particle resampling (systematic & multinomial)")
    print(f"  ✓ Effective sample size and weight normalization")
    print(f"  ✓ Innovation and Kalman gain computation")
    print(f"  ✓ Control inputs and time-varying systems")
    print(f"  ✓ Nonlinear dynamics and observations")
    print(f"  ✓ Sequential filtering and reproducibility")
    print(f"  ✓ EKF/UKF tracker wrappers for EDH/LEDH algorithms")
    print(f"  ✓ Particle flow integration (Euler & RK4)")
    print(f"  ✓ EDH hybrid filtering with global mean path")
    print(f"  ✓ LEDH per-particle linearization and flow")
    print(f"  ✓ Multi-target acoustic tracking simulation (MAT)")
    print(f"  ✓ Constant-velocity dynamics and sensor grids")
    print(f"  ✓ MAT integration with all filter types (EKF/UKF/EDH/LEDH)")
    print(f"  ✓ Sensor network linear Gaussian (SNLG) spatial correlation")
    print(f"  ✓ Squared-exponential covariance kernels")
    print(f"  ✓ Cholesky decomposition with progressive jitter")
    print(f"  ✓ SNLG integration with all filter types (KF/UKF/EKF/EDH/LEDH)")
    print(f"  ✓ Sensor network GH skew-t dynamics (heavy tails & skewness)")
    print(f"  ✓ Inverse-gamma scale mixing for heavy-tailed distributions")
    print(f"  ✓ Skewness via location shift (gamma vector)")
    print(f"  ✓ Poisson count-based measurements")
    print(f"  ✓ SNLG Skew-t integration with all filters (EKF/UKF/PF/EDH/LEDH)")
    print(f"  ✓ Lorenz 96 chaotic dynamics and RK4 integration")
    print(f"  ✓ Ensemble forecasting and sparse observations")
    print(f"  ✓ Kernel Particle Filter with matrix-valued RBF kernels")
    print(f"  ✓ Diagonal and scalar (isotropic) kernel modes")
    print(f"  ✓ Gaspari-Cohn localization for spatial systems")
    print(f"  ✓ Pseudo-time integration with adaptive step control")
    print(f"  ✓ KPF integration with LGSSM and Lorenz 96 systems")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run tests for State-Space Models and Filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--phase",
        choices=["simulator", "lgssm", "sv", "mat", "snlg", "snlg-skewt", "lorenz96", "kf", "kalman", "ekf", "ukf", "pf", "particle", "edh", "ledh", "kpf", "kernel-pf", "filters", "integration", "mat-filters", "snlg-filters", "snlg-skewt-filters", "kpf-integration", "all"],
        default="all",
        help="Which test phase to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="Show test statistics summary"
    )
    
    args = parser.parse_args()
    
    # Show statistics summary if requested
    if args.summary:
        print_test_statistics()
        sys.exit(0)
    
    # Print banner
    print_header("State-Space Models & Filters Test Suite")
    
    import time
    start_time = time.time()
    
    # Run requested tests
    if args.phase == "simulator":
        success, passed, failed = run_simulator_tests(args.verbose)
    elif args.phase == "lgssm":
        success, passed, failed = run_lgssm_simulator_tests(args.verbose)
    elif args.phase == "sv":
        success, passed, failed = run_sv_simulator_tests(args.verbose)
    elif args.phase == "mat":
        success, passed, failed = run_mat_simulator_tests(args.verbose)
    elif args.phase == "snlg":
        success, passed, failed = run_snlg_simulator_tests(args.verbose)
    elif args.phase == "sn_skewt":
        success, passed, failed = run_sn_skewt_simulator_tests(args.verbose)
    elif args.phase == "lorenz96":
        success, passed, failed = run_lorenz96_simulator_tests(args.verbose)
    elif args.phase in ["kf", "kalman"]:
        success, passed, failed = run_kf_tests(args.verbose)
    elif args.phase == "ekf":
        success, passed, failed = run_ekf_tests(args.verbose)
    elif args.phase == "ukf":
        success, passed, failed = run_ukf_tests(args.verbose)
    elif args.phase in ["pf", "particle"]:
        success, passed, failed = run_pf_tests(args.verbose)
    elif args.phase == "edh":
        success, passed, failed = run_edh_tests(args.verbose)
    elif args.phase == "ledh":
        success, passed, failed = run_ledh_tests(args.verbose)
    elif args.phase in ["kpf", "kernel-pf"]:
        success, passed, failed = run_kernel_pf_tests(args.verbose)
    elif args.phase == "filters":
        success, passed, failed = run_all_filter_tests(args.verbose)
    elif args.phase == "integration":
        success, passed, failed = run_integration_tests(args.verbose)
    elif args.phase == "mat-filters":
        success, passed, failed = run_mat_filter_integration_tests(args.verbose)
    elif args.phase == "snlg-filters":
        success, passed, failed = run_snlg_filter_integration_tests(args.verbose)
    elif args.phase == "sn_skewt_filters":
        success, passed, failed = run_sn_skewt_filter_integration_tests(args.verbose)
    elif args.phase == "kpf-integration":
        success, passed, failed = run_kernel_pf_integration_tests(args.verbose)
    else:  # all
        success, passed, failed = run_all_tests(args.verbose)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print_summary(passed, failed, elapsed_time)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
