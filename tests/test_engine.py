# tests/test_engine.py
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.analytical import BlackScholesAnalytical
from core.monte_carlo import MonteCarloPricer
from core.heston_engine import HestonEngine
from core.iv_engine import ImpliedVolatilitySolver


# --- Fixtures for Standard Test Parameters ---
@pytest.fixture
def standard_market():
    return {
        "S0": 100.0,
        "K": 100.0,
        "r": 0.05,
        "sigma": 0.20,
        "T": 1.0
    }


# --- 1. Analytical Engine & Greeks Tests ---
def test_black_scholes_call_put_parity(standard_market):
    """Verifies Put-Call Parity: C - P = S - K * exp(-rT)."""
    p = standard_market
    call = BlackScholesAnalytical.price_european(p["S0"], p["K"], p["r"], p["sigma"], p["T"], "call")
    put = BlackScholesAnalytical.price_european(p["S0"], p["K"], p["r"], p["sigma"], p["T"], "put")
    
    forward_diff = p["S0"] - p["K"] * np.exp(-p["r"] * p["T"])
    assert np.isclose(call - put, forward_diff, atol=1e-5)


def test_delta_bounds(standard_market):
    """European call delta must stay strictly between 0 and 1."""
    p = standard_market
    greeks = BlackScholesAnalytical.exact_greeks(p["S0"], p["K"], p["r"], p["sigma"], p["T"], "call")
    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["gamma"] > 0.0  # Long option gamma must be strictly positive


# --- 2. Monte Carlo Convergence Tests ---
def test_monte_carlo_convergence(standard_market):
    """Verifies MC price converges to Analytical Black-Scholes within 3 standard errors."""
    p = standard_market
    bs_price = BlackScholesAnalytical.price_european(p["S0"], p["K"], p["r"], p["sigma"], p["T"], "call")
    
    pricer = MonteCarloPricer(p["S0"], p["K"], p["r"], p["sigma"], p["T"])
    mc_price, mc_stderr = pricer.price_european(paths=150_000, steps=100, option_type="call")
    
    # 99.7% confidence interval test (3 sigma)
    abs_error = abs(mc_price - bs_price)
    assert abs_error <= 3.0 * mc_stderr, f"MC Error ({abs_error:.4f}) exceeded 3-sigma tolerance ({3.0 * mc_stderr:.4f})"


def test_asian_cheaper_than_vanilla(standard_market):
    """Arithmetic average Asian options must be cheaper than vanilla due to reduced volatility."""
    p = standard_market
    pricer = MonteCarloPricer(p["S0"], p["K"], p["r"], p["sigma"], p["T"])
    vanilla_price, _ = pricer.price_european(paths=100_000, steps=100, option_type="call")
    asian_price, _ = pricer.price_asian_arithmetic(paths=100_000, steps=100, option_type="call")
    
    assert asian_price < vanilla_price


# --- 3. Newton-Raphson Inversion Engine Tests ---
def test_newton_raphson_iv_recovery(standard_market):
    """Checks if IV engine correctly inverts known Black-Scholes prices back to original sigma."""
    p = standard_market
    known_sigma = 0.285  # Target 28.5%
    
    # Generate exact analytical price
    market_price = BlackScholesAnalytical.price_european(p["S0"], p["K"], p["r"], known_sigma, p["T"], "call")
    
    # Invert price using Newton-Raphson
    recovered_iv = ImpliedVolatilitySolver.solve_call_iv(
        market_price=market_price,
        S=p["S0"],
        K=p["K"],
        r=p["r"],
        T=p["T"]
    )
    
    assert recovered_iv is not None
    assert np.isclose(recovered_iv, known_sigma, atol=1e-4)


# --- 4. Heston Engine Numerical Stability ---
def test_heston_non_negative_variance():
    """Verifies that the Full Truncation Euler scheme handles high vol-of-vol without NaNs."""
    S_paths, v_paths = HestonEngine.generate_heston_paths(
        S0=100.0,
        v0=0.04,
        r=0.03,
        kappa=1.0,
        theta=0.04,
        xi=0.8,   # High vol-of-vol violating Feller condition
        rho=-0.7,
        T=0.5,
        steps=50,
        paths=5_000
    )
    
    assert not np.isnan(S_paths).any()
    assert not np.isnan(v_paths).any()
    assert (S_paths > 0.0).all()