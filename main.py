# main.py
from core.analytical import BlackScholesAnalytical
from core.heston_engine import HestonEngine
from core.iv_engine import analyze_live_option_chain
from core.monte_carlo import MonteCarloPricer
from data.database import MarketDatabase


def main():
    ticker = "TSLA"
    risk_free_rate = 0.045  # 4.5%
    maturity_years = 0.5  # 6 months

    print("=" * 65)
    print(f"       QUANTITATIVE PRICING & RISK ENGINE: {ticker}")
    print("=" * 65)

    # 1. Historical Data Layer
    print("\n[1] Ingesting Historical Data & Calculating Realized Metrics...")
    db = MarketDatabase("quant_data.db")
    db.fetch_and_store(ticker, period="1y")
    spot, sigma = db.get_market_parameters(ticker)
    strike = round(spot * 1.05, 2)

    print(f"  • Spot Price (S0):         ${spot:.2f}")
    print(f"  • Target Strike (K):       ${strike:.2f}")
    print(f"  • 1-Year Realized Vol (σ): {sigma * 100:.2f}%")

    # 2. Closed-Form Black-Scholes
    print("\n[2] Analytical Black-Scholes (Constant Volatility Benchmark):")
    bs_price = BlackScholesAnalytical.price_european(
        spot, strike, risk_free_rate, sigma, maturity_years, "call"
    )
    greeks = BlackScholesAnalytical.exact_greeks(
        spot, strike, risk_free_rate, sigma, maturity_years, "call"
    )
    print(f"  • Theoretical BS Price:    ${bs_price:.4f}")
    print(f"  • Delta:                   {greeks['delta']:.4f}")
    print(f"  • Vega:                    ${greeks['vega']:.4f}")

    # 3. Geometric Brownian Motion Monte Carlo
    print("\n[3] GBM Monte Carlo (200,000 Paths):")
    mc = MonteCarloPricer(spot, strike, risk_free_rate, sigma, maturity_years)
    mc_price, mc_err = mc.price_european(
        paths=200_000, steps=126, option_type="call"
    )
    print(f"  • GBM Simulated Price:     ${mc_price:.4f} (±${mc_err:.4f} SE)")

    # 4. Heston Stochastic Volatility Monte Carlo
    print(
        "\n[4] Heston Stochastic Volatility Model (Coupled SDEs, 200,000 Paths):"
    )
    v0 = sigma**2  # Initial variance matching realized volatility
    theta = sigma**2  # Long-term equilibrium variance
    kappa = 1.6  # Mean-reversion speed
    xi = 0.7  # Volatility of variance
    rho = -0.50  # Equity-volatility negative correlation

    heston_price, heston_err = HestonEngine.price_european(
        S0=spot,
        K=strike,
        v0=v0,
        r=risk_free_rate,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        T=maturity_years,
        steps=126,
        paths=200_000,
        option_type="call",
    )
    print(
        f"  • Heston Simulated Price:  ${heston_price:.4f} (±${heston_err:.4f} SE)"
    )
    print(
        f"  • Stochastic Skew Impact:  ${(heston_price - bs_price):.4f} difference vs Flat BS"
    )

    # 5. Live Market Implied Volatility Surface (Newton-Raphson)
    print("\n[5] Scanning Live Option Chain for Implied Volatility...")
    print("-" * 65)
    analyze_live_option_chain(ticker, risk_free_rate)
    print("-" * 65)


    # =====================================================================
    # MARKET COMPARISON & ACCURACY BENCHMARK
    # =====================================================================
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import yfinance as yf

    def get_closest_market_option(ticker_str: str, target_T: float, target_K: float, option_type: str = "call"):
        ticker = yf.Ticker(ticker_str)
        today = datetime.today()
        
        # 1. Select expiration date closest to target_T
        expiries = ticker.options
        t_list = [(datetime.strptime(exp, "%Y-%m-%d") - today).days / 365.0 for exp in expiries]
        best_exp_idx = int(np.argmin(np.abs(np.array(t_list) - target_T)))
        best_expiry_str = expiries[best_exp_idx]
        actual_T = max(t_list[best_exp_idx], 1e-4)
        
        # 2. Pull chain and select strike closest to target_K
        chain = ticker.option_chain(best_expiry_str)
        df_options = chain.calls if option_type.lower() == "call" else chain.puts
        best_k_idx = int((df_options['strike'] - target_K).abs().argmin())
        row = df_options.iloc[best_k_idx]
        actual_K = float(row['strike'])
        
        # 3. Midpoint price if bid/ask exist, else lastPrice
        if row['bid'] > 0 and row['ask'] > 0:
            market_price = float((row['bid'] + row['ask']) / 2.0)
        else:
            market_price = float(row['lastPrice'])
            
        return market_price, actual_T, actual_K, best_expiry_str


    # Fetch closest matching exchange-traded contract
    market_price, actual_T, actual_K, expiry_date = get_closest_market_option(ticker, maturity_years, strike)
    print(f"\nMatched Contract: Expiry={expiry_date} (T={actual_T:.3f}y), Strike=${actual_K:.2f}")

    # Re-run models against the exact exchange contract parameters
   

    # Output aligned benchmarking table
    comparison = pd.DataFrame({
        "Pricing Method": [
            f"Exchange Market Mid ({expiry_date})",
            "Analytical Black-Scholes",
            "GBM Monte Carlo",
            "Heston Stochastic Vol"
        ],
        "Price ($)": [market_price, bs_price, mc_price, heston_price]
    })

    comparison["Dollar Error ($)"] = comparison["Price ($)"] - market_price
    comparison["% Error"] = (comparison["Dollar Error ($)"] / market_price) * 100

    print("\n" + "=" * 65)
    print("             THEORETICAL VS LIVE MARKET BENCHMARK")
    print("=" * 65)
    print(comparison.to_string(index=False))
    print("=" * 65)




if __name__ == "__main__":
    main()
