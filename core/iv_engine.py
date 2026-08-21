import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf


class BlackScholesCore:
    @staticmethod
    def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        # Vega = ∂C/∂σ = S * φ(d1) * √T
        return float(S * norm.pdf(d1) * np.sqrt(T))


class ImpliedVolatilitySolver:
    @staticmethod
    def solve_call_iv(
        market_price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma_init: float = 0.3,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float | None:
        """Calculates Implied Volatility for a European/American Call using Newton-Raphson.

        Returns None if no convergence or arbitrage bounds are violated.
        """
        # Intrinsic value arbitrage check
        intrinsic_val = max(0.0, S - K * np.exp(-r * T))
        if market_price < intrinsic_val or market_price >= S:
            return None  # Price violates no-arbitrage bounds

        sigma = sigma_init
        for _ in range(max_iter):
            price = BlackScholesCore.call_price(S, K, r, sigma, T)
            diff = price - market_price

            if abs(diff) < tol:
                return float(sigma)

            v = BlackScholesCore.vega(S, K, r, sigma, T)
            if abs(v) < 1e-8:
                # Vega too small (slope ~ 0), Newton-Raphson will divide by zero
                break

            # Newton-Raphson step
            sigma = sigma - diff / v

            # Bound volatility to realistic physical bounds to prevent divergence
            if sigma <= 1e-4:
                sigma = 1e-4
            elif sigma > 5.0:  # 500% cap
                sigma = 5.0

        return None  # Failed to converge


def analyze_live_option_chain(ticker_symbol: str = "AAPL", risk_free_rate: float = 0.045):
    print(f"[*] Fetching live market data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    # Get underlying spot price
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Could not retrieve spot price for {ticker_symbol}")
    spot_price = float(history["Close"].iloc[-1])

    # Available expiration dates
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No active option chain found for {ticker_symbol}")

    # Select an expiration roughly 25-60 days out
    today = datetime.date.today()
    target_exp = None
    target_T = None

    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_to_exp = (exp_date - today).days
        if days_to_exp >= 25:
            target_exp = exp_str
            target_T = days_to_exp / 365.0
            break

    if target_exp is None:
        target_exp = expirations[0]
        days_to_exp = (
            datetime.datetime.strptime(target_exp, "%Y-%m-%d").date() - today
        ).days
        target_T = max(days_to_exp, 1) / 365.0

    print(f"[*] Analyzing Expiration: {target_exp} (T = {target_T:.4f} years, {int(target_T*365)} days)")
    print(f"[*] Spot Price (S): ${spot_price:.2f}\n")

    # Fetch Call Option Chain
    opt_chain = ticker.option_chain(target_exp)
    calls = opt_chain.calls.copy()

    # Filter for strikes within ±15% of spot
    calls = calls[
        (calls["strike"] >= spot_price * 0.85)
        & (calls["strike"] <= spot_price * 1.15)
    ]

    results = []
    for _, row in calls.iterrows():
        strike = float(row["strike"])

        # Determine valid market price
        if row["bid"] > 0 and row["ask"] > 0:
            market_price = float((row["bid"] + row["ask"]) / 2.0)
        elif row["lastPrice"] > 0:
            market_price = float(row["lastPrice"])
        else:
            continue

        yf_iv = float(row["impliedVolatility"])

        calculated_iv = ImpliedVolatilitySolver.solve_call_iv(
            market_price=market_price,
            S=spot_price,
            K=strike,
            r=risk_free_rate,
            T=target_T,
        )

        if calculated_iv is not None:
            results.append({
                "Strike": strike,
                "Moneyness (K/S)": round(strike / spot_price, 3),
                "Market Price ($)": round(market_price, 2),
                "Calculated IV (%)": round(calculated_iv * 100, 2),
                "Yahoo IV (%)": round(yf_iv * 100, 2),
                "Diff (%)": round((calculated_iv - yf_iv) * 100, 2),
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No valid option data found within pricing parameters.")
    else:
        print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    analyze_live_option_chain("AAPL")