import numpy as np
from scipy.stats import norm

class BlackScholesAnalytical:
    @staticmethod
    def d1_d2(S: float, K: float, r: float, sigma: float, T: float):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def price_european(cls, S: float, K: float, r: float, sigma: float, T: float, option_type="call") -> float:
        d1, d2 = cls.d1_d2(S, K, r, sigma, T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    @classmethod
    def exact_greeks(cls, S: float, K: float, r: float, sigma: float, T: float, option_type="call") -> dict:
        d1, d2 = cls.d1_d2(S, K, r, sigma, T)
        pdf_d1 = norm.pdf(d1)
        
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T)  # Sensitivity per 1.0 unit (100%) volatility shift
        
        if option_type == "call":
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega / 100,  # Scaled per 1% vol change
            "theta": theta / 365 # Scaled per 1 day decay
        }