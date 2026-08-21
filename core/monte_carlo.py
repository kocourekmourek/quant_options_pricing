import numpy as np
from core.sde_engine import StochasticEngine

class MonteCarloPricer:
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def price_european(self, paths=200_000, steps=252, option_type="call") -> tuple[float, float]:
        sim_paths = StochasticEngine.generate_gbm_paths(self.S0, self.r, self.sigma, self.T, steps, paths)
        S_T = sim_paths[:, -1]
        
        payoffs = np.maximum(S_T - self.K, 0.0) if option_type == "call" else np.maximum(self.K - S_T, 0.0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(paths)
        return float(price), float(std_error)

    def price_asian_arithmetic(self, paths=200_000, steps=252, option_type="call") -> tuple[float, float]:
        """Payoff depends on arithmetic average price over the full trajectory."""
        sim_paths = StochasticEngine.generate_gbm_paths(self.S0, self.r, self.sigma, self.T, steps, paths)
        S_avg = np.mean(sim_paths, axis=1)
        
        payoffs = np.maximum(S_avg - self.K, 0.0) if option_type == "call" else np.maximum(self.K - S_avg, 0.0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        return float(np.mean(discounted_payoffs)), float(np.std(discounted_payoffs) / np.sqrt(paths))

    def price_knockout_barrier(self, barrier: float, paths=200_000, steps=252) -> tuple[float, float]:
        """Up-and-Out Call: option expires worthless if price ever crosses the barrier level."""
        sim_paths = StochasticEngine.generate_gbm_paths(self.S0, self.r, self.sigma, self.T, steps, paths)
        max_prices = np.max(sim_paths, axis=1)
        S_T = sim_paths[:, -1]
        
        # Valid only if barrier was never breached
        active_mask = (max_prices < barrier)
        payoffs = np.maximum(S_T - self.K, 0.0) * active_mask
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        return float(np.mean(discounted_payoffs)), float(np.std(discounted_payoffs) / np.sqrt(paths))