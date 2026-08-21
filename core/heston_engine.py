import numpy as np


class HestonEngine:
    @staticmethod
    def generate_heston_paths(
        S0: float,
        v0: float,
        r: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        T: float,
        steps: int,
        paths: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulates coupled Asset Price (S) and Variance (v) paths using the Full

        Truncation Euler-Maruyama scheme.

        Returns:
            S: np.ndarray of shape (paths, steps + 1)
            v: np.ndarray of shape (paths, steps + 1)
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)

        # 1. Generate Correlated Brownian Shocks via Cholesky decomposition
        z1 = np.random.standard_normal((paths, steps))
        z2 = np.random.standard_normal((paths, steps))

        # Correlate shocks: Z_S and Z_v have exact correlation rho
        dW_S = z1 * sqrt_dt
        dW_v = (rho * z1 + np.sqrt(1.0 - rho**2) * z2) * sqrt_dt

        # 2. Pre-allocate arrays
        S = np.zeros((paths, steps + 1))
        v = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        # 3. Step through time (Full Truncation Scheme)
        for t in range(steps):
            # Truncate negative variance before taking square root
            v_pos = np.maximum(v[:, t], 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Variance SDE (CIR process): dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
            v[:, t + 1] = (
                v[:, t]
                + kappa * (theta - v_pos) * dt
                + xi * sqrt_v * dW_v[:, t]
            )

            # Asset Price SDE (Log-Euler step for numerical stability):
            # d(ln S) = (r - 0.5 * v) * dt + sqrt(v) * dW_S
            S[:, t + 1] = S[:, t] * np.exp(
                (r - 0.5 * v_pos) * dt + sqrt_v * dW_S[:, t]
            )

        return S, v

    @classmethod
    def price_european(
        cls,
        S0: float,
        K: float,
        v0: float,
        r: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        T: float,
        steps: int = 126,
        paths: int = 200_000,
        option_type: str = "call",
    ) -> tuple[float, float]:
        """Prices European options under the Heston model via Monte Carlo."""
        S_paths, _ = cls.generate_heston_paths(
            S0, v0, r, kappa, theta, xi, rho, T, steps, paths
        )
        S_T = S_paths[:, -1]

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(K - S_T, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_payoffs = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs) / np.sqrt(paths))
        return price, stderr