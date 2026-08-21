import numpy as np

class StochasticEngine:
    @staticmethod
    def generate_gbm_paths(
        S0: float, 
        r: float, 
        sigma: float, 
        T: float, 
        steps: int, 
        paths: int,
        antithetic: bool = True
    ) -> np.ndarray:
        """
        Generates simulated price trajectories using exact discretized GBM.
        Output shape: (paths, steps + 1)
        """
        dt = T / steps
        
        if antithetic:
            half_paths = paths // 2
            z = np.random.standard_normal((half_paths, steps))
            # Combine standard random draws with their exact negatives
            gaussian_shocks = np.vstack([z, -z])
        else:
            gaussian_shocks = np.random.standard_normal((paths, steps))

        # Vectorized drift and diffusion accumulation
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * gaussian_shocks
        
        log_increments = drift + diffusion
        log_paths = np.cumsum(log_increments, axis=1)

        # Prepend initial state log(S0)
        full_log_paths = np.hstack([np.zeros((paths, 1)), log_paths])
        return S0 * np.exp(full_log_paths)