# Quantitative Options Pricing & Stochastic Volatility Engine

A high-performance quantitative finance library in Python implementing closed-form analytical valuation, vectorized Monte Carlo pricing with variance reduction, live Implied Volatility surface extraction via Newton-Raphson root finding, and Heston Stochastic Volatility simulations under the Full Truncation Euler-Maruyama scheme.

---

## 1. Mathematical Architecture

### Geometric Brownian Motion (GBM)
Asset price dynamics follow the standard risk-neutral SDE:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

Integrated analytically via Itô's Lemma:

$$S_T = S_0 \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z\right), \quad Z \sim \mathcal{N}(0, 1)$$

### Heston Stochastic Volatility Model
To account for market-implied skew and stochastic variance, the model solves two coupled SDEs:

$$\begin{aligned}
dS_t &= r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \\
dv_t &= \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v
\end{aligned}$$

where $\mathbb{E}[dW_t^S \, dW_t^v] = \rho \, dt$, discretized using the **Full Truncation Scheme** to handle Feller condition violations ($2\kappa\theta \lt \xi^2$).

### Numerical Inversion (Newton-Raphson IV Engine)
Inverts market prices $C_{\text{Market}}$ to extract implied volatility $\sigma^*$ where:

$$\begin{aligned}
f(\sigma) &= C_{\text{BS}}(\sigma) - C_{\text{Market}} = 0 \\
\sigma_{n+1} &= \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{Market}}}{\mathcal{V}_{\text{BS}}(\sigma_n)}
\end{aligned}$$

where $\mathcal{V}_{\text{BS}} = \frac{\partial C}{\partial \sigma} = S \sqrt{T} \phi(d_1)$ is option Vega.

---

## 2. Project Structure

```text
quant_engine/
│
├── data/
│   ├── __init__.py
│   └── database.py         # SQLite ingestion & realized volatility calculation
│
├── core/
│   ├── __init__.py
│   ├── sde_engine.py       # Vectorized GBM simulation & antithetic variates
│   ├── analytical.py       # Closed-form Black-Scholes formula & exact Greeks
│   ├── monte_carlo.py      # European & Asian arithmetic path pricer
│   ├── heston_engine.py    # Coupled SDE simulator (Full Truncation scheme)
│   └── iv_engine.py        # Newton-Raphson implied volatility solver
│
├── viz/
│   ├── __init__.py
│   └── surface_plot.py     # 3D interactive Plotly surface interpolation
│
├── tests/
│   ├── __init__.py
│   └── test_engine.py      # Pytest validation (Parity, Convergence, Inversion)
│
└── main.py                 # Pipeline execution driver
```

---

## 3. Installation & Usage

### Setup Environment
```bash
git clone https://github.com/kocourekmourek/quant_options_pricing.git
cd quant_engine
pip install numpy scipy pandas yfinance plotly pytest
```

### Run Pricing Pipeline
```bash
python main.py
```

### Run 3D Surface Generator
```bash
python -m viz.surface_plot
```

### Execute Test Suite
```bash
python -m pytest -v
```

---

## 4. Verification & Testing

* **Put-Call Parity:** Verified within $10^{-5}$ tolerance.
* **Monte Carlo Convergence:** Empirical prices verified to fall within theoretical $3\sigma$ confidence intervals of closed-form Black-Scholes.
* **Boundary Validation:** Call Greek delta bounded strictly in $(0, 1)$, and positive gamma maintained.
* **Numerical Robustness:** Variance process verified non-negative and free of `NaN` outputs under extreme volatility-of-volatility.

---

## 📊 Real-World Market Benchmark

Theoretical model prices are dynamically aligned and benchmarked against live exchange option chains on **AAPL** at matched strikes ($K$) and expiration horizons ($T$):

| Pricing Engine | Option Price ($) | Dollar Error ($) | % Error vs. Market | Model Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Exchange Traded (Market Mid)** | **$16.90** | **$0.00** | **0.00%** | Live quote midpoint (`yfinance`) |
| **Analytical Black-Scholes** | $18.17 | +$1.27 | +7.53% | Assumes constant flat volatility ($\sigma_{\text{realized}}$) |
| **GBM Monte Carlo (100k paths)** | $18.16 | +$1.26 | +7.48% | Standard Brownian motion with antithetic variates |
| **Heston Stochastic Volatility** | **$17.06** | **+$0.16** | **+0.92%** | Captures market skew & leverage effect ($\rho = -0.70$) |

### Key Observations
* **GBM Numerical Stability:** Monte Carlo convergence matches closed-form Black-Scholes within cents ($18.16$ vs $18.17$), validating the antithetic sampling engine.
* **Stochastic Volatility Outperformance:** Plain Black-Scholes overprices OTM call options by ~7.5% due to its constant volatility assumption. The **Heston SDE Engine** captures the negative spot-volatility correlation ($\rho = -0.70$), reducing pricing error to under 1% against live exchange pricing.

---

## 📈 3D Volatility Surface

The `viz/surface_plot.py` module evaluates option chains across multiple expirations ($T$) and strike ranges ($K$), solving for implied volatility $\sigma_{\text{IV}}$ at each grid intersection and rendering an interactive 3D Volatility Surface:

$$\sigma_{\text{IV}} = f(K, T)$$

This visualizes market phenomena including the volatility smile/skew across moneyness and the term structure of volatility across maturities.

---
