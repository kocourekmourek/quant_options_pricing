# viz/surface_plot.py
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import yfinance as yf

from core.iv_engine import ImpliedVolatilitySolver


def build_volatility_surface(
    ticker_symbol: str = "AAPL",
    risk_free_rate: float = 0.045,
    max_expirations: int = 6,
):
    print(f"[*] Extracting full Volatility Surface for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Failed to fetch spot for {ticker_symbol}")
    spot_price = float(history["Close"].iloc[-1])

    expirations = ticker.options[:max_expirations]
    today = datetime.date.today()

    data_points = []

    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_to_exp = (exp_date - today).days
        if days_to_exp < 7:
            continue

        T = days_to_exp / 365.0
        chain = ticker.option_chain(exp_str).calls

        # Keep strikes within ±20% of spot
        chain = chain[
            (chain["strike"] >= spot_price * 0.80)
            & (chain["strike"] <= spot_price * 1.20)
        ]

        for _, row in chain.iterrows():
            strike = float(row["strike"])
            if row["bid"] > 0 and row["ask"] > 0:
                price = float((row["bid"] + row["ask"]) / 2.0)
            elif row["lastPrice"] > 0:
                price = float(row["lastPrice"])
            else:
                continue

            iv = ImpliedVolatilitySolver.solve_call_iv(
                market_price=price,
                S=spot_price,
                K=strike,
                r=risk_free_rate,
                T=T,
            )

            if iv is not None:
                data_points.append({
                    "Strike": strike,
                    "Maturity_Days": days_to_exp,
                    "Maturity_Years": T,
                    "IV": iv * 100.0,
                })

    df = pd.DataFrame(data_points)
    if df.empty:
        print("[!] Not enough market data to build a surface.")
        return

    # Create uniform 2D grid for 3D surface interpolation
    strike_grid = np.linspace(df["Strike"].min(), df["Strike"].max(), 40)
    maturity_grid = np.linspace(
        df["Maturity_Days"].min(), df["Maturity_Days"].max(), 40
    )
    X, Y = np.meshgrid(strike_grid, maturity_grid)

    # 2D cubic interpolation across (Strike, Maturity) -> IV
    Z = griddata(
        (df["Strike"], df["Maturity_Days"]),
        df["IV"],
        (X, Y),
        method="linear",
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                colorbar_title="IV (%)",
            )
        ]
    )

    fig.update_layout(
        title=f"Live Implied Volatility Surface: {ticker_symbol} (Spot: ${spot_price:.2f})",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Days to Expiration (T)",
            zaxis_title="Implied Volatility (%)",
        ),
        autosize=True,
        width=950,
        height=650,
        margin=dict(l=40, r=40, b=40, t=60),
    )

    output_html = f"volatility_surface_{ticker_symbol}.html"
    fig.write_html(output_html)
    print(f"[+] Interactive 3D surface generated: {output_html}")
    fig.show()


if __name__ == "__main__":
    build_volatility_surface("AAPL")