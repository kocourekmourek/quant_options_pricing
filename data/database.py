import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf

class MarketDatabase:
    def __init__(self, db_name="quant_data.db"):
        self.conn = sqlite3.connect(db_name)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_history (
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    log_return REAL,
                    PRIMARY KEY (ticker, date)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS options_risk_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    spot REAL,
                    strike REAL,
                    maturity REAL,
                    implied_vol REAL,
                    fair_price REAL,
                    delta REAL,
                    gamma REAL,
                    vega REAL,
                    theta REAL
                );
            """)

    def fetch_and_store(self, ticker: str, period="1y"):
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=1, axis=1)

        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna().reset_index()

        records = [
            (ticker, row['Date'].strftime('%Y-%m-%d'), float(row['Close']), float(row['log_return']))
            for _, row in df.iterrows()
        ]

        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO asset_history (ticker, date, close, log_return)
                VALUES (?, ?, ?, ?);
            """, records)

    def get_market_parameters(self, ticker: str):
        query = "SELECT close, log_return FROM asset_history WHERE ticker = ? ORDER BY date ASC;"
        df = pd.read_sql_query(query, self.conn, params=(ticker,))
        if df.empty:
            raise ValueError(f"No records found for {ticker}. Run fetch_and_store first.")

        spot = df['close'].iloc[-1]
        # Annualize sample standard deviation of daily log returns (252 trading days)
        realized_vol = df['log_return'].std() * np.sqrt(252)
        return float(spot), float(realized_vol)

    def log_pricing_result(self, record: dict):
        with self.conn:
            self.conn.execute("""
                INSERT INTO options_risk_cache 
                (ticker, spot, strike, maturity, implied_vol, fair_price, delta, gamma, vega, theta)
                VALUES (:ticker, :spot, :strike, :maturity, :vol, :price, :delta, :gamma, :vega, :theta);
            """, record)