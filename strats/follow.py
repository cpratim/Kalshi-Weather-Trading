from trade.algorithm import Algorithm, Signal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from util.backtest import Backtest
from dotenv import load_dotenv
from keys.get_key import load_key
import pandas as pd
import os
from datetime import datetime
from math import floor
from util.update import get_all_updates


load_dotenv()
load_key()


pipeline = Pipeline(
    [
        ("robust_scaler", RobustScaler(unit_variance=True)),
        (
            "voting",
            VotingRegressor(
                estimators=[
                    (
                        "gb",
                        GradientBoostingRegressor(
                            n_estimators=50,
                            learning_rate=0.01,
                            max_depth=4,
                            subsample=0.5,
                            min_samples_split=50,
                            random_state=42,
                        ),
                    ),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=50, max_depth=4, random_state=42
                        ),
                    ),
                    ("lasso", LinearRegression()),
                    ("knn", KNeighborsRegressor(n_neighbors=1))
                ],
                n_jobs=-1,
            ),
        ),
    ]
)


class FollowSignal(Signal):

    def __init__(self, pipeline: Pipeline, **kwargs):
        super().__init__(pipeline, **kwargs)

    def set_features(self, columns: pd.Index):
        return [
            f
            for f in columns
            if f not in ["time", "result", "impact", "ticker", "trade_id", "outcome"] and not f[-1].isdigit()
        ]
        

class FollowTrader(Algorithm):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        super().__init__(ticker, date, signal, **kwargs)
        self.alpha = kwargs.get("alpha", 0.1)
        self.slack = kwargs.get("slack", 0.01)

    def decision_function(self, impact: float) -> float:
        qty = floor(2 * (2 / (1 + np.exp(-impact * self.alpha)) - 1))
        qty = np.clip(qty, -2, 2)
        return qty

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict):
        signal = self.signal(signal_trade)
        qty = self.decision_function(signal)
        # order = {}
        response = {}
        if (trade["taker_side"] == "yes" and qty >= 1) or (trade["taker_side"] == "no" and qty <= -1):
            # order = {"action": "buy", "side": "yes", 'ticker': trade['ticker'], 'taker_side': trade['taker_side'], 'qty': abs(qty), 'price': trade['yes_price'] / 100}
            # self._log(f"BUY YES - {order}")
            response = self.kernel.buy_yes(
                trade["ticker"], abs(qty), (trade["yes_price"] / 100), slack=self.slack
            )
        if (trade["taker_side"] == "no" and qty >= 1) or (trade["taker_side"] == "yes" and qty <= -1):
            # order = {"action": "buy", "side": "no", 'ticker': trade['ticker'], 'taker_side': trade['taker_side'], 'qty': abs(qty), 'price': trade['no_price'] / 100}
            # self._log(f"BUY NO - {order}")
            response = self.kernel.buy_no(
                trade["ticker"], abs(qty), (trade["no_price"] / 100), slack=self.slack
            )
        return {"signal": signal, "response": response}
            

def start_algorithm(ticker: str, **kwargs):
    get_all_updates()
    signal = FollowSignal(pipeline)
    kalshi_rsa_private_key = os.getenv("KALSHI_API_KEY")
    kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID")
    bot = FollowTrader(
        ticker,
        date="realtime",
        signal=signal,
        rsa_private_key=kalshi_rsa_private_key,
        api_key_id=kalshi_api_key_id,
        **kwargs
    )
    bot.start()


def backtest_algorithm(ticker: str, **kwargs):
    signal = FollowSignal(pipeline, metric="result")
    backtest = Backtest(
        ticker, 
        data_dir="../data", 
        backtest_window=kwargs.get("backtest_window", 1),
        min_window_size=kwargs.get("min_window_size", 30),
        max_window_size=kwargs.get("max_window_size", 30)
    )
    backtest.run_backtest(
        FollowTrader,
        signal,
    )


if __name__ == "__main__":
    # start_algorithm("kxhighny", train_window=20)
    backtest_algorithm("kxhighny")
