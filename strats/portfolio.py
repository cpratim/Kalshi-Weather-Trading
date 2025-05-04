from trade.algorithm import Algorithm, Signal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from util.backtest import Backtest
from dotenv import load_dotenv
from keys.get_key import load_key
import pandas as pd
import os
from math import floor
from util.update import get_all_updates
from util.load import DataLoader
from sklearn.linear_model import LogisticRegression


class PortfolioSignal(Signal):

    def __init__(self, **kwargs):
        pipeline = Pipeline(
            [
                ("robust_scaler", RobustScaler(unit_variance=True)),
                (
                    "quantile_transformer",
                    QuantileTransformer(n_quantiles=100, output_distribution="uniform"),
                ),
                ("logistic", LogisticRegression()),
            ]
        )
        super().__init__(pipeline, **kwargs)
        self.features = None
        self.metric = kwargs.get("metric", "outcome")

    def __call__(self, trade_data: pd.DataFrame):
        return {
            "signal": self.pipeline.predict_proba([trade_data[self.features]])[0, 1]
        }

    def fit(self, train_data: pd.DataFrame):
        if self.features is None:
            self.features = [
                f
                for f in train_data.columns
                if f
                not in ["time", "result", "impact", "ticker", "trade_id", "outcome"]
                and not f[-1].isdigit()
            ]
        self.pipeline.fit(train_data[self.features], train_data[self.metric])


class PortfolioTrader(Algorithm):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        super().__init__(ticker, date, signal, **kwargs)
        self.signal_dist = {}
        self.prices = {}
        self.loader = DataLoader(data_dir=kwargs.get("data_dir", "../data"))

    def get_dist(self):
        dist = {}
        mv = min(self.signal_dist.values())
        for k, v in self.signal_dist.items():
            dist[k] = v - mv
        s = sum(dist.values())
        for k in dist:
            dist[k] = dist[k] / max(s, 0.01)
        return dist

    def update_dist(self, ticker: str, signal: float, price: float):
        if ticker not in self.signal_dist:
            self.signal_dist[ticker] = 0
        self.prices[ticker] = price
        for t in self.signal_dist:
            self.signal_dist[t] = self.signal_dist[t] * self.signal_dist[t]
        self.signal_dist[ticker] = float(signal)

    def print_dist(self):
        dist = self.get_dist()
        keys = [(k, k.split("-")[-1][1:]) for k in dist.keys()]
        keys = sorted(keys, key=lambda x: x[1])
        for k, s in keys:
            print(f"{s}: {float(dist[k]):.2f} | ", end="")
        print()

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict) -> dict:
        ticker = trade["ticker"]
        signal = self.signal(signal_trade)["signal"]
        print(signal)
        self.update_dist(ticker, signal, trade["yes_price"])
        dist = self.get_dist()

        self.print_dist()
        return {}


def backtest_algorithm(ticker: str, **kwargs):
    signal = PortfolioSignal(metric="outcome")
    backtest = Backtest(
        ticker,
        data_dir="../data",
        backtest_window=kwargs.get("backtest_window", 1),
        min_window_size=kwargs.get("min_window_size", 15),
        max_window_size=kwargs.get("max_window_size", 30),
    )
    backtest.run_backtest(
        PortfolioTrader,
        signal,
    )


if __name__ == "__main__":
    # start_algorithm("kxhighny", train_window=20)
    backtest_algorithm("kxhighny")
