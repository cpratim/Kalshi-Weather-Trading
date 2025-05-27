from trade.algorithm import Algorithm, Signal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from util.backtest import Backtest
import pandas as pd
import os
from math import floor, ceil


class FairSignal(Signal):

    def __init__(self, **kwargs):
        self.features = [
            'yes_price',
            'day_forecast_strike_dev',
            'kalshi_day_forecast_dev',
            'time_to_strike',
        ]
        self.n_neighbors = kwargs.get("n_neighbors", 500)
        self.pipeline = Pipeline([
            ('robust', RobustScaler(unit_variance=True)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)),
        ])

    def remove_outliers(self, df, features, quantile_range=(0.05, 0.95)):
        return_df = df.copy()
        for f in features:
            return_df = return_df[
                (return_df[f] >= return_df[f].quantile(quantile_range[0])) & 
                (return_df[f] <= return_df[f].quantile(quantile_range[1]))
            ]
        return return_df

    def fit(self, train_data: pd.DataFrame):
        train_data = self.remove_outliers(train_data, self.features)
        train_data = train_data[train_data['time_to_strike'] < 60000]
        self.pipeline.fit(train_data[self.features], train_data['outcome'])

    def __call__(self, trade_data: pd.DataFrame):
        return {
            "signal": self.pipeline.predict_proba([trade_data[self.features]])[0][1],
        }


class FairTrader(Algorithm):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        super().__init__(ticker, date, signal, **kwargs)
        self.alpha = kwargs.get("alpha", 0.1)
        self.slack = kwargs.get("slack", 0.01)
        self.trades_between_signals = kwargs.get("trades_between_signals", 60)
        self.last_trade_times = {}
        self.signals = {}
        

    def __name__(self):
        return "FairTrader"

    def decision_function(self, impact: float) -> float:
        pass

    def _normalize_signals(self):
        T = sum(self.signals.values())
        for ticker in self.signals:
            self.signals[ticker] = self.signals[ticker] / T

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict) -> dict:

        if signal_trade['time_to_strike'] > 60000:
            return {"signal": 0, "response": {}}
        
        if trade['ticker'] not in self.last_trade_times:
            self.last_trade_times[trade['ticker']] = float('inf')

        self.signals[trade['ticker']] = self.signal(signal_trade)["signal"]
        self._normalize_signals()

        yes_signal = self.signal(signal_trade)["signal"]
        no_signal = 1 - yes_signal
        yes_price = trade['yes_price']
        no_price = trade['no_price']

        yes_edge = yes_signal - yes_price / 100
        no_edge = no_signal - no_price / 100

        response = {}
            
        if (yes_edge > self.alpha):
            response = self.kernel.buy_yes(
                    trade["ticker"],
                    1,
                    (trade["yes_price"] / 100),
                    slack=self.slack,
                )
        if (no_edge > self.alpha):
            response = self.kernel.buy_no(
                    trade["ticker"],
                    1,
                    (trade["no_price"] / 100),
                    slack=self.slack,
                )

        return {"signal": yes_signal, "response": response}
    

def backtest_algorithm(ticker: str, **kwargs):
    signal = FairSignal()
    backtest = Backtest(
        ticker,
        data_dir="../data",
        backtest_window=kwargs.get("backtest_window", 50),
        min_window_size=kwargs.get("min_window_size", 200),
        max_window_size=kwargs.get("max_window_size", 200),
    )
    backtest.run_backtest(
        FairTrader,
        signal,
    )


if __name__ == "__main__":
    backtest_algorithm("kxhighny")