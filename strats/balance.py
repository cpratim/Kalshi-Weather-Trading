from trade.algorithm import Algorithm, Signal
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from util.backtest import Backtest
import pandas as pd
import os
from math import floor, ceil
import scipy as sp
from bisect import bisect_left


class BalanceSignal(Signal):

    def __init__(self, **kwargs):
        self.features = [
            # 'kalshi_strike_delta',
            # 'kalshi_forecast_delta',
            # 'forecast_strike_delta',
            "time_to_strike",
            'yes_price',
        ]
        
        self.n_neighbors = kwargs.get("n_neighbors", 100)
        self.pipeline = Pipeline([
            # ('robust', RobustScaler(unit_variance=True)),
            # ('scaler', StandardScaler()),
            ('voting', VotingClassifier(estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)),
            ], voting='soft')),
        ])
        self.impact = 0

    def remove_outliers(self, df, features, quantile_range=(0.01, 0.99)):
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


class BalanceTrader(Algorithm):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        super().__init__(ticker, date, signal, **kwargs)
        self.alpha = kwargs.get("alpha", 0.10)
        self.slack = kwargs.get("slack", 0.01)
        self.decay_rate = kwargs.get("decay_rate", 0.1)
        self.trades_between_signals = kwargs.get("trades_between_signals", 60)
        self.prices = {}
        self.signals = {}
        self.last_bought = None
        self.impact = 0

    def __name__(self):
        return "BalanceTrader"

    def decision_function(self, impact: float) -> float:
        pass

    def _normalize_signals(self):
        T = sum(self.signals.values())
        for ticker in self.signals:
            self.signals[ticker] = self.signals[ticker] / T

    def _get_fair(self):
        fair = 0
        for ticker in self.signals:
            fair += self.signals[ticker] * self.strikes[ticker]
        return fair

    def calculate_resistance(self, ticker: str, side: str) -> float:

        if len(self.resistance[ticker][side]) < 2:
            return 0
        s, intercept, r, p, std_err = sp.stats.linregress(
            np.arange(len(self.resistance[ticker][side])),
            self.resistance[ticker][side]
        )
        return (s * np.sqrt(len(self.resistance[ticker][side]))) / 100
    
    def get_mid_tickers(self, fair: float) -> list:
        tickers, values = list(self.strikes.keys()), list(self.strikes.values())
        for i in range(len(values)):
            if fair < values[i]:
                break
        if i > 0:
            i -= 1
        return tickers[i:i+2]

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict, realtime: bool = True) -> dict:

        if trade['ticker'] not in self.prices:
            self.prices[trade['ticker']] = {
                'yes': 0, 'no': 0
            }

        self.prices[trade['ticker']]['yes'] = trade['yes_price']
        self.prices[trade['ticker']]['no'] = trade['no_price']

        self.signals[trade['ticker']] = self.signal(signal_trade)["signal"]
        self._normalize_signals()

        if signal_trade['time_to_strike'] > 60000 or not realtime:
            return {"signal": 0, "response": {}}

        fair = self._get_fair()
    
        responses = []
        buy_tickers = self.get_mid_tickers(fair)

        theo, price = 0, 0
        for ticker in buy_tickers:
            if ticker not in self.prices or ticker not in self.signals:
                return {'fair': fair, 'responses': []}
            theo += self.signals[ticker] 
            price += self.prices[ticker]['yes']
        
        edge = theo - price / 100
        if edge > self.alpha:
        
            for ticker in self.strikes:
                if ticker not in self.prices:
                    continue
                if ticker in buy_tickers:
                    response = self.kernel.buy_yes(
                        ticker, 1,
                        (self.prices[ticker]['yes'] / 100),
                        slack=self.slack,
                    )
                else:
                    response = self.kernel.buy_no(
                        ticker, 1,
                        (self.prices[ticker]['no'] / 100),
                        slack=self.slack,
                    )
                responses.append(response)

        return {"fair": fair, "responses": responses}
    

def backtest_algorithm(ticker: str, **kwargs):
    signal = BalanceSignal()
    backtest = Backtest(
        ticker,
        data_dir="../data",
        backtest_window=kwargs.get("backtest_window", 20),
        min_window_size=kwargs.get("min_window_size", 275),
        max_window_size=kwargs.get("max_window_size", 500),
    )
    backtest.run_backtest(
        BalanceTrader,
        signal,
    )


if __name__ == "__main__":
    backtest_algorithm("kxhighny")