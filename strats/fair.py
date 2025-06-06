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


class FairSignal(Signal):

    def __init__(self, **kwargs):
        self.features = [
            'kalshi_strike_delta',
            'kalshi_forecast_delta',
            'forecast_strike_delta',
            "time_to_strike",
            'yes_price',
        ]
        
        self.n_neighbors = kwargs.get("n_neighbors", 100)
        self.pipeline = Pipeline([
            ('voting', VotingClassifier(estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)),
            ], voting='soft')),
        ])

    def remove_outliers(self, df, features, quantile_range=(0.01, 0.99)):
        return_df = df.copy()
        for f in features:
            return_df = return_df[
                (return_df[f] >= return_df[f].quantile(quantile_range[0])) & 
                (return_df[f] <= return_df[f].quantile(quantile_range[1]))
            ]
        return return_df

    def fit(self, train_data: pd.DataFrame):
        self.pipeline.fit(train_data[self.features], train_data['outcome'])

    def __call__(self, trade_data: pd.DataFrame):
        return {
            "signal": self.pipeline.predict_proba([trade_data[self.features]])[0][1],
        }


class FairTrader(Algorithm):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        super().__init__(ticker, date, signal, **kwargs)
        self.alpha = kwargs.get("alpha", 0.075)
        self.slack = kwargs.get("slack", 0.01)
        self.signals = {}
        self.price_threshold = kwargs.get("price_threshold", 10)
        self.signal_data, self.kalshi_trades = self.kernel.get_existing_trades()
        if self.signal_data is not None:
            for i in range(len(self.signal_data)):
                self.on_signal_callback(self.signal_data.iloc[i], None, realtime=False)

        print(self.signals)

    def __name__(self):
        return "FairTrader"

    def decision_function(self, impact: float) -> float:
        pass

    def _normalize(self, dist: dict):
        if isinstance(dist, list):
            dist = {i: dist[i] for i in range(len(dist))}
        elif isinstance(dist, dict):
            pass
        else:
            raise ValueError(f"Invalid distribution type: {type(dist)}")
        T = sum(dist.values())
        if T == 0:
            T = 1
        for k in dist:
            dist[k] = dist[k] / T
        return dist

    def _get_fair(self):
        fair = 0
        for ticker in self.signals:
            fair += self.signals[ticker] * self.strikes[ticker]
        return fair
    
    def _kl_divergence(self, dist1: list, dist2: list):
        # kl divergence between two distributions safe with zeros
        return sum([dist1[i] * np.log(dist1[i] / dist2[i]) for i in range(len(dist1)) if dist1[i] > 0 and dist2[i] > 0])

    def calculate_resistance(self, ticker: str, side: str) -> float:

        if len(self.resistance[ticker][side]) < 2:
            return 0
        s, intercept, r, p, std_err = sp.stats.linregress(
            np.arange(len(self.resistance[ticker][side])),
            self.resistance[ticker][side]
        )
        return (s * np.sqrt(len(self.resistance[ticker][side]))) / 100
    
    def check_regions(self, yes_price: float, regions: list = [(25, 35), (60, 80)]):
        for region in regions:
            if yes_price >= region[0] and yes_price <= region[1]:
                return True
        return False

    def on_signal_callback(self, trade: pd.Series, _: dict, realtime: bool = True) -> dict:

        self.signals[trade['ticker']] = float(self.signal(trade)["signal"])
        self.signals = self._normalize(self.signals)

        if trade['time_to_strike'] > 60000 or not realtime or not self.check_regions(trade['yes_price']):
            return {"signal": 0, "response": {}}

        yes_signal = self.signals[trade['ticker']]
        no_signal = 1 - yes_signal
        yes_price = trade['yes_price']
        no_price = trade['no_price']

        yes_edge = yes_signal - yes_price / 100
        no_edge = no_signal - no_price / 100

        response = {}
            
        if (yes_edge > self.alpha and yes_price > self.price_threshold):
            response = self.kernel.buy_yes(
                trade["ticker"],
                1,
                (trade["yes_price"] / 100),
                slack=self.slack,
            )

        if (no_edge > self.alpha and no_price > self.price_threshold):
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
        backtest_window=kwargs.get("backtest_window", 1),
        min_window_size=kwargs.get("min_window_size", 250),
        max_window_size=kwargs.get("max_window_size", 500),
    )
    backtest.run_backtest(
        FairTrader,
        signal,
    )


if __name__ == "__main__":
    backtest_algorithm("kxhighny")