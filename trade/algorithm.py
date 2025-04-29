from util.load import DataLoader
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from trade.kernel import HistoricalKernel, RealTimeKernel
from pprint import pprint
import logging
from datetime import datetime
import json
import os


date = datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(
    filename=f"../logs/{date}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Signal(object):

    def __init__(self, pipeline: Pipeline, **kwargs):
        self.pipeline = pipeline
        self.features = kwargs.get("features", None)
        self.metric = kwargs.get("metric", "result")

    def __call__(self, trade_data: pd.DataFrame):
        return self.pipeline.predict([trade_data[self.features]])[0]

    def set_features(self, train_columns: list[str]):
        return train_columns

    def fit(self, train_data: pd.DataFrame):
        if self.features is None:
            self.features = self.set_features(train_data.columns)
        self.pipeline.fit(train_data[self.features], train_data[self.metric])


class Algorithm(object):

    def __init__(self, ticker: str, date: str, signal: Signal, **kwargs):
        self.signal = signal
        self.ticker = ticker
        self.date = date
        self.kwargs = kwargs
        self.data_dir = kwargs.get("data_dir", "../data")

    def init_algorithm(self):
        if self.date == "realtime":
            context = DataLoader(self.data_dir).load_consolidated_daily_data(
                self.ticker, self.kwargs.get("train_window", 20), type_="polysignal", verbose=False
            )
            self.fit_signal(context)
            self.kernel = RealTimeKernel(
                self.ticker, on_signal_callback=self._on_signal_callback, **self.kwargs
            )
            self.logger = logging.getLogger(__name__)
            self.log_runtime_data = True
        else:
            self.kernel = HistoricalKernel(
                self.ticker, self.date, on_signal_callback=self._on_signal_callback, **self.kwargs
            )
            self.logger = None
            self.log_runtime_data = False
        self.runtime_data_file = (
            f"../runtime/{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        if self.log_runtime_data and not os.path.exists(self.runtime_data_file):
            open(self.runtime_data_file, "w")

    def fit_signal(self, context: pd.DataFrame):
        self.signal.fit(context)

    def save_runtime_data(self, signal_trade: pd.Series, result: dict):
        signal_dict = signal_trade.to_dict()
        signal_dict["time"] = str(signal_dict["time"])
        event = {"trade": signal_dict, "result": result}
        with open(self.runtime_data_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def decision_function(self, impact: float) -> float:
        return 0

    def get_portfolio(self):
        return self.kernel.exchange.get_portfolio()

    def on_trade(self, trade):
        pass

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict):
        pass

    def _on_signal_callback(self, signal_trade: pd.Series, trade: dict):
        result = self.on_signal_callback(signal_trade, trade)
        if self.log_runtime_data:
            self.save_runtime_data(signal_trade, result)
        return result

    def start(self):
        self.kernel.start_stream()

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)


if __name__ == "__main__":
    algo = Algorithm("kxhighny", "2025-04-24")
    algo.kernel.start_stream()
    pprint(algo.kernel.exchange.get_full_orderbook())
