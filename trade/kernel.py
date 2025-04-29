from trade.exchange import HistoricalExchange, RealTimeExchange
from trade.stream import HistoricalDataStream, RealTimeDataStream
import logging
import pandas as pd

class HistoricalKernel(object):

    def __init__(
        self, ticker: str, date: str, on_signal_callback: callable = None, **kwargs
    ):
        self.ticker = ticker
        self.date = date
        self.exchange = HistoricalExchange(**kwargs)
        self.stream = HistoricalDataStream(
            ticker, date, on_signal_callback=self.on_signal_callback, **kwargs
        )
        self._on_signal_callback = on_signal_callback

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict):
        self.exchange.on_trade(trade)
        self._on_signal_callback(signal_trade, trade)

    def buy_yes(self, *args, **kwargs):
        return self.exchange.buy_yes(*args, **kwargs)
    
    def buy_no(self, *args, **kwargs):
        return self.exchange.buy_no(*args, **kwargs)

    def start_stream(self):
        self.stream.start()

    def set_stream_callbacks(self, **kwargs):
        if "on_signal_callback" in kwargs:
            self.on_signal_callback = kwargs["on_signal_callback"]


class RealTimeKernel(object):

    def __init__(self, ticker: str, on_signal_callback: callable = None, **kwargs):
        self.ticker = ticker
        self.exchange = RealTimeExchange(**kwargs)
        self.stream = RealTimeDataStream(ticker, on_signal_callback=self.on_signal_callback, **kwargs)
        self._on_signal_callback = on_signal_callback
        self.logger = logging.getLogger(__name__)
        
    def set_stream_callbacks(self, **kwargs):
        if "on_signal_callback" in kwargs:
            self.on_signal_callback = kwargs["on_signal_callback"]

    def on_signal_callback(self, signal_trade: pd.Series, trade: dict):
        self._on_signal_callback(signal_trade, trade)

    def buy_yes(self, *args, **kwargs):
        response = self.exchange.buy_yes(*args, **kwargs)
        if response['status'] == "executed":
            self.stream.set_just_executed()
        return response
    
    def buy_no(self, *args, **kwargs):
        response = self.exchange.buy_no(*args, **kwargs)
        if response['status'] == "executed":
            self.stream.set_just_executed()
        return response

    def start_stream(self):
        self.stream.start()
