import json
import os
from api.kalshi import KalshiAPI
from api.weather import OpenMeteoAPI, NWSAPI
import pandas as pd
from datetime import datetime
from api.kalshi import KalshiWS, KalshiAPI
from api.polymarket import PolyMarketAPI
from api.weather import OpenMeteoAPI
import os
from concurrent.futures import ThreadPoolExecutor
import time
from util.load import DataLoader
from threading import Thread
import logging
import pytz


class HistoricalDataStream(object):

    def __init__(
        self,
        ticker: str,
        date: str,
        data_dir: str = "../data",
        process_type: str = "processed",
        on_signal_callback: callable = None,
        **kwargs,
    ):
        self.ticker = ticker
        self.date = date
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.date_ptr = datetime.strptime(date, "%Y-%m-%d")
        self.kalshi = KalshiAPI()
        self.dataloader = DataLoader(data_dir=data_dir)
        with open(
            os.path.join(self.data_dir, "kalshi", ticker, "events", f"{date}.json"), "r"
        ) as f:
            self.events = json.load(f)
        with open(
            os.path.join(self.data_dir, process_type, ticker, f"{date}.csv"), "r"
        ) as f:
            self.trades = pd.read_csv(f)
        self.strikes, self.mean_strike = self.dataloader.get_strikes(self.events)
        self.trades["time"] = pd.to_datetime(self.trades["time"])
        self.on_signal_callback = on_signal_callback

    def get_events(self):
        return self.events

    def get_results(self):
        return self.kalshi.get_market_results(self.ticker, self.date)

    def set_callbacks(self, **kwargs):
        self.callbacks = kwargs

    def start(self):
        for index, signal_trade in self.trades.iterrows():
            if signal_trade["time"] < self.date_ptr:
                continue
            trade = {
                "time": signal_trade["time"],
                "ticker": signal_trade["ticker"],
                "yes_price": signal_trade["yes_price"],
                "no_price": signal_trade["no_price"],
                "taker_side": "yes" if signal_trade["taker_side"] == 1 else "no",
            }
            self.on_signal_callback(signal_trade, trade)


class RealTimeDataStream(object):

    def __init__(self, ticker: str, on_signal_callback: callable, **kwargs):
        self.ticker = ticker
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        self.latest_update = None
        self.kalshi_api = KalshiAPI()
        self.dataloader = DataLoader(data_dir="../data")
        self.event_data = self.kalshi_api.get_current_weather_event_data(ticker)
        self.tickers = [event["ticker"] for event in self.event_data["markets"]]
        self.strikes, self.mean_strike = self.dataloader.get_strikes(self.event_data)
        self.strike_time = self.dataloader.get_strike_times(self.event_data)
        self.orderbook_dir = '../runtime/orderbook'
        self.kalshi_ws = KalshiWS(
            tickers=self.tickers,
            private_key=kwargs.get("rsa_private_key"),
            api_key_id=kwargs.get("api_key_id"),
            on_message_callback=self.on_kalshi_message,
        )
        self.weather_api = OpenMeteoAPI() if kwargs.get('use_openmeteo', False) else NWSAPI()
        self.on_signal_callback = on_signal_callback
        self.forecast = self.weather_api.get_current_forecast(self.ticker)
        self.kalshi_trades = self.kalshi_api.get_current_trade_data(self.event_data)
        self.latest_forecast_update_time = None
        self.kalshi_dist = {}
        self.trade_idx = {}
        self.signal_data = self.load_existing_trades()
        self.n_messages = 0
        self.log_interval = 100
        self.just_executed = False

    def save_orderbook_state(self, orderbook: dict):
        date = datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(f'{self.orderbook_dir}/{date}.jsonl'):
            open(f'{self.orderbook_dir}/{date}.jsonl', 'w')
        with open(f'{self.orderbook_dir}/{date}.jsonl', 'a') as f:
            f.write(json.dumps(orderbook) + '\n')

    def set_just_executed(self):
        self.just_executed = True

    def get_existing_trades(self):
        return self.signal_data, self.kalshi_trades

    def load_existing_trades(self):
        for ticker in self.kalshi_trades:
            self.kalshi_trades[ticker] = self.kalshi_trades[ticker][::-1]
            strike = float(self.strikes[ticker])
            self.trade_idx[ticker] = len(self.kalshi_trades[ticker])
            self.kalshi_dist[strike] = float(self.kalshi_trades[ticker][-1]["yes_price"])
        return self.dataloader.process_trade_data(
            self.kalshi_trades,
            self.event_data,
            self.forecast,
            forecast_shift=0.0,
        )

    def get_trade_idx(self, ticker: str):
        idx = self.trade_idx[ticker]
        self.trade_idx[ticker] += 1
        return idx

    def get_weather_forecast(self):
        if (
            self.latest_forecast_update_time is None
            or (datetime.now() - self.latest_forecast_update_time).total_seconds()
            > 3600
        ):
            self.forecast = (
                self.weather_api.get_current_forecast(self.ticker)
            )
            self.latest_forecast_update_time = datetime.now()
        return self.forecast

    def handle_trade(self, trade):
        forecast = self.get_weather_forecast()
        trade["ticker"] = trade["market_ticker"]
        trade["time"] = datetime.fromtimestamp(trade["ts"], tz=pytz.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        strike = self.strikes[trade["ticker"]]
        self.kalshi_dist[float(strike)] = float(trade["yes_price"])
        features = self.dataloader.process_trade(
            trade,
            self.kalshi_dist,
            forecast,
            self.strike_time,
            strike,
            self.mean_strike,
            forecast_shift=0.0,
        )
        self.signal_data = pd.concat([self.signal_data, pd.DataFrame([features])])
        signal_trade = self.signal_data.iloc[-1]
        self.on_signal_callback(signal_trade, trade)

    async def on_kalshi_message(self, message):
        if self.n_messages % self.log_interval == 0:
            self.logger.info(f"PULSE [{self.n_messages}] - {self.kalshi_dist}")
        self.n_messages += 1
        try:
            message = json.loads(message)
        except:
            return
        if message["type"] != "trade":
            return
        if self.just_executed:
            self.just_executed = False
            return
        trade = message["msg"]
        self.logger.info(f"TRADE - {trade}")
        Thread(target=self.handle_trade, args=(trade,)).start()

    def start(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.kalshi_ws.start)
            print(f"Started Kalshi WS | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                while True:
                    state = self.kalshi_api.get_orderbook(self.ticker)
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    orderbook = {
                        'time': ts,
                        'orderbook': state,
                    }
                    self.save_orderbook_state(orderbook)

                    time.sleep(1)
            except KeyboardInterrupt:
                print(
                    f"Closing Kalshi WS | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.kalshi_ws.ws.close()
