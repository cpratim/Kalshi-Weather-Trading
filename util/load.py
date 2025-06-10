import pandas as pd
import json
import os
from pprint import pprint
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional
from collections import deque
from api.kalshi import KalshiAPI
from warnings import filterwarnings
import pytz
import numpy as np


filterwarnings("ignore")


class DataLoader(object):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def _add_trend_features(self, df, row, feature, window):
        rows = df[
            (df["time"] <= row["time"])
            & (df["time"] > row["time"] - timedelta(minutes=window))
            & (df["shift"] == row["shift"])
        ]
        diff = rows[feature].iloc[-1] - rows[feature].iloc[0]
        return diff

    def load_weather_forecast(self, ticker: str):
        daily_weather_forecast_df = pd.read_csv(
            f"{self.data_dir}/weather/{ticker}/daily_weather_forecast.csv"
        )
        daily_weather_forecast_df["time"] = pd.to_datetime(
            daily_weather_forecast_df["time"]
        )
        daily_weather_forecast = {}
        for i, row in daily_weather_forecast_df.iterrows():
            forecast = {k: v for k, v in row.to_dict().items()}
            daily_weather_forecast[forecast["time"].strftime("%Y-%m-%d")] = forecast
        return daily_weather_forecast

    def get_strikes(self, event_data: dict, mean_center: bool = False):
        tickers = sorted(
            [event["ticker"] for event in event_data["markets"]],
            key = lambda x: float(x.split("-")[-1][1:])
        )
        # print(tickers)
        strikes = {}
        for i, t in enumerate(tickers):
            v = t.split("-")[-1]
            k = float(v[1:])
            if i == 0:
                k -= 0
            if i == len(tickers) - 1:
                k += 0
            strikes[t] = k
        mean_strike = sum(strikes.values()) / len(strikes)
        if mean_center:
            strikes = {ticker: strikes[ticker] - mean_strike for ticker in tickers}
        return strikes, mean_strike

    def parse_trade_time(self, timestamp: str, timezone: str = "US/Eastern"):
        utc_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        est_tz = pytz.timezone(timezone)
        est_dt = utc_dt.astimezone(est_tz)
        naive_est_dt = datetime(
            year=est_dt.year,
            month=est_dt.month,
            day=est_dt.day,
            hour=est_dt.hour,
            minute=est_dt.minute,
            second=est_dt.second,
            microsecond=est_dt.microsecond,
        )
        return naive_est_dt

    def get_strike_times(self, event_data: dict):
        if "strike_date" not in event_data["event"]:
            strike_time = self.parse_trade_time(event_data["event"]["close_time"])
        else:
            strike_time = self.parse_trade_time(event_data["event"]["strike_date"])
        return strike_time

    def _get_data(self, ticker: str, date: str, data_dir: str, data_type: str):
        with open(
            os.path.join(self.data_dir, data_dir, ticker, data_type, f"{date}.json"),
            "r",
        ) as f:
            return json.load(f)

    def _get_dates(self, path: str, max_days: Optional[int] = 300):
        max_days = len(os.listdir(os.path.join(self.data_dir, path))) if max_days is None else max_days
        return sorted(
            [
                x.split(".")[0]
                for x in os.listdir(os.path.join(self.data_dir, path))
            ]
        )[-max_days:]

    def _get_fair(self, dist: dict):
        total_p = sum(dist.values())
        probabilities = [k / max(1, total_p) for k in dist.values()]
        values = list(dist.keys())
        return sum([p * v for p, v in zip(probabilities, values)])

    def _load_trades(self, trade_data: dict):
        trades = []
        for series in trade_data:
            trades.extend(
                [
                    {
                        "idx": i,
                        "ticker": trade["ticker"],
                        "trade_id": trade["trade_id"],
                        "taker_side": trade["taker_side"],
                        "time": trade["created_time"],
                        "count": trade["count"],
                        "no_price": trade["no_price"],
                        "yes_price": trade["yes_price"],
                    }
                    for i, trade in enumerate(trade_data[series])
                ]
            )
        trades = sorted(trades, key=lambda x: x["time"])
        return trades

    def _get_trade_post_average(self, trade_data: list[dict]):
        trade_price_postfix, trade_count_postfix = [0], [0]
        for i, trade in enumerate(sorted(trade_data, key=lambda x: x["created_time"])):
            trade_price_postfix.append(
                trade["yes_price"] * trade["count"] + trade_price_postfix[-1]
            )
            trade_count_postfix.append(trade["count"] + trade_count_postfix[-1])

        total_trade_price, total_trade_count = (
            trade_price_postfix[-1],
            trade_count_postfix[-1],
        )
        trade_price_average = [0] * len(trade_data)
        for i, trade in enumerate(trade_data):
            trade_price_average[i] = (total_trade_price - trade_price_postfix[i]) / (
                total_trade_count - trade_count_postfix[i]
            )
        return trade_price_average

    def process_current_trade_data(
        self, verbose: bool = True, max_days: int = 100
    ):
        for ticker in self.metadata["polymarket"]:
            trade_dates = set(
                [
                    x.split(".")[0]
                    for x in os.listdir(
                        os.path.join(self.data_dir, "polysignal", ticker)
                    )
                ]
            )
            dates = []
            date_ptr = datetime.now() - timedelta(days=1)
            iter = 0
            while iter < max_days and date_ptr.strftime("%Y-%m-%d") not in trade_dates:
                dates.append(date_ptr.strftime("%Y-%m-%d"))
                date_ptr -= timedelta(days=1)
                iter += 1
            self.process_historical_trade_data(
                ticker, dates, verbose=verbose
            )

    def get_best_forecast(self, forecast: dict, kalshi_fair: float):
        min_dev, best_forecast = float('inf'), None
        for k, v in forecast.items():
            if not k.startswith('temperature'):
                continue
            dev = abs(v - kalshi_fair)
            if dev < min_dev and not dev == 0:
                min_dev = dev
                best_forecast = v
        return best_forecast
    
    def _get_bucket_delta(self, forecast_rounded: float, strike: float, mean_strike: float):
        mid = strike + mean_strike
        if mid != int(mid):
            bucket = (mid - .5, mid + .5)
        else:
            if strike < 0:
                bucket = (0, mid - 1)
            else:
                bucket = (mid + 1, 10e10)

        delta = min(
            abs(forecast_rounded - bucket[0]),
            abs(forecast_rounded - bucket[1]),
        )
        return delta


    def process_trade(
        self,
        trade: dict,
        kalshi_dist: dict,
        forecast: dict,
        strike_time: datetime,
        strike: float,
        mean_strike: float,
        forecast_shift: float = 0,
    ):
        trade_ticker = trade["ticker"]
        trade_time = self.parse_trade_time(trade["time"])
        time_to_strike = (strike_time - trade_time).total_seconds()
        kalshi_fair = self._get_fair(kalshi_dist)
        best_forecast = forecast['forecast'] + forecast_shift
        rounded_forecast = round(best_forecast, 2)
        features = {
            "time": trade_time,
            "trade_id": trade["trade_id"],
            "ticker": trade_ticker,
            "time_to_strike": time_to_strike,
            "count": trade["count"],
            "shift": strike - mean_strike,
            "kalshi_fair": kalshi_fair,
            "forecast": best_forecast,
            "yes_price": trade["yes_price"],
            "no_price": trade["no_price"],
            "taker_side": int(trade["taker_side"] == "yes"),
            "kalshi_strike_delta": kalshi_fair - strike,
            "forecast_strike_delta": best_forecast - strike,
            "forecast_bucket_delta": self._get_bucket_delta(rounded_forecast, strike, mean_strike),
            "kalshi_forecast_delta": best_forecast - kalshi_fair,
        }
        for k, v in kalshi_dist.items():
            if k - mean_strike < 0:
                features[f'strike_m_{abs(k - mean_strike)}'] = v
            else:
                features[f'strike_p_{k - mean_strike}'] = v
        
        dist = list(kalshi_dist.values())
        entropy = -sum([p * np.log(p) for p in dist if p > 0])
        features['entropy'] = entropy

        return features

    def process_trade_data(
        self,
        trades_data: dict,
        events_data: dict,
        forecast: dict,
        results: dict = None,
        strike_time: datetime = None,
        trades_post_average: dict = None,
        decay_rate: float = 0.1,
        forecast_shift: float = 0.0,
    ):
        if strike_time is None:
            strike_time = self.get_strike_times(events_data)
        strikes, mean_strike = self.get_strikes(events_data)
        S = sorted(list(strikes.values()))
        
        # print(S, [x - mean_strike for x in S], mean_strike)
        trades = self._load_trades(trades_data)
        k_values = [s for s in strikes.values()]
        kalshi_dist = {k: 0 for k in k_values}
        trade_data = []
        for i, trade in enumerate(trades):
            trade_ticker, trade_idx = trade["ticker"], trade["idx"]
            strike = strikes[trade_ticker]
            kalshi_dist = {k: v * (1 - decay_rate) for k, v in kalshi_dist.items()}
            kalshi_dist[strike] = trade["yes_price"]
            features = self.process_trade(
                trade,
                kalshi_dist,
                forecast,
                strike_time,
                strike,
                mean_strike,
                forecast_shift,
            )
            if results:
                if trade["taker_side"] == "yes":
                    features["impact"] = (
                        trades_post_average[trade_ticker][trade_idx]
                        - trade["yes_price"]
                    )
                    features["result"] = (
                        (100 - trade["yes_price"])
                        if results[trade_ticker] == "yes"
                        else -trade["yes_price"]
                    )
                    features["outcome"] = int(results[trade_ticker] == "yes")
                else:
                    features["impact"] = (
                        100 - trades_post_average[trade_ticker][trade_idx]
                    ) - trade["no_price"]
                    features["result"] = (
                        -trade["no_price"]
                        if results[trade_ticker] == "yes"
                        else (100 - trade["no_price"])
                    )
                    features["outcome"] = int(results[trade_ticker] == "yes")
            trade_data.append(features)
        if len(trade_data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(trade_data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time")
        df = df.fillna(0)
        df = df.round(3)
        return df

    def process_historical_trade_data(
        self,
        ticker: str,
        dates: Optional[list[str]] = None,
        verbose: bool = True,
        max_days: int = 300,
    ):
        daily_forecast = self.load_weather_forecast(ticker)
        if dates is None:
            dates = self._get_dates(f'{self.data_dir}/kalshi/{ticker}/trades', max_days=max_days)
        idx = 0
        iterator = (
            tqdm(dates, desc=f"Processing {ticker} for {dates[idx]}")
            if verbose
            else dates
        )
        for date in iterator:
            try:
                events_data, trades_data = (
                    self._get_data(ticker, date, "kalshi", "events"),
                    self._get_data(ticker, date, "kalshi", "trades"),
                )
            except Exception as e:
                continue
            # strike_time = self.get_strike_times(events_data)
            strike_time = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1) - timedelta(minutes=1)
            # print(strike_time, date)
            forecast = daily_forecast[strike_time.strftime("%Y-%m-%d")]
            trades_post_average = {
                k: self._get_trade_post_average(v) for k, v in trades_data.items()
            }
            city = self.metadata['city_map'][ticker]
            results = KalshiAPI().get_market_results(ticker, date)
            df = self.process_trade_data(
                trades_data,
                events_data,
                forecast,
                results=results,
                strike_time=strike_time,
                trades_post_average=trades_post_average,
                forecast_shift=self.metadata["cities"][city]["shift"],
            )
            if ticker not in os.listdir(os.path.join(self.data_dir, "processed")):
                os.makedirs(os.path.join(self.data_dir, "processed", ticker))
            df.to_csv(
                os.path.join(self.data_dir, "processed", ticker, f"{date}.csv"),
                index=False,
            )
            idx += 1
            if idx < len(dates) and verbose:
                iterator.set_description(f"Processing {ticker} for {dates[idx]}")

    def load_trade_data(self, event_data: dict, trade_data: dict):
        strike_time = datetime.strptime(
            event_data["event"]["strike_date"], "%Y-%m-%dT%H:%M:%SZ"
        )
        strikes, mean_strike = self.get_strikes(event_data)
        trade_data_flattened = []
        for series in trade_data:
            for _, trade in enumerate(
                sorted(trade_data[series], key=lambda x: x["created_time"])
            ):
                features = self.get_basic_features(
                    trade, strike_time, strikes, mean_strike
                )
                trade_data_flattened.append(features)
        trade_data_flattened = pd.DataFrame(trade_data_flattened)
        trade_data_flattened = self.add_window_features(trade_data_flattened)
        trade_data_flattened["time"] = pd.to_datetime(trade_data_flattened["time"])
        trade_data_flattened = trade_data_flattened.sort_values(by="time")
        trade_data_flattened = trade_data_flattened.fillna(0)
        trade_data_flattened = trade_data_flattened.reset_index(drop=True)
        return trade_data_flattened

    def load_consolidated_daily_data(
        self,
        ticker: str,
        type_: str = "processed",
        verbose: bool = True,
        max_days: int = 300,
    ) -> pd.DataFrame:
        dates = self._get_dates(f'{self.data_dir}/{type_}/{ticker}', max_days=max_days)
        result_df = pd.DataFrame()
        idx = 0
        iterator = (
            tqdm(sorted(dates), desc=f"Loading {ticker.ljust(9)} for {dates[idx]}")
            if verbose
            else sorted(dates)
        )
        for date in iterator:
            df = pd.read_csv(os.path.join(self.data_dir, type_, ticker, f"{date}.csv"))
            result_df = pd.concat([result_df, df])
            idx += 1
            if idx < len(dates) and verbose:
                iterator.set_description(f"Loading {ticker.ljust(9)} for {dates[idx]}")
        result_df = result_df.fillna(0)
        result_df = result_df.reset_index(drop=True)
        result_df["time"] = pd.to_datetime(result_df["time"])
        return result_df

    def load_daily_data(
        self, ticker: str, type_: str = "processed", max_days: int = 300, verbose: bool = True
    ) -> pd.DataFrame:
        daily_data = {}
        dates = self._get_dates(f'{self.data_dir}/{type_}/{ticker}', max_days=max_days)
        idx = 0
        iterator = (
            tqdm(sorted(dates), desc=f"Loading {ticker.ljust(9)} for {dates[idx]}")
            if verbose
            else sorted(dates)
        )
        for date in iterator:
            df = pd.read_csv(os.path.join(self.data_dir, type_, ticker, f"{date}.csv"))
            df = df.fillna(0)
            df["time"] = pd.to_datetime(df["time"])
            daily_data[date] = df
            idx += 1
            if idx < len(dates) and verbose:
                iterator.set_description(f"Loading {ticker.ljust(9)} for {dates[idx]}")
        return daily_data
    
    def get_all_market_results(self, ticker: str, max_days: int = 300):
        dates = self._get_dates(f'{self.data_dir}/kalshi/{ticker}/trades', max_days=max_days)
        results = {}
        for date in dates:
            res = KalshiAPI().get_market_results(ticker, date)
            key, first = None, True
            for k, v in res.items():
                if v == 'yes':
                    key = k
                    break
                first = False
            key = key.split('-')[-1]
            if key[0] == 'T':
                val = float(key[1:]) - 1.5 if first else float(key[1:]) + 1.5
            else:
                val = float(key[1:])
            results[date] = val
        return results


if __name__ == "__main__":
    data_dir = "../data"
    loader = DataLoader(data_dir)
    loader.process_historical_trade_data("kxhighny", max_days=300, verbose=True)
