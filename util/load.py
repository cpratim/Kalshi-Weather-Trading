import pandas as pd
import json
import os
from pprint import pprint
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional
from api.kalshi import KalshiAPI
from warnings import filterwarnings
import pytz

filterwarnings("ignore")


class DataLoader(object):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def get_valid_date_range(self, ticker: str, max_days: int = 365) -> list[str]:
        kalshi = KalshiAPI(data_dir=self.data_dir)
        return kalshi.get_valid_date_range(ticker, max_days=max_days)

    def _add_agg_features(self, df, row, feature, window):
        rows = df[
            (df["time"] <= row["time"])
            & (df["time"] > row["time"] - timedelta(minutes=window))
        ]
        agg_count = rows[feature].sum()
        return agg_count

    def _add_vol_features(self, df, row, feature, window):
        rows = df[
            (df["time"] <= row["time"])
            & (df["time"] > row["time"] - timedelta(minutes=window))
            & (df["shift"] == row["shift"])
        ]
        vol = rows[feature].std()
        return vol

    def _add_trend_features(self, df, row, feature, window):
        rows = df[
            (df["time"] <= row["time"])
            & (df["time"] > row["time"] - timedelta(minutes=window))
            & (df["shift"] == row["shift"])
        ]
        diff = rows[feature].iloc[0] - rows[feature].iloc[-1]
        return diff

    def _add_sentiment_features(self, df, row, feature, window):
        rows = df[
            (df["time"] <= row["time"])
            & (df["time"] > row["time"] - timedelta(minutes=window))
            & (df["shift"] == row["shift"])
        ]
        sentiment = rows[feature].mean()
        return sentiment

    def get_valid_dates(
        self, ticker: str, max_days: int = 200, type_: str = "processed"
    ) -> list[str]:
        return sorted(
            [
                x.split(".")[0]
                for x in os.listdir(os.path.join(self.data_dir, type_, ticker))
            ]
        )[-max_days:]

    def add_window_features(
        self,
        df,
        windows=[2, 5, 15, 60, 120],
        agg_features=["count"],
        vol_features=["yes_price", "count"],
        sentiment_features=["taker_side", "yes_price", "count"],
        trend_features=["yes_price"],
    ):

        for window in windows:
            for feature in agg_features:
                df[f"{feature}_agg_{window}"] = df.apply(
                    lambda row: self._add_agg_features(df, row, feature, window=window),
                    axis=1,
                )
            for feature in vol_features:
                df[f"{feature}_vol_{window}"] = df.apply(
                    lambda row: self._add_vol_features(df, row, feature, window=window),
                    axis=1,
                )
            for feature in sentiment_features:
                df[f"{feature}_sentiment_{window}"] = df.apply(
                    lambda row: self._add_sentiment_features(
                        df, row, feature, window=window
                    ),
                    axis=1,
                )
            for feature in trend_features:
                df[f"{feature}_trend_{window}"] = df.apply(
                    lambda row: self._add_trend_features(
                        df, row, feature, window=window
                    ),
                    axis=1,
                )
        return df

    def add_window_features_last_trade(
        self,
        df,
        windows=[2, 5, 15, 60, 120],
        agg_features=["count"],
        vol_features=["yes_price", "count"],
        sentiment_features=["taker_side", "yes_price", "count"],
        trend_features=["yes_price"],
    ):
        for window in windows:
            for feature in agg_features:
                df[f"{feature}_agg_{window}"].iloc[-1] = self._add_agg_features(
                    df, df.iloc[-1], feature, window=window
                )
            for feature in vol_features:
                df[f"{feature}_vol_{window}"].iloc[-1] = self._add_vol_features(
                    df, df.iloc[-1], feature, window=window
                )
            for feature in sentiment_features:
                df[f"{feature}_sentiment_{window}"].iloc[-1] = (
                    self._add_sentiment_features(
                        df, df.iloc[-1], feature, window=window
                    )
                )
            for feature in trend_features:
                df[f"{feature}_trend_{window}"].iloc[-1] = self._add_trend_features(
                    df, df.iloc[-1], feature, window=window
                )
        df = df.fillna(0)
        return df

    def normalize_features(self, df, features: list[str]):
        for feature in features:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        return df

    def load_weather_forecast(self, ticker: str):
        daily_weather_forecast_df = pd.read_csv(
            f"{self.data_dir}/weather/{ticker}/daily_weather_forecast.csv"
        )
        daily_weather_forecast_df["time"] = pd.to_datetime(
            daily_weather_forecast_df["time"]
        )
        hourly_weather_forecast_df = pd.read_csv(
            f"{self.data_dir}/weather/{ticker}/hourly_weather_forecast.csv"
        )
        hourly_weather_forecast_df["time"] = pd.to_datetime(
            hourly_weather_forecast_df["time"]
        )
        daily_weather_forecast = {}
        for i, row in daily_weather_forecast_df.iterrows():
            forecast = {k: v for k, v in row.to_dict().items()}
            daily_weather_forecast[forecast["time"].strftime("%Y-%m-%d")] = forecast

        hourly_weather_forecast = {}
        for i, row in hourly_weather_forecast_df.iterrows():
            forecast = {k: v for k, v in row.to_dict().items()}
            hourly_weather_forecast[forecast["time"].strftime("%Y-%m-%dT%H:00")] = (
                forecast
            )
        return daily_weather_forecast, hourly_weather_forecast

    def add_weather_features(
        self,
        ticker: str,
        features,
        daily_weather_forecast,
        hourly_weather_forecast,
        strike_date: str,
        trade_hour: str,
        short_term_hour: str,
        strikes: dict,
    ):
        day_weather_forecast = daily_weather_forecast[strike_date]
        current_weather = hourly_weather_forecast[trade_hour]
        short_term_forecast = hourly_weather_forecast[short_term_hour]
        for key in day_weather_forecast:
            if "time" in key:
                continue
            if "temperature" in key:
                features[f"DF_{key}"] = day_weather_forecast[key] - strikes[ticker]
                features[f"DF_{key}_dev"] = (
                    day_weather_forecast[key] - current_weather["temperature_2m"]
                )
            else:
                features[f"DF_{key}"] = day_weather_forecast[key]
        for key in current_weather:
            if "time" in key:
                continue
            if "temperature" in key:
                features[f"CF_{key}"] = current_weather[key] - strikes[ticker]
                features[f"CF_{key}_dev"] = (
                    day_weather_forecast["temperature_2m_max"] - current_weather[key]
                )
            else:
                features[f"CF_{key}"] = current_weather[key]
        for key in short_term_forecast:
            if "time" in key:
                continue
            if "temperature" in key:
                features[f"STF_{key}"] = short_term_forecast[key] - strikes[ticker]
                features[f"STF_{key}_dev"] = (
                    day_weather_forecast["temperature_2m_max"]
                    - short_term_forecast[key]
                )
            else:
                features[f"STF_{key}"] = short_term_forecast[key]
        return features

    def get_strikes(self, event_data: dict, mean_center: bool = False):
        strikes = {}
        for i, event in enumerate(event_data["markets"]):
            lines = [event[k] for k in event if k.endswith("strike")]
            strikes[event["ticker"]] = sum(lines) / max(1, len(lines))
            if i == 0:
                strikes[event["ticker"]] -= 1.5
            if i == len(event_data["markets"]) - 1:
                strikes[event["ticker"]] += 1.5
        mean_strike = sum(strikes.values()) / len(strikes)
        if mean_center:
            for ticker in strikes:
                strikes[ticker] = strikes[ticker] - mean_strike
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

    def _get_dates(self, data_dir: str, ticker: str):
        return sorted(
            [
                x.split(".")[0]
                for x in os.listdir(os.path.join(self.data_dir, data_dir, ticker))
            ]
        )

    def _get_fair(self, dist: dict):
        total_p = sum(dist.values())
        probabilities = [k / total_p for k in dist.values()]
        values = list(dist.keys())
        return sum([p * v for p, v in zip(probabilities, values)])

    def _load_trades(self, trade_data: dict, type_: str = "kalshi"):
        trades = []
        if type_ == "kalshi":
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
        elif type_ == "polymarket":
            for series in trade_data:
                trades.extend(
                    [
                        {
                            "idx": i,
                            "strike": series,
                            "time": trade["t"],
                            "yes_price": trade["p"] * 100,
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

    def process_current_poly_signal_trade_data(
        self, verbose: bool = True, max_days: int = 50
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
            self.process_historical_poly_signal_trade_data(
                ticker, dates, verbose=verbose
            )

    def process_poly_signal_trade(
        self,
        trade: dict,
        kalshi_dist: dict,
        polymk_dist: dict,
        day_forecast: dict,
        hour_forecast: dict,
        strike_time: datetime,
        strike: float,
        mean_strike: float,
    ):
        trade_ticker = trade["ticker"]
        trade_time = self.parse_trade_time(trade["time"])
        time_to_strike = (strike_time - trade_time).total_seconds()
        kalshi_fair, polymk_fair = (
            self._get_fair(kalshi_dist),
            self._get_fair(polymk_dist),
        )
        features = {
            "time": trade_time,
            "trade_id": trade["trade_id"],
            "ticker": trade_ticker,
            "time_to_strike": time_to_strike,
            "day_forecast_strike_dev": day_forecast["temperature_2m_max"] - strike,
            "current_forecast_strike_dev": hour_forecast["temperature_2m"] - strike,
            "day_current_forecast_dev": day_forecast["temperature_2m_max"] - hour_forecast["temperature_2m"],
            "kalshi_strike_dev": kalshi_fair - strike,
            "polymk_strike_dev": polymk_fair - strike,
            "kalshi_polymk_dev": kalshi_fair - polymk_fair,
            "kalshi_day_forecast_dev": day_forecast["temperature_2m_max"] - kalshi_fair,
            "polymk_day_forecast_dev": day_forecast["temperature_2m_max"] - polymk_fair,
            "day_forecast_percipitation": day_forecast["precipitation_probability_max"],
            "day_forecast_rain": day_forecast["rain_sum"],
            "hour_forecast_rain": hour_forecast["rain"],
            "yes_price": trade["yes_price"],
            "no_price": trade["no_price"],
            "taker_side": int(trade["taker_side"] == "yes"),
            "count": trade["count"],
            "shift": strike - mean_strike,
        }
        return features
    
    def process_poly_signal_trade_data(
        self, 
        trades_data: dict,
        polymk_data: dict,
        events_data: dict,
        day_forecast: dict,
        hourly_forecast: dict,
        results: dict = None,
        trades_post_average: dict = None,
        threshold: int = 3,
        max_tts: int = 3600 * 24,
    ):
        strike_time = self.get_strike_times(events_data)
        strikes, mean_strike = self.get_strikes(events_data)
        trades, prices = (
            self._load_trades(trades_data, type_="kalshi"),
            self._load_trades(polymk_data, type_="polymarket"),
        )
        kalshi_dist, polymk_dist = {}, {}
        trade_data, p_idx = [], 0
        for i, trade in enumerate(trades):
            trade_ticker, trade_idx = trade["ticker"], trade["idx"]
            trade_time = self.parse_trade_time(trade["time"])
            tts = (strike_time - trade_time).total_seconds()
            if (
                min(trade["yes_price"], trade["no_price"]) < threshold
                or tts < 0
                or tts > max_tts
            ):
                continue
            trade_ts = datetime.fromisoformat(
                trade["time"].replace("Z", "+00:00")
            ).timestamp()
            while p_idx < len(prices) and prices[p_idx]["time"] < trade_ts:
                polymk_dist[float(prices[p_idx]["strike"])] = prices[p_idx][
                    "yes_price"
                ]
                p_idx += 1
            strike = strikes[trade_ticker]
            kalshi_dist[strike] = trade["yes_price"]
            trade_hour = trade_time.strftime("%Y-%m-%dT%H:00")
            hour_forecast = hourly_forecast[trade_hour]
            if (
                min(trade["yes_price"], trade["no_price"]) < threshold
                or tts < 0
                or tts > max_tts
            ):
                continue
            features = self.process_poly_signal_trade(
                trade,
                kalshi_dist,
                polymk_dist,
                day_forecast,
                hour_forecast,
                strike_time,
                strike,
                mean_strike,
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
                    features['outcome'] = int(results[trade_ticker] == "yes")
                else:
                    features["impact"] = (
                        100 - trades_post_average[trade_ticker][trade_idx]
                    ) - trade["no_price"]
                    features["result"] = (
                        -trade["no_price"]
                        if results[trade_ticker] == "yes"
                        else (100 - trade["no_price"])
                    )
                    features['outcome'] = int(results[trade_ticker] == "yes")
            trade_data.append(features)
        if len(trade_data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(trade_data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time")
        df = self.add_window_features(df)
        df = df.fillna(0)
        df = df.round(3)
        return df

    def process_historical_poly_signal_trade_data(
        self,
        ticker: str,
        dates: Optional[list[str]] = None,
        threshold: int = 3,
        max_tts: int = 3600 * 24,
        verbose: bool = True,
    ):
        daily_forecast, hourly_forecast = self.load_weather_forecast(ticker)
        if dates is None:
            dates = self._get_dates("polymarket", ticker)
        dates = sorted(dates)
        idx = 0
        iterator = (
            tqdm(dates, desc=f"Processing {ticker} for {dates[idx]}")
            if verbose
            else dates
        )
        for date in iterator:
            events_data, trades_data, polymk_data = (
                self._get_data(ticker, date, "kalshi", "events"),
                self._get_data(ticker, date, "kalshi", "trades"),
                self._get_data(ticker, date, "polymarket", ""),
            )
            strike_time = self.get_strike_times(events_data)
            day_forecast = daily_forecast[strike_time.strftime("%Y-%m-%d")]
            trades_post_average = {
                k: self._get_trade_post_average(v) for k, v in trades_data.items()
            }
            results = {
                event["ticker"]: event["result"] for event in events_data["markets"]
            }
            df = self.process_poly_signal_trade_data(
                trades_data,
                polymk_data,
                events_data,
                day_forecast,
                hourly_forecast,
                results=results,
                trades_post_average=trades_post_average,
                threshold=threshold,
                max_tts=max_tts,
            )
            if ticker not in os.listdir(os.path.join(self.data_dir, "polysignal")):
                os.makedirs(os.path.join(self.data_dir, "polysignal", ticker))
            df.to_csv(
                os.path.join(self.data_dir, "polysignal", ticker, f"{date}.csv"),
                index=False,
            )
            idx += 1
            if idx < len(dates):
                iterator.set_description(f"Processing {ticker} for {dates[idx]}")

    def process_consolidated_trade_data(
        self,
        ticker: str,
        dates: Optional[list[str]] = None,
        threshold: int = 3,
        verbose: bool = True,
    ) -> pd.DataFrame:
        daily_weather_forecast, hourly_weather_forecast = self.load_weather_forecast(
            ticker
        )
        dates = sorted(dates)
        idx = 0
        iterator = (
            tqdm(dates, desc=f"Processing {ticker} for {dates[idx]}")
            if verbose
            else dates
        )
        for date in iterator:
            with open(
                os.path.join(self.data_dir, "kalshi", ticker, "events", f"{date}.json"),
                "r",
            ) as f:
                events_data = json.load(f)
            with open(
                os.path.join(self.data_dir, "kalshi", ticker, "trades", f"{date}.json"),
                "r",
            ) as f:
                trades_data = json.load(f)
            results = {
                event["ticker"]: event["result"] for event in events_data["markets"]
            }
            if "strike_date" not in events_data["event"]:
                strike_time = datetime.strptime(
                    events_data["markets"][0]["close_time"], "%Y-%m-%dT%H:%M:%SZ"
                )
            else:
                strike_time = datetime.strptime(
                    events_data["event"]["strike_date"], "%Y-%m-%dT%H:%M:%SZ"
                )
            strike_date_obj = strike_time.date() - timedelta(days=1)
            strike_date = strike_date_obj.strftime("%Y-%m-%d")
            strikes, mean_strike = self.get_strikes(events_data)
            trade_data_flattened = []
            for series in trades_data:
                trade_price_postfix, trade_count_postfix = [0], [0]
                for i, trade in enumerate(
                    sorted(trades_data[series], key=lambda x: x["created_time"])
                ):
                    trade_price_postfix.append(
                        trade["yes_price"] * trade["count"] + trade_price_postfix[-1]
                    )
                    trade_count_postfix.append(trade["count"] + trade_count_postfix[-1])

                total_trade_price, total_trade_count = (
                    trade_price_postfix[-1],
                    trade_count_postfix[-1],
                )
                trade_price_average = [0] * len(trades_data[series])
                for i, trade in enumerate(trades_data[series]):
                    trade_price_average[i] = (
                        total_trade_price - trade_price_postfix[i]
                    ) / (total_trade_count - trade_count_postfix[i])

                for i, trade in enumerate(
                    sorted(trades_data[series], key=lambda x: x["created_time"])
                ):
                    if (trade["yes_price"] < threshold) or (
                        trade["no_price"] < threshold
                    ):
                        continue
                    trade_time = datetime.strptime(
                        trade["created_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    short_term_hour = (trade_time + timedelta(hours=1)).strftime(
                        "%Y-%m-%dT%H:00"
                    )
                    trade_hour = trade_time.strftime("%Y-%m-%dT%H:00")
                    event_ticker = trade["ticker"]
                    features = {
                        "time": trade_time,
                        "day_of_year": trade_time.timetuple().tm_yday,
                        "time_to_strike": (strike_time - trade_time).total_seconds(),
                        "time_of_day": trade_time.hour
                        + trade_time.minute / 60
                        + trade_time.second / 3600,
                        "count": trade["count"],
                        "yes_price": trade["yes_price"],
                        "no_price": trade["no_price"],
                        "taker_side": int(trade["taker_side"] == "yes"),
                        "shift": strikes[event_ticker] - mean_strike,
                    }
                    try:
                        features = self.add_weather_features(
                            event_ticker,
                            features,
                            daily_weather_forecast,
                            hourly_weather_forecast,
                            strike_date,
                            trade_hour,
                            short_term_hour,
                            strikes,
                        )
                    except Exception as e:
                        continue
                    if trade["taker_side"] == "yes":
                        features["impact"] = trade_price_average[i] - trade["yes_price"]
                        features["result"] = (
                            (100 - trade["yes_price"])
                            if results[event_ticker] == "yes"
                            else -trade["yes_price"]
                        )
                    else:
                        features["impact"] = trade["no_price"] - (
                            (100 - trade_price_average[i])
                        )
                        features["result"] = (
                            -trade["no_price"]
                            if results[event_ticker] == "yes"
                            else (100 - trade["no_price"])
                        )
                    trade_data_flattened.append(features)
            trade_data_flattened.sort(key=lambda x: x["time"])
            trade_data_flattened = pd.DataFrame(trade_data_flattened)
            
            trade_data_flattened["time"] = pd.to_datetime(trade_data_flattened["time"])
            trade_data_flattened = self.add_window_features(trade_data_flattened)
            if not os.path.exists(os.path.join(self.data_dir, "processed", ticker)):
                os.makedirs(os.path.join(self.data_dir, "processed", ticker))
            trade_data_flattened.to_csv(
                os.path.join(self.data_dir, "processed", ticker, f"{date}.csv"),
                index=False,
            )
            idx += 1
            if idx < len(dates):
                iterator.set_description(f"Processing {ticker} for {dates[idx]}")

    def process_current_weather_event_trade_data(self, verbose: bool = True):
        for ticker in self.metadata["weather"]:
            if ticker not in os.listdir(os.path.join(self.data_dir, "processed")):
                os.makedirs(os.path.join(self.data_dir, "processed", ticker))
            event_dates = set(
                [
                    x.split(".")[0]
                    for x in os.listdir(
                        os.path.join(self.data_dir, "kalshi", ticker, "events")
                    )
                ]
            )
            processed_dates = set(
                [
                    x.split(".")[0]
                    for x in os.listdir(
                        os.path.join(self.data_dir, "processed", ticker)
                    )
                ]
            )
            valid_dates = list(event_dates - processed_dates)
            if len(valid_dates) == 0:
                continue
            self.process_consolidated_trade_data(
                ticker, sorted(list(valid_dates)), verbose=verbose
            )

    def get_basic_features(
        self,
        trade: dict,
        strike_time: datetime,
        strikes: dict,
        mean_strike: float = None,
    ):
        trade_time = datetime.strptime(trade["created_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
        event_ticker = trade["ticker"]
        features = {
            "time": trade_time,
            "day_of_year": trade_time.timetuple().tm_yday,
            "time_to_strike": (strike_time - trade_time).total_seconds(),
            "time_of_day": trade_time.hour
            + trade_time.minute / 60
            + trade_time.second / 3600,
            "count": trade["count"],
            "yes_price": trade["yes_price"],
            "no_price": trade["no_price"],
            "taker_side": int(trade["taker_side"] == "yes"),
            "shift": (
                (strikes[event_ticker] - mean_strike)
                if mean_strike is not None
                else strikes[event_ticker]
            ),
        }
        return features

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
        self, ticker: str, max_days: int = 200, type_: str = "processed"
    ) -> pd.DataFrame:
        dates = self.get_valid_dates(ticker, max_days=max_days, type_=type_)
        result_df = pd.DataFrame()
        for date in tqdm(sorted(dates)):
            df = pd.read_csv(os.path.join(self.data_dir, type_, ticker, f"{date}.csv"))
            result_df = pd.concat([result_df, df])
        result_df = result_df.fillna(0)
        result_df = result_df.reset_index(drop=True)
        return result_df

    def load_daily_data(
        self, ticker: str, max_days: int = 200, type_: str = "processed"
    ) -> pd.DataFrame:
        daily_data = {}
        dates = self.get_valid_dates(ticker, max_days=max_days, type_=type_)
        for date in tqdm(sorted(dates)):
            df = pd.read_csv(os.path.join(self.data_dir, type_, ticker, f"{date}.csv"))
            df = df.fillna(0)
            daily_data[date] = df
        return daily_data


if __name__ == "__main__":
    data_dir = "../data"
    loader = DataLoader(data_dir)
    # loader.process_current_weather_event_trade_data()
    # loader.process_poly_signal_trade_data("kxhighny", to_csv=True)
    # loader.process_poly_signal_trade_data("kxhighny", to_csv=True)
    loader.process_historical_poly_signal_trade_data("kxhighny")
