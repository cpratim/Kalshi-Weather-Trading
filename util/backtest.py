import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from collections import deque
from sklearn.metrics import r2_score
from hashlib import sha256
from datetime import datetime
from math import ceil
from util.load import DataLoader


class Backtest(object):

    def __init__(
        self,
        ticker: str,
        data_dir: str = "../data",
        backtest_window: int = 30,
        min_window_size: int = 15,
        max_window_size: int = 20,
    ):
        self.ticker = ticker
        self.data_dir = data_dir
        self.loader = DataLoader(self.data_dir)
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.data = self.loader.load_daily_data(
            self.ticker, max_days=backtest_window + min_window_size, type_="polysignal"
        )

    def concat_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        data = pd.concat(data)
        data = data.fillna(0)
        return data

    def normalize_feature(self, feature: pd.Series, scaler: Any) -> pd.Series:
        return scaler.fit_transform(feature.values.reshape(-1, 1)).flatten()

    def get_backtest_stats(
        self,
        pipeline: Any,
        min_window_size: int,
        max_window_size: int,
        output_metric: str = "impact",
        latest_n_days: int = None,
        scaler: Any = None,
    ) -> Dict[str, List[float]]:

        stats_df = {
            "r2_train": [],
            "r2_test": [],
            "corr_train": [],
            "corr_test": [],
        }
        predictions_df = {
            "y_real": [],
            "y_pred": [],
        }
        day_predictions = {}

        dates = sorted(list(self.data.keys()))
        if latest_n_days is not None:
            dates = dates[-(latest_n_days + min_window_size) :]

        iterator = tqdm(dates)
        rolling_df = deque(maxlen=max_window_size)
        predictions = pd.DataFrame()
        for date in iterator:
            if len(rolling_df) > max_window_size:
                rolling_df.popleft()
            if (
                len(rolling_df) >= min_window_size
                and len(rolling_df) <= max_window_size
            ):
                train_data = self.concat_data(rolling_df)
                input_data = self.data[date]
                if scaler is not None:
                    train_data[output_metric] = self.normalize_feature(
                        train_data[output_metric], scaler
                    )
                    input_data[output_metric] = self.normalize_feature(
                        input_data[output_metric], scaler
                    )
                input_outputs = self.data[date][output_metric]
                train_outputs = train_data[output_metric]
                train_predictions, input_predictions = pipeline(train_data, input_data)

                # merge input_data and input_predictions
                input_predictions_df = pd.DataFrame(
                    {f"{output_metric}_pred": input_predictions}
                )
                day_pred = pd.concat([input_data, input_predictions_df], axis=1)
                day_pred.columns = [*input_data.columns, *input_predictions_df.columns]
                predictions = pd.concat([predictions, day_pred])
                r2_train = r2_score(train_outputs, train_predictions)
                r2_test = r2_score(input_outputs, input_predictions)
                corr_train = np.corrcoef(train_outputs, train_predictions)[0, 1]
                corr_test = np.corrcoef(input_outputs, input_predictions)[0, 1]

                stats_df["r2_train"].append(r2_train)
                stats_df["r2_test"].append(r2_test)
                stats_df["corr_train"].append(corr_train)
                stats_df["corr_test"].append(corr_test)
                predictions_df["y_real"].extend(input_outputs)
                predictions_df["y_pred"].extend(input_predictions)
                day_predictions[date] = {
                    "y_real": input_outputs,
                    "y_pred": input_predictions,
                }

            rolling_df.append(self.data[date])

        return (
            pd.DataFrame(stats_df),
            pd.DataFrame(predictions_df),
            day_predictions,
            predictions,
        )

    def run_backtest(
        self,
        algorithm: Any,
        min_window_size: int,
        max_window_size: int,
        latest_n_days: int = None,
        verbose: bool = False,
    ):
        dates = sorted(list(self.data.keys()))
        if latest_n_days is not None:
            dates = dates[-(latest_n_days + min_window_size) :]
        iterator = tqdm(dates) if verbose else dates
        rolling_df = deque(maxlen=max_window_size)

        for date in iterator:
            if len(rolling_df) > max_window_size:
                rolling_df.popleft()
            if (
                len(rolling_df) >= min_window_size
                and len(rolling_df) <= max_window_size
            ):
                train_data = self.concat_data(rolling_df)
                input_data = self.data[date]
                predictions = algorithm.predict(train_data, input_data)
                trade_data = self.data[date]
                trade_data["impact_prediction"] = predictions
                for _, row in trade_data.iterrows():
                    trade = row.to_dict()
                    order = algorithm.on_trade(trade)
                    # matched_orders, order_id = self.exchange.on_trade(trade, order)
                    # if len(matched_orders):
                    #     algorithm.on_matched_orders(matched_orders, order_id)
            algorithm.reset_day(
                date,
                update=(
                    len(rolling_df) >= min_window_size
                    and len(rolling_df) <= max_window_size
                ),
            )
            rolling_df.append(self.data[date])

    def run_backtest(
        self,
        Algorithm: Any,
        signal: Any,
        verbose: bool = False,
    ):
        dates = sorted(list(self.data.keys()))
        iterator = tqdm(dates) if verbose else dates
        rolling_df = deque(maxlen=self.max_window_size)
        for date in iterator:
            if len(rolling_df) > self.max_window_size:
                rolling_df.popleft()
            if len(rolling_df) >= self.min_window_size:
                train_data = self.concat_data(rolling_df)
                # signal.fit(train_data)
                algo = Algorithm(self.ticker, date, signal)
                algo.init_algorithm()
                algo.fit_signal(train_data)
                algo.start()

                results = algo.kernel.stream.get_results()
                # portfolio = algo.get_portfolio()
                day_results = algo.kernel.exchange.settle_day(results)
                print(
                    f"[{date}] - Profit: {day_results['profit']:8.2f}  | Quantity: {day_results['quantity']:8.2f} | Volume: {day_results['volume']:8.2f} | Fees: {day_results['fees_paid']:8.2f}"
                )
                # for ticker in positions:
                #     day[ticker] = {'p': positions[ticker], 'r': results[ticker]}
                # print(day)
                # print(profit)
                # print()
            rolling_df.append(self.data[date])
