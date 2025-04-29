import requests
import uuid
from api.kalshi import KalshiAPI
import numpy as np
from math import ceil
from datetime import datetime
import logging


class Exchange(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.portfolio = {}
        self.max_exposure = kwargs.get("max_exposure", 50)

    def buy_yes(self, ticker: str, count: int, price: float, slack: float = 0.01):
        exposure = self.get_exposure(ticker)
        if abs(exposure) + count > self.max_exposure:
            return {"status": "rejected", "reason": "max_exposure"}
        if exposure >= 0:
            action = "buy"
            limit_price = price + slack
            side = "yes"
        else:
            action = "sell"
            limit_price = (1 - price) - slack
            side = "no"
            count = min(count, abs(exposure))
        return self.submit_order(ticker, action, side, limit_price, count)

    def buy_no(self, ticker: str, count: int, price: float, slack: float = 0.01):
        exposure = self.get_exposure(ticker)
        if abs(exposure) + count > self.max_exposure and exposure < 0:
            return {"status": "rejected", "reason": "max_exposure"}
        if exposure <= 0:
            action = "buy"
            limit_price = price + slack
            side = "no"
        else:
            action = "sell"
            limit_price = (1 - price) - slack
            side = "yes"
            count = min(count, abs(exposure))
        return self.submit_order(ticker, action, side, limit_price, count)

    def get_portfolio(self):
        return {}
    
    def get_exposure(self, ticker: str):
        return self.get_portfolio().get(ticker, 0) 

    def submit_order(
        self, ticker: str, action: str, side: str, limit_price: float, count: int
    ):
        return {}


class RealTimeExchange(Exchange):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kalshi_api = KalshiAPI(
            api_key_id=kwargs.get("api_key_id"),
            private_key=kwargs.get("rsa_private_key"),
        )
        self.logger = logging.getLogger(__name__)
        self.kwargs = kwargs

    def submit_order(
        self, ticker: str, action: str, side: str, limit_price: float, count: int
    ):
        order_json = {
            "client_order_id": str(uuid.uuid4()),
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": int(count),
            "type": "limit",
            "expiration_ts": int(datetime.now().timestamp()),
        }
        if side == "yes":
            order_json["yes_price"] = int(limit_price * 100)
        else:
            order_json["no_price"] = int(limit_price * 100)
        logging.info(f"SUBMIT - {order_json}")
        response = requests.post(
            f"{self.kalshi_api.base}/portfolio/orders",
            headers=self.kalshi_api.get_auth_headers(
                "POST", "/trade-api/v2/portfolio/orders"
            ),
            json=order_json,
        ).json()
        if "order" in response:
            if response['order']['status'] == "executed":
                logging.info(f"EXECUTED - {response['order']}")
            else:
                logging.info(f"CANCELLED - {response['order']}")
        else:
            logging.error(f"ERROR - {response}")
            return {"status": "rejected", "reason": "error"}
        return response['order']

    def get_portfolio(self):
        response = requests.get(
            f"{self.kalshi_api.base}/portfolio/positions",
            headers=self.kalshi_api.get_auth_headers(
                "GET", "/trade-api/v2/portfolio/positions"
            ),
        )
        portfolio = {}
        for position in response.json()['market_positions']:
            portfolio[position['ticker']] = position['position']
        return portfolio


class HistoricalExchange(Exchange):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orderbook = {}
        self.mm_spread_p = [0.1, 0.5, 0.3, 0.1]
        self.portfolio = {}
        self.balance = kwargs.get("balance", 500)
        self.quantity = 0
        self.fees_paid = 0
        self.volume = 0

    def _get_trade_noise(self):
        return float(
            np.random.choice(
                [x for x in range(len(self.mm_spread_p))],
                p=self.mm_spread_p,
            )
        )

    def get_fee(self, count: int, price: float):
        fee = 0.07 * count * price * (1 - price)
        return ceil(fee * 100) / 100

    def add_position(self, action: str, ticker: str, count: int, side: str):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = 0
        if action == "buy":
            self.portfolio[ticker] += count if side == "yes" else -count
        else:
            self.portfolio[ticker] -= count if side == "yes" else -count

    def on_trade(self, trade: dict):
        ticker = trade["ticker"]
        if ticker not in self.orderbook:
            self.orderbook[ticker] = {}
        self.orderbook[ticker]["yes"] = {
            "bid": (trade["yes_price"] - self._get_trade_noise()) / 100,
            "ask": (trade["yes_price"] + self._get_trade_noise()) / 100,
        }
        self.orderbook[ticker]["no"] = {
            "bid": (trade["no_price"] - self._get_trade_noise()) / 100,
            "ask": (trade["no_price"] + self._get_trade_noise()) / 100,
        }

    def get_portfolio(self):
        return self.portfolio

    def submit_order(
        self, ticker: str, action: str, side: str, limit_price: float, count
    ):
        if action == "buy":
            book_price = self.orderbook[ticker][side]["ask"]
            if limit_price >= book_price:
                execute_price = min(limit_price, book_price)
                fee = self.get_fee(count, execute_price)
                cost = -(fee + count * execute_price)
                self.balance += cost
                self.fees_paid += fee
                self.add_position(action, ticker, int(count), side)
                self.quantity += count
                self.volume += count * execute_price
                return {
                    "action": action,
                    "side": side,
                    "status": "executed",
                    "fee": fee,
                    "price": execute_price,
                    "cost": round(float(cost), 2),
                }
        else:
            book_price = self.orderbook[ticker][side]["bid"]
            if limit_price <= book_price:
                execute_price = max(limit_price, book_price)
                fee = self.get_fee(count, execute_price)
                cost = count * execute_price - fee
                self.balance += cost
                self.fees_paid += fee
                self.add_position(action, ticker, int(count), side)
                self.quantity += count
                self.volume += count * execute_price
                return {
                    "action": action,
                    "side": side,
                    "status": "executed",
                    "fee": fee,
                    "price": execute_price,
                    "cost": round(float(cost), 2),
                }
        return {"status": "rejected"}

    def settle_day(self, results: dict):
        for ticker, result in results.items():
            rt = -1 if result == "no" else 1
            if ticker in self.portfolio:
                self.balance += max(0, self.portfolio[ticker] * rt)
        profit = self.balance - self.kwargs.get("balance", 500)
        return {
            "profit": profit,
            "quantity": self.quantity,
            "volume": self.volume,
            "fees_paid": self.fees_paid,
        }

    def get_full_orderbook(self):
        return self.orderbook
