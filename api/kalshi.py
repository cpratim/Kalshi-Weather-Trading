import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend
import base64
import time
from typing import Dict, Any
import websockets
import asyncio


load_dotenv()


class KalshiAuth(object):

    def __init__(self, private_key: str, api_key_id: str):
        self.private_key_str = private_key
        self.api_key_id = api_key_id
        if private_key is not None:
            self.private_key = load_pem_private_key(
                private_key.encode("utf-8"), password=None, backend=default_backend()
            )

    def _load_private_key(self):
        if self._private_key is None:
            private_key_bytes = self.private_key_str.encode()
            self._private_key = load_pem_private_key(private_key_bytes, password=None)
        return self._private_key

    def _generate_signature(self, timestamp):
        private_key = self._load_private_key()
        message = f"{self.api_key_id}{timestamp}".encode()
        signature = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
        return base64.b64encode(signature).decode()

    def get_auth_headers(self, method: str, path: str) -> Dict[str, Any]:
        current_time_milliseconds = int(time.time() * 1000)
        timestamp_str = str(current_time_milliseconds)
        path_parts = path.split("?")

        msg_string = timestamp_str + method + path_parts[0]
        signature = self.sign_pss_text(msg_string)

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }
        return headers

    def sign_pss_text(self, text: str) -> str:
        """Signs the text using RSA-PSS and returns the base64 encoded signature."""
        message = text.encode("utf-8")
        try:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception as e:
            print(e)
            raise ValueError("RSA sign PSS failed") from e


class KalshiAPI(KalshiAuth):

    def __init__(
        self, private_key: str = None, api_key_id: str = None, data_dir: str = "../data"
    ):
        super().__init__(private_key, api_key_id)
        self.base = "https://api.elections.kalshi.com/trade-api/v2"
        self.headers = {"accept": "application/json"}
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def _make_request(self, endpoint: str, params: dict = None, as_json: bool = True):
        response = requests.get(
            f"{self.base}/{endpoint}", headers=self.headers, params=params
        )
        if as_json:
            return response.json()
        return response

    def _log(self, message: str):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def get_current_weather_events(self):
        tickers = {}
        for market in self.metadata["weather"]:
            events = self._make_request(
                "events/" + f"{market}-{datetime.now().strftime('%y%b%d')}".upper(),
                params={"status": "open", "with_nested_markets": True},
            )
            tickers[market] = [event["ticker"] for event in events["event"]["markets"]]
        return tickers

    def get_current_weather_event_data(self, market: str):
        events = self._get_event_data(market, datetime.now(), with_nested_markets=False)
        return events

    def _get_event_data(
        self, ticker: str, date: str, with_nested_markets: bool = False
    ):
        response = self._make_request(
            "events/" + f"{ticker}-{date.strftime('%y%b%d')}".upper(),
            params={"with_nested_markets": with_nested_markets},
        )
        return response

    def _get_trade_data(
        self,
        event_data: Dict[str, Any],
        date_ptr: datetime,
        buffer_days: int = 5,
        limit: int = 1000,
    ):
        trade_data = {}
        if "markets" not in event_data:
            return
        for market in event_data["markets"]:
            ticker = market["ticker"]
            params = {
                "ticker": ticker,
                "start_ts": int((date_ptr - timedelta(days=buffer_days)).timestamp()),
                "end_ts": int((date_ptr + timedelta(days=buffer_days)).timestamp()),
                "limit": limit,
            }
            response = self._make_request(f"markets/trades", params=params)
            trades = response["trades"]
            while response["cursor"]:
                params["cursor"] = response["cursor"]
                response = self._make_request(f"markets/trades", params=params)
                if "trades" in response and len(response["trades"]) > 0:
                    trades.extend(response["trades"])
            trade_data[ticker] = trades
        return trade_data

    def get_current_trade_data(self, event_data: dict):
        return self._get_trade_data(event_data, datetime.now())

    def update_current_weather_event_data(
        self, max_days: int = 200, verbose: bool = False
    ):
        for market in self.metadata["weather"]:
            if market not in os.listdir(os.path.join(self.data_dir, "kalshi")):
                os.makedirs(os.path.join(self.data_dir, "kalshi", market, "events"))
            events_files = set(
                os.listdir(os.path.join(self.data_dir, "kalshi", market, "events"))
            )
            date_ptr = datetime.now() - timedelta(days=1)
            ticker = market
            iter = 0
            while (
                date_ptr.strftime("%Y-%m-%d") + ".json"
            ) not in events_files and iter < max_days:
                events = self._get_event_data(ticker, date_ptr)
                if "error" in events:
                    ticker = ticker[2:]
                    events = self._get_event_data(ticker, date_ptr)
                    if "error" in events:
                        break
                with open(
                    os.path.join(
                        self.data_dir,
                        "kalshi",
                        market,
                        "events",
                        f"{date_ptr.strftime('%Y-%m-%d')}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(events, f, indent=4)
                if verbose:
                    self._log(
                        f"Downloaded {market} for {date_ptr.strftime('%Y-%m-%d')}"
                    )
                date_ptr -= timedelta(days=1)
                iter += 1

    def update_current_weather_trade_data(
        self, max_days: int = 200, verbose: bool = False
    ):
        for market in self.metadata["weather"]:
            if "trades" not in os.listdir(
                os.path.join(self.data_dir, "kalshi", market)
            ):
                os.makedirs(os.path.join(self.data_dir, "kalshi", market, "trades"))
            trade_files = set(
                os.listdir(os.path.join(self.data_dir, "kalshi", market, "trades"))
            )
            date_ptr = datetime.now() - timedelta(days=1)
            iter = 0
            while (
                date_ptr.strftime("%Y-%m-%d") + ".json"
            ) not in trade_files and iter < max_days:
                with open(
                    os.path.join(
                        self.data_dir,
                        "kalshi",
                        market,
                        "events",
                        f"{date_ptr.strftime('%Y-%m-%d')}.json",
                    ),
                    "r",
                ) as f:
                    event_data = json.load(f)
                trade_data = self._get_trade_data(event_data, date_ptr)
                if "error" in trade_data:
                    break
                with open(
                    os.path.join(
                        self.data_dir,
                        "kalshi",
                        market,
                        "trades",
                        f"{date_ptr.strftime('%Y-%m-%d')}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(trade_data, f, indent=4)
                if verbose:
                    self._log(
                        f"Downloaded {market} for {date_ptr.strftime('%Y-%m-%d')}"
                    )
                date_ptr -= timedelta(days=1)
                iter += 1

    def get_market_results(self, ticker: str, date: str):
        with open(os.path.join(self.data_dir, "kalshi", ticker, "trades", f"{date}.json"), "r") as f:
            trade_data = json.load(f)
        results = {}
        for market, trades in trade_data.items():
            no_price, yes_price = trades[0]["no_price"], trades[0]["yes_price"]
            if no_price > yes_price:
                results[market] = "no"
            else:
                results[market] = "yes"
        return results


class KalshiWS(KalshiAuth):

    def __init__(
        self,
        tickers: list[str],
        private_key: str,
        api_key_id: str,
        on_message_callback: callable = None,
    ):
        super().__init__(private_key, api_key_id)
        self.tickers = tickers
        self.base = "wss://api.elections.kalshi.com"
        self.url_suffix = "/trade-api/ws/v2"
        self.channels = ["orderbook_delta", "ticker", "trade", "fill"]
        self.ws = None
        self.idx = 1
        self.on_message_callback = on_message_callback

    async def on_open(self):
        command = {
            "id": self.idx,
            "cmd": "subscribe",
            "params": {"channels": self.channels, "market_tickers": self.tickers},
        }
        await self.ws.send(json.dumps(command))
        self.idx += 1

    async def on_message(self, message):
        if self.on_message_callback is not None:
            await self.on_message_callback(message)

    async def on_error(self, arg1, arg2):
        print(arg1, arg2)

    async def handler(self):
        try:
            async for message in self.ws:
                await self.on_message(message)
        except websockets.ConnectionClosed as e:
            await self.on_close(e.code, e.reason)
            await self.start_ws()
        except Exception as e:
            await self.on_error(e)

    async def on_close(self, code, reason):
        print(f"Connection closed: {code} - {reason}")

    async def start_ws(self):
        host = self.base + self.url_suffix
        auth_headers = self.get_auth_headers("GET", self.url_suffix)
        async with websockets.connect(
            host, additional_headers=auth_headers
        ) as websocket:
            self.ws = websocket
            await self.on_open()
            await self.handler()

    def start(self):
        asyncio.run(self.start_ws())


if __name__ == "__main__":
    kalshi_api = KalshiAPI()
    # kalshi_api.update_current_weather_event_data(max_days=250, verbose=True)
    print(kalshi_api.get_market_results("kxhighny", "2025-04-29"))
