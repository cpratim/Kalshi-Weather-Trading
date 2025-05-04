import requests
import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import TradeParams, BookParams
from pprint import pprint
import json

load_dotenv()
from datetime import datetime, timedelta
from time import sleep, time


class PolyMarketAPI(object):

    def __init__(
        self, data_dir: str = "../data", event: str = "highest-temperature-in-nyc"
    ):
        self.host = "https://clob.polymarket.com/"
        self.key = os.getenv("POLYMARKET_CLOB_API_KEY")
        self.chain_id = 137
        self.client = ClobClient(self.host, key=self.key, chain_id=self.chain_id)
        self.event = event
        self.data_dir = data_dir
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds)
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def get_strike(self, title):
        s = title.split("°")[0]
        t = [int(x) for x in s.split("-")]
        return sum(t) / len(t)

    def get_polymarket_markets(self, date_ptr: datetime):
        month_lower = date_ptr.strftime("%B").lower()
        day_lower = str(int(date_ptr.strftime("%d"))).lower()
        event_slug = f"{self.event}-on-{month_lower}-{day_lower}"
        response = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"slug": event_slug},
        ).json()
        if len(response) == 0:
            return {}
        return response[0]["markets"]

    def get_polymarket_data(self, date_ptr: datetime):
        poly_markets = self.get_polymarket_markets(date_ptr)

        trade_data = {}
        for market in poly_markets:
            strike = self.get_strike(market["groupItemTitle"])
            tokens = json.loads(market["clobTokenIds"])
            trades = {}
            for token in tokens:
                data = requests.get(
                    f"{self.host}/prices-history",
                    params={
                        "market": token,
                        "interval": "max",
                        "fidelity": "1",
                    },
                ).json()
                for h in data["history"]:
                    trades[h["t"]] = h["p"]
            trade_data[strike] = [{"t": t, "p": p} for t, p in sorted(trades.items())]
        return trade_data

    def update_current_poly_signal_data(
        self, max_days: int = 100, verbose: bool = False
    ):
        for ticker in self.metadata["polymarket"]:
            dates = set(
                [
                    x.split(".")[0]
                    for x in os.listdir(
                        os.path.join(self.data_dir, "polymarket", ticker)
                    )
                ]
            )
            date_ptr = datetime.now() - timedelta(days=1)
            iter = 0
            while iter < max_days and date_ptr.strftime("%Y-%m-%d") not in dates:
                trade_data = self.get_polymarket_data(date_ptr)
                if len(trade_data) == 0:
                    if verbose:
                        print(f"No data for {date_ptr.strftime('%Y-%m-%d')}")
                    break

                with open(
                    os.path.join(
                        self.data_dir,
                        "polymarket",
                        ticker,
                        f"{date_ptr.strftime('%Y-%m-%d')}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(trade_data, f, indent=4)
                if verbose:
                    print(f"Saved data for {date_ptr.strftime('%Y-%m-%d')}")
                iter += 1
                date_ptr = date_ptr - timedelta(days=1)
                sleep(0.5)

    def get_market_token_map(self, poly_markets: list):
        token_map = {}
        for market in poly_markets:
            market_tokens = json.loads(market["clobTokenIds"])
            token_map[market["groupItemTitle"]] = market_tokens[0]
        return token_map

    def get_polymarket_dist(self, token_map: dict):
        params = [BookParams(token_id=v) for v in token_map.values()]
        data = self.client.get_midpoints(params=params)
        dist = {}
        for title, token in token_map.items():
            strike = self.get_strike(title)
            if token in data:
                dist[strike] = float(data[token])
        return dist


class PolyMarketWS(object):
    def __init__(self, event: str = "highest-temperature-in-nyc"):
        self.event = event
        self.host = "wss://gamma-api.polymarket.com/events"
        self.ws = None


if __name__ == "__main__":
    api = PolyMarketAPI()
    poly_markets = api.get_polymarket_markets(datetime.now() + timedelta(days=1))
    poly_token_map = api.get_market_token_map(poly_markets)
