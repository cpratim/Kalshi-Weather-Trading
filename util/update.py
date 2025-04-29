from api.kalshi import KalshiAPI
from util.load import DataLoader
from api.weather import OpenMeteoAPI
from api.polymarket import PolyMarketAPI


def get_all_updates(data_dir: str = "../data"):
    loader = DataLoader(data_dir=data_dir)
    kalshi = KalshiAPI(data_dir=data_dir)
    weather = OpenMeteoAPI(data_dir=data_dir)
    polymarket = PolyMarketAPI(data_dir=data_dir)
    kalshi.update_current_weather_event_data(verbose=False)
    print("UPDATED   | Kalshi     [event data]")
    kalshi.update_current_weather_trade_data(verbose=False)
    print("UPDATED   | Kalshi     [trade data]")
    weather.update_historical_forecast_data(verbose=False)
    print("UPDATED   | OpenMeteo  [forecast data]")
    polymarket.update_current_poly_signal_data(verbose=False)
    print("UPDATED   | PolyMarket [trade data]")
    loader.process_current_weather_event_trade_data(verbose=False)
    print("PROCESSED | Kalshi     [event and trade data]")
    loader.process_current_poly_signal_trade_data(verbose=False)
    print("PROCESSED | PolyMarket [trade data]")
    print("FINISHED  | All APIs   [updated and processed]")


if __name__ == "__main__":
    get_all_updates()
