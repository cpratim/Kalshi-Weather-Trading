import requests
import os
from dotenv import load_dotenv
import json
import pandas as pd
from warnings import filterwarnings
from tqdm import tqdm
from datetime import timedelta, datetime
from pprint import pprint
filterwarnings("ignore")
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY")


class WeatherAPI(object):

    def __init__(self, API_KEY: str = None, data_dir: str = None):
        self.API_KEY = API_KEY
        self.data_dir = data_dir

    def get_forecast(self, city: str) -> dict:
        pass


class OpenMeteoAPI(WeatherAPI):

    def __init__(self, data_dir: str = "../data"):
        super().__init__(data_dir=data_dir)
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.historical_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.daily_features = [
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "showers_sum",
            "snowfall_sum",
            "precipitation_sum",
            "precipitation_hours",
            "rain_sum",
            "precipitation_probability_max",
        ]
        self.hourly_features = [
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
        ]
    def get_historical_forecast_data(
        self, ticker: str, city: str, start_date: str, end_date: str
    ) -> dict:
        if ticker not in os.listdir(os.path.join(self.data_dir, "weather")):
            os.makedirs(os.path.join(self.data_dir, "weather", ticker))
        lat, lon = (
            self.metadata["cities"][city]["lat"],
            self.metadata["cities"][city]["lon"],
        )
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(self.daily_features),
            "hourly": ",".join(self.hourly_features),
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        response = requests.get(self.historical_url, params=params).json()
        daily_df = pd.DataFrame(response["daily"])
        daily_df = daily_df.fillna(0)
        hourly_df = pd.DataFrame(response["hourly"])
        hourly_df = hourly_df.fillna(0)
        return daily_df, hourly_df

    def update_historical_forecast_data(
        self, max_days: int = 250, verbose: bool = False
    ):
        for ticker in self.metadata["city_map"]:
            city = self.metadata["city_map"][ticker]
            if ticker not in os.listdir(os.path.join(self.data_dir, "weather")):
                os.makedirs(os.path.join(self.data_dir, "weather", ticker))
            if os.path.exists(
                os.path.join(
                    self.data_dir, "weather", ticker, f"daily_weather_forecast.csv"
                )
            ):
                hourly_df = pd.read_csv(
                    os.path.join(
                        self.data_dir, "weather", ticker, f"hourly_weather_forecast.csv"
                    )
                )
                daily_df = pd.read_csv(
                    os.path.join(
                        self.data_dir, "weather", ticker, f"daily_weather_forecast.csv"
                    )
                )
            else:
                daily_df = pd.DataFrame()
                hourly_df = pd.DataFrame()
            if not daily_df.empty:
                latest_date = datetime.strptime(
                    daily_df.iloc[-1]["time"], "%Y-%m-%d"
                ) + timedelta(days=1)
            else:
                latest_date = datetime.strptime(
                    datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d"
                ) - timedelta(days=max_days)
            curr_date = datetime.strptime(
                datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d"
            )
            if latest_date >= curr_date:
                continue
            latest_daily_df, latest_hourly_df = self.get_historical_forecast_data(
                ticker,
                city,
                latest_date.strftime("%Y-%m-%d"),
                curr_date.strftime("%Y-%m-%d"),
            )
            daily_df = pd.concat([daily_df, latest_daily_df])
            hourly_df = pd.concat([hourly_df, latest_hourly_df])
            daily_df.reset_index(drop=True)
            hourly_df.reset_index(drop=True)
            daily_df.to_csv(
                os.path.join(
                    self.data_dir, "weather", ticker, f"daily_weather_forecast.csv"
                ),
                index=False,
            )
            hourly_df.to_csv(
                os.path.join(
                    self.data_dir, "weather", ticker, f"hourly_weather_forecast.csv"
                ),
                index=False,
            )
            if verbose:
                print(f"Updated historical weather forecast data for {city}")

    def get_current_forecast(self, ticker: str) -> dict:
        city = self.metadata["city_map"][ticker]
        lat, lon = (
            self.metadata["cities"][city]["lat"],
            self.metadata["cities"][city]["lon"],
        )
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(self.daily_features),
            "hourly": ",".join(self.hourly_features),
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": 1,
        }
        response = requests.get(self.forecast_url, params=params).json()
        day_forecast = {}
        for key in response["daily"]:
            day_forecast[key] = response["daily"][key][0]
        hourly_forecast = {}
        for i, hour in enumerate(response["hourly"]['time']):
            hourly_forecast[hour] = {}
            for key in response["hourly"]:
                if key == "time":
                    continue
                hourly_forecast[hour][key] = response["hourly"][key][i]
        return day_forecast, hourly_forecast


class TomorrowAPI(WeatherAPI):

    def __init__(self):
        tomorrow_api_key = os.getenv("TOMORROW_API_KEY")
        super().__init__(tomorrow_api_key)
        self.base_url = "https://api.tomorrow.io/v4/weather/forecast"

    def get_forecast(self, city: str) -> dict:
        pass


if __name__ == "__main__":
    openmeteo = OpenMeteoAPI(data_dir="../data")
    # openmeteo.update_historical_forecast_data(max_days=255, verbose=True)
    current_forecast = openmeteo.get_current_forecast("kxhighny")
    pprint(current_forecast)

