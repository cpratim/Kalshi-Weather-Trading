import requests
import os
from dotenv import load_dotenv
import json
import pandas as pd
from warnings import filterwarnings
from tqdm import tqdm
from datetime import timedelta, datetime
from pprint import pprint
from util.load import DataLoader
from sklearn.linear_model import Lasso 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
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


class TomorrowAPI(WeatherAPI):

    def __init__(self, data_dir: str = "../data", realtime: bool = False):
        super().__init__(data_dir=data_dir)
        self.forecast_url = "https://api.tomorrow.io/v4/weather/forecast"
        self.historical_url = (
            "https://historical-forecast-api.open-meteo.com/v1/forecast"
        )

    def get_historical_forecast_data(self, ticker: str, city: str, start_date: str, end_date: str) -> dict:
        if fields is None:
            fields = [
                "temperature",
                "temperatureMax", 
                "temperatureMin",
                "humidity",
                "windSpeed",
                "windDirection", 
                "precipitationIntensity",
                "weatherCode",
                "cloudCover",
                "uvIndex",
                "visibility"
            ]
        
        url = f"{self.base_url}/weather/history/recent"


class NWSAPI(WeatherAPI):

    def __init__(self, data_dir: str = "../data", realtime: bool = False):
        super().__init__(data_dir=data_dir)
        self.forecast_url = "https://api.weather.gov/points/{lat},{lon}"
        self.historical_url = "https://api.weather.gov/points/{lat},{lon}/history"
        self.base = 'https://api.weather.gov'
        self.headers = {'User-Agent': '(kalshi-weather, cpratim18@gmail.com)'}
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def get_stations(self) -> list:
        url = f'{self.base}/zones'
        response = requests.get(url).json()
        with open('zones.json', 'w') as f:
            json.dump(response, f, indent=4)
        return response
    
    def get_station_forecast(self, station: str, type_: str = 'public'):
        url = f'{self.base}/zones/forecast/{station}/observations'
        response = requests.get(url).json()
        with open('forecast.json', 'w') as f:
            json.dump(response, f, indent=4)
        return response
    
    def get_current_forecast(self, ticker: str):
        city = self.metadata["city_map"][ticker]
        lat, lon = (
            self.metadata["cities"][city]["lat"],
            self.metadata["cities"][city]["lon"],
        )
        url = self.forecast_url.format(lat=lat, lon=lon)
        response = requests.get(url, headers=self.headers).json()
        hourly_url = response['properties']['forecastHourly']

        data = requests.get(hourly_url, headers=self.headers).json()
        
        max_temp = 0
        periods = data['properties']['periods']
        for period in periods:
            time = period['startTime']
            date = datetime.fromisoformat(time).strftime('%Y-%m-%d')
            if date != datetime.now().strftime('%Y-%m-%d'):
                continue
            max_temp = max(max_temp, period['temperature'])
        return { 'forecast': max_temp, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S') }


class OpenMeteoAPI(WeatherAPI):

    def __init__(self, data_dir: str = "../data", realtime: bool = False):
        super().__init__(data_dir=data_dir)
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.historical_url = (
            "https://historical-forecast-api.open-meteo.com/v1/forecast"
        )
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.daily_features = [
            "temperature_2m_max",
            'daylight_duration',
            'precipitation_hours',
            'precipitation_probability_max',
            'wind_speed_10m_max',
        ]
        self.hourly_features = []
        self.providers = [
            'bom_access_global',
            'cma_grapes_global',
            'ecmwf_ifs025',
            'best_match',
            'knmi_seamless',
            'dmi_seamless',
            'gfs_seamless',
        ]
        self.loader = DataLoader(data_dir=data_dir)

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
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            'models': ','.join(self.providers),
        }
        response = requests.get(self.historical_url, params=params).json()
        daily_df = pd.DataFrame(response["daily"])
        daily_df = daily_df.fillna(0)
        return daily_df

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
                daily_df = pd.read_csv(
                    os.path.join(
                        self.data_dir, "weather", ticker, f"daily_weather_forecast.csv"
                    )
                )
            else:
                daily_df = pd.DataFrame()
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
            latest_df = self.get_historical_forecast_data(
                ticker,
                city,
                latest_date.strftime("%Y-%m-%d"),
                curr_date.strftime("%Y-%m-%d"),
            )
            daily_df = pd.concat([daily_df, latest_df])
            daily_df['forecast'] = daily_df['temperature_2m_max_best_match']
            daily_df.reset_index(drop=True)
            daily_df.to_csv(
                os.path.join(
                    self.data_dir, "weather", ticker, f"daily_weather_forecast.csv"
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
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": 1,
            'models': ','.join(self.providers),
        }
        response = requests.get(self.forecast_url, params=params).json()
        day_forecast = {}
        for key in response["daily"]:
            day_forecast[key] = response["daily"][key][0]
        day_forecast['forecast'] = day_forecast['temperature_2m_max_best_match']
        return day_forecast


class TomorrowAPI(WeatherAPI):

    def __init__(self):
        tomorrow_api_key = os.getenv("TOMORROW_API_KEY")
        super().__init__(tomorrow_api_key)
        self.base_url = "https://api.tomorrow.io/v4/weather/forecast"

    def get_forecast(self, city: str) -> dict:
        pass


if __name__ == "__main__":
    openmeteo = OpenMeteoAPI(data_dir="../data")
    # current_forecast = openmeteo.get_current_forecast("kxhighny")
    # pprint(current_forecast)
    openmeteo.update_historical_forecast_data(max_days=300, verbose=True)
    # openmeteo.set_model("kxhighny")
    # pprint(openmeteo.get_current_forecast("kxhighny")[0])
    # openmeteo.process_historical_forecast_data("kxhighny")
    # nws = NWSAPI(data_dir="../data")

    # pprint(nws.get_daily_high_forecast('kxhighny'))
