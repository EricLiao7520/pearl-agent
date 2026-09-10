from typing import Dict, Any, Optional
import requests
from geopy.geocoders import Nominatim
from pearl.tools.base import BaseTool, GetWeatherParams
WMO_WEATHER_CODES: Dict[str, str] = {
    "0": "Clear sky",
    "1": "Mainly clear",
    "2": "Partly cloudy",
    "3": "Overcast",
    "45": "Fog",
    "48": "Depositing rime fog",
    "51": "Light drizzle",
    "53": "Moderate drizzle",
    "55": "Dense drizzle",
    "61": "Slight rain",
    "63": "Moderate rain",
    "65": "Heavy rain",
    "71": "Slight snow fall",
    "73": "Moderate snow fall",
    "75": "Heavy snow fall",
    "77": "Snow grains",
    "80": "Slight rain showers",
    "81": "Moderate rain showers",
    "82": "Violent rain showers",
    "85": "Slight snow showers",
    "86": "Heavy snow showers",
    "95": "Thunderstorm",
    "96": "Thunderstorm with slight hail",
    "99": "Thunderstorm with heavy hail"
}
class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Fetches weather data (temperature, humidity, wind speed) for a specific location, YYYY-MM-DD date, and HH hour (24-hour format)."
    args_schema = GetWeatherParams

    def __init__(self, user_agent: str = "omniverify_agent"):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.weather_codes = WMO_WEATHER_CODES
    def run(self, location: str, date: str, hour: str = "12", **kwargs: Any) -> Dict[str, Any]:
            try:
                loc = self.geolocator.geocode(location)
                if not loc:
                    return {"error": f"Could not geocode location: {location}"}
                lat, lon = loc.latitude, loc.longitude
                try:
                    h = int(hour)
                except (ValueError, TypeError):
                    h = 12
                h = max(0, min(23, h))
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date,
                    "end_date": date,
                    "hourly": ["temperature_2m", "wind_speed_10m", "relative_humidity_2m", "weather_code"],
                }

                responses = requests.get(url, params=params)
                data = responses.json()
                if "error" in data or "hourly" not in data:
                    url = "https://api.open-meteo.com/v1/forecast"
                    responses = requests.get(url, params=params)
                    data = responses.json()
                    
                hourly = data.get("hourly", {})
                temps = hourly.get("temperature_2m", [])
                if not temps or len(temps) <= h:
                    return {"error": data.get("reason", "No weather data found for given date/hour")}

                temp = temps[h]
                humidity = hourly.get("relative_humidity_2m", [None])[h]
                wind_speed = hourly.get("wind_speed_10m", [None])[h]
                w_code = hourly.get("weather_code", [None])[h]
                desc = self.weather_codes.get(str(w_code), "Unknown")
                return {
                    "temperature": temp,
                    "weather_description": desc,
                    "humidity": humidity,
                    "wind_speed": wind_speed
                }
            except Exception as e:
                return {"error": str(e)}