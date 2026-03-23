# src/ingestion.py
# Fetches EIA demand + Open-Meteo weather

import requests
import pandas as pd
import numpy as np
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEZONE = "America/Los_Angeles"

CITIES = {
    "Los_Angeles"   : (34.0522,  -118.2437),
    "San_Francisco" : (37.7749,  -122.4194),
    "San_Diego"     : (32.7157,  -117.1611),
    "Sacramento"    : (38.5816,  -121.4944),
    "San_Jose"      : (37.3382,  -121.8863),
    "Fresno"        : (36.7378,  -119.7871),
    "Oakland"       : (37.8044,  -122.2711),
    "Bakersfield"   : (35.3733,  -119.0187),
    "Riverside"     : (33.9806,  -117.3755),
}

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
]


def make_session():
    session = requests.Session()
    retry = Retry(
    total            = 7,
    backoff_factor   = 5,
    status_forcelist = [429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = make_session()


def fetch_eia_day(api_key, date_str):
    """
    Fetch one day of EIA hourly demand.
    Returns DataFrame with columns: timestamp, demand_mwh
    Returns None on failure.
    """
    url      = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_rows = []
    offset   = 0

    while True:
        params = {
            "api_key"              : api_key,
            "frequency"            : "hourly",
            "data[0]"              : "value",
            "facets[respondent][]" : "CISO",
            "facets[type][]"       : "D",
            "start"                : date_str,
            "end"                  : date_str,
            "sort[0][column]"      : "period",
            "sort[0][direction]"   : "asc",
            "offset"               : offset,
            "length"               : 5000,
        }
        try:
            resp = SESSION.get(url, params=params, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            print(f"  EIA fetch error: {e}")
            return None

        payload = resp.json()
        rows    = payload["response"].get("data", [])
        total   = int(payload["response"].get("total", 0))

        if not rows:
            break
        all_rows.extend(rows)
        if len(all_rows) >= total:
            break
        offset += 5000
        time.sleep(1)

    if not all_rows:
        print(f"  EIA: no data returned for {date_str}")
        return None

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"period": "timestamp",
                             "value" : "demand_mwh"})
    df["timestamp"]  = pd.to_datetime(
        df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    df["demand_mwh"] = pd.to_numeric(
        df["demand_mwh"], errors="coerce")
    df = (df[["timestamp", "demand_mwh"]]
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True))

    print(f"  EIA: fetched {len(df)} rows for {date_str}")
    return df


def _fetch_weather_from_url(url, params):
    """Internal helper — fetch weather from one URL."""
    city_dfs = []

    for city, (lat, lon) in CITIES.items():
        p = dict(params)
        p["latitude"]  = lat
        p["longitude"] = lon

        try:
            resp = SESSION.get(url, params=p, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Weather error {city}: {e}")
            continue

        if "hourly" not in data:
            continue

        df = pd.DataFrame(data["hourly"])
        df["timestamp"] = pd.to_datetime(
            df["time"]
        ).dt.tz_localize(
            TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        df = df.drop(columns=["time"]).rename(columns={
            "temperature_2m"       : "temperature_c",
            "relative_humidity_2m" : "humidity_pct",
            "wind_speed_10m"       : "wind_speed_kmh",
            "shortwave_radiation"  : "solar_radiation_wm2",
            "precipitation"        : "precipitation_mm",
        })
        city_dfs.append(df)
        time.sleep(0.3)

    if not city_dfs:
        return None

    weather = (
        pd.concat(city_dfs, ignore_index=True)
          .groupby("timestamp")[
              ["temperature_c", "humidity_pct",
               "wind_speed_kmh", "solar_radiation_wm2",
               "precipitation_mm"]]
          .mean()
          .reset_index()
          .dropna(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    return weather


def fetch_weather_archive(date_str):
    """
    Fetch historical weather from Open-Meteo archive.
    Used for past dates during retraining.
    """
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "start_date"     : date_str,
        "end_date"       : date_str,
        "hourly"         : WEATHER_VARIABLES,
        "timezone"       : TIMEZONE,
        "wind_speed_unit": "kmh",
    }
    result = _fetch_weather_from_url(url, params)
    if result is not None:
        print(f"  Weather archive: {len(result)} rows "
              f"for {date_str}")
    return result


def fetch_weather_forecast():
    """
    Fetch next 2 days weather forecast from Open-Meteo.
    Used for inference — predicting today or tomorrow.
    """
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "forecast_days"  : 3,
        "hourly"         : WEATHER_VARIABLES,
        "timezone"       : TIMEZONE,
        "wind_speed_unit": "kmh",
    }
    result = _fetch_weather_from_url(url, params)
    if result is not None:
        print(f"  Weather forecast: {len(result)} rows")
    return result
