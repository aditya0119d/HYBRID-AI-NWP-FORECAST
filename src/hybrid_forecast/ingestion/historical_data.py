"""
WeatherFusion V2 — Historical training data ingestion.

Purpose:
    Fetch archived forecast data and reference weather data
    for training the GFS/ECMWF bias-correction models.
"""

from __future__ import annotations

import pandas as pd
import requests

from hybrid_forecast.ingestion.cities import CITIES


# =========================================================
# CONFIGURATION
# =========================================================

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUEST_TIMEOUT = 60

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]


# =========================================================
# GENERIC REQUEST
# =========================================================

def _get_json(url: str, params: dict) -> dict:
    """Send an Open-Meteo request and return JSON."""

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(
            f"Open-Meteo response missing 'hourly': {payload}"
        )

    return payload


# =========================================================
# REFERENCE WEATHER — ERA5
# =========================================================

def fetch_reference_weather(
    city_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical reference weather from Open-Meteo ERA5.
    """

    city_key = city_name.strip().lower()

    if city_key not in CITIES:
        raise ValueError(f"Unknown city: {city_name}")

    location = CITIES[city_key]

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }

    payload = _get_json(
        HISTORICAL_WEATHER_URL,
        params,
    )

    df = pd.DataFrame(payload["hourly"])

    df["valid_time"] = pd.to_datetime(
        df.pop("time"),
        utc=True,
    )

    df["city_name"] = location["name"]
    df["lat"] = location["latitude"]
    df["lon"] = location["longitude"]

    return df[
        [
            "city_name",
            "lat",
            "lon",
            "valid_time",
            *WEATHER_VARIABLES,
        ]
    ]


# =========================================================
# HISTORICAL MODEL RUNS
# =========================================================

def fetch_previous_runs(
    city_name: str,
    model: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch archived model forecasts at fixed lead times.

    model examples:
        gfs_seamless
        ecmwf_ifs025
    """

    city_key = city_name.strip().lower()

    if city_key not in CITIES:
        raise ValueError(f"Unknown city: {city_name}")

    location = CITIES[city_key]

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            f"{variable}_previous_day{lead}"
            for variable in WEATHER_VARIABLES
            for lead in range(1, 4)
        ),
        "models": model,
        "timezone": "UTC",
    }

    payload = _get_json(
        PREVIOUS_RUNS_URL,
        params,
    )

    hourly = payload["hourly"]

    rows = []

    timestamps = pd.to_datetime(
        hourly["time"],
        utc=True,
    )

    for lead in range(1, 4):
        for variable in WEATHER_VARIABLES:

            column = f"{variable}_previous_day{lead}"

            if column not in hourly:
                continue

            values = hourly[column]

            for timestamp, value in zip(
                timestamps,
                values,
            ):
                rows.append(
                    {
                        "city_name": location["name"],
                        "lat": location["latitude"],
                        "lon": location["longitude"],
                        "valid_time": timestamp,
                        "lead_days": lead,
                        "variable": variable,
                        "forecast_value": value,
                        "model": model,
                    }
                )

    return pd.DataFrame(rows)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WeatherFusion V2 — Historical Data Test")
    print("=" * 60)

    city = "Indore"

    start_date = "2025-09-01"
    end_date = "2025-09-03"

    print(f"\nCity: {city}")
    print(f"Period: {start_date} → {end_date}")

    print("\nFetching ERA5 reference data...")

    reference = fetch_reference_weather(
        city,
        start_date,
        end_date,
    )

    print(f"Reference rows: {len(reference)}")
    print(reference.head())

    print("\nFetching GFS previous runs...")

    gfs = fetch_previous_runs(
        city,
        "gfs_seamless",
        start_date,
        end_date,
    )

    print(f"GFS rows: {len(gfs)}")
    print(gfs.head())

    print("\nFetching ECMWF previous runs...")

    ecmwf = fetch_previous_runs(
        city,
        "ecmwf_ifs025",
        start_date,
        end_date,
    )

    print(f"ECMWF rows: {len(ecmwf)}")
    print(ecmwf.head())

    print("\nHistorical ingestion test PASSED.")