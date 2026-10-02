"""
Open-Meteo NWP data ingestion for WeatherFusion V2.

Fetches:
    - GFS forecast data
    - ECMWF forecast data

and converts both into the common WeatherFusion schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from hybrid_forecast.ingestion.cities import CITIES


# =========================================================
# CONFIGURATION
# =========================================================

GFS_URL = "https://api.open-meteo.com/v1/gfs"
ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

REQUEST_TIMEOUT = 30

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]


# =========================================================
# API FETCH
# =========================================================

def fetch_open_meteo(
    url: str,
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from an Open-Meteo endpoint.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(
            "Open-Meteo response does not contain hourly data."
        )

    hourly = payload["hourly"]

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:
        raise ValueError(
            "Open-Meteo response does not contain time values."
        )

    df["valid_time"] = pd.to_datetime(
        df.pop("time"),
        utc=True,
    )

    return df


# =========================================================
# STANDARDIZATION
# =========================================================

def standardize_nwp_data(
    df: pd.DataFrame,
    city_name: str,
    latitude: float,
    longitude: float,
    model_name: str,
) -> pd.DataFrame:
    """
    Convert Open-Meteo data into the WeatherFusion
    internal schema.
    """

    result = df.copy()

    result["city_name"] = city_name
    result["lat"] = latitude
    result["lon"] = longitude
    result["model"] = model_name

    result = result[
        [
            "city_name",
            "lat",
            "lon",
            "valid_time",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "model",
        ]
    ]

    return result


# =========================================================
# FETCH GFS
# =========================================================

def fetch_gfs(
    city_name: str,
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch GFS forecast for one location.
    """

    raw = fetch_open_meteo(
        GFS_URL,
        latitude,
        longitude,
        forecast_days,
    )

    return standardize_nwp_data(
        raw,
        city_name,
        latitude,
        longitude,
        "GFS",
    )


# =========================================================
# FETCH ECMWF
# =========================================================

def fetch_ecmwf(
    city_name: str,
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch ECMWF forecast for one location.
    """

    raw = fetch_open_meteo(
        ECMWF_URL,
        latitude,
        longitude,
        forecast_days,
    )

    return standardize_nwp_data(
        raw,
        city_name,
        latitude,
        longitude,
        "ECMWF",
    )


# =========================================================
# FETCH BOTH MODELS
# =========================================================

def fetch_city_forecasts(
    city_name: str,
    forecast_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch GFS and ECMWF forecasts for one city.
    """

    city_key = city_name.strip().lower()
    if city_key not in CITIES:
        raise ValueError(
            f"Unknown city: {city_name}. "
            f"Available cities: {[city['name'] for city in CITIES.values()]}"
        )

    location = CITIES[city_key]
    city_name = location["name"]

    gfs = fetch_gfs(
        city_name,
        location["latitude"],
        location["longitude"],
        forecast_days,
    )

    ecmwf = fetch_ecmwf(
        city_name,
        location["latitude"],
        location["longitude"],
        forecast_days,
    )

    return gfs, ecmwf


# =========================================================
# SAVE DATA
# =========================================================

def save_city_forecasts(
    city_name: str,
    output_dir: str | Path = "data/raw/open_meteo",
    forecast_days: int = 3,
) -> tuple[Path, Path]:
    """
    Fetch and save GFS + ECMWF data as Parquet.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    gfs, ecmwf = fetch_city_forecasts(
        city_name,
        forecast_days,
    )

    city_slug = city_name.lower().replace(
        " ",
        "_",
    )

    gfs_path = (
        output_path
        / f"gfs_{city_slug}.parquet"
    )

    ecmwf_path = (
        output_path
        / f"ecmwf_{city_slug}.parquet"
    )

    gfs.to_parquet(
        gfs_path,
        index=False,
    )

    ecmwf.to_parquet(
        ecmwf_path,
        index=False,
    )

    return gfs_path, ecmwf_path


# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WeatherFusion V2 — Open-Meteo ingestion test")
    print("=" * 60)

    city = "Delhi"

    print(f"\nFetching GFS + ECMWF for {city}...")

    gfs_df, ecmwf_df = fetch_city_forecasts(
        city,
        forecast_days=3,
    )

    print(f"\nGFS rows:   {len(gfs_df)}")
    print(f"ECMWF rows: {len(ecmwf_df)}")

    print("\nGFS columns:")
    print(gfs_df.columns.tolist())

    print("\nECMWF columns:")
    print(ecmwf_df.columns.tolist())

    print("\nGFS preview:")
    print(gfs_df.head())

    print("\nECMWF preview:")
    print(ecmwf_df.head())

    print("\nOpen-Meteo ingestion test PASSED.")