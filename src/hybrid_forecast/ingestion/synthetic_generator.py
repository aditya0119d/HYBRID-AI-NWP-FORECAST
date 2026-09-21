"""Deterministic synthetic NWP data generator for the prototype.

WARNING
-------
THIS IS SYNTHETIC DATA FOR PROTOTYPE PURPOSES.

This module deliberately mimics structured, learnable NWP forecast error.
It must NOT be presented as real GFS/ECMWF/ERA5 output.

The downstream schema is designed to match the future real-ingestion path so
alignment, feature engineering, modelling, evaluation, backend, and frontend
modules do not need to change when real data ingestion is restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_DATA_WARNING = (
    "THIS IS SYNTHETIC DATA FOR PROTOTYPE PURPOSES. "
    "It is not real GFS/ECMWF/ERA5 data."
)

RANDOM_SEED = 20260920
FORECAST_HOURS = np.arange(0, 121, dtype=np.int16)
FORECAST_INIT_FREQUENCY_HOURS = 6


@dataclass(frozen=True)
class CityClimate:
    """Climate parameters used to construct a city's synthetic weather."""

    city_name: str
    lat: float
    lon: float
    base_temperature: float
    seasonal_amplitude: float
    diurnal_amplitude: float
    seasonal_peak_day: float
    rainy_months: tuple[int, ...]
    rain_factor: float
    region: str


CITIES: tuple[CityClimate, ...] = (
    CityClimate("Mumbai", 19.0760, 72.8777, 27.0, 3.2, 3.0, 120, (6, 7, 8, 9), 0.85, "coastal"),
    CityClimate("Chennai", 13.0827, 80.2707, 29.0, 2.4, 3.2, 135, (10, 11, 12), 0.75, "coastal"),
    CityClimate("Kolkata", 22.5726, 88.3639, 27.0, 5.0, 4.0, 125, (6, 7, 8, 9), 0.90, "monsoon"),
    CityClimate("Delhi", 28.6139, 77.2090, 25.0, 11.0, 7.0, 155, (7, 8, 9), 0.55, "inland"),
    CityClimate("Jaipur", 26.9124, 75.7873, 25.5, 12.0, 7.5, 155, (7, 8, 9), 0.40, "arid"),
    CityClimate("Bengaluru", 12.9716, 77.5946, 23.0, 3.8, 5.0, 120, (6, 7, 8, 9, 10), 0.65, "inland"),
    CityClimate("Guwahati", 26.1445, 91.7362, 24.5, 5.0, 4.5, 125, (5, 6, 7, 8, 9), 0.95, "monsoon"),
    CityClimate("Shimla", 31.1048, 77.1734, 12.5, 8.0, 5.0, 175, (6, 7, 8, 9), 0.65, "himalayan"),
    CityClimate("Hyderabad", 17.3850, 78.4867, 26.0, 8.0, 6.0, 140, (6, 7, 8, 9), 0.50, "inland"),
)

FORECAST_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_u_10m",
    "wind_v_10m",
    "cloud_cover",
    "dew_point_2m",
)


def _is_rainy_season(months: np.ndarray, rainy_months: tuple[int, ...]) -> np.ndarray:
    """Return a boolean mask for a city's rainy season."""
    return np.isin(months, rainy_months)


def _dew_point_celsius(
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
) -> np.ndarray:
    """Estimate dew point using the Magnus approximation."""
    safe_rh = np.clip(relative_humidity, 1.0, 100.0)
    gamma = np.log(safe_rh / 100.0) + (
        17.625 * temperature / (243.04 + temperature)
    )
    return 243.04 * gamma / (17.625 - gamma)


def _base_observations(
    timestamps: pd.DatetimeIndex,
    city: CityClimate,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a deterministic hourly observed weather series for one city."""
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)
    hour = timestamps.hour.to_numpy(dtype=float)
    months = timestamps.month.to_numpy(dtype=int)

    seasonal = city.seasonal_amplitude * np.sin(
        2.0 * np.pi * (day_of_year - city.seasonal_peak_day) / 365.25
    )
    diurnal = city.diurnal_amplitude * np.sin(
        2.0 * np.pi * (hour - 8.0) / 24.0
    )

    temperature = city.base_temperature + seasonal + diurnal
    temperature += rng.normal(0.0, 0.35, len(timestamps))

    rainy = _is_rainy_season(months, city.rainy_months)

    # Persistent wet-state process: rainfall appears in clusters rather than
    # as independent uniform random noise.
    rain = np.zeros(len(timestamps), dtype=float)
    wet_previous = False

    for index in range(len(timestamps)):
        probability = 0.018 + (
            0.075 * city.rain_factor if rainy[index] else 0.008
        )
        if wet_previous:
            probability += 0.20

        wet_now = rng.random() < min(probability, 0.50)
        if wet_now:
            rain[index] = rng.gamma(
                shape=1.8,
                scale=2.2 + 5.5 * city.rain_factor,
            )
            wet_previous = True
        else:
            wet_previous = False

    humidity = 73.0 - 1.25 * (temperature - city.base_temperature)
    humidity += 12.0 * rainy.astype(float)
    humidity += rng.normal(0.0, 3.0, len(timestamps))
    humidity = np.clip(humidity, 25.0, 100.0)

    pressure = 1012.0 - 0.55 * (temperature - 25.0)
    pressure += 2.5 * np.cos(2.0 * np.pi * day_of_year / 365.25)
    pressure += rng.normal(0.0, 1.2, len(timestamps))

    cloud = 20.0 + 0.95 * humidity
    cloud += np.where(rain > 0, 8.0, 0.0)
    cloud += rng.normal(0.0, 5.0, len(timestamps))
    cloud = np.clip(cloud, 0.0, 100.0)

    wind_speed = 2.5 + 0.018 * np.maximum(temperature - 18.0, 0.0)
    wind_speed += 1.8 * rainy.astype(float)
    wind_speed += rng.normal(0.0, 0.7, len(timestamps))
    wind_speed = np.clip(wind_speed, 0.2, 18.0)

    wind_direction = (
        180.0
        + 50.0 * np.sin(2.0 * np.pi * day_of_year / 365.25)
        + rng.normal(0.0, 20.0, len(timestamps))
    )
    direction_rad = np.deg2rad(wind_direction)
    wind_u = wind_speed * np.cos(direction_rad)
    wind_v = wind_speed * np.sin(direction_rad)
    dew_point = _dew_point_celsius(temperature, humidity)

    return pd.DataFrame(
        {
            "valid_time": timestamps,
            "temperature_2m": temperature.astype(np.float32),
            "precipitation": rain.astype(np.float32),
            "relative_humidity_2m": humidity.astype(np.float32),
            "surface_pressure": pressure.astype(np.float32),
            "wind_speed_10m": wind_speed.astype(np.float32),
            "wind_u_10m": wind_u.astype(np.float32),
            "wind_v_10m": wind_v.astype(np.float32),
            "cloud_cover": cloud.astype(np.float32),
            "dew_point_2m": dew_point.astype(np.float32),
        }
    )


def _forecast_bias(
    valid_times: pd.DatetimeIndex,
    city: CityClimate,
    lead_hours: np.ndarray,
    model: str,
) -> dict[str, np.ndarray]:
    """Return structured, model-specific forecast biases."""
    months = valid_times.month.to_numpy(dtype=int)
    rainy = _is_rainy_season(months, city.rainy_months)
    lead = lead_hours.astype(float)

    if model == "gfs":
        temperature_bias = (
            0.30
            + 0.014 * lead
            + 0.75 * rainy.astype(float)
            + (0.45 if city.region == "coastal" else 0.0)
            + (
                0.35
                if city.region == "arid"
                else 0.0
            )
            * np.isin(months, (4, 5, 6)).astype(float)
        )
        precipitation_multiplier = (
            1.0 + 0.22 * rainy.astype(float) + 0.002 * lead
        )
        humidity_bias = -2.0 - 0.025 * lead
        pressure_bias = 0.8 + 0.006 * lead
        wind_bias = 0.12 + 0.004 * lead

    elif model == "ecmwf":
        temperature_bias = (
            -0.12
            + 0.006 * lead
            + 0.30 * rainy.astype(float)
            + (
                0.35
                if city.region == "inland"
                else 0.0
            )
            * np.isin(months, (4, 5, 6)).astype(float)
        )
        precipitation_multiplier = (
            1.0 - 0.20 * rainy.astype(float) - 0.001 * lead
        )
        humidity_bias = 1.0 + 0.012 * lead
        pressure_bias = -0.35 + 0.003 * lead
        wind_bias = 0.08 + 0.002 * lead

    else:
        raise ValueError(f"Unsupported synthetic model: {model}")

    return {
        "temperature_bias": temperature_bias,
        "precipitation_multiplier": precipitation_multiplier,
        "humidity_bias": humidity_bias,
        "pressure_bias": pressure_bias,
        "wind_bias": wind_bias,
    }


def _make_forecast_frame(
    observations: pd.DataFrame,
    city: CityClimate,
    model: str,
    init_times: pd.DatetimeIndex,
    rng_seed: int,
) -> pd.DataFrame:
    """Build all forecast rows for one city and one model."""
    init_naive = init_times.tz_convert("UTC").tz_localize(None)
    lead_delta = FORECAST_HOURS.astype("timedelta64[h]")
    valid_matrix = (
        init_naive.to_numpy(dtype="datetime64[ns]")[:, None]
        + lead_delta[None, :]
    )
    valid_flat = pd.DatetimeIndex(valid_matrix.ravel()).tz_localize("UTC")
    init_flat = pd.DatetimeIndex(
        np.repeat(init_naive.to_numpy(dtype="datetime64[ns]"), len(FORECAST_HOURS))
    ).tz_localize("UTC")
    lead_flat = np.tile(FORECAST_HOURS, len(init_times)).astype(np.int16)

    # Restrict rows whose valid time falls outside the observation period.
    valid_mask = valid_flat <= observations["valid_time"].max()
    valid_flat = valid_flat[valid_mask]
    init_flat = init_flat[valid_mask]
    lead_flat = lead_flat[valid_mask]

    observed = observations.set_index("valid_time").reindex(valid_flat)
    if observed.isna().any().any():
        raise RuntimeError("Observation lookup failed for synthetic forecast rows.")

    bias = _forecast_bias(valid_flat, city, lead_flat, model)
    rng = np.random.default_rng(rng_seed)

    temperature = (
        observed["temperature_2m"].to_numpy()
        + bias["temperature_bias"]
        + rng.normal(0.0, 0.28, len(valid_flat))
    )

    precipitation = np.maximum(
        0.0,
        observed["precipitation"].to_numpy()
        * bias["precipitation_multiplier"]
        + rng.normal(0.0, 0.18, len(valid_flat)),
    )

    humidity = np.clip(
        observed["relative_humidity_2m"].to_numpy()
        + bias["humidity_bias"]
        + rng.normal(0.0, 1.4, len(valid_flat)),
        5.0,
        100.0,
    )

    pressure = (
        observed["surface_pressure"].to_numpy()
        + bias["pressure_bias"]
        + rng.normal(0.0, 0.7, len(valid_flat))
    )

    wind_speed = np.maximum(
        0.1,
        observed["wind_speed_10m"].to_numpy()
        + bias["wind_bias"]
        + rng.normal(0.0, 0.25, len(valid_flat)),
    )

    wind_u = observed["wind_u_10m"].to_numpy() + rng.normal(
        0.0, 0.15, len(valid_flat)
    )
    wind_v = observed["wind_v_10m"].to_numpy() + rng.normal(
        0.0, 0.15, len(valid_flat)
    )

    cloud_offset = 2.0 if model == "gfs" else -1.0
    cloud = np.clip(
        observed["cloud_cover"].to_numpy()
        + cloud_offset
        + rng.normal(0.0, 2.0, len(valid_flat)),
        0.0,
        100.0,
    )

    dew_point = _dew_point_celsius(temperature, humidity)

    frame = pd.DataFrame(
        {
            "model": model,
            "init_time": init_flat,
            "valid_time": valid_flat,
            "forecast_hour": lead_flat,
            "lat": city.lat,
            "lon": city.lon,
            "city_name": city.city_name,
            "temperature_2m": temperature.astype(np.float32),
            "precipitation": precipitation.astype(np.float32),
            "relative_humidity_2m": humidity.astype(np.float32),
            "surface_pressure": pressure.astype(np.float32),
            "wind_speed_10m": wind_speed.astype(np.float32),
            "wind_u_10m": wind_u.astype(np.float32),
            "wind_v_10m": wind_v.astype(np.float32),
            "cloud_cover": cloud.astype(np.float32),
            "dew_point_2m": dew_point.astype(np.float32),
        }
    )

    for variable in FORECAST_VARIABLES:
        frame[f"observed_{variable}"] = observed[variable].to_numpy(
            dtype=np.float32
        )

    return frame


def generate_synthetic_dataset(
    output_dir: Path,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31 23:00",
) -> dict[str, object]:
    """Generate deterministic synthetic observations and forecasts.

    Observations are hourly. Forecast initialisation occurs every six hours,
    with lead times 0-120 hours inclusive.

    Args:
        output_dir: Destination directory.
        start_date: Inclusive UTC start timestamp.
        end_date: Inclusive UTC end timestamp.

    Returns:
        Generation manifest containing paths and row counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range(start_date, end_date, freq="h", tz="UTC")
    init_times = pd.date_range(start_date, end_date, freq="6h", tz="UTC")

    observation_frames: list[pd.DataFrame] = []
    forecast_paths: list[str] = []
    forecast_rows = 0

    for city_index, city in enumerate(CITIES):
        city_rng = np.random.default_rng(RANDOM_SEED + city_index)
        observations = _base_observations(timestamps, city, city_rng)

        observations.insert(0, "city_name", city.city_name)
        observations.insert(1, "lat", city.lat)
        observations.insert(2, "lon", city.lon)
        observation_frames.append(observations)

        for model_index, model in enumerate(("gfs", "ecmwf")):
            frame = _make_forecast_frame(
                observations,
                city,
                model,
                init_times,
                RANDOM_SEED + city_index * 10_000 + model_index * 1_000_000,
            )

            filename = f"{model}_{city.city_name.lower()}.parquet"
            path = output_dir / filename
            frame.to_parquet(path, index=False)
            forecast_paths.append(str(path))
            forecast_rows += len(frame)

    observations_df = pd.concat(observation_frames, ignore_index=True)
    observations_path = output_dir / "observations_hourly.parquet"
    observations_df.to_parquet(observations_path, index=False)

    manifest = {
        "warning": SYNTHETIC_DATA_WARNING,
        "seed": RANDOM_SEED,
        "cities": [city.city_name for city in CITIES],
        "observation_rows": len(observations_df),
        "forecast_rows": forecast_rows,
        "forecast_files": forecast_paths,
        "observations_file": str(observations_path),
        "forecast_initialisation_frequency": "6 hours",
        "forecast_lead_hours": "0-120 inclusive",
        "observation_start": str(observations_df["valid_time"].min()),
        "observation_end": str(observations_df["valid_time"].max()),
    }

    (output_dir / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = generate_synthetic_dataset(Path("data/raw/synthetic"))
    print(SYNTHETIC_DATA_WARNING)
    print(__import__("json").dumps(result, indent=2))
#if __name__ == "__main__":
    #main()