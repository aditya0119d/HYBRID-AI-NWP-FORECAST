"""
WeatherFusion V2 — Historical forecast alignment.

Aligns historical GFS/ECMWF forecasts with ERA5 reference
observations and calculates forecast residuals.

Residual:
    observed_value - forecast_value
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hybrid_forecast.ingestion.cities import CITIES
from hybrid_forecast.ingestion.historical_data import (
    fetch_previous_runs,
    fetch_reference_weather,
)


WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]


MODEL_NAMES = {
    "GFS": "gfs_seamless",
    "ECMWF": "ecmwf_ifs025",
}


def reference_to_long(reference: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 reference data from wide to long format."""

    required = {
        "city_name",
        "lat",
        "lon",
        "valid_time",
        *WEATHER_VARIABLES,
    }

    missing = required - set(reference.columns)

    if missing:
        raise ValueError(
            f"Reference data missing columns: {sorted(missing)}"
        )

    long_df = reference.melt(
        id_vars=[
            "city_name",
            "lat",
            "lon",
            "valid_time",
        ],
        value_vars=WEATHER_VARIABLES,
        var_name="variable",
        value_name="observed_value",
    )

    return long_df


def align_forecasts(
    forecasts: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align historical NWP forecasts with ERA5 observations.

    Matching keys:
        city_name
        valid_time
        variable

    lead_days is preserved as a forecast characteristic.
    """

    required_forecast = {
        "city_name",
        "lat",
        "lon",
        "valid_time",
        "lead_days",
        "variable",
        "forecast_value",
        "model",
    }

    missing = required_forecast - set(forecasts.columns)

    if missing:
        raise ValueError(
            f"Forecast data missing columns: {sorted(missing)}"
        )

    reference_long = reference_to_long(reference)

    aligned = forecasts.merge(
        reference_long[
            [
                "city_name",
                "valid_time",
                "variable",
                "observed_value",
            ]
        ],
        on=[
            "city_name",
            "valid_time",
            "variable",
        ],
        how="inner",
        validate="many_to_one",
    )

    aligned["forecast_value"] = pd.to_numeric(
        aligned["forecast_value"],
        errors="coerce",
    )

    aligned["observed_value"] = pd.to_numeric(
        aligned["observed_value"],
        errors="coerce",
    )

    aligned["residual"] = (
        aligned["observed_value"]
        - aligned["forecast_value"]
    )

    aligned = aligned.sort_values(
        [
            "city_name",
            "model",
            "variable",
            "valid_time",
            "lead_days",
        ]
    ).reset_index(drop=True)

    return aligned


def build_city_training_data(
    city_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Build aligned GFS + ECMWF training data for one city."""

    reference = fetch_reference_weather(
        city_name,
        start_date,
        end_date,
    )

    model_frames = []

    for model_label, model_id in MODEL_NAMES.items():

        forecasts = fetch_previous_runs(
            city_name,
            model_id,
            start_date,
            end_date,
        )

        aligned = align_forecasts(
            forecasts,
            reference,
        )

        model_frames.append(aligned)

    combined = pd.concat(
        model_frames,
        ignore_index=True,
    )

    return combined


def validate_alignment(
    aligned: pd.DataFrame,
) -> None:
    """Run basic integrity checks on aligned training data."""

    required_columns = {
        "city_name",
        "lat",
        "lon",
        "valid_time",
        "lead_days",
        "variable",
        "forecast_value",
        "observed_value",
        "residual",
        "model",
    }

    missing = required_columns - set(aligned.columns)

    if missing:
        raise AssertionError(
            f"Missing required columns: {sorted(missing)}"
        )

    if aligned.empty:
        raise AssertionError("Aligned dataset is empty.")

    if aligned["observed_value"].isna().any():
        raise AssertionError(
            "Aligned dataset contains missing observations."
        )

    if aligned["forecast_value"].isna().any():
        raise AssertionError(
            "Aligned dataset contains missing forecasts."
        )

    calculated_residual = (
        aligned["observed_value"]
        - aligned["forecast_value"]
    )

    if not (
        calculated_residual
        .sub(aligned["residual"])
        .abs()
        .lt(1e-9)
        .all()
    ):
        raise AssertionError(
            "Residual calculation is incorrect."
        )


def save_training_data(
    aligned: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save aligned training data as Parquet."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned.to_parquet(
        path,
        index=False,
    )

    return path


if __name__ == "__main__":

    print("=" * 60)
    print("WeatherFusion V2 — Historical Alignment Test")
    print("=" * 60)

    city = "Indore"
    start_date = "2025-09-01"
    end_date = "2025-09-03"

    print(f"\nCity: {city}")
    print(f"Period: {start_date} → {end_date}")

    print("\nBuilding aligned GFS + ECMWF dataset...")

    aligned = build_city_training_data(
        city,
        start_date,
        end_date,
    )

    print(f"\nAligned rows: {len(aligned)}")

    print("\nModels:")
    print(aligned["model"].value_counts())

    print("\nVariables:")
    print(aligned["variable"].value_counts())

    print("\nLead days:")
    print(aligned["lead_days"].value_counts().sort_index())

    print("\nResidual statistics:")
    print(
        aligned["residual"].describe()
    )

    print("\nSample:")
    print(
        aligned.head(10).to_string(index=False)
    )

    print("\nRunning validation...")

    validate_alignment(aligned)

    output = save_training_data(
        aligned,
        "data/processed/indore_aligned_training.parquet",
    )

    print("\nPASS — Historical alignment validated.")
    print(f"Saved: {output}")