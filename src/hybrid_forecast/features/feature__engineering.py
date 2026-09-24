"""
Feature engineering for the Hybrid AI-NWP Multi-Model
Forecast Blending System.

Input:
    data/processed/aligned_forecasts.parquet

Output:
    data/processed/ml_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aligned_forecasts.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.parquet"
)


# =========================================================
# Weather variables
# =========================================================

WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_u_10m",
    "wind_v_10m",
    "cloud_cover",
    "dew_point_2m",
]


# =========================================================
# Load aligned data
# =========================================================

def load_data() -> pd.DataFrame:
    """Load the aligned GFS, ECMWF and observation dataset."""

    print("Loading aligned dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# =========================================================
# Time features
# =========================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical time features."""

    df = df.copy()

    df["valid_time"] = pd.to_datetime(df["valid_time"])

    # Hour of day
    hour = df["valid_time"].dt.hour

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    # Day of year
    day_of_year = df["valid_time"].dt.dayofyear

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # Month
    df["month"] = df["valid_time"].dt.month

    return df


# =========================================================
# Forecast error features
# =========================================================

def create_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical forecast errors.

    Error:
        observation - forecast
    """

    df = df.copy()

    for variable in WEATHER_VARIABLES:

        gfs_column = f"gfs_{variable}"
        ecmwf_column = f"ecmwf_{variable}"
        observed_column = f"observed_{variable}"

        if (
            gfs_column in df.columns
            and observed_column in df.columns
        ):
            df[f"gfs_error_{variable}"] = (
                df[observed_column]
                - df[gfs_column]
            )

        if (
            ecmwf_column in df.columns
            and observed_column in df.columns
        ):
            df[f"ecmwf_error_{variable}"] = (
                df[observed_column]
                - df[ecmwf_column]
            )

    return df


# =========================================================
# Forecast difference features
# =========================================================

def create_model_difference_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the disagreement between GFS and ECMWF.

    Large disagreement can indicate uncertainty.
    """

    df = df.copy()

    for variable in WEATHER_VARIABLES:

        gfs_column = f"gfs_{variable}"
        ecmwf_column = f"ecmwf_{variable}"

        if (
            gfs_column in df.columns
            and ecmwf_column in df.columns
        ):
            df[f"gfs_ecmwf_diff_{variable}"] = (
                df[gfs_column]
                - df[ecmwf_column]
            )

    return df


# =========================================================
# Rolling error features
# =========================================================

def create_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create rolling forecast-error statistics.

    Rolling windows are calculated separately for each city.
    """

    df = df.copy()

    df = df.sort_values(
        ["city_name", "valid_time"]
    )

    # Temperature is our primary demonstration variable.
    # Additional weather variables can be added later.
    for model in ["gfs", "ecmwf"]:

        error_column = (
            f"{model}_error_temperature_2m"
        )

        if error_column in df.columns:

            grouped = df.groupby(
                "city_name"
            )[error_column]

            df[
                f"{model}_rolling_error_7"
            ] = grouped.transform(
                lambda x: x.rolling(
                    window=7,
                    min_periods=1,
                ).mean()
            )

            df[
                f"{model}_rolling_abs_error_7"
            ] = grouped.transform(
                lambda x: x.abs().rolling(
                    window=7,
                    min_periods=1,
                ).mean()
            )

    return df


# =========================================================
# Lead-time features
# =========================================================

def create_lead_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create lead-time features.

    forecast_hour comes from the NWP datasets.
    """

    df = df.copy()

    if "forecast_hour" in df.columns:
        df["lead_time_hours"] = df[
            "forecast_hour"
        ]

    return df


# =========================================================
# Final cleanup
# =========================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and prepare the ML feature dataset."""

    df = df.copy()

    # Remove duplicate records.
    df = df.drop_duplicates(
        subset=[
            "city_name",
            "lat",
            "lon",
            "valid_time",
        ]
    )

    # Replace infinite values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Sort chronologically.
    df = df.sort_values(
        ["city_name", "valid_time"]
    )

    # Remove rows that don't have the target
    # observation for temperature.
    target = "observed_temperature_2m"

    if target in df.columns:
        before = len(df)

        df = df.dropna(
            subset=[target]
        )

        removed = before - len(df)

        print(
            f"Rows removed due to missing target: "
            f"{removed:,}"
        )

    return df.reset_index(drop=True)


# =========================================================
# Main feature-engineering pipeline
# =========================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nCreating time features...")
    df = create_time_features(df)

    print("Creating forecast error features...")
    df = create_error_features(df)

    print("Creating GFS-ECMWF difference features...")
    df = create_model_difference_features(df)

    print("Creating rolling error features...")
    df = create_rolling_features(df)

    print("Creating lead-time features...")
    df = create_lead_time_features(df)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    return df


# =========================================================
# Validation
# =========================================================

def validate_features(
    df: pd.DataFrame,
) -> None:

    print("\n========== FEATURE VALIDATION ==========")

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nNew feature columns:")

    original_prefixes = (
        "gfs_",
        "ecmwf_",
        "observed_",
    )

    new_features = [
        column
        for column in df.columns
        if not column.startswith(
            original_prefixes
        )
        and column
        not in [
            "city_name",
            "lat",
            "lon",
            "valid_time",
            "model",
            "init_time",
            "forecast_hour",
        ]
    ]

    for column in new_features:
        print(f"  {column}")

    print("\nMissing values in feature dataset:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" Hybrid AI-NWP Feature Engineering")
    print("==============================================")

    df = load_data()

    df = build_features(df)

    validate_features(df)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================================")
    print("Feature engineering completed successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print("==============================================")


if __name__ == "__main__":
    main()