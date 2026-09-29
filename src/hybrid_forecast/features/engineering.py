
"""
WeatherFusion V2 — Feature Engineering

Converts aligned historical GFS/ECMWF forecasts into
XGBoost-ready bias-correction features.

Input:
    Aligned long-format dataframe containing:
        city_name
        lat
        lon
        valid_time
        lead_days
        variable
        forecast_value
        model
        observed_value
        residual

Output:
    Feature-engineered dataframe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]


# =========================================================
# LOAD
# =========================================================

def load_aligned_data(path: str | Path) -> pd.DataFrame:
    """Load aligned historical forecast data."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Aligned dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    required = {
        "city_name",
        "lat",
        "lon",
        "valid_time",
        "lead_days",
        "variable",
        "forecast_value",
        "model",
        "observed_value",
        "residual",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


# =========================================================
# TIME FEATURES
# =========================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical temporal features."""

    df = df.copy()

    df["valid_time"] = pd.to_datetime(
        df["valid_time"],
        utc=True,
    )

    hour = df["valid_time"].dt.hour
    day_of_year = df["valid_time"].dt.dayofyear

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return df


# =========================================================
# LEAD-TIME FEATURES
# =========================================================

def create_lead_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical forecast lead-time features."""

    df = df.copy()

    df["lead_hours"] = (
        pd.to_numeric(
            df["lead_days"],
            errors="coerce",
        )
        * 24
    )

    return df


# =========================================================
# RESIDUAL HISTORY
# =========================================================

def create_residual_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical forecast-error features.

    Grouping is performed independently for:
        city
        model
        variable

    The current residual is never used to create its own
    historical features.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "city_name",
            "model",
            "variable",
            "valid_time",
            "lead_days",
        ]
    ).reset_index(drop=True)

    group_columns = [
        "city_name",
        "model",
        "variable",
    ]

    # Previous residual.
    df["residual_lag_1"] = (
        df.groupby(group_columns)["residual"]
        .shift(1)
    )

    # Residual from two records earlier.
    df["residual_lag_2"] = (
        df.groupby(group_columns)["residual"]
        .shift(2)
    )

    # Historical rolling features.
    # shift(1) is essential: it prevents the current residual
    # from leaking into the feature.
    shifted_residual = (
        df.groupby(group_columns)["residual"]
        .shift(1)
    )

    df["residual_rolling_mean_7"] = (
        shifted_residual
        .groupby(
            [
                df["city_name"],
                df["model"],
                df["variable"],
            ]
        )
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    df["residual_rolling_abs_mean_7"] = (
        shifted_residual
        .abs()
        .groupby(
            [
                df["city_name"],
                df["model"],
                df["variable"],
            ]
        )
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    return df
    """
    Create historical forecast-error features.

    Grouping is performed independently for:
        city
        model
        variable

    This prevents information from another city/model/variable
    leaking into the feature values.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "city_name",
            "model",
            "variable",
            "valid_time",
            "lead_days",
        ]
    ).reset_index(drop=True)

    group_columns = [
        "city_name",
        "model",
        "variable",
    ]

    grouped = df.groupby(
        group_columns,
        sort=False,
    )["residual"]

    # Previous available residual.
    df["residual_lag_1"] = grouped.shift(1)

    # Two previous records.
    df["residual_lag_2"] = grouped.shift(2)

    # Rolling historical residual mean.
    # shift(1) ensures the current observation is NOT used.
    df["residual_rolling_mean_7"] = (
        grouped
        .shift(1)
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
        .reset_index(level=group_columns, drop=True)
    )

    # Rolling historical absolute error.
    df["residual_rolling_abs_mean_7"] = (
        grouped
        .shift(1)
        .abs()
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
        .reset_index(level=group_columns, drop=True)
    )

    return df


# =========================================================
# LOCATION FEATURES
# =========================================================

def create_location_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare geographic features."""

    df = df.copy()

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    return df


# =========================================================
# DATA CLEANING
# =========================================================

def clean_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the feature dataset."""

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.drop_duplicates(
        subset=[
            "city_name",
            "model",
            "variable",
            "valid_time",
            "lead_days",
        ]
    )

    df = df.sort_values(
        [
            "city_name",
            "model",
            "variable",
            "valid_time",
            "lead_days",
        ]
    ).reset_index(drop=True)

    return df


# =========================================================
# FEATURE PIPELINE
# =========================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete WeatherFusion V2 feature pipeline."""

    print("Creating time features...")
    df = create_time_features(df)

    print("Creating lead-time features...")
    df = create_lead_features(df)

    print("Creating location features...")
    df = create_location_features(df)

    print("Creating historical residual features...")
    df = create_residual_history(df)

    print("Cleaning feature dataset...")
    df = clean_features(df)

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_features(
    df: pd.DataFrame,
) -> None:
    """Validate generated feature dataset."""

    required_features = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "lead_hours",
        "residual_lag_1",
        "residual_lag_2",
        "residual_rolling_mean_7",
        "residual_rolling_abs_mean_7",
    ]

    missing = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing engineered features: {missing}"
        )

    if df.empty:
        raise AssertionError(
            "Feature dataset is empty."
        )

    print("\n========== FEATURE VALIDATION ==========")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Cities:  {df['city_name'].nunique()}"
    )
    print(
        f"Models:  {df['model'].nunique()}"
    )
    print(
        f"Variables: {df['variable'].nunique()}"
    )

    print("\nEngineered features:")

    for column in required_features:
        print(f"  PASS — {column}")

    print("\nMissing values:")

    feature_missing = (
        df[required_features]
        .isna()
        .sum()
    )

    print(feature_missing)

    print("\nFeature engineering validation PASSED.")


# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WeatherFusion V2 — Feature Engineering Test")
    print("=" * 60)

    input_file = (
        "data/processed/"
        "indore_aligned_training.parquet"
    )

    output_file = (
        "data/processed/"
        "indore_ml_features.parquet"
    )

    print(f"\nInput: {input_file}")

    df = load_aligned_data(input_file)

    print(f"Input rows: {len(df)}")

    print("\nBuilding features...")

    features = build_features(df)

    validate_features(features)

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Rows:   {len(features)}")
    print(f"Cols:   {len(features.columns)}")
