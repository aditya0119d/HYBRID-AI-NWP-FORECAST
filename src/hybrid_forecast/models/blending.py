"""
Stage 5: Forecast-Time Adaptive Multi-Model Blending

Combines bias-corrected GFS and ECMWF forecasts.

IMPORTANT:
Model weights are calculated using ONLY historical forecast
performance. Current/future observations are never used to
calculate the current forecast weight.

Workflow:

    Past GFS errors
            ↓
    Rolling GFS MAE
            ↓
    GFS skill score
            ↓
            ┐
            ├── Dynamic weights
            ┘
    ECMWF skill score
            ↓
    Hybrid forecast

Hybrid forecast:

    hybrid =
        (GFS_corrected * GFS_weight)
        +
        (ECMWF_corrected * ECMWF_weight)

Weights always sum to 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GFS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gfs_corrected.parquet"
)

ECMWF_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecmwf_corrected.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid_forecast.parquet"
)


# =========================================================
# Configuration
# =========================================================

# Assuming hourly data:
# 24 hours × 7 days = 168 observations
ROLLING_WINDOW = 168

# Prevent division by zero.
EPSILON = 1e-6


# =========================================================
# Load corrected forecasts
# =========================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("Loading corrected forecasts...")

    gfs = pd.read_parquet(GFS_FILE)
    ecmwf = pd.read_parquet(ECMWF_FILE)

    print(f"GFS rows:   {len(gfs):,}")
    print(f"ECMWF rows: {len(ecmwf):,}")

    return gfs, ecmwf


# =========================================================
# Prepare blending dataset
# =========================================================

def prepare_data(
    gfs: pd.DataFrame,
    ecmwf: pd.DataFrame,
) -> pd.DataFrame:

    print("\nPreparing blending dataset...")

    keys = [
        "city_name",
        "lat",
        "lon",
        "valid_time",
    ]

    ecmwf_subset = ecmwf[
        keys
        + [
            "ecmwf_temperature_2m",
            "ecmwf_predicted_error",
            "ecmwf_corrected_temperature_2m",
        ]
    ].copy()

    result = gfs.merge(
        ecmwf_subset,
        on=keys,
        how="inner",
        suffixes=("", "_ecmwf"),
    )

    # Make sure time is actually datetime.
    result["valid_time"] = pd.to_datetime(
        result["valid_time"]
    )

    # Sort chronologically within each location.
    result = result.sort_values(
        ["city_name", "valid_time"]
    ).reset_index(drop=True)

    print(f"Matched rows: {len(result):,}")

    return result


# =========================================================
# Calculate historical errors
# =========================================================

def calculate_historical_errors(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nCalculating historical forecast errors...")

    result = df.copy()

    observed = result[
        "observed_temperature_2m"
    ]

    # Error after bias correction.
    result["gfs_corrected_error"] = (
        observed
        - result["gfs_corrected_temperature_2m"]
    )

    result["ecmwf_corrected_error"] = (
        observed
        - result["ecmwf_corrected_temperature_2m"]
    )

    # Absolute errors.
    result["gfs_abs_error"] = (
        result["gfs_corrected_error"].abs()
    )

    result["ecmwf_abs_error"] = (
        result["ecmwf_corrected_error"].abs()
    )

    return result


# =========================================================
# Calculate forecast-time weights
# =========================================================

def calculate_weights(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\nCalculating forecast-time adaptive weights..."
    )

    result = df.copy()

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # shift(1) means:
    #
    # Current forecast DOES NOT use current observation.
    #
    # Example:
    #
    # Weight at 12:00
    #     ↓
    # Uses errors up to 11:00
    #
    # This prevents target leakage.
    # -----------------------------------------------------

    result["gfs_recent_mae"] = (
        result
        .groupby("city_name")["gfs_abs_error"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                ROLLING_WINDOW,
                min_periods=24,
            )
            .mean()
        )
    )

    result["ecmwf_recent_mae"] = (
        result
        .groupby("city_name")["ecmwf_abs_error"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                ROLLING_WINDOW,
                min_periods=24,
            )
            .mean()
        )
    )

    # -----------------------------------------------------
    # Convert error into skill score.
    #
    # Lower MAE = higher skill.
    #
    # skill = 1 / MAE
    # -----------------------------------------------------

    result["gfs_skill"] = (
        1.0
        / (
            result["gfs_recent_mae"]
            + EPSILON
        )
    )

    result["ecmwf_skill"] = (
        1.0
        / (
            result["ecmwf_recent_mae"]
            + EPSILON
        )
    )

    total_skill = (
        result["gfs_skill"]
        + result["ecmwf_skill"]
    )

    # Normalize so weights sum to 1.
    result["gfs_weight"] = (
        result["gfs_skill"]
        / total_skill
    )

    result["ecmwf_weight"] = (
        result["ecmwf_skill"]
        / total_skill
    )

    # -----------------------------------------------------
    # Early rows do not have enough historical data.
    #
    # Use equal weights until sufficient history exists.
    # -----------------------------------------------------

    insufficient_history = (
        result["gfs_recent_mae"].isna()
        | result["ecmwf_recent_mae"].isna()
    )

    result.loc[
        insufficient_history,
        "gfs_weight"
    ] = 0.5

    result.loc[
        insufficient_history,
        "ecmwf_weight"
    ] = 0.5

    return result


# =========================================================
# Generate hybrid forecast
# =========================================================

def generate_hybrid_forecast(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nGenerating hybrid forecasts...")

    result = df.copy()

    result["hybrid_temperature_2m"] = (
        result["gfs_corrected_temperature_2m"]
        * result["gfs_weight"]
        +
        result["ecmwf_corrected_temperature_2m"]
        * result["ecmwf_weight"]
    )

    return result


# =========================================================
# Evaluate forecast
# =========================================================

def evaluate_forecast(
    df: pd.DataFrame,
) -> None:

    print(
        "\n========== FORECAST-TIME EVALUATION =========="
    )

    observed = df[
        "observed_temperature_2m"
    ]

    gfs_mae = (
        df["gfs_corrected_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    ecmwf_mae = (
        df["ecmwf_corrected_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    hybrid_mae = (
        df["hybrid_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    print(
        f"GFS corrected MAE:    {gfs_mae:.4f}"
    )

    print(
        f"ECMWF corrected MAE:  {ecmwf_mae:.4f}"
    )

    print(
        f"Hybrid forecast MAE:  {hybrid_mae:.4f}"
    )


# =========================================================
# Weight diagnostics
# =========================================================

def print_weight_statistics(
    df: pd.DataFrame,
) -> None:

    print(
        "\n========== WEIGHT STATISTICS =========="
    )

    print(
        f"Average GFS weight:   "
        f"{df['gfs_weight'].mean():.4f}"
    )

    print(
        f"Average ECMWF weight: "
        f"{df['ecmwf_weight'].mean():.4f}"
    )

    print(
        f"Minimum GFS weight:   "
        f"{df['gfs_weight'].min():.4f}"
    )

    print(
        f"Maximum GFS weight:   "
        f"{df['gfs_weight'].max():.4f}"
    )

    print(
        f"Minimum ECMWF weight: "
        f"{df['ecmwf_weight'].min():.4f}"
    )

    print(
        f"Maximum ECMWF weight: "
        f"{df['ecmwf_weight'].max():.4f}"
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" Hybrid AI-NWP Forecast Blending")
    print(" Forecast-Time Adaptive Weighting")
    print("==============================================")

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    gfs, ecmwf = load_data()

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    df = prepare_data(
        gfs,
        ecmwf,
    )

    # -----------------------------------------------------
    # Historical errors
    # -----------------------------------------------------

    df = calculate_historical_errors(df)

    # -----------------------------------------------------
    # Forecast-time weights
    # -----------------------------------------------------

    df = calculate_weights(df)

    # -----------------------------------------------------
    # Hybrid forecast
    # -----------------------------------------------------

    df = generate_hybrid_forecast(df)

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    evaluate_forecast(df)

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print_weight_statistics(df)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("\nSaving hybrid forecast...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================================")
    print("Hybrid forecast completed successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print("==============================================")


if __name__ == "__main__":
    main()