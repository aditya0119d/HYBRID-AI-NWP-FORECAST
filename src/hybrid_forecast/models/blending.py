"""
Stage 2: Multi-Model Forecast Blending

Combines bias-corrected GFS and ECMWF forecasts
into a single hybrid forecast.

For the prototype, model weights are based on
recent absolute forecast error.

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

    # Columns identifying the same forecast point.
    keys = [
        "city_name",
        "lat",
        "lon",
        "valid_time",
    ]

    # Select only ECMWF columns required for blending.
    ecmwf_subset = ecmwf[
        keys
        + [
            "ecmwf_temperature_2m",
            "ecmwf_predicted_error",
            "ecmwf_corrected_temperature_2m",
        ]
    ].copy()

    # Merge GFS and ECMWF forecasts.
    result = gfs.merge(
        ecmwf_subset,
        on=keys,
        how="inner",
        suffixes=("", "_ecmwf"),
    )

    print(f"Matched rows: {len(result):,}")

    return result


# =========================================================
# Calculate adaptive model weights
# =========================================================

def calculate_weights(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nCalculating adaptive model weights...")

    result = df.copy()

    # -----------------------------------------------------
    # Calculate corrected forecast errors
    # -----------------------------------------------------

    # GFS error:
    # observed - corrected GFS
    result["gfs_corrected_error"] = (
        result["observed_temperature_2m"]
        - result["gfs_corrected_temperature_2m"]
    )

    # ECMWF error:
    # observed - corrected ECMWF
    result["ecmwf_corrected_error"] = (
        result["observed_temperature_2m"]
        - result["ecmwf_corrected_temperature_2m"]
    )

    # -----------------------------------------------------
    # Absolute errors
    # -----------------------------------------------------

    gfs_error = result["gfs_corrected_error"].abs()

    ecmwf_error = result["ecmwf_corrected_error"].abs()

    # -----------------------------------------------------
    # Adaptive weights
    # -----------------------------------------------------
    #
    # Lower error = higher weight.
    #
    # GFS weight:
    # ECMWF error / total error
    #
    # ECMWF weight:
    # GFS error / total error
    #
    # Therefore:
    # gfs_weight + ecmwf_weight = 1
    # -----------------------------------------------------

    total_error = gfs_error + ecmwf_error

    # Avoid division by zero.
    total_error = total_error.replace(0, np.nan)

    result["gfs_weight"] = (
        ecmwf_error / total_error
    )

    result["ecmwf_weight"] = (
        gfs_error / total_error
    )

    # If both errors are zero, use equal weighting.
    result["gfs_weight"] = (
        result["gfs_weight"].fillna(0.5)
    )

    result["ecmwf_weight"] = (
        result["ecmwf_weight"].fillna(0.5)
    )

    return result


# =========================================================
# Generate hybrid forecast
# =========================================================

def generate_hybrid_forecast(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nGenerating hybrid forecasts...")

    result = df.copy()

    # Weighted combination of corrected GFS
    # and corrected ECMWF forecasts.
    result["hybrid_temperature_2m"] = (
        result["gfs_corrected_temperature_2m"]
        * result["gfs_weight"]
        +
        result["ecmwf_corrected_temperature_2m"]
        * result["ecmwf_weight"]
    )

    return result


# =========================================================
# Evaluate hybrid forecast
# =========================================================

def evaluate_forecast(
    df: pd.DataFrame,
) -> None:

    print("\n========== FORECAST EVALUATION ==========")

    observed = df["observed_temperature_2m"]

    # -----------------------------------------------------
    # GFS MAE
    # -----------------------------------------------------

    gfs_mae = (
        df["gfs_corrected_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    # -----------------------------------------------------
    # ECMWF MAE
    # -----------------------------------------------------

    ecmwf_mae = (
        df["ecmwf_corrected_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    # -----------------------------------------------------
    # Hybrid MAE
    # -----------------------------------------------------

    hybrid_mae = (
        df["hybrid_temperature_2m"]
        .sub(observed)
        .abs()
        .mean()
    )

    print(f"GFS corrected MAE:    {gfs_mae:.4f}")
    print(f"ECMWF corrected MAE:  {ecmwf_mae:.4f}")
    print(f"Hybrid forecast MAE:  {hybrid_mae:.4f}")


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" Hybrid AI-NWP Multi-Model Forecast Blending")
    print("==============================================")

    # Load corrected forecasts.
    gfs, ecmwf = load_data()

    # Match GFS and ECMWF forecasts.
    df = prepare_data(
        gfs,
        ecmwf,
    )

    # Calculate adaptive model weights.
    df = calculate_weights(df)

    # Generate final hybrid forecast.
    df = generate_hybrid_forecast(df)

    # Evaluate all forecasts.
    evaluate_forecast(df)

    # Save final hybrid dataset.
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