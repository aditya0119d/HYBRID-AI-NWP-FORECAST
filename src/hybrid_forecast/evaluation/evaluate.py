"""
Stage 3: Forecast Evaluation

Compares:
1. Raw GFS forecast
2. Corrected GFS forecast
3. Raw ECMWF forecast
4. Corrected ECMWF forecast
5. Hybrid forecast

Metrics:
- MAE
- RMSE
- Improvement percentage
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid_forecast.parquet"
)


# =========================================================
# Load data
# =========================================================

def load_data() -> pd.DataFrame:

    print("Loading hybrid forecast dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# =========================================================
# Calculate metrics
# =========================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, float]:

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    return mae, rmse


# =========================================================
# Evaluate forecasts
# =========================================================

def evaluate(df: pd.DataFrame) -> pd.DataFrame:

    print("\n========== FORECAST EVALUATION ==========")

    actual = df["observed_temperature_2m"]

    forecasts = {
        "Raw GFS": df["gfs_temperature_2m"],
        "Corrected GFS": df["gfs_corrected_temperature_2m"],
        "Raw ECMWF": df["ecmwf_temperature_2m"],
        "Corrected ECMWF": df["ecmwf_corrected_temperature_2m"],
        "Hybrid": df["hybrid_temperature_2m"],
    }

    results = []

    for name, forecast in forecasts.items():

        valid = actual.notna() & forecast.notna()

        mae, rmse = calculate_metrics(
            actual[valid],
            forecast[valid],
        )

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    results_df = pd.DataFrame(results)

    return results_df


# =========================================================
# Calculate improvement
# =========================================================

def calculate_improvement(
    results: pd.DataFrame,
) -> None:

    hybrid_mae = results.loc[
        results["Model"] == "Hybrid",
        "MAE",
    ].iloc[0]

    raw_gfs_mae = results.loc[
        results["Model"] == "Raw GFS",
        "MAE",
    ].iloc[0]

    raw_ecmwf_mae = results.loc[
        results["Model"] == "Raw ECMWF",
        "MAE",
    ].iloc[0]

    gfs_improvement = (
        (raw_gfs_mae - hybrid_mae)
        / raw_gfs_mae
        * 100
    )

    ecmwf_improvement = (
        (raw_ecmwf_mae - hybrid_mae)
        / raw_ecmwf_mae
        * 100
    )

    print("\n========== IMPROVEMENT ==========")

    print(
        f"Hybrid improvement vs Raw GFS:   "
        f"{gfs_improvement:.2f}%"
    )

    print(
        f"Hybrid improvement vs Raw ECMWF: "
        f"{ecmwf_improvement:.2f}%"
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" Hybrid AI-NWP Forecast Evaluation")
    print("==============================================")

    df = load_data()

    results = evaluate(df)

    print("\n========== MODEL COMPARISON ==========")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    calculate_improvement(results)

    print("\n==============================================")
    print("Evaluation completed successfully!")
    print("==============================================")


if __name__ == "__main__":
    main()