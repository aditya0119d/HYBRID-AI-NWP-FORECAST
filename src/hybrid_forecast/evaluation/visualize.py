"""
Stage 4: Forecast Visualization

Visualizes:
1. Observed vs Raw GFS vs Corrected GFS
2. Observed vs Raw ECMWF vs Corrected ECMWF
3. Observed vs Hybrid Forecast
4. Model MAE comparison
5. Forecast error comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "plots"
)


# =========================================================
# Load data
# =========================================================

def load_data() -> pd.DataFrame:

    print("Loading hybrid forecast dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# =========================================================
# Plot forecast comparison
# =========================================================

def plot_forecast_comparison(df: pd.DataFrame) -> None:

    print("\nGenerating forecast comparison plot...")

    # Use a limited number of points so the graph
    # remains readable.
    plot_df = df.sort_values("valid_time").head(300)

    plt.figure(figsize=(14, 6))

    plt.plot(
        plot_df["valid_time"],
        plot_df["observed_temperature_2m"],
        label="Observed",
        linewidth=2,
    )

    plt.plot(
        plot_df["valid_time"],
        plot_df["gfs_temperature_2m"],
        label="Raw GFS",
        alpha=0.7,
    )

    plt.plot(
        plot_df["valid_time"],
        plot_df["gfs_corrected_temperature_2m"],
        label="Corrected GFS",
        alpha=0.8,
    )

    plt.plot(
        plot_df["valid_time"],
        plot_df["ecmwf_temperature_2m"],
        label="Raw ECMWF",
        alpha=0.7,
    )

    plt.plot(
        plot_df["valid_time"],
        plot_df["ecmwf_corrected_temperature_2m"],
        label="Corrected ECMWF",
        alpha=0.8,
    )

    plt.plot(
        plot_df["valid_time"],
        plot_df["hybrid_temperature_2m"],
        label="Hybrid",
        linewidth=2,
    )

    plt.title("NWP Forecast Comparison")
    plt.xlabel("Valid Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output = OUTPUT_DIR / "forecast_comparison.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# =========================================================
# Calculate errors
# =========================================================

def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:

    result = pd.DataFrame()

    observed = df["observed_temperature_2m"]

    result["Raw GFS"] = (
        df["gfs_temperature_2m"] - observed
    )

    result["Corrected GFS"] = (
        df["gfs_corrected_temperature_2m"] - observed
    )

    result["Raw ECMWF"] = (
        df["ecmwf_temperature_2m"] - observed
    )

    result["Corrected ECMWF"] = (
        df["ecmwf_corrected_temperature_2m"] - observed
    )

    result["Hybrid"] = (
        df["hybrid_temperature_2m"] - observed
    )

    return result


# =========================================================
# Plot error distribution
# =========================================================

def plot_error_distribution(
    df: pd.DataFrame,
) -> None:

    print("Generating error distribution plot...")

    errors = calculate_errors(df)

    plt.figure(figsize=(12, 6))

    for column in errors.columns:
        plt.hist(
            errors[column].dropna(),
            bins=50,
            alpha=0.35,
            label=column,
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.title("Forecast Error Distribution")
    plt.xlabel("Forecast Error (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = OUTPUT_DIR / "error_distribution.png"

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# =========================================================
# Calculate MAE
# =========================================================

def calculate_mae(df: pd.DataFrame) -> dict:

    observed = df["observed_temperature_2m"]

    predictions = {
        "Raw GFS": df["gfs_temperature_2m"],
        "Corrected GFS": df["gfs_corrected_temperature_2m"],
        "Raw ECMWF": df["ecmwf_temperature_2m"],
        "Corrected ECMWF": df[
            "ecmwf_corrected_temperature_2m"
        ],
        "Hybrid": df["hybrid_temperature_2m"],
    }

    mae = {}

    for name, prediction in predictions.items():

        mae[name] = (
            prediction
            .sub(observed)
            .abs()
            .mean()
        )

    return mae


# =========================================================
# Plot MAE comparison
# =========================================================

def plot_mae_comparison(df: pd.DataFrame) -> None:

    print("Generating MAE comparison plot...")

    mae = calculate_mae(df)

    names = list(mae.keys())
    values = list(mae.values())

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        names,
        values,
    )

    plt.title("Forecast Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("MAE (°C)")
    plt.xticks(rotation=20)
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    # Display values above bars.
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output = OUTPUT_DIR / "mae_comparison.png"

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" Hybrid AI-NWP Forecast Visualization")
    print("==============================================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    plot_forecast_comparison(df)

    plot_error_distribution(df)

    plot_mae_comparison(df)

    print("\n==============================================")
    print("Visualization completed successfully!")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print("==============================================")


if __name__ == "__main__":
    main()