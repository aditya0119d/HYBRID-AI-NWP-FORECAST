"""
Stage 1: NWP Bias Correction using XGBoost.

The model learns the error of the ECMWF temperature forecast:

    error = observed_temperature - ECMWF_temperature

Then:

    corrected_ECMWF = ECMWF_temperature + predicted_error
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecmwf_corrected.parquet"
)


# =========================================================
# Load data
# =========================================================

def load_data() -> pd.DataFrame:
    """Load the feature-engineered dataset."""

    print("Loading ML feature dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# =========================================================
# Prepare ECMWF training data
# =========================================================

def prepare_training_data(
    df: pd.DataFrame,
):
    """
    Prepare features and target for ECMWF temperature
    bias correction.
    """

    # Features used by the ECMWF XGBoost model.
    feature_columns = [
        "ecmwf_temperature_2m",
        "gfs_temperature_2m",
        "gfs_ecmwf_diff_temperature_2m",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month",
        "lat",
        "lon",
        "ecmwf_rolling_error_7",
        "ecmwf_rolling_abs_error_7",
    ]

    # Check that all required features exist.
    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    # Target = observed temperature - ECMWF temperature.
    target_column = "ecmwf_error_temperature_2m"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    data = df[
        feature_columns + [target_column]
    ].dropna()

    X = data[feature_columns]
    y = data[target_column]

    return X, y, feature_columns


# =========================================================
# Train / validation split
# =========================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Time-aware train/validation split.

    First 80% = training
    Last 20%  = validation
    """

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =========================================================
# Train XGBoost
# =========================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:

    print("\nTraining ECMWF XGBoost bias-correction model...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("XGBoost training completed.")

    return model


# =========================================================
# Evaluate correction
# =========================================================

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:

    predicted_error = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predicted_error,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted_error,
        )
    )

    print("\n========== ECMWF XGBOOST ERROR PREDICTION ==========")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


# =========================================================
# Generate corrected ECMWF forecast
# =========================================================

def generate_corrected_forecast(
    df: pd.DataFrame,
    model: XGBRegressor,
    feature_columns: list[str],
) -> pd.DataFrame:

    result = df.copy()

    # Determine rows where all model features are available.
    valid_rows = result[
        feature_columns
    ].notna().all(axis=1)

    predicted_error = pd.Series(
        np.nan,
        index=result.index,
    )

    # Predict ECMWF forecast error.
    predicted_error.loc[valid_rows] = model.predict(
        result.loc[
            valid_rows,
            feature_columns,
        ]
    )

    result["ecmwf_predicted_error"] = predicted_error

    # Corrected ECMWF forecast:
    #
    # corrected forecast
    # = original forecast + predicted error
    result["ecmwf_corrected_temperature_2m"] = (
        result["ecmwf_temperature_2m"]
        + result["ecmwf_predicted_error"]
    )

    return result


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("==============================================")
    print(" ECMWF XGBoost Bias Correction")
    print("==============================================")

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # Prepare training data
    # -----------------------------------------------------

    X, y, feature_columns = prepare_training_data(
        df
    )

    print(
        f"\nTraining samples available: {len(X):,}"
    )

    # -----------------------------------------------------
    # Split data
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    print(
        f"Training samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_test):,}"
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Evaluate model
    # -----------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Generate corrected forecasts
    # -----------------------------------------------------

    print("\nGenerating corrected ECMWF forecasts...")

    result = generate_corrected_forecast(
        df,
        model,
        feature_columns,
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save corrected forecast
    # -----------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================================")
    print("ECMWF bias correction completed successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print("==============================================")


if __name__ == "__main__":
    main()