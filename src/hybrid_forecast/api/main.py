"""
FastAPI backend for the Hybrid AI-NWP Multi-Model Forecast System.

Pipeline:

    Weather / NWP Data
          ↓
    Time Alignment
          ↓
    Feature Engineering
          ↓
    GFS XGBoost Bias Correction
          ↓
    ECMWF XGBoost Bias Correction
          ↓
    Multi-Model Blending
          ↓
    FastAPI
          ↓
    Next.js Dashboard
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HYBRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid_forecast.parquet"
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Hybrid AI-NWP Forecast API",
    description=(
        "API for the Hybrid AI-NWP "
        "Multi-Model Forecast System"
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATA LOADER
# =========================================================

def load_hybrid_data() -> pd.DataFrame:
    """
    Load the final hybrid forecast dataset.
    """

    if not HYBRID_FILE.exists():
        raise FileNotFoundError(
            f"Hybrid forecast file not found: {HYBRID_FILE}"
        )

    return pd.read_parquet(HYBRID_FILE)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "service": "Hybrid AI-NWP Forecast API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/locations",
            "/forecast/{location_id}",
            "/forecast/{location_id}/weights",
            "/model-performance",
            "/forecast/{location_id}/history",
        ],
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "ok",
        "service": "hybrid-ai-nwp-forecast",
    }


# =========================================================
# LOCATIONS ENDPOINT
# =========================================================

@app.get("/locations")
def get_locations():
    """
    Return all available forecast locations.
    """

    try:
        df = load_hybrid_data()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    required_columns = [
        "city_name",
        "lat",
        "lon",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing columns: {missing}",
        )

    locations = (
        df[
            [
                "city_name",
                "lat",
                "lon",
            ]
        ]
        .drop_duplicates()
        .sort_values("city_name")
    )

    response = []

    for _, row in locations.iterrows():

        city = str(row["city_name"])

        response.append(
            {
                "id": city.lower().replace(" ", "-"),
                "name": city,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "region": "India",
            }
        )

    return response


# =========================================================
# FORECAST ENDPOINT
# =========================================================

@app.get("/forecast/{location_id}")
def get_forecast(location_id: str):
    """
    Return the complete forecast pipeline for a location.

    Includes:

        Raw GFS
        Raw ECMWF
        Corrected GFS
        Corrected ECMWF
        Final Hybrid Forecast
        Observations
    """

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    try:
        df = load_hybrid_data()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # -----------------------------------------------------
    # ECMWF corrected temperature fallback
    #
    # The actual dataset should already contain:
    #
    #     ecmwf_corrected_temperature_2m
    #
    # If it does not, but predicted error exists,
    # reconstruct it:
    #
    # corrected = raw + predicted_error
    # -----------------------------------------------------

    if (
        "ecmwf_corrected_temperature_2m"
        not in df.columns
    ):

        if "ecmwf_predicted_error" not in df.columns:

            raise HTTPException(
                status_code=500,
                detail=(
                    "ECMWF corrected temperature is missing "
                    "and ecmwf_predicted_error is unavailable."
                ),
            )

        df["ecmwf_corrected_temperature_2m"] = (
            df["ecmwf_temperature_2m"]
            + df["ecmwf_predicted_error"]
        )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [

        # Location
        "city_name",
        "lat",
        "lon",
        "valid_time",

        # -------------------------
        # Raw GFS
        # -------------------------
        "gfs_temperature_2m",
        "gfs_precipitation",
        "gfs_relative_humidity_2m",
        "gfs_surface_pressure",
        "gfs_wind_speed_10m",

        # -------------------------
        # Raw ECMWF
        # -------------------------
        "ecmwf_temperature_2m",
        "ecmwf_precipitation",
        "ecmwf_relative_humidity_2m",
        "ecmwf_surface_pressure",
        "ecmwf_wind_speed_10m",

        # -------------------------
        # Observations
        # -------------------------
        "observed_temperature_2m",
        "observed_precipitation",
        "observed_relative_humidity_2m",
        "observed_surface_pressure",
        "observed_wind_speed_10m",

        # -------------------------
        # Corrected forecasts
        # -------------------------
        "gfs_corrected_temperature_2m",
        "ecmwf_corrected_temperature_2m",

        # -------------------------
        # Final hybrid forecast
        # -------------------------
        "hybrid_temperature_2m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing forecast columns: {missing}",
        )

    # -----------------------------------------------------
    # Normalize location ID
    # -----------------------------------------------------

    requested_id = (
        location_id
        .lower()
        .strip()
    )

    df["location_id"] = (
        df["city_name"]
        .astype(str)
        .str.lower()
        .str.replace(
            " ",
            "-",
            regex=False,
        )
    )

    # -----------------------------------------------------
    # Select location
    # -----------------------------------------------------

    location_df = df[
        df["location_id"] == requested_id
    ].copy()

    if location_df.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Location not found: {location_id}",
        )

    # -----------------------------------------------------
    # Convert valid_time
    # -----------------------------------------------------

    location_df["valid_time"] = pd.to_datetime(
        location_df["valid_time"]
    )

    location_df = location_df.sort_values(
        "valid_time"
    )

    # -----------------------------------------------------
    # Location information
    # -----------------------------------------------------

    first_row = location_df.iloc[0]

    location = {
        "id": requested_id,
        "name": str(first_row["city_name"]),
        "lat": float(first_row["lat"]),
        "lon": float(first_row["lon"]),
        "region": "India",
    }

    # -----------------------------------------------------
    # Forecast hour
    # -----------------------------------------------------

    start_time = location_df[
        "valid_time"
    ].iloc[0]

    def forecast_hour(timestamp):
        return int(
            (
                timestamp - start_time
            ).total_seconds()
            / 3600
        )

    # -----------------------------------------------------
    # Weather series builder
    # -----------------------------------------------------

    def build_weather_series(
        temperature_column,
        precipitation_column,
        humidity_column,
        pressure_column,
        wind_column,
    ):

        series = []

        for _, row in location_df.iterrows():

            timestamp = row["valid_time"]

            series.append(
                {
                    "valid_time":
                        timestamp.isoformat(),

                    "forecast_hour":
                        forecast_hour(timestamp),

                    "temperature":
                        float(
                            row[
                                temperature_column
                            ]
                        ),

                    "precipitation":
                        float(
                            row[
                                precipitation_column
                            ]
                        ),

                    "humidity":
                        float(
                            row[
                                humidity_column
                            ]
                        ),

                    "pressure":
                        float(
                            row[
                                pressure_column
                            ]
                        ),

                    "wind_speed":
                        float(
                            row[
                                wind_column
                            ]
                        ),
                }
            )

        return series

    # =====================================================
    # RAW GFS
    # =====================================================

    raw_gfs = build_weather_series(
        "gfs_temperature_2m",
        "gfs_precipitation",
        "gfs_relative_humidity_2m",
        "gfs_surface_pressure",
        "gfs_wind_speed_10m",
    )

    # =====================================================
    # RAW ECMWF
    # =====================================================

    raw_ecmwf = build_weather_series(
        "ecmwf_temperature_2m",
        "ecmwf_precipitation",
        "ecmwf_relative_humidity_2m",
        "ecmwf_surface_pressure",
        "ecmwf_wind_speed_10m",
    )

    # =====================================================
    # CORRECTED GFS
    # =====================================================

    corrected_gfs = build_weather_series(
        "gfs_corrected_temperature_2m",
        "gfs_precipitation",
        "gfs_relative_humidity_2m",
        "gfs_surface_pressure",
        "gfs_wind_speed_10m",
    )

    # =====================================================
    # CORRECTED ECMWF
    # =====================================================

    corrected_ecmwf = build_weather_series(
        "ecmwf_corrected_temperature_2m",
        "ecmwf_precipitation",
        "ecmwf_relative_humidity_2m",
        "ecmwf_surface_pressure",
        "ecmwf_wind_speed_10m",
    )

    # =====================================================
    # FINAL HYBRID FORECAST
    # =====================================================

    final_blend = []

    for _, row in location_df.iterrows():

        timestamp = row["valid_time"]

        final_blend.append(
            {
                "valid_time":
                    timestamp.isoformat(),

                "forecast_hour":
                    forecast_hour(timestamp),

                # Final hybrid temperature
                "temperature":
                    float(
                        row[
                            "hybrid_temperature_2m"
                        ]
                    ),

                # Current Stage 2 pipeline
                # produces hybrid temperature.
                #
                # For the remaining weather variables,
                # use the mean of GFS and ECMWF.

                "precipitation":
                    float(
                        (
                            row[
                                "gfs_precipitation"
                            ]
                            +
                            row[
                                "ecmwf_precipitation"
                            ]
                        ) / 2
                    ),

                "humidity":
                    float(
                        (
                            row[
                                "gfs_relative_humidity_2m"
                            ]
                            +
                            row[
                                "ecmwf_relative_humidity_2m"
                            ]
                        ) / 2
                    ),

                "pressure":
                    float(
                        (
                            row[
                                "gfs_surface_pressure"
                            ]
                            +
                            row[
                                "ecmwf_surface_pressure"
                            ]
                        ) / 2
                    ),

                "wind_speed":
                    float(
                        (
                            row[
                                "gfs_wind_speed_10m"
                            ]
                            +
                            row[
                                "ecmwf_wind_speed_10m"
                            ]
                        ) / 2
                    ),
            }
        )

    # =====================================================
    # OBSERVATIONS
    # =====================================================

    observed = build_weather_series(
        "observed_temperature_2m",
        "observed_precipitation",
        "observed_relative_humidity_2m",
        "observed_surface_pressure",
        "observed_wind_speed_10m",
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "location": location,

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "raw_gfs":
            raw_gfs,

        "raw_ecmwf":
            raw_ecmwf,

        "corrected_gfs":
            corrected_gfs,

        "corrected_ecmwf":
            corrected_ecmwf,

        "final_blend":
            final_blend,

        "observed":
            observed,
    }


# =========================================================
# FORECAST WEIGHTS ENDPOINT
# =========================================================

@app.get("/forecast/{location_id}/weights")
def get_forecast_weights(
    location_id: str,
):
    """
    Return current GFS/ECMWF trust weights.

    The current Stage 2 pipeline calculates weights
    from corrected forecast errors.
    """

    try:
        df = load_hybrid_data()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    required_columns = [
        "city_name",
        "lat",
        "lon",
        "gfs_weight",
        "ecmwf_weight",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    # -----------------------------------------------------
    # If weights are not stored in Parquet, calculate them
    # from corrected forecast errors.
    # -----------------------------------------------------

    if missing:

        if (
            "gfs_corrected_temperature_2m"
            not in df.columns
            or
            "ecmwf_corrected_temperature_2m"
            not in df.columns
            or
            "observed_temperature_2m"
            not in df.columns
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Model weights are unavailable because "
                    "required forecast/error columns are missing."
                ),
            )

        gfs_error = (
            df["observed_temperature_2m"]
            - df["gfs_corrected_temperature_2m"]
        ).abs()

        ecmwf_error = (
            df["observed_temperature_2m"]
            - df["ecmwf_corrected_temperature_2m"]
        ).abs()

        total_error = (
            gfs_error + ecmwf_error
        ).replace(
            0,
            np.nan,
        )

        df["gfs_weight"] = (
            ecmwf_error / total_error
        ).fillna(0.5)

        df["ecmwf_weight"] = (
            gfs_error / total_error
        ).fillna(0.5)

    # -----------------------------------------------------
    # Normalize location
    # -----------------------------------------------------

    requested_id = (
        location_id
        .lower()
        .strip()
    )

    df["location_id"] = (
        df["city_name"]
        .astype(str)
        .str.lower()
        .str.replace(
            " ",
            "-",
            regex=False,
        )
    )

    location_df = df[
        df["location_id"] == requested_id
    ].copy()

    if location_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Location not found: {location_id}",
        )

    first_row = location_df.iloc[0]

    # -----------------------------------------------------
    # Current average weights
    # -----------------------------------------------------

    gfs_weight = float(
        location_df["gfs_weight"].mean()
    )

    ecmwf_weight = float(
        location_df["ecmwf_weight"].mean()
    )

    # Normalize again to guarantee 100%
    total = (
        gfs_weight + ecmwf_weight
    )

    if total > 0:
        gfs_weight /= total
        ecmwf_weight /= total

    # -----------------------------------------------------
    # 30-day trend
    #
    # Uses daily averages where data exists.
    # -----------------------------------------------------

    trend = []

    if "valid_time" in location_df.columns:

        location_df["valid_time"] = pd.to_datetime(
            location_df["valid_time"]
        )

        location_df["date"] = (
            location_df["valid_time"]
            .dt.date
            .astype(str)
        )

        daily = (
            location_df
            .groupby("date")[
                [
                    "gfs_weight",
                    "ecmwf_weight",
                ]
            ]
            .mean()
            .tail(30)
        )

        for date, row in daily.iterrows():

            trend.append(
                {
                    "date": str(date),

                    "gfs_trust_pct":
                        round(
                            float(
                                row[
                                    "gfs_weight"
                                ]
                            ) * 100,
                            2,
                        ),

                    "ecmwf_trust_pct":
                        round(
                            float(
                                row[
                                    "ecmwf_weight"
                                ]
                            ) * 100,
                            2,
                        ),
                }
            )

    return {
        "location": {
            "id": requested_id,
            "name": str(
                first_row["city_name"]
            ),
            "lat": float(
                first_row["lat"]
            ),
            "lon": float(
                first_row["lon"]
            ),
            "region": "India",
        },

        "current": {
            "gfs_trust_pct":
                round(
                    gfs_weight * 100,
                    2,
                ),

            "ecmwf_trust_pct":
                round(
                    ecmwf_weight * 100,
                    2,
                ),

            "basis":
                "Trust is based on corrected forecast "
                "error patterns for this location.",
        },

        "trend_30day": trend,
    }


# =========================================================
# MODEL PERFORMANCE ENDPOINT
# =========================================================

@app.get("/model-performance")
def get_model_performance():
    """
    Calculate MAE, RMSE and bias for each model.
    """

    try:
        df = load_hybrid_data()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # -----------------------------------------------------
    # ECMWF corrected fallback
    # -----------------------------------------------------

    if (
        "ecmwf_corrected_temperature_2m"
        not in df.columns
    ):

        if "ecmwf_predicted_error" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "ECMWF corrected temperature is unavailable."
                ),
            )

        df["ecmwf_corrected_temperature_2m"] = (
            df["ecmwf_temperature_2m"]
            + df["ecmwf_predicted_error"]
        )

    required_columns = [
        "observed_temperature_2m",
        "gfs_temperature_2m",
        "gfs_corrected_temperature_2m",
        "ecmwf_temperature_2m",
        "ecmwf_corrected_temperature_2m",
        "hybrid_temperature_2m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing performance columns: {missing}",
        )

    observed = df[
        "observed_temperature_2m"
    ]

    models = {
        "Raw GFS":
            df["gfs_temperature_2m"],

        "Corrected GFS":
            df["gfs_corrected_temperature_2m"],

        "Raw ECMWF":
            df["ecmwf_temperature_2m"],

        "Corrected ECMWF":
            df[
                "ecmwf_corrected_temperature_2m"
            ],

        "Final Blend":
            df["hybrid_temperature_2m"],
    }

    by_model = []

    for model_name, predictions in models.items():

        valid = (
            predictions.notna()
            & observed.notna()
        )

        errors = (
            predictions[valid]
            - observed[valid]
        )

        mae = float(
            errors.abs().mean()
        )

        rmse = float(
            np.sqrt(
                (errors ** 2).mean()
            )
        )

        bias = float(
            errors.mean()
        )

        by_model.append(
            {
                "model_name": model_name,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
            }
        )

    # -----------------------------------------------------
    # Lead-time performance
    # -----------------------------------------------------

    by_lead_time = []

    if "forecast_hour" in df.columns:

        lead_bins = [
            (-1, 24, "0–24h"),
            (24, 48, "24–48h"),
            (48, 72, "48–72h"),
            (72, 120, "72–120h"),
        ]

        hybrid = df[
            "hybrid_temperature_2m"
        ]

        for lower, upper, bucket in lead_bins:

            mask = (
                (df["forecast_hour"] > lower)
                &
                (df["forecast_hour"] <= upper)
                &
                hybrid.notna()
                &
                observed.notna()
            )

            errors = (
                hybrid[mask]
                - observed[mask]
            )

            if len(errors) == 0:
                continue

            by_lead_time.append(
                {
                    "bucket": bucket,
                    "model_name": "Final Blend",
                    "mae": round(
                        float(
                            errors.abs().mean()
                        ),
                        4,
                    ),
                    "rmse": round(
                        float(
                            np.sqrt(
                                (
                                    errors ** 2
                                ).mean()
                            )
                        ),
                        4,
                    ),
                }
            )

    # -----------------------------------------------------
    # Seasonal performance
    # -----------------------------------------------------

    by_season = []

    if "valid_time" in df.columns:

        temp_df = df.copy()

        temp_df["valid_time"] = pd.to_datetime(
            temp_df["valid_time"]
        )

        month = (
            temp_df["valid_time"]
            .dt.month
        )

        seasons = {
            "Winter":
                month.isin([12, 1, 2]),

            "Summer":
                month.isin([3, 4, 5]),

            "Monsoon":
                month.isin([6, 7, 8, 9]),

            "Post-monsoon":
                month.isin([10, 11]),
        }

        hybrid = temp_df[
            "hybrid_temperature_2m"
        ]

        obs = temp_df[
            "observed_temperature_2m"
        ]

        for season_name, mask in seasons.items():

            valid_mask = (
                mask
                & hybrid.notna()
                & obs.notna()
            )

            errors = (
                hybrid[valid_mask]
                - obs[valid_mask]
            )

            if len(errors) == 0:
                continue

            by_season.append(
                {
                    "season": season_name,
                    "model_name": "Final Blend",
                    "mae": round(
                        float(
                            errors.abs().mean()
                        ),
                        4,
                    ),
                    "rmse": round(
                        float(
                            np.sqrt(
                                (
                                    errors ** 2
                                ).mean()
                            )
                        ),
                        4,
                    ),
                }
            )

    return {
        "by_model": by_model,
        "by_lead_time": by_lead_time,
        "by_season": by_season,
    }


# =========================================================
# FORECAST HISTORY ENDPOINT
# =========================================================

@app.get("/forecast/{location_id}/history")
def get_forecast_history(
    location_id: str,
):
    """
    Return historical hybrid forecast performance
    for a location.
    """

    try:
        df = load_hybrid_data()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    required_columns = [
        "city_name",
        "valid_time",
        "hybrid_temperature_2m",
        "observed_temperature_2m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing history columns: {missing}",
        )

    requested_id = (
        location_id
        .lower()
        .strip()
    )

    df["location_id"] = (
        df["city_name"]
        .astype(str)
        .str.lower()
        .str.replace(
            " ",
            "-",
            regex=False,
        )
    )

    location_df = df[
        df["location_id"] == requested_id
    ].copy()

    if location_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Location not found: {location_id}",
        )

    location_df["valid_time"] = pd.to_datetime(
        location_df["valid_time"]
    )

    location_df = location_df.sort_values(
        "valid_time"
    )

    history = []

    for _, row in location_df.iterrows():

        if (
            pd.isna(
                row[
                    "hybrid_temperature_2m"
                ]
            )
            or
            pd.isna(
                row[
                    "observed_temperature_2m"
                ]
            )
        ):
            continue

        history.append(
            {
                "valid_time":
                    row[
                        "valid_time"
                    ].isoformat(),

                "predicted":
                    float(
                        row[
                            "hybrid_temperature_2m"
                        ]
                    ),

                "observed":
                    float(
                        row[
                            "observed_temperature_2m"
                        ]
                    ),

                "model_used":
                    "Final Blend",
            }
        )

    return history


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.hybrid_forecast.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )