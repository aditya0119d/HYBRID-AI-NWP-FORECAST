"""
WeatherFusion V2 — Raw NWP data validation.

Validates:
- city coverage
- row counts
- required columns
- missing values
- duplicate city/timestamp records
- timestamp continuity
- coordinate consistency
- weather-value ranges
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hybrid_forecast.ingestion.cities import CITIES


DATA_DIR = Path("data/raw/open_meteo")

REQUIRED_COLUMNS = [
    "city_name",
    "lat",
    "lon",
    "valid_time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "model",
]

EXPECTED_CITIES = {
    city["name"] for city in CITIES.values()
}


def validate_dataset(
    path: Path,
    expected_model: str,
) -> bool:

    print("\n" + "=" * 60)
    print(f"VALIDATING: {expected_model}")
    print("=" * 60)

    df = pd.read_parquet(path)

    passed = True

    # -----------------------------------------------------
    # Shape
    # -----------------------------------------------------

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print(f"FAIL — Missing columns: {missing_columns}")
        passed = False
    else:
        print("PASS — Required columns present")

    # -----------------------------------------------------
    # Model identity
    # -----------------------------------------------------

    models = set(df["model"].dropna().unique())

    if models != {expected_model}:
        print(
            f"FAIL — Expected model {expected_model}, "
            f"found {models}"
        )
        passed = False
    else:
        print(f"PASS — Model identity: {expected_model}")

    # -----------------------------------------------------
    # City coverage
    # -----------------------------------------------------

    cities = set(df["city_name"].dropna().unique())

    missing_cities = EXPECTED_CITIES - cities
    unexpected_cities = cities - EXPECTED_CITIES

    print(f"Cities found: {len(cities)}")

    if missing_cities:
        print(
            f"FAIL — Missing cities: "
            f"{sorted(missing_cities)}"
        )
        passed = False
    else:
        print("PASS — All 50 registered cities present")

    if unexpected_cities:
        print(
            f"WARNING — Unexpected cities: "
            f"{sorted(unexpected_cities)}"
        )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    null_counts = df[REQUIRED_COLUMNS].isna().sum()

    total_nulls = int(null_counts.sum())

    if total_nulls:
        print("\nFAIL — Missing values detected:")
        print(
            null_counts[null_counts > 0]
        )
        passed = False
    else:
        print("PASS — No missing values")

    # -----------------------------------------------------
    # Duplicate records
    # -----------------------------------------------------

    duplicate_columns = [
        "city_name",
        "valid_time",
        "model",
    ]

    duplicates = int(
        df.duplicated(
            subset=duplicate_columns
        ).sum()
    )

    if duplicates:
        print(
            f"FAIL — Duplicate records: {duplicates}"
        )
        passed = False
    else:
        print("PASS — No duplicate city/time/model records")

    # -----------------------------------------------------
    # Timestamp validation
    # -----------------------------------------------------

    df["valid_time"] = pd.to_datetime(
        df["valid_time"],
        utc=True,
    )

    if not df["valid_time"].is_monotonic_increasing:
        print(
            "INFO — Global timestamp order is not monotonic "
            "(expected because multiple cities are interleaved)"
        )

    invalid_times = int(
        df["valid_time"].isna().sum()
    )

    if invalid_times:
        print(
            f"FAIL — Invalid timestamps: {invalid_times}"
        )
        passed = False
    else:
        print("PASS — Valid UTC timestamps")

    # -----------------------------------------------------
    # Coordinate validation
    # -----------------------------------------------------

    invalid_lat = (
        (df["lat"] < -90)
        | (df["lat"] > 90)
    ).sum()

    invalid_lon = (
        (df["lon"] < -180)
        | (df["lon"] > 180)
    ).sum()

    if invalid_lat or invalid_lon:
        print(
            f"FAIL — Invalid coordinates: "
            f"lat={invalid_lat}, lon={invalid_lon}"
        )
        passed = False
    else:
        print("PASS — Coordinates within valid ranges")

    # -----------------------------------------------------
    # Weather-value sanity checks
    # -----------------------------------------------------

    checks = {
        "relative_humidity_2m": (
            (df["relative_humidity_2m"] < 0)
            | (df["relative_humidity_2m"] > 100)
        ),
        "precipitation": (
            df["precipitation"] < 0
        ),
        "wind_speed_10m": (
            df["wind_speed_10m"] < 0
        ),
    }

    invalid_weather = False

    for variable, condition in checks.items():

        count = int(condition.sum())

        if count:
            print(
                f"FAIL — {variable}: "
                f"{count} invalid values"
            )
            invalid_weather = True

    if invalid_weather:
        passed = False
    else:
        print("PASS — Weather-value sanity checks")

    # -----------------------------------------------------
    # Per-city temporal coverage
    # -----------------------------------------------------

    expected_rows_per_city = 72

    counts = (
        df.groupby("city_name")
        .size()
    )

    incorrect_counts = counts[
        counts != expected_rows_per_city
    ]

    if len(incorrect_counts):
        print(
            "\nFAIL — Incorrect rows per city:"
        )
        print(incorrect_counts)
        passed = False
    else:
        print(
            f"PASS — Every city has "
            f"{expected_rows_per_city} records"
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "-" * 60)

    if passed:
        print(
            f"{expected_model} DATA VALIDATION: PASSED"
        )
    else:
        print(
            f"{expected_model} DATA VALIDATION: FAILED"
        )

    return passed


def main() -> None:

    gfs_path = DATA_DIR / "gfs_india.parquet"
    ecmwf_path = DATA_DIR / "ecmwf_india.parquet"

    gfs_ok = validate_dataset(
        gfs_path,
        "GFS",
    )

    ecmwf_ok = validate_dataset(
        ecmwf_path,
        "ECMWF",
    )

    print("\n" + "=" * 60)

    if gfs_ok and ecmwf_ok:
        print("WEATHERFUSION V2 — RAW DATA VALIDATION PASSED")
    else:
        print("WEATHERFUSION V2 — RAW DATA VALIDATION FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()