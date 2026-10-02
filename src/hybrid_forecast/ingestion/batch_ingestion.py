"""
WeatherFusion V2 — Multi-city Open-Meteo batch ingestion.

Fetches GFS + ECMWF forecasts for all registered Indian cities
and stores the results as Parquet files.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from hybrid_forecast.ingestion.cities import CITIES
from hybrid_forecast.ingestion.open_meteo import fetch_city_forecasts


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_DIR = Path("data/raw/open_meteo")

FORECAST_DAYS = 3

REQUEST_DELAY_SECONDS = 0.5


# =========================================================
# SINGLE-CITY BATCH WORKER
# =========================================================

def process_city(city_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch GFS and ECMWF forecasts for one registered city.
    """

    city = CITIES[city_key]

    city_name = city["name"]

    print(f"\nFetching: {city_name}")

    gfs, ecmwf = fetch_city_forecasts(
        city_name,
        forecast_days=FORECAST_DAYS,
    )

    print(
        f"  GFS:   {len(gfs)} rows"
    )

    print(
        f"  ECMWF: {len(ecmwf)} rows"
    )

    return gfs, ecmwf


# =========================================================
# BATCH INGESTION
# =========================================================

def run_batch_ingestion() -> None:
    """
    Fetch GFS + ECMWF data for all registered cities.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_gfs: list[pd.DataFrame] = []
    all_ecmwf: list[pd.DataFrame] = []

    total_cities = len(CITIES)

    print("=" * 60)
    print("WeatherFusion V2 — 50-City Open-Meteo Ingestion")
    print("=" * 60)

    print(f"\nCities registered: {total_cities}")
    print(f"Forecast horizon: {FORECAST_DAYS} days")

    successful = 0
    failed = 0

    for index, city_key in enumerate(CITIES, start=1):

        city_name = CITIES[city_key]["name"]

        print(
            f"\n[{index}/{total_cities}] {city_name}"
        )

        try:

            gfs, ecmwf = process_city(city_key)

            all_gfs.append(gfs)
            all_ecmwf.append(ecmwf)

            successful += 1

            print("  Status: SUCCESS")

        except Exception as exc:

            failed += 1

            print(
                f"  Status: FAILED — {exc}"
            )

        if index < total_cities:
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # =====================================================
    # COMBINE RESULTS
    # =====================================================

    if not all_gfs or not all_ecmwf:
        raise RuntimeError(
            "No weather data was successfully collected."
        )

    gfs_combined = pd.concat(
        all_gfs,
        ignore_index=True,
    )

    ecmwf_combined = pd.concat(
        all_ecmwf,
        ignore_index=True,
    )

    # =====================================================
    # SAVE
    # =====================================================

    gfs_path = (
        OUTPUT_DIR
        / "gfs_india.parquet"
    )

    ecmwf_path = (
        OUTPUT_DIR
        / "ecmwf_india.parquet"
    )

    gfs_combined.to_parquet(
        gfs_path,
        index=False,
    )

    ecmwf_combined.to_parquet(
        ecmwf_path,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("BATCH INGESTION COMPLETE")
    print("=" * 60)

    print(f"\nSuccessful cities: {successful}")
    print(f"Failed cities:     {failed}")

    print(f"\nGFS total rows:     {len(gfs_combined)}")
    print(f"ECMWF total rows:   {len(ecmwf_combined)}")

    print(f"\nGFS output:")
    print(gfs_path)

    print(f"\nECMWF output:")
    print(ecmwf_path)


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    run_batch_ingestion()