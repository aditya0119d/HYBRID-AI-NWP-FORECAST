"""
WeatherFusion V2 — Multi-city historical alignment.

Builds aligned GFS + ECMWF + ERA5 training data
for all registered WeatherFusion cities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hybrid_forecast.ingestion.cities import CITIES
from hybrid_forecast.alignment.historical_alignment import (
    build_city_training_data,
    validate_alignment,
    save_training_data,
)


START_DATE = "2025-09-01"
END_DATE = "2025-09-03"

OUTPUT_DIR = Path("data/processed/historical")


def city_slug(city_name: str) -> str:
    """Convert city name into a filesystem-safe slug."""

    return (
        city_name
        .strip()
        .lower()
        .replace(" ", "_")
    )


def build_all_cities() -> None:
    """Build historical aligned datasets for all registered cities."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = []
    failed = []

    print("=" * 60)
    print("WeatherFusion V2 — Multi-City Historical Alignment")
    print("=" * 60)

    print(f"\nCities: {len(CITIES)}")
    print(f"Period: {START_DATE} → {END_DATE}")

    for index, city in enumerate(CITIES.values(), start=1):

        city_name = city["name"]

        print("\n" + "-" * 60)
        print(f"[{index}/{len(CITIES)}] {city_name}")
        print("-" * 60)

        try:
            print("Building GFS + ECMWF aligned dataset...")

            aligned = build_city_training_data(
                city_name,
                START_DATE,
                END_DATE,
            )

            validate_alignment(aligned)

            filename = (
                f"{city_slug(city_name)}_aligned.parquet"
            )

            output_path = OUTPUT_DIR / filename

            save_training_data(
                aligned,
                output_path,
            )

            print(f"Rows: {len(aligned)}")
            print(f"Saved: {output_path}")
            print("Status: SUCCESS")

            successful.append(city_name)

        except Exception as exc:
            print(f"Status: FAILED — {exc}")
            failed.append(city_name)

    print("\n" + "=" * 60)
    print("MULTI-CITY HISTORICAL ALIGNMENT COMPLETE")
    print("=" * 60)

    print(f"\nSuccessful cities: {len(successful)}")
    print(f"Failed cities:     {len(failed)}")

    if failed:
        print("\nFailed cities:")
        for city in failed:
            print(f"  - {city}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    build_all_cities()
    