"""
Time alignment for the Hybrid AI-NWP Multi-Model Forecast Blending System.

Combines:
    1. GFS forecasts
    2. ECMWF forecasts
    3. Independent weather observations

using:
    city_name + lat + lon + valid_time

Output:
    data/processed/aligned_forecasts.parquet
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GFS_FILE = RAW_DIR / "gfs_delhi.parquet"
ECMWF_FILE = RAW_DIR / "ecmwf_delhi.parquet"
OBS_FILE = RAW_DIR / "observations_hourly.parquet"

OUTPUT_FILE = PROCESSED_DIR / "aligned_forecasts.parquet"


# ---------------------------------------------------------
# Columns used to identify the same weather record
# ---------------------------------------------------------

KEY_COLUMNS = [
    "city_name",
    "lat",
    "lon",
    "valid_time",
]


# ---------------------------------------------------------
# Weather variables
# ---------------------------------------------------------

WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_u_10m",
    "wind_v_10m",
    "cloud_cover",
    "dew_point_2m",
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load GFS, ECMWF and observation datasets."""

    print("Loading datasets...")

    gfs = pd.read_parquet(GFS_FILE)
    ecmwf = pd.read_parquet(ECMWF_FILE)
    observations = pd.read_parquet(OBS_FILE)

    print(f"GFS rows:          {len(gfs):,}")
    print(f"ECMWF rows:        {len(ecmwf):,}")
    print(f"Observation rows:  {len(observations):,}")

    return gfs, ecmwf, observations


# ---------------------------------------------------------
# Prepare NWP forecasts
# ---------------------------------------------------------

def prepare_nwp(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Keep forecast variables from one NWP model.

    The observed_* columns already present inside the synthetic
    NWP files are deliberately ignored. The independent
    observations dataset will provide the target values.
    """

    columns = KEY_COLUMNS + WEATHER_VARIABLES

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{model_name} dataset is missing columns: {missing}"
        )

    result = df[columns].copy()

    # Add model prefix to forecast variables.
    rename_map = {
        column: f"{model_name}_{column}"
        for column in WEATHER_VARIABLES
    }

    result = result.rename(columns=rename_map)

    return result


# ---------------------------------------------------------
# Prepare observations
# ---------------------------------------------------------

def prepare_observations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare independent observation data."""

    columns = KEY_COLUMNS + WEATHER_VARIABLES

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Observation dataset is missing columns: {missing}"
        )

    result = df[columns].copy()

    # Prefix observations so they are clearly targets.
    rename_map = {
        column: f"observed_{column}"
        for column in WEATHER_VARIABLES
    }

    result = result.rename(columns=rename_map)

    return result


# ---------------------------------------------------------
# Remove duplicate records
# ---------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Remove duplicate records using the alignment keys."""

    before = len(df)

    df = df.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="first",
    )

    after = len(df)

    print(
        f"{name}: removed {before - after:,} duplicate rows"
    )

    return df


# ---------------------------------------------------------
# Align datasets
# ---------------------------------------------------------

def align_datasets(
    gfs: pd.DataFrame,
    ecmwf: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align GFS, ECMWF and observations using the same
    location and valid forecast time.
    """

    print("\nPreparing GFS...")
    gfs = prepare_nwp(gfs, "gfs")

    print("Preparing ECMWF...")
    ecmwf = prepare_nwp(ecmwf, "ecmwf")

    print("Preparing observations...")
    observations = prepare_observations(observations)

    # Remove duplicates before merging.
    gfs = remove_duplicates(gfs, "GFS")
    ecmwf = remove_duplicates(ecmwf, "ECMWF")
    observations = remove_duplicates(
        observations,
        "Observations",
    )

    # Merge GFS and ECMWF.
    print("\nAligning GFS + ECMWF...")

    aligned = pd.merge(
        gfs,
        ecmwf,
        on=KEY_COLUMNS,
        how="inner",
    )

    print(
        f"GFS + ECMWF matched rows: {len(aligned):,}"
    )

    # Add observations.
    print("Adding observations...")

    aligned = pd.merge(
        aligned,
        observations,
        on=KEY_COLUMNS,
        how="inner",
    )

    print(
        f"Final aligned rows: {len(aligned):,}"
    )

    return aligned


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the aligned dataset."""

    print("\n========== VALIDATION ==========")

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    print("\nModels represented:")

    # Model is known from the separate forecast columns.
    print("GFS + ECMWF + observations")

    print("\nColumns:")

    for column in df.columns:
        print(f"  {column}")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:
    """Run the complete alignment pipeline."""

    print("==============================================")
    print(" Hybrid AI-NWP Forecast Alignment")
    print("==============================================")

    # Make sure output directory exists.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Check input files.
    for file in [GFS_FILE, ECMWF_FILE, OBS_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Input file not found: {file}"
            )

    # Load.
    gfs, ecmwf, observations = load_data()

    # Align.
    aligned = align_datasets(
        gfs,
        ecmwf,
        observations,
    )

    # Validate.
    validate_dataset(aligned)

    # Save.
    aligned.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================================")
    print("Alignment completed successfully!")
    print(f"Output: {OUTPUT_FILE}")
    print("==============================================")


if __name__ == "__main__":
    main()