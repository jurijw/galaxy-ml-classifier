"""
Galaxy Zoo Data Filtering Script

This script loads Galaxy Zoo data from a CSV file, filters for galaxies
with reliable classifications, removes outliers in magnitude measurements,
and exports a cleaned, compact table for further analysis.
"""

from pathlib import Path
import pandas as pd


def filter_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Remove rows that fall outside the 1.5×IQR range for any of the given columns.

    Args:
        df: Input DataFrame to filter.
        columns: Names of columns to apply the IQR filter to.

    Returns:
        Filtered DataFrame with outliers removed.
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        col_mask = df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        mask &= col_mask
    return df[mask]


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load Galaxy Zoo data from a CSV file.

    Args:
        filepath: Path to the input CSV file containing Galaxy Zoo data.

    Returns:
        The raw, unfiltered Galaxy Zoo DataFrame.
    """
    df_raw = pd.read_csv(filepath)
    print(f"Loaded {len(df_raw):,} objects from '{filepath.name}'.")
    return df_raw


def filter_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filtering steps to the raw Galaxy Zoo data.

    Args:
        df_raw: The unfiltered Galaxy Zoo DataFrame.

    Returns:
        A cleaned and filtered DataFrame containing galaxy information.
    """
    filters = [
        "nvote > 20",                    # Assert a minimum vote threshold.
        "type == 3",                     # Ensure object is a galaxy.
        "insideMask == False",           # Remove if in problematic survey regions.
        "uncertain != 1",                # Exclude uncertain classifications.
        "(p_el > 0.9 or p_cs > 0.9)",    # Keep confidently classified objects.
    ]

    df = df_raw.query(" and ".join(filters))

    model_mag_cols = [
        "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
    ]
    psf_mag_cols = [
        "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
    ]

    # Apply IQR-based outlier removal on magnitude columns.
    df = filter_iqr(df, model_mag_cols)
    df = filter_iqr(df, psf_mag_cols)

    # Drop rows with duplicate objids (unique identifier).
    df = df.drop_duplicates(subset="objid", keep="first")

    # Filter out malformed objids. These should be 19-digit numbers.
    df = df[df["objid"].apply(lambda x: len(str(x)) == 19)]

    # Keep only relevant columns.
    df_final = df[["objid", "ra", "dec", "petroR90_r", "spiral", "elliptical"]]

    # Sanity check: spiral + elliptical should match total number of rows.
    assert (
        df_final["spiral"].sum() + df_final["elliptical"].sum() == len(df_final)
    ), "Mismatch between spiral and elliptical counts."

    return df_final


def main() -> None:
    """
    Main entry point for the script.

    Loads the raw Galaxy Zoo dataset, applies filtering, and saves the
    cleaned result to a new CSV file.
    """
    datasets_dir = Path("tables")
    input_file = datasets_dir / "ZooSpecPhotoDR19.csv"
    output_file = datasets_dir / "ZooSpecPhotoDR19_filtered.csv"

    df_raw = load_data(input_file)
    df_final = filter_data(df_raw)
    df_final.to_csv(output_file, index=False)

    print(f"Filtered data saved to: {output_file}")
    print(f"Final dataset contains {len(df_final):,} galaxies.")


if __name__ == "__main__":
    main()
