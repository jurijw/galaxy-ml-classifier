"""
Galaxy Zoo Image Downloader (Recursive Version)

This script loads a filtered Galaxy Zoo dataset, requests SDSS images
for each galaxy using astroquery.hips2fits, and saves the images
locally in directories according to galaxy type.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from datetime import datetime

import numpy as np
import pandas as pd
from astroquery.hips2fits import hips2fitsClass
from astropy import units as u
from tqdm import tqdm
from PIL import Image


# Set a base directory to store images in.
BASE_IMAGE_PATH = Path("images")


def create_image_dirs(base_path: Path = BASE_IMAGE_PATH) -> None:
    """Create image directories for each galaxy type.

    Args:
        base_path: Base path under which to create subdirectories.
    """
    for label in ("elliptical", "spiral"):
        (base_path / label).mkdir(parents=True, exist_ok=True)


def log(msg: str, path: Path = Path("log.txt")) -> None:
    """Write a timestamped message to a log file.

    Args:
        msg: Message to log.
        path: Path to the log file.
    """
    with open(path, "a") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def load_dataframe(filtered_data_path: Path, download_table_path: Path) -> pd.DataFrame:
    """Load the filtered Galaxy Zoo dataset.

    If a previous download table exists, resume from it; otherwise,
    create the downloaded status columns.

    Args:
        filtered_data_path: Path to the filtered CSV file.
        download_table_path: Path to the download table CSV file.

    Returns:
        DataFrame containing galaxy data with download tracking columns.
    """
    if download_table_path.exists():
        df = pd.read_csv(download_table_path)
    else:
        df = pd.read_csv(filtered_data_path)
        df['downloaded'] = False
        df['num_download_attempts'] = 0
    print(f"Loaded dataframe with {len(df):,} rows.")
    return df


def request_jpg(ra: float, dec: float, fov: float) -> np.ndarray:
    """Request a JPEG image from the SDSS survey via hips2fits.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        fov: Field of view in degrees.

    Returns:
        NumPy array containing the JPEG image.
    """
    hips = hips2fitsClass()
    jpg = hips.query(
        hips="sdss9/color",
        ra=ra * u.deg,
        dec=dec * u.deg,
        fov=fov * u.deg,
        width=64,
        height=64,
        projection="tan",
        coordsys="icrs",
        format="jpg"
    )
    return jpg


def request_jpg_from_row(df: pd.DataFrame, i: int) -> tuple[int, np.ndarray]:
    """Request a JPEG image for a row in the DataFrame.

    Args:
        df: DataFrame containing galaxy data.
        i: Row index to request an image for.

    Returns:
        Tuple containing the galaxy objid and the JPEG image as a NumPy array.
    """
    row = df.iloc[i]
    ra, dec, objid, petroR90_r = row['ra'], row['dec'], row['objid'], row['petroR90_r']
    fov = petroR90_r * 2.0 / 3600.0
    jpg = request_jpg(ra, dec, fov)
    return int(objid), jpg


def downloaded(df: pd.DataFrame, i: int) -> bool:
    """Check if a row has already been downloaded.

    Args:
        df: DataFrame containing download information.
        i: Row index.

    Returns:
        True if the image has already been downloaded, False otherwise.
    """
    return df['downloaded'].iloc[i]


def save_image(df: pd.DataFrame, i: int, retries: int = 10, base_path: Path = BASE_IMAGE_PATH) -> None:
    """Save a galaxy image locally with recursive retries.

    Args:
        df: DataFrame with galaxy data.
        i: Index of the row to download.
        retries: Remaining number of retries for this row.
        base_path: Base path where images will be saved.
    """
    if retries == 0:
        log(f"Unable to download image for objid {df['objid'].iloc[i]} at index {i}")
        return

    if downloaded(df, i):
        return

    try:
        df.loc[i, 'num_download_attempts'] += 1
        objid, jpg = request_jpg_from_row(df, i)
        label = 'elliptical' if df.loc[i, 'elliptical'] else 'spiral'
        save_path = base_path / label / f"{objid}.jpg"
        Image.fromarray(jpg).save(save_path)
        df.loc[i, 'downloaded'] = True
    except Exception as e:
        log(f"Exception downloading objid {df.loc[i, 'objid']} at index {i}, "
            f"attempt {df.loc[i, 'num_download_attempts']}: {e}")
        sleep(1)
        save_image(df, i, retries - 1)


def download_images(df: pd.DataFrame, max_workers: int = 50) -> None:
    """Download images for all galaxies in the DataFrame using multithreading.

    Args:
        df: DataFrame containing galaxy data and download tracking.
        max_workers: Maximum number of threads to use.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda i: save_image(df, i), df.index),
                  total=len(df),
                  desc="Downloading images"))
    print("Finished downloading images.")


def main() -> None:
    """Main entry point to create directories and download all images."""
    create_image_dirs()

    filtered_data_path = Path("tables/ZooSpecPhotoDR19_filtered.csv")
    download_table_path = Path("tables/download_table.csv")

    df = load_dataframe(filtered_data_path, download_table_path)
    download_images(df)

    failed = df[df['downloaded'] == False]
    if not failed.empty:
        print(f"{len(failed)} images failed to download:")
        print(failed)
    else:
        print("All images downloaded successfully.")

    df.to_csv(download_table_path, index=False)


if __name__ == "__main__":
    main()
