"""Download NISAR products from ASF with parallel support and validation."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import requests

log = logging.getLogger(__name__)


def validate_h5_quick(filepath: Path) -> bool:
    """Quick validation — check file is valid HDF5 and can open."""
    try:
        with h5py.File(filepath, "r") as f:
            # Verify it has at least the top-level group
            _ = list(f.keys())
        return True
    except Exception:
        return False


def validate_h5_thorough(filepath: Path) -> bool:
    """Thorough validation — verify NISAR structure and read a small dataset.

    Checks that the file:
    1. Is valid HDF5
    2. Contains ``science/LSAR/``
    3. Contains ``identification/productType``
    4. Can read the product type dataset (catches truncated files)
    """
    try:
        with h5py.File(filepath, "r") as f:
            if "science/LSAR" not in f:
                log.warning("Missing science/LSAR: %s", filepath.name)
                return False

            pt_path = "science/LSAR/identification/productType"
            if pt_path not in f:
                log.warning("Missing productType: %s", filepath.name)
                return False

            # Actually read a value to catch truncated files
            _ = f[pt_path][()]

            # Try to access the grids group to verify data section exists
            product_type = f[pt_path][()]
            if isinstance(product_type, bytes):
                product_type = product_type.decode().strip()

            grids_path = f"science/LSAR/{product_type}/grids"
            if grids_path in f:
                grp = f[grids_path]
                # Read one key to verify data is accessible
                _ = list(grp.keys())

        return True
    except Exception as e:
        log.warning("Validation failed for %s: %s", filepath.name, e)
        return False


def download_urls(
    urls: list[str],
    out_directory: str | Path,
    reprocess: bool = False,
    max_workers: int = 4,
    retries: int = 3,
    timeout: int = 60,
    validate: bool = True,
) -> list[Path]:
    """Download files from a list of URLs in parallel with validation.

    Uses connection pooling and multithreading for fast downloads.
    Skips files that already exist (unless ``reprocess=True``).
    Optionally validates downloaded HDF5 files and retries corrupted ones.

    Parameters
    ----------
    urls : list of str
        Download URLs (typically from :func:`find_nisar`).
    out_directory : str or Path
        Directory to save files to. Created if it does not exist.
    reprocess : bool
        If ``True``, re-download files even if they already exist.
    max_workers : int
        Number of parallel download threads. Default 4.
    retries : int
        Number of retry attempts per URL on failure. Default 3.
    timeout : int
        Request timeout in seconds. Default 60.
    validate : bool
        If ``True`` (default), validate downloaded ``.h5`` files and retry
        corrupted ones.

    Returns
    -------
    list of Path
        Paths to downloaded files, sorted alphabetically.
    """
    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def _download_one(url: str) -> Path:
        out_fp = out_directory / Path(url).name

        if out_fp.exists() and out_fp.stat().st_size > 0 and not reprocess:
            log.debug("Skipping (exists): %s", out_fp.name)
            return out_fp

        for attempt in range(retries):
            try:
                r = session.get(url, stream=True, timeout=timeout)
                r.raise_for_status()
                with open(out_fp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                log.debug("Downloaded: %s", out_fp.name)
                return out_fp
            except Exception:
                if attempt == retries - 1:
                    raise
                log.warning("Retry %d for %s", attempt + 1, url)
                time.sleep(1.5)

        return out_fp

    # Download all files
    download_fps: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, url): url for url in urls}
        for future in as_completed(futures):
            download_fps.append(future.result())

    # Post-download validation
    if validate:
        url_by_name = {Path(url).name: url for url in urls}
        corrupted = []

        for fp in download_fps:
            if fp.suffix.lower() == ".h5" and not validate_h5_thorough(fp):
                corrupted.append(fp)

        if corrupted:
            log.warning("Found %d corrupted files, retrying...", len(corrupted))
            for fp in corrupted:
                download_fps.remove(fp)
                if fp.exists():
                    os.remove(fp)

            # Retry corrupted
            corrupted_urls = [url_by_name[fp.name] for fp in corrupted if fp.name in url_by_name]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_one, url): url for url in corrupted_urls}
                for future in as_completed(futures):
                    retry_fp = future.result()
                    if validate_h5_thorough(retry_fp):
                        download_fps.append(retry_fp)
                    else:
                        log.warning("Still corrupted after retry: %s", retry_fp.name)

    return sorted(download_fps)
