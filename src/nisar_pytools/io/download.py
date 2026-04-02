"""Download NISAR products from ASF with parallel support and validation."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import requests

log = logging.getLogger(__name__)

# Thread-local storage for per-thread sessions
_thread_local = threading.local()


def validate_h5_quick(filepath: Path) -> bool:
    """Quick validation — check file is valid HDF5 and can open."""
    try:
        with h5py.File(filepath, "r") as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False


def validate_h5_thorough(filepath: Path) -> bool:
    """Thorough validation — verify NISAR structure and read a small dataset.

    Checks that the file:
    1. Is valid HDF5
    2. Contains a NISAR science band (LSAR or SSAR)
    3. Contains ``identification/productType``
    4. Can read the product type and access the data group
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Detect band (LSAR or SSAR)
            band = None
            if "science/LSAR" in f:
                band = "LSAR"
            elif "science/SSAR" in f:
                band = "SSAR"
            if band is None:
                log.warning("Missing science/LSAR or science/SSAR: %s", filepath.name)
                return False

            pt_path = f"science/{band}/identification/productType"
            if pt_path not in f:
                log.warning("Missing productType: %s", filepath.name)
                return False

            # Read product type (single read)
            product_type = f[pt_path][()]
            if isinstance(product_type, bytes):
                product_type = product_type.decode().strip()

            # Try to access the data group
            for data_group in ["grids", "swaths"]:
                grids_path = f"science/{band}/{product_type}/{data_group}"
                if grids_path in f:
                    _ = list(f[grids_path].keys())
                    break

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

    Downloads are written to a temporary file first and renamed atomically
    on success, preventing partial/corrupt files from being left on disk.

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
        Paths to successfully downloaded files, sorted alphabetically.
        Files that fail after all retries are excluded (with a warning logged).
    """
    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    # Build URL → filename mapping, warn on duplicates
    url_to_filename: dict[str, str] = {}
    seen_filenames: dict[str, str] = {}
    for url in urls:
        fname = Path(url).name
        if fname in seen_filenames:
            log.warning(
                "Duplicate filename '%s' from URLs:\n  %s\n  %s",
                fname, seen_filenames[fname], url,
            )
        seen_filenames[fname] = url
        url_to_filename[url] = fname

    def _get_session() -> requests.Session:
        """Get a thread-local requests session."""
        if not hasattr(_thread_local, "session"):
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=2,
                pool_maxsize=2,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _thread_local.session = session
        return _thread_local.session

    def _download_one(url: str) -> Path | None:
        out_fp = out_directory / url_to_filename[url]

        if out_fp.exists() and out_fp.stat().st_size > 0 and not reprocess:
            log.debug("Skipping (exists): %s", out_fp.name)
            return out_fp

        session = _get_session()

        for attempt in range(retries):
            # Write to temp file, rename on success (atomic)
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=out_directory, suffix=".tmp"
                )
                r = session.get(url, stream=True, timeout=timeout)
                r.raise_for_status()
                with os.fdopen(tmp_fd, "wb") as f:
                    tmp_fd = None  # fdopen takes ownership
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                # Atomic rename
                os.replace(tmp_path, out_fp)
                tmp_path = None
                log.debug("Downloaded: %s", out_fp.name)
                return out_fp
            except Exception:
                if attempt == retries - 1:
                    log.error("Failed after %d retries: %s", retries, url)
                    return None
                log.warning("Retry %d for %s", attempt + 1, url)
                time.sleep(1.5)
            finally:
                # Clean up temp file on failure
                if tmp_fd is not None:
                    os.close(tmp_fd)
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return None

    # Download all files, catching individual failures
    download_fps: list[Path] = []
    failed_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if result is not None:
                    download_fps.append(result)
                else:
                    failed_urls.append(url)
            except Exception as e:
                log.error("Download failed for %s: %s", url, e)
                failed_urls.append(url)

    if failed_urls:
        log.warning("%d downloads failed: %s", len(failed_urls), failed_urls)

    # Post-download validation
    if validate:
        corrupted: list[Path] = []
        for fp in download_fps:
            if fp.suffix.lower() == ".h5" and not validate_h5_thorough(fp):
                corrupted.append(fp)

        if corrupted:
            log.warning("Found %d corrupted files, retrying...", len(corrupted))

            # Build set for O(1) lookup
            corrupted_set = set(corrupted)
            download_fps = [fp for fp in download_fps if fp not in corrupted_set]

            # Find URLs for corrupted files
            filename_to_url = {Path(url).name: url for url in urls}
            corrupted_urls = [
                filename_to_url[fp.name]
                for fp in corrupted
                if fp.name in filename_to_url
            ]

            # Remove corrupted files from disk
            for fp in corrupted:
                if fp.exists():
                    os.remove(fp)

            # Retry corrupted
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_download_one, url): url
                    for url in corrupted_urls
                }
                for future in as_completed(futures):
                    try:
                        retry_fp = future.result()
                        if retry_fp is not None and validate_h5_thorough(retry_fp):
                            download_fps.append(retry_fp)
                        elif retry_fp is not None:
                            log.warning("Still corrupted after retry: %s", retry_fp.name)
                    except Exception as e:
                        log.warning("Retry failed: %s", e)

    return sorted(download_fps)
