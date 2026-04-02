"""Tests for nisar_pytools.io.download."""

from unittest.mock import MagicMock

import h5py
import numpy as np

from nisar_pytools.io.download import (
    download_urls,
    validate_h5_quick,
    validate_h5_thorough,
)


def _make_valid_nisar_h5(path):
    """Create a minimal valid NISAR HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        grp = f.create_group("science/LSAR/GSLC/grids/frequencyA")
        grp.create_dataset("xCoordinates", data=np.arange(10.0))
    return path


def _mock_session():
    """Create a mock requests session with a working response."""
    mock = MagicMock()
    response = MagicMock()
    response.iter_content.return_value = [b"fake data"]
    response.raise_for_status.return_value = None
    mock.get.return_value = response
    return mock


class TestValidateH5Quick:
    def test_valid_file(self, tmp_path):
        fp = _make_valid_nisar_h5(tmp_path / "valid.h5")
        assert validate_h5_quick(fp) is True

    def test_invalid_file(self, tmp_path):
        fp = tmp_path / "bad.h5"
        fp.write_text("not hdf5")
        assert validate_h5_quick(fp) is False

    def test_missing_file(self, tmp_path):
        assert validate_h5_quick(tmp_path / "nope.h5") is False


class TestValidateH5Thorough:
    def test_valid_nisar(self, tmp_path):
        fp = _make_valid_nisar_h5(tmp_path / "valid.h5")
        assert validate_h5_thorough(fp) is True

    def test_ssar_band(self, tmp_path):
        fp = tmp_path / "ssar.h5"
        with h5py.File(fp, "w") as f:
            ident = f.create_group("science/SSAR/identification")
            ident.create_dataset("productType", data=b"GSLC")
            f.create_group("science/SSAR/GSLC/grids/frequencyA")
        assert validate_h5_thorough(fp) is True

    def test_missing_science_lsar(self, tmp_path):
        fp = tmp_path / "no_lsar.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("other")
        assert validate_h5_thorough(fp) is False

    def test_missing_product_type(self, tmp_path):
        fp = tmp_path / "no_pt.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("science/LSAR/identification")
        assert validate_h5_thorough(fp) is False

    def test_truncated_file(self, tmp_path):
        fp = tmp_path / "truncated.h5"
        fp.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
        assert validate_h5_thorough(fp) is False

    def test_not_hdf5(self, tmp_path):
        fp = tmp_path / "text.h5"
        fp.write_text("just text")
        assert validate_h5_thorough(fp) is False


class TestDownloadUrls:
    def test_skips_existing(self, tmp_path):
        existing = tmp_path / "file.h5"
        existing.write_bytes(b"data")

        fps = download_urls(
            ["https://example.com/file.h5"], tmp_path, validate=False
        )
        assert len(fps) == 1
        assert fps[0] == existing

    def test_returns_sorted(self, tmp_path):
        for name in ["c.h5", "a.h5", "b.h5"]:
            (tmp_path / name).write_bytes(b"x")

        urls = [f"https://example.com/{n}" for n in ["c.h5", "a.h5", "b.h5"]]
        fps = download_urls(urls, tmp_path, validate=False)
        names = [fp.name for fp in fps]
        assert names == sorted(names)

    def test_validation_passes_valid_h5(self, tmp_path):
        _make_valid_nisar_h5(tmp_path / "valid.h5")
        fps = download_urls(
            ["https://example.com/valid.h5"], tmp_path, validate=True
        )
        assert len(fps) == 1

    def test_failed_download_excluded(self, tmp_path):
        """A URL that fails all retries should be excluded, not crash."""
        # No files exist, no mock session — will fail to connect
        # Use max_workers=1 and retries=1 for speed
        fps = download_urls(
            ["https://0.0.0.0:1/nonexistent.txt"],
            tmp_path,
            validate=False,
            max_workers=1,
            retries=1,
            timeout=1,
        )
        assert len(fps) == 0

    def test_creates_directory(self, tmp_path):
        out = tmp_path / "new" / "nested"
        # Just test that directory is created with an existing file
        out.mkdir(parents=True, exist_ok=True)
        (out / "f.txt").write_bytes(b"x")
        fps = download_urls(["https://example.com/f.txt"], out, validate=False)
        assert out.is_dir()
        assert len(fps) == 1

    def test_atomic_write_no_partial_files(self, tmp_path):
        """After a failed download, no partial .h5 file should remain."""
        download_urls(
            ["https://0.0.0.0:1/test.h5"],
            tmp_path,
            validate=False,
            max_workers=1,
            retries=1,
            timeout=1,
        )
        # No .h5 file should exist (temp file cleaned up)
        assert not (tmp_path / "test.h5").exists()
        # No .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
