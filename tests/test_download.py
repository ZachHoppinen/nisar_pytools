"""Tests for nisar_pytools.io.download."""

from unittest.mock import MagicMock, patch

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
    @patch("nisar_pytools.io.download.requests.Session")
    def test_downloads_files(self, mock_session_cls, tmp_path):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake data"]
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        urls = ["https://example.com/file1.h5", "https://example.com/file2.h5"]
        fps = download_urls(urls, tmp_path, validate=False)
        assert len(fps) == 2

    def test_skips_existing(self, tmp_path):
        existing = tmp_path / "file.h5"
        existing.write_bytes(b"data")

        with patch("nisar_pytools.io.download.requests.Session"):
            fps = download_urls(["https://example.com/file.h5"], tmp_path, validate=False)
        assert len(fps) == 1
        assert fps[0] == existing

    def test_reprocess_redownloads(self, tmp_path):
        existing = tmp_path / "file.txt"
        existing.write_bytes(b"old data")

        with patch("nisar_pytools.io.download.requests.Session") as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"new data"]
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response

            fps = download_urls(
                ["https://example.com/file.txt"], tmp_path, reprocess=True, validate=False
            )
        assert fps[0].read_bytes() == b"new data"

    def test_creates_directory(self, tmp_path):
        out = tmp_path / "new" / "nested"
        with patch("nisar_pytools.io.download.requests.Session") as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"data"]
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response

            download_urls(["https://example.com/f.txt"], out, validate=False)
        assert out.is_dir()

    def test_returns_sorted(self, tmp_path):
        for name in ["c.h5", "a.h5", "b.h5"]:
            (tmp_path / name).write_bytes(b"x")

        urls = [f"https://example.com/{n}" for n in ["c.h5", "a.h5", "b.h5"]]
        with patch("nisar_pytools.io.download.requests.Session"):
            fps = download_urls(urls, tmp_path, validate=False)

        names = [fp.name for fp in fps]
        assert names == sorted(names)

    def test_validation_passes_valid_h5(self, tmp_path):
        """Download + validate with a real valid HDF5 file."""
        _make_valid_nisar_h5(tmp_path / "valid.h5")
        # File already exists, so download is skipped, but validation runs
        with patch("nisar_pytools.io.download.requests.Session"):
            fps = download_urls(
                ["https://example.com/valid.h5"], tmp_path, validate=True
            )
        assert len(fps) == 1

    def test_validation_catches_bad_h5(self, tmp_path):
        """A corrupted .h5 should be detected during validation."""
        bad = tmp_path / "bad.h5"
        bad.write_text("not hdf5")

        with patch("nisar_pytools.io.download.requests.Session") as mock_cls:
            mock_session = MagicMock()
            mock_cls.return_value = mock_session

            # Retry still produces bad file (can't easily mock real h5 via streaming)
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"still not hdf5"]
            mock_session.get.return_value = mock_response

            fps = download_urls(
                ["https://example.com/bad.h5"], tmp_path, validate=True
            )
        # Bad file should be excluded from results
        assert len(fps) == 0
