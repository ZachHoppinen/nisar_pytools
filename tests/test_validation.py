"""Tests for nisar_pytools.utils.validation."""

import h5py
import pytest

from nisar_pytools.utils.validation import (
    detect_product_type,
    validate_nisar_hdf5,
)


def _make_nisar_h5(path, product_type="GSLC"):
    """Create a minimal valid NISAR HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=product_type.encode())
    return path


class TestValidateNisarHdf5:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_nisar_hdf5(tmp_path / "nonexistent.h5")

    def test_directory_not_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_nisar_hdf5(tmp_path)

    def test_not_hdf5(self, tmp_path):
        path = tmp_path / "fake.h5"
        path.write_text("not an hdf5 file")
        with pytest.raises(ValueError, match="Not a valid HDF5 file"):
            validate_nisar_hdf5(path)

    def test_hdf5_missing_science_lsar(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_group("some/other/group")
        with pytest.raises(ValueError, match="Missing required group"):
            validate_nisar_hdf5(path)

    def test_hdf5_missing_product_type(self, tmp_path):
        path = tmp_path / "no_type.h5"
        with h5py.File(path, "w") as f:
            f.create_group("science/LSAR/identification")
        with pytest.raises(ValueError, match="Missing.*productType"):
            validate_nisar_hdf5(path)

    def test_valid_gslc(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "gslc.h5", "GSLC")
        h5file = validate_nisar_hdf5(path)
        try:
            assert isinstance(h5file, h5py.File)
            assert h5file.id.valid
        finally:
            h5file.close()

    def test_valid_gunw(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "gunw.h5", "GUNW")
        h5file = validate_nisar_hdf5(path)
        try:
            assert isinstance(h5file, h5py.File)
            assert h5file.id.valid
        finally:
            h5file.close()

    def test_accepts_string_path(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "str.h5", "GSLC")
        h5file = validate_nisar_hdf5(str(path))
        try:
            assert h5file.id.valid
        finally:
            h5file.close()


class TestDetectProductType:
    def test_gslc(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "gslc.h5", "GSLC")
        with h5py.File(path, "r") as f:
            assert detect_product_type(f) == "GSLC"

    def test_gunw(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "gunw.h5", "GUNW")
        with h5py.File(path, "r") as f:
            assert detect_product_type(f) == "GUNW"

    def test_unsupported_type(self, tmp_path):
        path = _make_nisar_h5(tmp_path / "rslc.h5", "RSLC")
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="Unsupported product type"):
                detect_product_type(f)

    def test_missing_dataset(self, tmp_path):
        path = tmp_path / "no_type.h5"
        with h5py.File(path, "w") as f:
            f.create_group("science/LSAR/identification")
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="Missing"):
                detect_product_type(f)
