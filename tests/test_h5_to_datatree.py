"""Tests for nisar_pytools.io.h5_to_datatree."""

import dask.array as da
import h5py
import numpy as np
import pytest

from nisar_pytools.io.h5_to_datatree import h5_to_datatree


class TestGSLCDataTree:
    def test_tree_structure(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        # Key groups should exist as tree nodes
        assert "science" in dt.children
        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"]
        freq_b = dt["science/LSAR/GSLC/grids/frequencyB"]
        assert freq_a is not None
        assert freq_b is not None

    def test_lazy_arrays(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert "HH" in freq_a
        assert "HV" in freq_a
        # Data must be dask-backed
        assert isinstance(freq_a["HH"].data, da.Array)
        assert isinstance(freq_a["HV"].data, da.Array)

    def test_coordinates_assigned(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert "x" in freq_a.coords
        assert "y" in freq_a.coords
        assert freq_a["HH"].dims == ("y", "x")
        assert len(freq_a.coords["x"]) == 10
        assert len(freq_a.coords["y"]) == 8

    def test_different_frequency_resolutions(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        freq_b = dt["science/LSAR/GSLC/grids/frequencyB"].dataset
        assert len(freq_a.coords["x"]) == 10
        assert len(freq_b.coords["x"]) == 5

    def test_scalars_as_attrs(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        attrs = dt["science/LSAR/GSLC/grids/frequencyA"].dataset.attrs
        assert "centerFrequency" in attrs
        assert attrs["centerFrequency"] == pytest.approx(1.257e9)
        assert isinstance(attrs["centerFrequency"], float)

    def test_string_datasets_decoded(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        ident = dt["science/LSAR/identification"].dataset.attrs
        assert ident["productType"] == "GSLC"
        assert isinstance(ident["productType"], str)

    def test_list_datasets_decoded(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        attrs = dt["science/LSAR/GSLC/grids/frequencyA"].dataset.attrs
        assert "listOfPolarizations" in attrs
        assert attrs["listOfPolarizations"] == ["HH", "HV"]

    def test_data_var_attrs_preserved(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        hh = dt["science/LSAR/GSLC/grids/frequencyA"].dataset["HH"]
        assert "description" in hh.attrs
        assert hh.attrs["description"] == "Focused SLC image (HH)"

    def test_root_attrs(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f)

        assert dt.dataset.attrs["Conventions"] == "CF-1.7"


class TestGUNWDataTree:
    def test_tree_structure(self, gunw_h5):
        with h5py.File(gunw_h5, "r") as f:
            dt = h5_to_datatree(f)

        base = "science/LSAR/GUNW/grids/frequencyA"
        assert dt[f"{base}/unwrappedInterferogram/HH"] is not None
        assert dt[f"{base}/wrappedInterferogram/HH"] is not None
        assert dt[f"{base}/pixelOffsets/HH"] is not None

    def test_multi_resolution(self, gunw_h5):
        with h5py.File(gunw_h5, "r") as f:
            dt = h5_to_datatree(f)

        base = "science/LSAR/GUNW/grids/frequencyA"
        unwrapped = dt[f"{base}/unwrappedInterferogram/HH"].dataset
        wrapped = dt[f"{base}/wrappedInterferogram/HH"].dataset

        # Different resolutions
        assert unwrapped["unwrappedPhase"].shape == (6, 8)
        assert wrapped["wrappedInterferogram"].shape == (12, 16)

        # Each has its own coordinates
        assert len(unwrapped.coords["x"]) == 8
        assert len(wrapped.coords["x"]) == 16

    def test_lazy_arrays(self, gunw_h5):
        with h5py.File(gunw_h5, "r") as f:
            dt = h5_to_datatree(f)

        base = "science/LSAR/GUNW/grids/frequencyA"
        ds = dt[f"{base}/unwrappedInterferogram/HH"].dataset
        assert isinstance(ds["unwrappedPhase"].data, da.Array)
        assert isinstance(ds["coherenceMagnitude"].data, da.Array)

    def test_coordinates_assigned(self, gunw_h5):
        with h5py.File(gunw_h5, "r") as f:
            dt = h5_to_datatree(f)

        base = "science/LSAR/GUNW/grids/frequencyA"
        ds = dt[f"{base}/pixelOffsets/HH"].dataset
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert ds["alongTrackOffset"].dims == ("y", "x")


class TestChunking:
    def test_auto_uses_h5_chunks(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f, chunks="auto")

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        # HDF5 chunks are (4, 4), so dask chunks should match
        assert freq_a["HH"].data.chunksize == (4, 4)

    def test_explicit_chunks(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f, chunks={"y": 2, "x": 5})

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert freq_a["HH"].data.chunksize == (2, 5)

    def test_none_loads_eagerly(self, gslc_h5):
        with h5py.File(gslc_h5, "r") as f:
            dt = h5_to_datatree(f, chunks=None)

        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert isinstance(freq_a["HH"].data, np.ndarray)


class TestEdgeCases:
    def test_named_type_skipped(self, tmp_path):
        """Ensure h5py.Datatype objects at root don't cause errors."""
        path = tmp_path / "with_named_type.h5"
        with h5py.File(path, "w") as f:
            f["mytype"] = np.dtype([("r", "f4"), ("i", "f4")])
            grp = f.create_group("data")
            grp.create_dataset("value", data=42.0)

        with h5py.File(path, "r") as f:
            dt = h5_to_datatree(f)
        # Should not raise, named type should be silently skipped
        assert "data" in dt.children

    def test_empty_group_skipped(self, tmp_path):
        """Groups with only subgroups produce no dataset."""
        path = tmp_path / "nested.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("a/b/c")
            grp.create_dataset("val", data=1.0)

        with h5py.File(path, "r") as f:
            dt = h5_to_datatree(f)
        assert "a" in dt.children
        assert dt["a/b/c"] is not None
