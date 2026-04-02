"""Tests for nisar_pytools.io._stack (stack_gslcs)."""

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import pytest

from nisar_pytools.io._stack import stack_gslcs


def _make_gslc_file(
    path,
    time_str="2025-11-03T12:46:15.000000000",
    track=77,
    frame=24,
    ny=8,
    nx=10,
    pols=("HH", "HV"),
    frequency="frequencyA",
    x_offset=0.0,
    y_offset=0.0,
):
    """Create a minimal synthetic GSLC HDF5 file for stacking tests."""
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"

        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        ident.create_dataset("zeroDopplerStartTime", data=time_str.encode())
        ident.create_dataset("trackNumber", data=np.uint32(track))
        ident.create_dataset("frameNumber", data=np.uint16(frame))

        grp = f.create_group(f"science/LSAR/GSLC/grids/{frequency}")
        x = np.arange(nx, dtype="f8") * 100.0 + 500000.0 + x_offset
        y = np.arange(ny, dtype="f8") * 100.0 + 4000000.0 + y_offset
        xds = grp.create_dataset("xCoordinates", data=x)
        yds = grp.create_dataset("yCoordinates", data=y)
        xds.make_scale("xCoordinates")
        yds.make_scale("yCoordinates")

        rng = np.random.default_rng(hash(time_str) % 2**31)
        for pol in pols:
            data = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))).astype(np.complex64)
            ds = grp.create_dataset(pol, data=data, chunks=(4, 4))
            ds.dims[0].attach_scale(yds)
            ds.dims[1].attach_scale(xds)

        grp.create_dataset("projection", data=np.uint32(32611))
    return path


@pytest.fixture
def three_gslcs(tmp_path):
    """Create three GSLC files with different times, same grid."""
    times = [
        "2025-11-03T12:46:15.000000000",
        "2025-11-15T12:46:15.000000000",
        "2025-11-27T12:46:15.000000000",
    ]
    paths = []
    for i, t in enumerate(times):
        p = tmp_path / f"gslc_{i}.h5"
        _make_gslc_file(p, time_str=t)
        paths.append(p)
    return paths, times


class TestStackGSLCs:
    def test_output_shape(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        assert stack.shape == (3, 8, 10)
        assert stack.dims == ("time", "y", "x")

    def test_sorted_by_time(self, three_gslcs):
        paths, times = three_gslcs
        # Pass in reversed order
        stack = stack_gslcs(paths[::-1])
        expected = pd.DatetimeIndex([pd.Timestamp(t) for t in times])
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(stack.time.values), expected
        )

    def test_lazy(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        assert isinstance(stack.data, da.Array)

    def test_coords_from_files(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        x_expected = np.arange(10, dtype="f8") * 100.0 + 500000.0
        y_expected = np.arange(8, dtype="f8") * 100.0 + 4000000.0
        np.testing.assert_array_equal(stack.x.values, x_expected)
        np.testing.assert_array_equal(stack.y.values, y_expected)

    def test_data_computable(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        values = stack.values
        assert values.shape == (3, 8, 10)
        assert np.iscomplexobj(values)

    def test_name_and_attrs(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        assert stack.name == "HH"
        assert stack.attrs["frequency"] == "frequencyA"
        assert stack.attrs["polarization"] == "HH"

    def test_select_polarization(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths, polarization="HV")
        assert stack.name == "HV"

    def test_file_handles_kept(self, three_gslcs):
        paths, _ = three_gslcs
        stack = stack_gslcs(paths)
        handles = stack.attrs["_h5files"]
        assert len(handles) == 3
        assert all(h.id.valid for h in handles)

    def test_single_file(self, tmp_path):
        p = _make_gslc_file(tmp_path / "single.h5")
        stack = stack_gslcs([p])
        assert stack.shape == (1, 8, 10)


class TestStackValidation:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            stack_gslcs([])

    def test_mismatched_x_raises(self, tmp_path):
        p1 = _make_gslc_file(tmp_path / "a.h5", time_str="2025-01-01T00:00:00")
        p2 = _make_gslc_file(
            tmp_path / "b.h5", time_str="2025-01-13T00:00:00", x_offset=50.0
        )
        with pytest.raises(ValueError, match="x coordinates do not match"):
            stack_gslcs([p1, p2])

    def test_mismatched_y_raises(self, tmp_path):
        p1 = _make_gslc_file(tmp_path / "a.h5", time_str="2025-01-01T00:00:00")
        p2 = _make_gslc_file(
            tmp_path / "b.h5", time_str="2025-01-13T00:00:00", y_offset=50.0
        )
        with pytest.raises(ValueError, match="y coordinates do not match"):
            stack_gslcs([p1, p2])

    def test_mismatched_track_raises(self, tmp_path):
        p1 = _make_gslc_file(tmp_path / "a.h5", time_str="2025-01-01T00:00:00", track=77)
        p2 = _make_gslc_file(tmp_path / "b.h5", time_str="2025-01-13T00:00:00", track=99)
        with pytest.raises(ValueError, match="Track numbers do not match"):
            stack_gslcs([p1, p2])

    def test_mismatched_frame_raises(self, tmp_path):
        p1 = _make_gslc_file(tmp_path / "a.h5", time_str="2025-01-01T00:00:00", frame=24)
        p2 = _make_gslc_file(tmp_path / "b.h5", time_str="2025-01-13T00:00:00", frame=25)
        with pytest.raises(ValueError, match="Frame numbers do not match"):
            stack_gslcs([p1, p2])

    def test_skip_track_frame_check(self, tmp_path):
        p1 = _make_gslc_file(tmp_path / "a.h5", time_str="2025-01-01T00:00:00", track=77)
        p2 = _make_gslc_file(tmp_path / "b.h5", time_str="2025-01-13T00:00:00", track=99)
        # Should not raise with check disabled
        stack = stack_gslcs([p1, p2], check_track_frame=False)
        assert stack.shape == (2, 8, 10)

    def test_missing_polarization_raises(self, tmp_path):
        p = _make_gslc_file(tmp_path / "a.h5", pols=("HH",))
        with pytest.raises(ValueError, match="Polarization 'VV' not found"):
            stack_gslcs([p], polarization="VV")

    def test_missing_frequency_raises(self, tmp_path):
        p = _make_gslc_file(tmp_path / "a.h5")
        with pytest.raises(ValueError, match="Frequency group.*not found"):
            stack_gslcs([p], frequency="frequencyC")
