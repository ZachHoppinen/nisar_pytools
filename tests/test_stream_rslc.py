"""Tests for nisar_pytools.io.stream_rslc.

The remote handle needs Earthdata and a real granule, so it is not unit-tested
here. What is testable offline is everything the streaming path relies on: the
skeleton writer, and that cropping from a handle matches cropping from a path.
"""

import h5py
import numpy as np
import pytest

from nisar_pytools.io.stream_rslc import write_skeleton
from nisar_pytools.processing.crop_rslc import crop_rslc, crop_rslc_from_handle
from tests.test_crop_rslc import _SWATHS, EXPECTED_WINDOW, N_LINES, _make_rslc


@pytest.fixture
def quad_h5(tmp_path):
    path = tmp_path / "quad_rslc.h5"
    _make_rslc(path, pols=("HH", "HV", "VH", "VV"), n_subswaths=4)
    return path


class TestWriteSkeleton:
    def test_images_are_empty_but_shaped(self, quad_h5, tmp_path):
        skel = tmp_path / "skeleton.h5"
        with h5py.File(quad_h5, "r") as src:
            write_skeleton(src, skel)

        with h5py.File(quad_h5, "r") as src, h5py.File(skel, "r") as out:
            for pol in ("HH", "HV", "VH", "VV"):
                path = f"{_SWATHS}/frequencyA/{pol}"
                assert out[path].shape == src[path].shape
                assert out[path].dtype == src[path].dtype
                # No pixels were written, so the image reads back as zeros.
                assert not out[path][()].any()

    def test_skeleton_is_small(self, quad_h5, tmp_path):
        """The point of the skeleton: structure without the imagery."""
        skel = tmp_path / "skeleton.h5"
        with h5py.File(quad_h5, "r") as src:
            write_skeleton(src, skel)
        assert skel.stat().st_size < quad_h5.stat().st_size

    def test_axes_and_metadata_survive(self, quad_h5, tmp_path):
        """The window solve reads these, so they must be real, not empty."""
        skel = tmp_path / "skeleton.h5"
        with h5py.File(quad_h5, "r") as src:
            write_skeleton(src, skel)

        with h5py.File(quad_h5, "r") as src, h5py.File(skel, "r") as out:
            assert out.attrs["mission_name"] == src.attrs["mission_name"]
            for path in (f"{_SWATHS}/zeroDopplerTime",
                         f"{_SWATHS}/frequencyA/slantRange",
                         f"{_SWATHS}/frequencyA/validSamplesSubSwath4",
                         "science/LSAR/RSLC/metadata/orbit/position"):
                np.testing.assert_array_equal(out[path][()], src[path][()])
            assert out[f"{_SWATHS}/zeroDopplerTime"].attrs["units"] == "seconds"

    def test_full_azimuth_axis_kept(self, quad_h5, tmp_path):
        skel = tmp_path / "skeleton.h5"
        with h5py.File(quad_h5, "r") as src:
            write_skeleton(src, skel)
        with h5py.File(skel, "r") as out:
            assert out[f"{_SWATHS}/zeroDopplerTime"].shape == (N_LINES,)


class TestCropFromHandle:
    def test_matches_path_based_crop(self, quad_h5, tmp_path):
        """Streaming only changes where bytes come from, not the output."""
        from_path = tmp_path / "from_path.h5"
        from_handle = tmp_path / "from_handle.h5"

        crop_rslc(quad_h5, from_path, EXPECTED_WINDOW)
        with h5py.File(quad_h5, "r") as src:
            crop_rslc_from_handle(src, from_handle, EXPECTED_WINDOW)

        with h5py.File(from_path, "r") as a, h5py.File(from_handle, "r") as b:
            paths = []
            a.visititems(lambda name, obj: paths.append(name)
                         if isinstance(obj, h5py.Dataset) else None)
            assert paths, "fixture produced no datasets"
            for path in paths:
                np.testing.assert_array_equal(a[path][()], b[path][()])
