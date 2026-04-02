"""Tests for nisar_pytools.utils.search_validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from nisar_pytools.utils.search_validation import (
    validate_aoi,
    validate_dates,
    validate_path,
    validate_urls,
)


class TestValidateDates:
    def test_valid_strings(self):
        start, end = validate_dates("2025-01-01", "2025-06-01")
        assert start == pd.Timestamp("2025-01-01")
        assert end == pd.Timestamp("2025-06-01")

    def test_valid_timestamps(self):
        s = pd.Timestamp("2025-03-01")
        e = pd.Timestamp("2025-04-01")
        start, end = validate_dates(s, e)
        assert start == s
        assert end == e

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_dates("2025-06-01", "2025-01-01")

    def test_equal_dates_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_dates("2025-01-01", "2025-01-01")

    def test_before_nisar_launch_raises(self):
        with pytest.raises(ValueError, match="before NISAR launch"):
            validate_dates("2020-01-01", "2025-01-01")

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            validate_dates("not-a-date", "2025-01-01")

    def test_datetime_objects(self):
        from datetime import datetime
        start, end = validate_dates(
            datetime(2025, 1, 1), datetime(2025, 6, 1)
        )
        assert start == pd.Timestamp("2025-01-01")

    def test_strips_timezone(self):
        start, end = validate_dates(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        )
        assert start.tz is None


class TestValidateAOI:
    def test_shapely_polygon(self):
        poly = box(-115, 43, -114, 44)
        result = validate_aoi(poly)
        assert isinstance(result, Polygon)

    def test_bounding_box_list(self):
        result = validate_aoi([-115, 43, -114, 44])
        assert isinstance(result, Polygon)
        assert result.bounds == (-115.0, 43.0, -114.0, 44.0)

    def test_point_tuple(self):
        result = validate_aoi((-115.0, 43.5))
        assert isinstance(result, Point)

    def test_dict_west_south_east_north(self):
        result = validate_aoi({"west": -115, "south": 43, "east": -114, "north": 44})
        assert isinstance(result, Polygon)

    def test_dict_xmin_ymin(self):
        result = validate_aoi({"xmin": -115, "ymin": 43, "xmax": -114, "ymax": 44})
        assert isinstance(result, Polygon)

    def test_swapped_bounds_auto_sorted(self):
        result = validate_aoi([-114, 44, -115, 43])
        assert result.bounds == (-115.0, 43.0, -114.0, 44.0)

    def test_empty_geometry_raises(self):
        empty = Polygon()
        with pytest.raises(ValueError, match="empty"):
            validate_aoi(empty)

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError, match="must be geometry"):
            validate_aoi(42)

    def test_wrong_list_length_raises(self):
        with pytest.raises(ValueError, match="2 or 4"):
            validate_aoi([1, 2, 3])

    def test_unrecognized_dict_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            validate_aoi({"foo": 1, "bar": 2})

    def test_numpy_array(self):
        result = validate_aoi(np.array([-115, 43, -114, 44]))
        assert isinstance(result, Polygon)


class TestValidateUrls:
    def test_valid_urls(self):
        urls = ["https://example.com/a.h5", "https://example.com/b.h5"]
        result = validate_urls(urls)
        assert result == urls

    def test_strips_whitespace(self):
        result = validate_urls(["  https://example.com/a.h5  "])
        assert result == ["https://example.com/a.h5"]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No URLs"):
            validate_urls([])

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_urls([123])

    def test_non_http_raises(self):
        with pytest.raises(ValueError, match="must be http/https"):
            validate_urls(["ftp://example.com/a.h5"])


class TestValidatePath:
    def test_existing_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hi")
        result = validate_path(f, should_exist=True)
        assert result == f

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_path(tmp_path / "nope", should_exist=True)

    def test_already_exists_raises(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="already exists"):
            validate_path(f, should_exist=False)

    def test_make_directory(self, tmp_path):
        d = tmp_path / "new" / "nested"
        result = validate_path(d, make_directory=True)
        assert result.is_dir()

    def test_string_input(self, tmp_path):
        result = validate_path(str(tmp_path))
        assert isinstance(result, Path)
