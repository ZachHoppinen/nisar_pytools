"""Tests for nisar_pytools.utils.filename."""

import pandas as pd
import pytest

from nisar_pytools.utils.filename import parse_filename


class TestParseFilename:
    def test_gslc(self):
        name = "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
        info = parse_filename(name)
        assert info.product_type == "GSLC"
        assert info.track == 77
        assert info.frame == 24
        assert info.cycle == 4
        assert info.direction == "Ascending"
        assert info.is_ascending
        assert info.start_time == pd.Timestamp("2025-11-03T12:46:15")
        assert info.end_time == pd.Timestamp("2025-11-03T12:46:50")
        assert not info.is_qa

    def test_gunw(self):
        name = "NISAR_L2_PR_GUNW_006_149_A_024_009_4000_SH_20251202T123756_20251202T123831_20260107T123757_20260107T123832_X05010_N_F_J_001.h5"
        info = parse_filename(name)
        assert info.product_type == "GUNW"
        assert info.track == 149
        assert info.frame == 24
        assert info.direction == "Ascending"
        assert info.start_time == pd.Timestamp("2025-12-02T12:37:56")

    def test_descending(self):
        name = "NISAR_L2_PR_GSLC_004_077_D_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
        info = parse_filename(name)
        assert info.direction == "Descending"
        assert not info.is_ascending

    def test_qa_file(self):
        name = "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001_QA_STATS.h5"
        info = parse_filename(name)
        assert info.is_qa
        assert info.product_type == "GSLC"

    def test_full_path(self):
        path = "/data/gslcs/NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
        info = parse_filename(path)
        assert info.product_type == "GSLC"
        assert info.track == 77

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="Not a recognized"):
            parse_filename("random_file.h5")

    def test_level(self):
        name = "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
        info = parse_filename(name)
        assert info.level == "L2"
