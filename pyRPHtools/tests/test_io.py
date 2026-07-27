import numpy as np
import pytest
from numpy.testing import assert_allclose

from rphtools import load_las

LAS_TEXT = """~VERSION INFORMATION
 VERS.                 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
 WRAP.                  NO : ONE LINE PER DEPTH STEP
~WELL INFORMATION
#MNEM.UNIT       DATA                    DESCRIPTION
STRT.M           1670.0000              : START DEPTH
STOP.M           1673.0000              : STOP DEPTH
STEP.M              1.0000              : STEP
NULL.            -999.2500              : NULL VALUE
WELL.            TEST WELL              : WELL
~CURVE INFORMATION
#MNEM.UNIT       API CODES              DESCRIPTION
 DEPT.M                                 : 1  DEPTH
 DT  .US/M       60 520 32 00           : 2  SONIC TRANSIT TIME
 RHOB.K/M3       45 350 01 00           : 3  BULK DENSITY
 NPHI.V/V        42 890 00 00           : 4  NEUTRON POROSITY
~PARAMETER INFORMATION
 BHT .DEGC          35.5000             : BOTTOM HOLE TEMPERATURE
~A  DEPTH     DT       RHOB     NPHI
1670.000   123.450   2550.000    0.450
1671.000   123.450  -999.250    0.450
1672.000  -999.250   2550.000    0.450
1673.000   124.500   2600.000    0.400
"""


@pytest.fixture
def las_path(tmp_path):
    p = tmp_path / "test.las"
    p.write_text(LAS_TEXT)
    return p


class TestLoadLas:
    def test_curve_names_sanitized_and_lowercased(self, las_path):
        las = load_las(las_path)
        assert las.columns == ["dept", "dt", "rhob", "nphi"]

    def test_data_shape_and_values(self, las_path):
        las = load_las(las_path)
        assert las.data.shape == (4, 4)
        assert_allclose(las.curves["dept"], [1670.0, 1671.0, 1672.0, 1673.0])
        assert las.curves["nphi"][0] == pytest.approx(0.45)

    def test_nulls_become_nan(self, las_path):
        las = load_las(las_path)
        assert np.isnan(las.curves["rhob"][1])
        assert np.isnan(las.curves["dt"][2])
        # Real values are untouched.
        assert las.curves["rhob"][0] == pytest.approx(2550.0)
        assert np.count_nonzero(np.isnan(las.data)) == 2

    def test_explicit_null_value(self, las_path):
        # Override: mask 2550 instead of the declared null.
        las = load_las(las_path, null_value=2550.0)
        assert np.isnan(las.curves["rhob"][0])
        assert las.curves["rhob"][1] == pytest.approx(-999.25)

    def test_header_captured(self, las_path):
        las = load_las(las_path)
        assert "CWLS LOG ASCII STANDARD" in las.header
        assert "~CURVE INFORMATION" in las.header
        # The data block is not part of the header (1670.0 also appears
        # in the STRT line, so check a value unique to the data).
        assert "123.450" not in las.header

    def test_curves_dict_matches_columns(self, las_path):
        las = load_las(las_path)
        assert set(las.curves) == set(las.columns)
        for k, name in enumerate(las.columns):
            assert_allclose(las.curves[name], las.data[:, k], equal_nan=True)

    def test_missing_curve_section(self, tmp_path):
        p = tmp_path / "bad.las"
        p.write_text("~VERSION\n VERS. 2.0 :\n~A\n1.0 2.0\n")
        with pytest.raises(ValueError, match="no ~CURVE section"):
            load_las(p)

    def test_ragged_data_rejected(self, tmp_path):
        p = tmp_path / "ragged.las"
        p.write_text("~CURVE INFORMATION\n DEPT.M :\n DT.US/M :\n~A DEPTH DT\n1.0 2.0\n3.0\n")
        with pytest.raises(ValueError, match="not a multiple"):
            load_las(p)

    def test_comment_lines_skipped_in_curve_block(self, las_path):
        las = load_las(las_path)
        # The "#MNEM.UNIT ..." comment line must not become a curve.
        assert not any(c.startswith("_mnem") or c == "mnem" for c in las.columns)
        assert len(las.columns) == 4

    def test_units_parsed_separately_from_names(self, las_path):
        # loadlas.m folded the unit into the name (rhob_k_m3); the port
        # splits the mnemonic and reports units on the side.
        las = load_las(las_path)
        assert las.units["rhob"] == "K/M3"
        assert las.units["dept"] == "M"
        assert las.units["dt"] == "US/M"
        assert set(las.units) == set(las.columns)

    def test_sections_after_curves_do_not_add_names(self, las_path):
        # ~PARAMETER sits between ~CURVE and ~A; BHT must not be a curve.
        las = load_las(las_path)
        assert "bht" not in las.columns
