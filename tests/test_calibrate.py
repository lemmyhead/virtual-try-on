import pytest

from backend.calibrate import (
    compute_mm_per_pixel,
    compute_target_pixel_sizes,
    parse_dimension_mm,
)
from backend.config import CREDIT_CARD_WIDTH_MM, US_QUARTER_DIAMETER_MM
from backend.models import BoundingBox


def test_compute_mm_per_pixel_credit_card_uses_average_of_both_dims():
    # A card photographed as an 856x540 px box -> 10x scale on both axes.
    bbox = BoundingBox(x_min=0, y_min=0, x_max=856, y_max=539.8)
    mm_per_pixel = compute_mm_per_pixel("credit_card", bbox)
    assert mm_per_pixel == pytest.approx(0.1, rel=1e-3)


def test_compute_mm_per_pixel_credit_card_single_dimension():
    # Height collapsed to 0 (e.g. detector only reported width confidently).
    bbox = BoundingBox(x_min=0, y_min=0, x_max=856, y_max=0)
    mm_per_pixel = compute_mm_per_pixel("credit_card", bbox)
    assert mm_per_pixel == pytest.approx(CREDIT_CARD_WIDTH_MM / 856, rel=1e-6)


def test_compute_mm_per_pixel_us_quarter():
    # A quarter photographed as a 200x196 px box.
    bbox = BoundingBox(x_min=10, y_min=10, x_max=210, y_max=206)
    mm_per_pixel = compute_mm_per_pixel("us_quarter", bbox)
    expected = US_QUARTER_DIAMETER_MM / ((200 + 196) / 2)
    assert mm_per_pixel == pytest.approx(expected, rel=1e-6)


def test_compute_mm_per_pixel_zero_size_raises():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)
    with pytest.raises(ValueError):
        compute_mm_per_pixel("us_quarter", bbox)


def test_parse_dimension_mm_pair():
    width, height = parse_dimension_mm("8x6mm")
    assert width == 8.0
    assert height == 6.0


def test_parse_dimension_mm_single():
    width, height = parse_dimension_mm("3mm")
    assert width == 3.0
    assert height is None


def test_parse_dimension_mm_unparsable_raises():
    with pytest.raises(ValueError):
        parse_dimension_mm("very wide")


def test_compute_target_pixel_sizes():
    dimensions = {"Stone Size": "8x6mm", "Band Width": "3mm"}
    mm_per_pixel = 0.1  # 1 real mm == 10 px
    sizes = compute_target_pixel_sizes(dimensions, mm_per_pixel)
    assert sizes["band_width_px"] == pytest.approx(30.0)
    assert sizes["stone_width_px"] == pytest.approx(80.0)
    assert sizes["stone_height_px"] == pytest.approx(60.0)


def test_compute_target_pixel_sizes_missing_dimensions_returns_empty():
    assert compute_target_pixel_sizes(None, 0.1) == {}


def test_compute_target_pixel_sizes_zero_scale_returns_empty():
    assert compute_target_pixel_sizes({"Band Width": "3mm"}, 0.0) == {}


def test_compute_target_pixel_sizes_includes_ring_height_when_present():
    dimensions = {"Band Width": "3mm", "Ring Height": "5mm"}
    sizes = compute_target_pixel_sizes(dimensions, mm_per_pixel=0.1)
    assert sizes["ring_height_px"] == pytest.approx(50.0)


def test_compute_target_pixel_sizes_omits_ring_height_when_absent():
    sizes = compute_target_pixel_sizes({"Band Width": "3mm"}, mm_per_pixel=0.1)
    assert "ring_height_px" not in sizes
