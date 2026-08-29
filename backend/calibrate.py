"""
Stage B -- scale calibrator.

Converts the reference object visible in the hand photo into a real-world
mm-per-pixel ratio, then derives the target pixel dimensions the ring should
render at. See PRD section 6.2.

The pixel math (`compute_mm_per_pixel`, `compute_target_pixel_sizes`) is a
pure function of bounding boxes and is unit-tested with hardcoded inputs
(no API calls). The Gemini detection call that produces those bounding boxes
lives in `detect_reference_and_finger`, which is the only part of this
module that hits the network -- and the only part `pipeline.py` needs to
mock in tests.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from . import generate
from .config import (
    AVERAGE_FINGER_WIDTH_MM,
    CREDIT_CARD_HEIGHT_MM,
    CREDIT_CARD_WIDTH_MM,
    MIN_DETECTION_CONFIDENCE,
    US_QUARTER_DIAMETER_MM,
)
from .models import BoundingBox, ProductSpec, ReferenceObject, ScaleResult, TargetFinger

logger = logging.getLogger(__name__)

_SIZE_MM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[x×]\s*(\d+(?:\.\d+)?))?\s*mm", re.IGNORECASE
)


class LowConfidenceDetectionError(Exception):
    """Raised internally when the reference object/finger can't be
    confidently located in the photo. Callers (calibrate_hand_photo) catch
    this and fall back to the average-hand-proportion estimate -- it should
    never propagate out of this module."""


# ---------------------------------------------------------------------------
# Pure pixel math -- unit-testable without any network access.
# ---------------------------------------------------------------------------


def compute_mm_per_pixel(
    reference_object: ReferenceObject, reference_bbox: BoundingBox
) -> float:
    """PRD 6.2 step 2: mm_per_pixel = known_real_world_size_mm / measured_pixel_size.

    Credit card: average width- and height-derived ratios when the box looks
    roughly frontal (i.e. both dimensions are present and non-trivial);
    otherwise use whichever dimension is available. US quarter: circular, so
    width and height of its bounding box should both approximate the
    diameter -- average them for robustness against a slightly non-square
    detection box.
    """
    width_px = reference_bbox.width
    height_px = reference_bbox.height

    if reference_object == "us_quarter":
        measured = [d for d in (width_px, height_px) if d > 0]
        if not measured:
            raise ValueError("Reference bounding box has zero size.")
        avg_px = sum(measured) / len(measured)
        return US_QUARTER_DIAMETER_MM / avg_px

    if reference_object == "credit_card":
        ratios = []
        if width_px > 0:
            ratios.append(CREDIT_CARD_WIDTH_MM / width_px)
        if height_px > 0:
            ratios.append(CREDIT_CARD_HEIGHT_MM / height_px)
        if not ratios:
            raise ValueError("Reference bounding box has zero size.")
        return sum(ratios) / len(ratios)

    raise ValueError(f"Unknown reference object: {reference_object!r}")


def parse_dimension_mm(value: str) -> tuple[float, Optional[float]]:
    """Parse a dimension string like "8x6mm" or "3mm" into (width_mm, height_mm).
    height_mm is None for single-value dimensions (e.g. band width)."""
    match = _SIZE_MM_RE.search(value)
    if not match:
        raise ValueError(f"Could not parse a mm dimension out of {value!r}")
    width = float(match.group(1))
    height = float(match.group(2)) if match.group(2) else None
    return width, height


def compute_target_pixel_sizes(
    dimensions: Optional[dict], mm_per_pixel: float
) -> dict:
    """PRD 6.2 step 3. Returns a dict with any of
    band_width_px / stone_width_px / stone_height_px / ring_height_px that
    could be derived from the available dimension strings. Missing/unparsable
    entries are simply omitted -- never fabricated."""
    out: dict = {}
    if not dimensions or mm_per_pixel <= 0:
        return out

    band_raw = dimensions.get("Band Width")
    if band_raw:
        try:
            width_mm, _ = parse_dimension_mm(band_raw)
            out["band_width_px"] = width_mm / mm_per_pixel
        except ValueError:
            pass

    stone_raw = dimensions.get("Stone Size")
    if stone_raw:
        try:
            width_mm, height_mm = parse_dimension_mm(stone_raw)
            out["stone_width_px"] = width_mm / mm_per_pixel
            out["stone_height_px"] = (height_mm or width_mm) / mm_per_pixel
        except ValueError:
            pass

    # How far the ring rises off the finger -- rarely listed on a product
    # page, but used when it is (see prompt_builder.py's _height_instruction
    # for why this matters: band width + stone footprint alone say nothing
    # about vertical profile, and an unconstrained image model tends to
    # invent an exaggerated one).
    height_raw = dimensions.get("Ring Height")
    if height_raw:
        try:
            height_mm, _ = parse_dimension_mm(height_raw)
            out["ring_height_px"] = height_mm / mm_per_pixel
        except ValueError:
            pass

    return out


# ---------------------------------------------------------------------------
# Gemini-backed detection (the only networked part of this module).
# ---------------------------------------------------------------------------

_DETECTION_PROMPT_TEMPLATE = """\
You are analyzing a photo containing a human hand and a size-reference \
object ({reference_label}). Respond with strict JSON only, no prose, no \
markdown fences, matching exactly this shape:

{{
  "reference_object": {{
    "found": true|false,
    "confidence": 0.0-1.0,
    "x_min": <pixels>, "y_min": <pixels>, "x_max": <pixels>, "y_max": <pixels>
  }},
  "finger": {{
    "found": true|false,
    "confidence": 0.0-1.0,
    "x_min": <pixels>, "y_min": <pixels>, "x_max": <pixels>, "y_max": <pixels>
  }}
}}

Bounding boxes are in pixel coordinates of the image as provided (origin \
top-left). "reference_object" is the {reference_label} visible in the \
frame. "finger" is the base/knuckle segment of the {hand_side} hand's \
{target_finger} finger (the finger the ring will be placed on -- never the \
thumb or pinky). If you cannot find something, set "found": false and leave \
the box fields as 0.
"""

_REFERENCE_LABELS = {
    "credit_card": "credit card or ID card",
    "us_quarter": "US quarter coin",
}


def detect_reference_and_finger(
    hand_photo: bytes,
    reference_object: ReferenceObject,
    target_finger: TargetFinger,
    hand_side: Optional[str],
) -> tuple[BoundingBox, BoundingBox]:
    """Calls Gemini once to locate both the reference object and the target
    finger in the photo. Raises LowConfidenceDetectionError if either box is
    missing, low-confidence, or degenerate (zero-size)."""
    prompt = _DETECTION_PROMPT_TEMPLATE.format(
        reference_label=_REFERENCE_LABELS[reference_object],
        hand_side=hand_side or "shown",
        target_finger=target_finger,
    )
    raw = generate.run_analysis(hand_photo, prompt)
    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise LowConfidenceDetectionError(f"Could not parse detection response: {exc}")

    ref_box = _box_from_detection(data.get("reference_object"))
    finger_box = _box_from_detection(data.get("finger"))

    if ref_box is None or finger_box is None:
        raise LowConfidenceDetectionError("Detection did not return both bounding boxes.")
    if ref_box.width <= 0 or ref_box.height <= 0 or finger_box.width <= 0:
        raise LowConfidenceDetectionError("Detection returned a degenerate bounding box.")

    return ref_box, finger_box


def _box_from_detection(node: Optional[dict]) -> Optional[BoundingBox]:
    if not isinstance(node, dict) or not node.get("found"):
        return None
    confidence = float(node.get("confidence", 0.0))
    if confidence < MIN_DETECTION_CONFIDENCE:
        return None
    try:
        return BoundingBox(
            x_min=float(node["x_min"]),
            y_min=float(node["y_min"]),
            x_max=float(node["x_max"]),
            y_max=float(node["y_max"]),
            confidence=confidence,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# Orchestration -- what pipeline.py calls.
# ---------------------------------------------------------------------------


def calibrate_hand_photo(
    hand_photo: bytes,
    product_spec: ProductSpec,
    reference_object: ReferenceObject,
    target_finger: TargetFinger,
    hand_side: Optional[str],
) -> ScaleResult:
    """Top-level Stage B entry point. Tries a real calibration; on any
    detection failure, falls back to the average-hand-proportion estimate
    (PRD 6.2 "Required fallback") rather than failing the pipeline."""
    try:
        ref_box, finger_box = detect_reference_and_finger(
            hand_photo, reference_object, target_finger, hand_side
        )
        mm_per_pixel = compute_mm_per_pixel(reference_object, ref_box)
        target_sizes = compute_target_pixel_sizes(product_spec.dimensions, mm_per_pixel)
        return ScaleResult(
            mm_per_pixel=mm_per_pixel,
            scale_source="calibrated",
            reference_object=reference_object,
            finger_bbox=finger_box,
            reference_bbox=ref_box,
            **target_sizes,
        )
    except LowConfidenceDetectionError as exc:
        logger.warning("Falling back to estimated scale: %s", exc)
        return _estimated_scale_result(product_spec)


def _estimated_scale_result(product_spec: ProductSpec) -> ScaleResult:
    """Assume an average finger width stands in for a calibrated
    measurement. We don't have a pixel measurement of anything in this
    branch, so we can't derive target pixel sizes either -- prompt_builder
    falls back to the qualitative size hint instead (PRD 6.1.1 / 6.3)."""
    return ScaleResult(
        mm_per_pixel=0.0,
        scale_source="estimated",
        warning=(
            "Reference object could not be confidently detected; sizing is "
            f"estimated from an average finger width (~{AVERAGE_FINGER_WIDTH_MM}mm) "
            "rather than calibrated from your photo."
        ),
    )
