"""Shared dataclasses/pydantic models for every stage's payloads."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ReferenceObject = Literal["credit_card", "us_quarter"]
TargetFinger = Literal["index", "middle", "ring"]
HandSide = Literal["left", "right"]
ScaleSource = Literal["calibrated", "estimated"]


class ProductSpec(BaseModel):
    """Output of Stage A (parser.py). Mirrors PRD section 6.1 exactly, plus
    one additive `attributes` field for extra metadata (metal/stone/etc.)
    that helps prompt_builder describe the ring but isn't required by the
    PRD's core shape."""

    name: str
    price: Optional[str] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    dimensions: Optional[dict] = None
    product_images: list = Field(default_factory=list)
    available_sizes: list = Field(default_factory=list)
    source_url: Optional[str] = None
    attributes: Optional[dict] = None


class BoundingBox(BaseModel):
    """Pixel-space bounding box, relative to the image the detection call
    was run against. x_min/y_min is the top-left corner."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)


class ScaleResult(BaseModel):
    """Output of Stage B (calibrate.py)."""

    mm_per_pixel: float
    scale_source: ScaleSource
    # Which reference object was actually confirmed in the photo -- only set
    # on the "calibrated" path. Lets prompt_builder.py anchor the ring's size
    # to that same visible object (a fixed, detection-error-free mm-to-mm
    # ratio) rather than only to a detected finger bounding box.
    reference_object: Optional[ReferenceObject] = None
    band_width_px: Optional[float] = None
    stone_width_px: Optional[float] = None
    stone_height_px: Optional[float] = None
    ring_height_px: Optional[float] = None
    finger_bbox: Optional[BoundingBox] = None
    reference_bbox: Optional[BoundingBox] = None
    warning: Optional[str] = None


class QAResult(BaseModel):
    """Output of Stage D (qa_check.py)."""

    ring_present: bool = False
    ring_finger: Optional[str] = None
    hand_looks_natural: bool = False
    anomalies: list = Field(default_factory=list)
    qa_passed: bool = False
    raw_response: Optional[str] = None


class ProductSource(BaseModel):
    """Input to pipeline.py's Stage A step. Exactly one of these should be
    set: a URL to fetch+parse, raw HTML to parse directly, or an
    already-parsed spec (e.g. re-submitted from the frontend after the user
    confirmed extraction in the /api/parse-product step, per PRD section 7)."""

    url: Optional[str] = None
    html: Optional[str] = None
    product_spec: Optional[ProductSpec] = None


class TryOnResult(BaseModel):
    """Output of pipeline.py -- what the FastAPI endpoint hands back."""

    image_base64: str
    qa_passed: bool
    scale_source: ScaleSource
    attempts: int
    warnings: list = Field(default_factory=list)
    product_name: Optional[str] = None
