"""
Environment/config loading and constants shared across the pipeline.

Nothing in here should call the network. Keep this a pure "settings object"
so tests can construct/override it without touching real env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env once at import time (no-op if the file doesn't exist).
load_dotenv()


# ---------------------------------------------------------------------------
# Gemini model IDs.
#
# NOTE (see PRD section 11 "Gemini model churn"): the Nano Banana model family
# has moved fast (2.5 Flash Image -> 3.1 Flash Image -> 3 Pro Image across
# 2025-2026). These are config values on purpose -- do not hardcode a model
# ID inside generate.py/calibrate.py/qa_check.py. If Google ships a newer
# model, override via env vars rather than editing code.
#
# Checked against https://ai.google.dev/gemini-api/docs/image-generation and
# https://ai.google.dev/gemini-api/docs/models at build time (Aug 2026):
#   - gemini-3.1-flash-image is documented as the model that "excels at
#     multiple reference image processing and consistency" -- exactly what
#     stage C needs (product photo(s) + hand photo -> one faithful composite).
#   - gemini-3.6-flash is the current general-purpose multimodal model, used
#     for the *analysis* calls (bounding-box detection in stage B, JSON QA
#     in stage D) that take an image in and return text/JSON, not a new image.
# ---------------------------------------------------------------------------
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
GEMINI_ANALYSIS_MODEL = os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-3.6-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MAX_REGENERATIONS_PER_SESSION = int(os.getenv("MAX_REGENERATIONS_PER_SESSION", "5"))

# Max total generation attempts per try-on request (1 initial + retries).
# PRD section 6.4: "retry generation up to 2 additional times (3 total attempts)".
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))

# ---------------------------------------------------------------------------
# Real-world reference object dimensions (mm), used by calibrate.py.
# ---------------------------------------------------------------------------
CREDIT_CARD_WIDTH_MM = 85.60   # ISO/IEC 7810 ID-1
CREDIT_CARD_HEIGHT_MM = 53.98  # ISO/IEC 7810 ID-1
US_QUARTER_DIAMETER_MM = 24.26

# Assumed average finger (base/knuckle) width for a ring-size-7-equivalent
# hand, used only when reference-object detection fails and we fall back to
# an estimate rather than a calibrated measurement (PRD section 6.2, "Required
# fallback").
AVERAGE_FINGER_WIDTH_MM = 17.0

# Below this confidence, a Gemini detection result is treated as "not
# confident enough to calibrate from" and calibrate.py falls back to the
# average-hand-proportion estimate instead.
MIN_DETECTION_CONFIDENCE = 0.5

# Cap on how many product images we carry forward from the parser / send to
# the generation model (PRD 6.1 point 3: "pass the top N candidate image
# URLs forward").
MAX_PRODUCT_IMAGES = 6

REQUEST_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class Settings:
    """Convenience bundle, mostly useful for tests that want to override a
    couple of values without monkeypatching module-level globals everywhere."""

    gemini_api_key: str = field(default_factory=lambda: GEMINI_API_KEY)
    image_model: str = field(default_factory=lambda: GEMINI_IMAGE_MODEL)
    analysis_model: str = field(default_factory=lambda: GEMINI_ANALYSIS_MODEL)
    max_regenerations_per_session: int = field(
        default_factory=lambda: MAX_REGENERATIONS_PER_SESSION
    )
    max_generation_attempts: int = field(
        default_factory=lambda: MAX_GENERATION_ATTEMPTS
    )


settings = Settings()
