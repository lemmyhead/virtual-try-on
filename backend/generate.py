"""
Stage C (generation half) -- all direct Gemini API calls live in this module.

`calibrate.py` and `qa_check.py` reuse `run_analysis` for their
text+image-in/JSON-out calls rather than each managing their own client, but
the actual image *generation* call (`generate_tryon_image`) is the one the
PRD calls out explicitly in section 6.3.
"""
from __future__ import annotations

import mimetypes
from functools import lru_cache
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import settings


class GenerationError(Exception):
    """Raised when Gemini returns no usable image/text in a response, or
    when the API call itself fails (bad key, quota, transient 5xx, etc).
    Every Gemini call in this module is wrapped so callers only ever need
    to catch this one exception type."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def _image_part(image_bytes: bytes, mime_type: str = "image/jpeg") -> types.Part:
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def run_analysis(image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
    """Cheap text+image -> text/JSON call, used for bounding-box detection
    (stage B) and output QA (stage D). Not an image-generation call."""
    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.analysis_model,
            contents=[_image_part(image_bytes, mime_type), prompt],
        )
    except genai_errors.APIError as exc:
        raise GenerationError(f"Gemini analysis call failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise GenerationError("Gemini analysis call returned no text.")
    return text


def generate_tryon_image(
    product_images: list[bytes],
    hand_photo: bytes,
    prompt: str,
    product_mime_types: Optional[list[str]] = None,
    hand_mime_type: str = "image/jpeg",
) -> bytes:
    """Stage C generation call.

    PRD 6.3 "Image ordering": send product photo(s) first, hand photo last --
    Gemini adopts the aspect ratio of the last image when inputs differ, and
    we want the output framed like the user's own selfie. The prompt is
    included as the final content part; the exact prompt/image interleaving
    is otherwise deliberately simple, since `prompt_builder.build_prompt`
    already encodes all the structured constraints as text.
    """
    client = get_client()
    product_mime_types = product_mime_types or ["image/jpeg"] * len(product_images)

    contents: list = []
    for img_bytes, mime in zip(product_images, product_mime_types):
        contents.append(_image_part(img_bytes, mime))
    contents.append(_image_part(hand_photo, hand_mime_type))
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=settings.image_model,
            contents=contents,
        )
    except genai_errors.APIError as exc:
        raise GenerationError(f"Gemini generation call failed: {exc}") from exc

    image_bytes = _extract_image_bytes(response)
    if image_bytes is None:
        raise GenerationError("Gemini generation call returned no image data.")
    return image_bytes


def _extract_image_bytes(response) -> Optional[bytes]:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and inline_data.data:
                return inline_data.data
    return None


def guess_mime_type(filename: Optional[str]) -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return "image/jpeg"
