"""
Orchestration: Stage A -> B -> C -> D. Single entry point (`run_tryon_pipeline`)
that the FastAPI layer calls, and that pipeline-level tests exercise directly
without spinning up HTTP (PRD section 6.5).
"""
from __future__ import annotations

import base64
import logging
from typing import Literal, Optional

from . import calibrate, generate, parser, qa_check
from .config import settings
from .models import ProductSource, ProductSpec, TryOnResult

logger = logging.getLogger(__name__)


def resolve_product_spec(product_source: ProductSource) -> ProductSpec:
    """Stage A entry point used by pipeline.py. `/api/parse-product` calls
    `parser.parse_product_page` directly; this wrapper additionally accepts
    an already-parsed spec so `/api/tryon` doesn't have to re-fetch/re-parse
    a page the user already confirmed (PRD section 7)."""
    if product_source.product_spec is not None:
        return product_source.product_spec
    if product_source.html is not None:
        return parser.parse_product_page(product_source.html, source_url=None)
    if product_source.url is not None:
        html = parser.fetch_html(product_source.url)
        return parser.parse_product_page(html, source_url=product_source.url)
    raise ValueError("ProductSource must set one of url, html, or product_spec.")


def run_tryon_pipeline(
    product_source: ProductSource,
    hand_photo: bytes,
    reference_object: Literal["credit_card", "us_quarter"],
    target_finger: Literal["index", "middle", "ring"],
    hand_side: Optional[Literal["left", "right"]] = None,
) -> TryOnResult:
    warnings: list[str] = []

    # Stage A
    product_spec = resolve_product_spec(product_source)
    if product_spec.dimensions is None:
        warnings.append(
            "We couldn't find exact measurements for this product — sizing in "
            "your result will be approximate."
        )

    # Stage B
    scale_result = calibrate.calibrate_hand_photo(
        hand_photo, product_spec, reference_object, target_finger, hand_side
    )
    if scale_result.warning:
        warnings.append(scale_result.warning)

    # Stage C + D, with retries (PRD 6.4: up to 3 total attempts)
    from . import prompt_builder  # local import keeps module import graph flat

    prompt = prompt_builder.build_prompt(product_spec, scale_result, target_finger, hand_side)
    product_images = _fetch_product_images(product_spec)

    best_image: Optional[bytes] = None
    best_qa = None
    best_score = None
    attempts = 0

    for attempt in range(1, settings.max_generation_attempts + 1):
        attempts = attempt
        try:
            image_bytes = generate.generate_tryon_image(product_images, hand_photo, prompt)
        except generate.GenerationError as exc:
            logger.warning("Generation attempt %d failed: %s", attempt, exc)
            continue

        qa_result = qa_check.run_qa(image_bytes, target_finger)
        attempt_score = qa_check.score(qa_result)

        if best_score is None or attempt_score > best_score:
            best_image, best_qa, best_score = image_bytes, qa_result, attempt_score

        if qa_result.qa_passed:
            break

    if best_image is None:
        raise generate.GenerationError(
            "All generation attempts failed to produce an image."
        )

    if best_qa is not None and not best_qa.qa_passed:
        warnings.append(
            "This result may not be fully accurate — our automated quality "
            "check flagged possible issues with the generated image."
        )

    return TryOnResult(
        image_base64=base64.b64encode(best_image).decode("ascii"),
        qa_passed=bool(best_qa and best_qa.qa_passed),
        scale_source=scale_result.scale_source,
        attempts=attempts,
        warnings=warnings,
        product_name=product_spec.name,
    )


def _fetch_product_images(product_spec: ProductSpec) -> list[bytes]:
    """Resolves the parser's candidate product images to raw bytes so they
    can be sent as reference images to the generation call. Each entry is
    either an http(s) URL (the normal parser.py path) or a `data:` URI (the
    direct-photo-upload path in app.py, which has no URL to host the image
    at). Skips any entry that fails to resolve rather than failing the whole
    pipeline over one broken image."""
    import requests

    images: list[bytes] = []
    for url in product_spec.product_images:
        try:
            if url.startswith("data:"):
                images.append(_decode_data_uri(url))
            else:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                images.append(resp.content)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Could not resolve product image %s: %s", url[:60], exc)
    return images


def _decode_data_uri(data_uri: str) -> bytes:
    header, _, payload = data_uri.partition(",")
    if "base64" not in header:
        raise ValueError("Only base64-encoded data URIs are supported.")
    return base64.b64decode(payload)
