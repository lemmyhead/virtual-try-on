"""FastAPI server. See PRD section 7 for the endpoint contract."""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import generate, parser, pipeline
from .config import MAX_PRODUCT_IMAGES, settings
from .models import ProductSource, ProductSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ring Try-On", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # v1 prototype; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Basic in-memory per-IP rate limiter for /api/tryon (PRD section 7 + 9:
# "Rate-limit /api/tryon per session/IP at a basic level (e.g. in-memory
# counter)"). Process-lifetime only, resets on server restart -- fine for a
# v1 prototype, not a substitute for real quota/auth.
# ---------------------------------------------------------------------------
_tryon_counts: dict[str, int] = {}


def _check_and_increment_rate_limit(client_key: str) -> None:
    count = _tryon_counts.get(client_key, 0)
    # +1 so the first (non-regeneration) call plus MAX_REGENERATIONS_PER_SESSION
    # regenerations are both allowed.
    limit = settings.max_regenerations_per_session + 1
    if count >= limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Regeneration limit reached ({settings.max_regenerations_per_session} "
                "per client). This is an in-memory, per-process counter (PRD section 7) "
                "-- refreshing the page won't reset it. Restart the server to reset it."
            ),
        )
    _tryon_counts[client_key] = count + 1


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": time.time()}


@app.post("/api/parse-product")
async def parse_product(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    product_name: Optional[str] = Form(None),
):
    # `images` is the third input option: the user has photos of the ring
    # but no product page at all. There's nothing to parse in that case --
    # we just wrap the photos (as data URIs, so pipeline.py can send them
    # straight to Gemini without needing them hosted anywhere) into a
    # ProductSpec with no price/dimensions, same as any other page that
    # doesn't expose measurements (PRD 6.1.1's fallback already covers a
    # spec with dimensions=None end-to-end).
    has_images = bool(images) and any(img.filename for img in images)
    if not url and not file and not has_images:
        raise HTTPException(
            400, "Provide a product page URL, an HTML file, or product photo(s)."
        )

    try:
        if file is not None:
            html_bytes = await file.read()
            spec = parser.parse_product_page(html_bytes.decode("utf-8", errors="ignore"))
        elif url:
            html = parser.fetch_html(url)
            spec = parser.parse_product_page(html, source_url=url)
        else:
            spec = await _build_spec_from_images(images, product_name)
    except Exception as exc:  # noqa: BLE001 -- surface as a clean 4xx to the user
        logger.exception("parse-product failed")
        raise HTTPException(400, f"Could not parse product: {exc}") from exc

    return JSONResponse(spec.model_dump())


async def _build_spec_from_images(
    images: List[UploadFile], product_name: Optional[str]
) -> ProductSpec:
    data_uris = []
    for upload in images[:MAX_PRODUCT_IMAGES]:
        if not upload.filename:
            continue
        raw = await upload.read()
        normalized = _normalize_image_to_jpeg(raw, upload.filename)
        data_uris.append("data:image/jpeg;base64," + base64.b64encode(normalized).decode("ascii"))

    if not data_uris:
        raise ValueError("No readable image files were uploaded.")

    return ProductSpec(name=product_name or "Uploaded ring", product_images=data_uris)


@app.post("/api/tryon")
async def tryon(
    request: Request,
    reference_object: str = Form(...),
    target_finger: str = Form(...),
    hand_side: Optional[str] = Form(None),
    product_url: Optional[str] = Form(None),
    product_html: Optional[str] = Form(None),
    product_spec_json: Optional[str] = Form(None),
    hand_photo: UploadFile = File(...),
):
    if target_finger in ("thumb", "pinky"):
        # Frontend must never offer these (PRD 4.3), but enforce server-side too.
        raise HTTPException(400, "target_finger must be one of: index, middle, ring.")

    client_key = request.client.host if request.client else "unknown"
    _check_and_increment_rate_limit(client_key)

    raw_bytes = await hand_photo.read()
    try:
        normalized_bytes = _normalize_image_to_jpeg(raw_bytes, hand_photo.filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            400,
            "Could not read the uploaded hand photo. Please upload a jpg, png, "
            f"or heic image. ({exc})",
        ) from exc

    if product_spec_json:
        # Photo-upload path (step 1's third option): the frontend already has
        # the parsed spec from /api/parse-product and just resends it as-is --
        # there's no raw page source to re-fetch/re-parse here.
        try:
            product_source = ProductSource(product_spec=ProductSpec.model_validate_json(product_spec_json))
        except ValueError as exc:
            raise HTTPException(400, f"Invalid product_spec_json: {exc}") from exc
    else:
        product_source = ProductSource(url=product_url, html=product_html)

    try:
        result = pipeline.run_tryon_pipeline(
            product_source=product_source,
            hand_photo=normalized_bytes,
            reference_object=reference_object,  # type: ignore[arg-type]
            target_finger=target_finger,  # type: ignore[arg-type]
            hand_side=hand_side,  # type: ignore[arg-type]
        )
    except generate.GenerationError as exc:
        raise HTTPException(502, f"Image generation failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return JSONResponse(result.model_dump())


def _normalize_image_to_jpeg(raw_bytes: bytes, filename: Optional[str]) -> bytes:
    """PRD 4.2: "jpg/png/heic -> normalize to jpg server-side." HEIC support
    depends on the optional pillow-heif plugin being importable; if it isn't,
    we raise a clear error rather than silently failing inside Pillow."""
    if filename and filename.lower().endswith((".heic", ".heif")):
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError as exc:
            raise ValueError(
                "HEIC support isn't installed on this server; please upload a jpg or png"
            ) from exc

    image = Image.open(io.BytesIO(raw_bytes))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# Serve the plain HTML/CSS/JS frontend at "/" (mounted last so it never
# shadows the /api/* routes above).
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
