"""
Stage A -- product page parser.

Turns a raw product page (URL or uploaded .html) into a structured
`ProductSpec`. See PRD section 6.1 for the extraction-priority rationale.

Design principle (PRD 6.1.1 and section 11 "Parser fragility"): prefer
returning None/empty over guessing. A wrong-but-confident extraction is worse
than an honest "we couldn't find this."
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import MAX_PRODUCT_IMAGES, REQUEST_TIMEOUT_SECONDS
from .models import ProductSpec

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 RingTryOnBot/1.0"
)

# Dimension label variants we recognize, mapped to a normalized label.
# Written to tolerate wording differences across retailers (PRD 6.1 point 2).
_DIMENSION_LABEL_ALIASES = {
    "stone size": "Stone Size",
    "center stone size": "Stone Size",
    "center stone": "Stone Size",
    "side stone sizes": "Side Stone Sizes",
    "side stone size": "Side Stone Sizes",
    "band width": "Band Width",
    "width": "Band Width",
    "ring width": "Band Width",
    "band": "Band Width",
    "total weight": "Weight",
    "weight": "Weight",
    # How far the ring rises off the finger -- rarely listed, but when it is,
    # it matters: without it, an image model has no real-world anchor for the
    # ring's vertical profile and tends to invent an exaggerated, chunky one
    # (see calibrate.py's compute_target_pixel_sizes and prompt_builder.py's
    # _height_instruction).
    "total height": "Ring Height",
    "ring height": "Ring Height",
    "face height": "Ring Height",
    "profile height": "Ring Height",
    "height off finger": "Ring Height",
    "stone height off band": "Ring Height",
}

# `Label: 8x6mm` or `Label: 3mm` (tabs/newlines/extra spaces tolerated).
_DIMENSION_LINE_RE = re.compile(
    r"([A-Za-z][A-Za-z \-]{1,40}?)\s*:\s*"
    r"(\d+(?:\.\d+)?\s*(?:[x×]\s*\d+(?:\.\d+)?)?\s*mm)",
    re.IGNORECASE,
)

# Filenames/paths that indicate "not a real product photo" -- favicons,
# storefront chrome, logos -- rather than the ring itself.
_NON_PRODUCT_IMAGE_HINTS = (
    "favicon",
    "apple-icon",
    "android-icon",
    "logo",
    "sprite",
    "placeholder",
)


def fetch_html(url: str) -> str:
    """Server-side fetch of a product page URL. Raises requests exceptions
    on network failure -- callers (app.py) turn that into a 4xx for the user."""
    resp = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.text


def parse_product_page(html: str, source_url: Optional[str] = None) -> ProductSpec:
    """Main entry point for Stage A. Never raises on "normal" missing-data
    cases -- returns a ProductSpec with None/[] fields instead."""
    soup = BeautifulSoup(html, "html.parser")

    node = _extract_jsonld_product(soup)

    if node is not None:
        name = _clean_text(node.get("name")) or "Unknown product"
        description = _clean_text(node.get("description"))
        price, currency = _extract_offer_price(node.get("offers"))
        images = _normalize_image_list(node.get("image"))
        attributes = _extract_additional_properties(node.get("additionalProperty"))
    else:
        name = _fallback_name(soup) or "Unknown product"
        description = None
        price, currency = None, None
        images = []
        attributes = None

    dimensions = _parse_dimensions(description) if description else None

    if not images:
        images = _fallback_extract_images(soup, source_url)
    images = _dedupe_preserve_order(images)[:MAX_PRODUCT_IMAGES]

    sizes = _extract_available_sizes(soup)

    return ProductSpec(
        name=name,
        price=price,
        currency=currency,
        description=description,
        dimensions=dimensions,
        product_images=images,
        available_sizes=sizes,
        source_url=source_url,
        attributes=attributes,
    )


# ---------------------------------------------------------------------------
# JSON-LD (priority 1 per PRD 6.1)
# ---------------------------------------------------------------------------


def _extract_jsonld_product(soup: BeautifulSoup) -> Optional[dict]:
    """Find a schema.org Product node in any <script type="application/ld+json">
    block. Handles both a top-level Product and a Product nested under
    `mainEntity` (as in the sample fixture, whose top-level type is WebPage)."""
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        product = _find_product_node(data)
        if product is not None:
            return product
    return None


def _find_product_node(data) -> Optional[dict]:
    """Recursively search a parsed JSON-LD payload (dict, list, or nested
    combination) for a node with "@type" == "Product"."""
    if isinstance(data, dict):
        if data.get("@type") == "Product":
            return data
        for value in data.values():
            found = _find_product_node(value)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_product_node(item)
            if found is not None:
                return found
    return None


def _extract_offer_price(offers) -> tuple[Optional[str], Optional[str]]:
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if not isinstance(offers, dict):
        return None, None
    return offers.get("price"), offers.get("priceCurrency")


def _normalize_image_list(image) -> list:
    if image is None:
        return []
    if isinstance(image, str):
        return [image]
    if isinstance(image, list):
        return [i for i in image if isinstance(i, str)]
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return [image["url"]]
    return []


def _extract_additional_properties(props) -> Optional[dict]:
    if not isinstance(props, list):
        return None
    out = {}
    for p in props:
        if isinstance(p, dict) and p.get("name") and "value" in p:
            out[str(p["name"])] = p["value"]
    return out or None


# ---------------------------------------------------------------------------
# Dimension parsing (priority 2 per PRD 6.1)
# ---------------------------------------------------------------------------


def _parse_dimensions(description: str) -> Optional[dict]:
    matches = _DIMENSION_LINE_RE.findall(description)
    if not matches:
        return None
    result = {}
    for raw_label, value in matches:
        label = raw_label.strip().lower()
        normalized = _DIMENSION_LABEL_ALIASES.get(label, raw_label.strip().title())
        result[normalized] = re.sub(r"\s+", "", value).lower().replace("×", "x")
    return result or None


# ---------------------------------------------------------------------------
# Product images (priority 3 per PRD 6.1) -- DOM fallback path, used only
# when JSON-LD didn't provide an image list.
# ---------------------------------------------------------------------------


def _fallback_extract_images(soup: BeautifulSoup, source_url: Optional[str]) -> list:
    candidates: list[str] = []

    # Prefer imagery inside markup associated with the main product gallery
    # (PRD names `configurator-assets-slider` / primary swiper container as
    # the pattern to look for) over other swiper instances on the page
    # (cross-sell carousels, "wear it with", etc).
    gallery = soup.select_one('[class*="configurator-assets-slider"]') or soup.select_one(
        '.swiper'
    )
    scopes = [gallery] if gallery else []
    scopes.append(soup)  # whole-page fallback if nothing gallery-scoped works out

    seen_scopes = []
    for scope in scopes:
        if scope is None or scope in seen_scopes:
            continue
        seen_scopes.append(scope)
        for img in scope.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if _looks_like_product_photo(src):
                candidates.append(urljoin(source_url or "", src) if source_url else src)
        if candidates:
            break  # gallery scope produced results; don't also pull page-wide noise

    return candidates


def _looks_like_product_photo(src: str) -> bool:
    if not src:
        return False
    if src.startswith("data:"):
        return False  # lazy-load base64 placeholder, not a real image
    lowered = src.lower()
    if any(hint in lowered for hint in _NON_PRODUCT_IMAGE_HINTS):
        return False
    if lowered.endswith(".svg"):
        return False  # site chrome/icons in this fixture are all svg
    return True


def _dedupe_preserve_order(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Available sizes (priority 4 per PRD 6.1)
# ---------------------------------------------------------------------------


def _extract_available_sizes(soup: BeautifulSoup) -> list:
    sizes = []
    for option in soup.find_all("option"):
        value = option.get("value") or ""
        if not value or not re.match(r"^S?\d", value):
            continue
        # Vue/Nuxt SSR often injects hydration comments (<!--[--> ... <!--]-->)
        # between the <option> tag and its visible text; strip those before
        # reading the label.
        label = re.sub(r"<!--.*?-->", "", option.decode_contents())
        label = re.sub(r"<[^>]+>", "", label).strip()
        sizes.append(label or value)
    return _dedupe_preserve_order(sizes)


# ---------------------------------------------------------------------------
# Misc fallbacks
# ---------------------------------------------------------------------------


def _fallback_name(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return _clean_text(h1.get_text())
    return None


def _clean_text(text) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    return cleaned or None
