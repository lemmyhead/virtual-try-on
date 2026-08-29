"""
Stage C (prompt half) -- pure prompt construction, independently
unit-testable without calling any API (PRD section 6.3).
"""
from __future__ import annotations

from typing import Optional

from . import calibrate
from .config import CREDIT_CARD_HEIGHT_MM, CREDIT_CARD_WIDTH_MM, US_QUARTER_DIAMETER_MM
from .models import ProductSpec, ScaleResult

_QUALITATIVE_BAND_HINT = {
    # Fallback wording used when we have no calibrated/estimated pixel target
    # at all -- e.g. dimensions were never found on the product page.
    None: "a medium-width band, based on typical ring proportions",
}

_NON_TARGET_FINGERS = {"thumb", "pinky"}

# Reference-object anchor info: a human-readable label, its known real-world
# characteristic dimension (used for the mm-to-mm ratio), and how to name
# that dimension in the sentence ("long edge" for a card, "diameter" for a
# coin). These ratios are exact -- derived from fixed physical constants, not
# from any pixel/bounding-box detection -- so they're immune to measurement
# error the way a finger-width ratio isn't.
_REFERENCE_OBJECT_INFO = {
    "credit_card": {
        "label": "a standard credit/ID card",
        "size_desc": f"{CREDIT_CARD_WIDTH_MM:.1f}mm x {CREDIT_CARD_HEIGHT_MM:.1f}mm",
        "characteristic_mm": CREDIT_CARD_WIDTH_MM,
        "dimension_name": f"long edge ({CREDIT_CARD_WIDTH_MM:.1f}mm)",
    },
    "us_quarter": {
        "label": "a US quarter",
        "size_desc": f"{US_QUARTER_DIAMETER_MM:.2f}mm diameter",
        "characteristic_mm": US_QUARTER_DIAMETER_MM,
        "dimension_name": f"diameter ({US_QUARTER_DIAMETER_MM:.2f}mm)",
    },
}


def build_prompt(
    spec: ProductSpec,
    scale_result: ScaleResult,
    finger: str,
    hand_orientation: Optional[str],
) -> str:
    """Builds the instruction prompt sent alongside [product photo(s)..., hand
    photo] to the Gemini image-generation call. Encodes every constraint
    called out in PRD 6.3:
      - preserve the hand exactly as-is, only add the ring
      - correct finger placement with an explicit anti-thumb/pinky constraint
      - target size (calibrated pixels, or a qualitative fallback)
      - match the product's actual design/metal/stone, not a reinterpretation
      - photorealism + lighting/color match to the original hand photo
    """
    hand_desc = f"the {hand_orientation} hand's" if hand_orientation else "the visible hand's"
    size_line = _size_instruction(spec, scale_result)
    design_line = _design_instruction(spec)

    lines = [
        "Edit the LAST provided image (the hand photo) by adding a photorealistic "
        "ring onto the hand. Do not generate a new hand or new background: preserve "
        "the original hand's skin tone, pose, lighting, shadows, and background "
        "exactly as shown in the last image. Only add the ring.",
        "",
        f"Placement: put the ring on {hand_desc} {finger} finger, at the base/knuckle "
        "position rings are normally worn. Do NOT place the ring on the thumb or the "
        "pinky finger under any circumstance, even if that would look more natural -- "
        f"it must be the {finger} finger specifically.",
        "",
        design_line,
        "",
        size_line,
        "",
        "Photorealism: match the ring's metal reflectivity and the scene's lighting "
        "and color temperature to the hand photo so the ring looks like it was "
        "physically photographed on that hand, not pasted on. Avoid warping the "
        "fingers, changing the number of fingers, or introducing any anatomical "
        "artifacts.",
    ]
    return "\n".join(lines)


def _design_instruction(spec: ProductSpec) -> str:
    details = []
    if spec.attributes:
        metal = spec.attributes.get("Metal")
        stone = spec.attributes.get("Stone")
        if metal:
            details.append(f"metal finish: {metal}")
        if stone:
            details.append(f"stone: {stone}")
    detail_str = f" ({'; '.join(details)})" if details else ""
    return (
        f"Design fidelity: use the ring design shown in the product photo(s) "
        f'above -- "{spec.name}"{detail_str} -- exactly as depicted. Do not '
        "reinterpret, simplify, or creatively alter its band shape, setting, or "
        "stone color/cut."
    )


_ANTI_OVERSIZE_INSTRUCTION = (
    "Rings in styled product photography (macro/hero shots) are usually shown "
    "dramatically larger than real life -- do NOT replicate that look here. "
    "The ring must read as the size it would actually be on a real hand in an "
    "ordinary, un-stylized snapshot. If you're unsure, err toward smaller and "
    "more modest, never larger."
)


def _size_instruction(spec: ProductSpec, scale_result: ScaleResult) -> str:
    if scale_result.scale_source == "estimated" or not scale_result.mm_per_pixel:
        qualitative = _qualitative_hint(spec)
        base = (
            "Sizing: exact pixel measurements are not available for this photo, "
            f"so size the ring qualitatively as {qualitative}, proportioned to look "
            "correctly scaled for a human finger."
        )
    else:
        parts = []

        # Primary anchor: a direct mm-to-mm comparison against the reference
        # object itself, which is visible in the same photo. This ratio comes
        # straight from fixed physical constants (the card's/coin's known
        # size and the ring's stated band width) -- it has no dependency on
        # how accurately any bounding box was detected, unlike a finger-width
        # ratio.
        reference_anchor = _reference_object_anchor(spec, scale_result)
        if reference_anchor:
            parts.append(reference_anchor)

        # Secondary anchor: ratio to the finger's own detected width, as an
        # independent cross-check (or the only anchor, if the band width
        # wasn't parseable but a finger box was still detected).
        finger_width_px = scale_result.finger_bbox.width if scale_result.finger_bbox else None
        if scale_result.band_width_px and finger_width_px:
            ratio_pct = (scale_result.band_width_px / finger_width_px) * 100
            parts.append(
                f"As a cross-check, the band should also span roughly {ratio_pct:.0f}% "
                "of the visible width of the finger it's on."
            )
        elif scale_result.band_width_px and not reference_anchor:
            parts.append(f"Band width ≈ {scale_result.band_width_px:.1f}px.")

        if not parts:
            qualitative = _qualitative_hint(spec)
            base = (
                "Sizing: the product page did not list parseable dimensions, so "
                f"size the ring qualitatively as {qualitative}."
            )
        else:
            base = "Sizing: " + " ".join(parts)

    height_line = _height_instruction(spec, scale_result)
    return f"{base} {height_line} {_ANTI_OVERSIZE_INSTRUCTION}"


def _height_instruction(spec: ProductSpec, scale_result: ScaleResult) -> str:
    """Band width and stone footprint (width x height as seen from directly
    above) say nothing about how far the ring actually rises off the finger
    -- a dimension almost no product page lists. Left unconstrained, an
    image model tends to invent an exaggerated dome/bezel height to fill
    that gap, which reads as "inflated" even when width is otherwise
    correct. Always returns a constraint -- data-driven if we have a real
    height dimension (rare), an explicit anti-inflation instruction if not."""
    height_raw = (spec.dimensions or {}).get("Ring Height") if spec.dimensions else None
    if not height_raw:
        return (
            "Height/profile: the product page doesn't list how far the ring "
            "rises off the finger (only its footprint from above), so keep "
            "this modest and true-to-life -- most rings sit only a few "
            "millimeters proud of the band unless it's an obvious tall "
            "cocktail/statement style. Do not invent a tall, domed, or chunky "
            "profile to compensate for the missing measurement."
        )

    try:
        height_mm, _ = calibrate.parse_dimension_mm(height_raw)
    except ValueError:
        return (
            f"Height/profile: the ring's stated height off the finger is "
            f"{height_raw} -- keep the vertical rise consistent with that, "
            "not taller."
        )

    reference_info = (
        _REFERENCE_OBJECT_INFO.get(scale_result.reference_object)
        if scale_result.reference_object
        else None
    )
    if reference_info:
        ratio_pct = (height_mm / reference_info["characteristic_mm"]) * 100
        return (
            f"Height/profile: the ring's actual height off the finger is "
            f"{height_raw}, about {ratio_pct:.0f}% of the card/coin's "
            f"{reference_info['dimension_name']} -- check its thickness "
            "against that same reference object too, not just its footprint."
        )

    return (
        f"Height/profile: the ring's actual height off the finger is "
        f"{height_raw} -- keep the vertical rise modest and proportionate to "
        "that, not domed or chunky."
    )


def _reference_object_anchor(spec: ProductSpec, scale_result: ScaleResult) -> Optional[str]:
    """Builds the primary size-anchor sentence tying the ring's real
    dimensions to the reference object visible in the same photo. Returns
    None if we don't know which reference object was confirmed present
    (e.g. scale was estimated, not calibrated) -- callers fall back to the
    finger-ratio/qualitative anchors instead."""
    if not scale_result.reference_object:
        return None
    info = _REFERENCE_OBJECT_INFO.get(scale_result.reference_object)
    if not info:
        return None

    band_mm_str = (spec.dimensions or {}).get("Band Width") if spec.dimensions else None
    if not band_mm_str:
        return (
            f"The hand photo also shows {info['label']} ({info['size_desc']}) -- "
            "use it as a real-world size anchor and scale the ring to look "
            "correctly sized next to it, the way it actually would in person."
        )

    try:
        band_mm, _ = calibrate.parse_dimension_mm(band_mm_str)
    except ValueError:
        return (
            f"The hand photo also shows {info['label']} ({info['size_desc']}) -- "
            "use it as a real-world size anchor and scale the ring to look "
            "correctly sized next to it, the way it actually would in person."
        )

    ratio_pct = (band_mm / info["characteristic_mm"]) * 100
    return (
        f"The hand photo also shows {info['label']} -- use it as your primary "
        f"real-world size anchor: this ring's actual band width is {band_mm_str}, "
        f"which is about {ratio_pct:.0f}% of the card/coin's {info['dimension_name']}. "
        "Check the ring's size directly against that card or coin the way you'd "
        "hold them side-by-side in person, not just against the finger."
    )


def _qualitative_hint(spec: ProductSpec) -> str:
    band = (spec.dimensions or {}).get("Band Width") if spec.dimensions else None
    if band:
        return f"a band matching the product's stated width ({band})"
    return _QUALITATIVE_BAND_HINT[None]
