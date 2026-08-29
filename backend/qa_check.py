"""
Stage D -- output QA. See PRD section 6.4.

Sends the generated image back to Gemini as a cheap text+image call (not
another generation call) and asks for strict JSON describing whether the
result is usable. `pipeline.py` owns the retry loop; this module just scores
a single attempt.
"""
from __future__ import annotations

import json
import logging
import re

from . import generate
from .models import QAResult, TargetFinger

logger = logging.getLogger(__name__)

_KNOWN_ANOMALIES = {
    "extra fingers",
    "missing fingers",
    "warped hand",
    "floating ring",
    "blurry ring",
    "wrong hand shape",
    "ring on wrong finger",
}

_QA_PROMPT_TEMPLATE = """\
You are checking the quality of an AI-generated image that should show a \
ring worn on a person's {target_finger} finger. Respond with strict JSON \
only, no prose, no markdown fences, matching exactly this shape:

{{
  "ring_present": true|false,
  "ring_finger": "thumb"|"index"|"middle"|"ring"|"pinky"|"unknown",
  "hand_looks_natural": true|false,
  "anomalies": [<zero or more of: "extra fingers", "missing fingers", \
"warped hand", "floating ring", "blurry ring", "wrong hand shape", \
"ring on wrong finger">]
}}

Be strict: if the ring is not clearly and fully visible on a real-looking \
hand, on the {target_finger} finger specifically, flag it accordingly.
"""


def run_qa(image_bytes: bytes, target_finger: TargetFinger) -> QAResult:
    """Runs a single QA pass over one generated image and returns a scored
    QAResult. Never raises on a malformed model response -- treats that as
    a failed QA pass instead, so a flaky QA call can't crash the pipeline."""
    prompt = _QA_PROMPT_TEMPLATE.format(target_finger=target_finger)
    try:
        raw = generate.run_analysis(image_bytes, prompt)
    except generate.GenerationError as exc:
        logger.warning("QA call produced no response: %s", exc)
        return QAResult(qa_passed=False, raw_response=str(exc))

    data = _parse_json(raw)
    if data is None:
        return QAResult(qa_passed=False, raw_response=raw)

    ring_present = bool(data.get("ring_present", False))
    ring_finger = data.get("ring_finger")
    hand_looks_natural = bool(data.get("hand_looks_natural", False))
    anomalies = [a for a in data.get("anomalies", []) if isinstance(a, str)]

    qa_passed = (
        ring_present
        and ring_finger == target_finger
        and hand_looks_natural
        and not anomalies
    )

    return QAResult(
        ring_present=ring_present,
        ring_finger=ring_finger,
        hand_looks_natural=hand_looks_natural,
        anomalies=anomalies,
        qa_passed=qa_passed,
        raw_response=raw,
    )


def score(result: QAResult) -> int:
    """Higher is better. Used by pipeline.py to pick the "best" attempt when
    every retry still fails QA (PRD 6.4: "return the best-scoring attempt").
    A passed attempt always outranks a failed one; among failed attempts,
    fewer problems wins."""
    if result.qa_passed:
        return 100
    points = 0
    points += 1 if result.ring_present else 0
    points += 1 if result.hand_looks_natural else 0
    points += 1 if result.ring_finger and result.ring_finger != "unknown" else 0
    points -= len(result.anomalies)
    return points


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("QA response was not valid JSON: %r", text[:200])
        return None
