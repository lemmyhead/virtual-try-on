"""
Integration-style pipeline tests. Per PRD section 10, these mock generate.py
(the actual network boundary: `run_analysis` for detection/QA calls,
`generate_tryon_image` for the image-generation call) so they run without
hitting the live Gemini API.
"""
import base64

from backend import generate, pipeline
from backend.models import ProductSource, ProductSpec

FIXTURE_SPEC = ProductSpec(
    name="Harvest Signet Ring",
    price="195.00",
    currency="USD",
    description="Stone Size: 8x6mm \n Band Width: 3mm",
    dimensions={"Stone Size": "8x6mm", "Band Width": "3mm"},
    product_images=[],  # empty on purpose -- avoids real HTTP fetches in tests
    available_sizes=[],
    source_url="https://stonefruit.com/product/harvest-signet-ring",
)

DETECTION_OK_RESPONSE = """{
  "reference_object": {"found": true, "confidence": 0.95, "x_min": 0, "y_min": 0, "x_max": 856, "y_max": 540},
  "finger": {"found": true, "confidence": 0.9, "x_min": 300, "y_min": 200, "x_max": 340, "y_max": 260}
}"""

DETECTION_LOW_CONFIDENCE_RESPONSE = """{
  "reference_object": {"found": false, "confidence": 0.1, "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
  "finger": {"found": true, "confidence": 0.9, "x_min": 300, "y_min": 200, "x_max": 340, "y_max": 260}
}"""

QA_PASS_RESPONSE = (
    '{"ring_present": true, "ring_finger": "index", '
    '"hand_looks_natural": true, "anomalies": []}'
)
QA_FAIL_RESPONSE = (
    '{"ring_present": true, "ring_finger": "index", '
    '"hand_looks_natural": true, "anomalies": ["floating ring"]}'
)


def _fake_run_analysis_factory(detection_response, qa_responses):
    """Builds a stand-in for generate.run_analysis that routes to a canned
    detection response or the next canned QA response based on which prompt
    template produced the call (the two prompts are shaped differently)."""
    qa_calls = {"count": 0}

    def _fake_run_analysis(image_bytes, prompt, mime_type="image/jpeg"):
        if "reference_object" in prompt:  # stage B detection prompt
            return detection_response
        if "ring_present" in prompt:  # stage D QA prompt
            idx = min(qa_calls["count"], len(qa_responses) - 1)
            qa_calls["count"] += 1
            return qa_responses[idx]
        raise AssertionError(f"Unexpected prompt sent to run_analysis: {prompt[:80]!r}")

    return _fake_run_analysis, qa_calls


def _product_source(spec=FIXTURE_SPEC):
    return ProductSource(product_spec=spec)


def _run(monkeypatch, detection_response, qa_responses, spec=FIXTURE_SPEC):
    fake_analysis, qa_calls = _fake_run_analysis_factory(detection_response, qa_responses)
    monkeypatch.setattr(generate, "run_analysis", fake_analysis)
    monkeypatch.setattr(generate, "generate_tryon_image", lambda *a, **k: b"fake-image-bytes")

    result = pipeline.run_tryon_pipeline(
        product_source=_product_source(spec),
        hand_photo=b"fake-hand-photo",
        reference_object="credit_card",
        target_finger="index",
        hand_side="left",
    )
    return result, qa_calls


def test_pipeline_success_on_first_attempt(monkeypatch):
    result, qa_calls = _run(monkeypatch, DETECTION_OK_RESPONSE, [QA_PASS_RESPONSE])

    assert result.qa_passed is True
    assert result.attempts == 1
    assert qa_calls["count"] == 1
    assert result.scale_source == "calibrated"
    assert base64.b64decode(result.image_base64) == b"fake-image-bytes"
    assert not any("measurements" in w for w in result.warnings)


def test_pipeline_retries_and_recovers_from_failed_qa(monkeypatch):
    """Definition of done: 'QA/retry loop demonstrably catches and retries
    at least one bad generation during testing.'"""
    result, qa_calls = _run(
        monkeypatch, DETECTION_OK_RESPONSE, [QA_FAIL_RESPONSE, QA_PASS_RESPONSE]
    )

    assert result.attempts == 2
    assert result.qa_passed is True
    assert qa_calls["count"] == 2


def test_pipeline_returns_best_attempt_when_every_attempt_fails_qa(monkeypatch):
    result, qa_calls = _run(
        monkeypatch,
        DETECTION_OK_RESPONSE,
        [QA_FAIL_RESPONSE, QA_FAIL_RESPONSE, QA_FAIL_RESPONSE],
    )

    assert result.attempts == 3  # 3 total attempts per PRD 6.4
    assert qa_calls["count"] == 3
    assert result.qa_passed is False
    assert any("may not be fully accurate" in w for w in result.warnings)


def test_pipeline_missing_dimensions_falls_back_without_crashing(monkeypatch):
    spec_without_dims = FIXTURE_SPEC.model_copy(update={"dimensions": None})
    result, _ = _run(
        monkeypatch, DETECTION_OK_RESPONSE, [QA_PASS_RESPONSE], spec=spec_without_dims
    )

    assert result.qa_passed is True  # missing dimensions must not crash the pipeline
    assert any("couldn't find exact measurements" in w for w in result.warnings)


def test_pipeline_falls_back_to_estimated_scale_on_low_confidence_detection(monkeypatch):
    result, _ = _run(monkeypatch, DETECTION_LOW_CONFIDENCE_RESPONSE, [QA_PASS_RESPONSE])

    assert result.scale_source == "estimated"
    assert any("could not be confidently detected" in w for w in result.warnings)


def test_pipeline_resolves_data_uri_product_images(monkeypatch):
    """Covers the direct-photo-upload input path (step 1's third option):
    product_images holds data: URIs instead of http(s) URLs, since an
    uploaded photo has nothing to be hosted at."""
    import base64

    from backend.pipeline import _fetch_product_images

    spec = FIXTURE_SPEC.model_copy(
        update={
            "product_images": [
                "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg-bytes").decode("ascii")
            ]
        }
    )
    resolved = _fetch_product_images(spec)
    assert resolved == [b"fake-jpeg-bytes"]


def test_pipeline_runs_end_to_end_with_photo_uploaded_product(monkeypatch):
    import base64

    spec = FIXTURE_SPEC.model_copy(
        update={
            "name": "Uploaded ring",
            "dimensions": None,
            "product_images": [
                "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg-bytes").decode("ascii")
            ],
        }
    )
    result, _ = _run(monkeypatch, DETECTION_OK_RESPONSE, [QA_PASS_RESPONSE], spec=spec)

    assert result.qa_passed is True
    assert result.product_name == "Uploaded ring"
