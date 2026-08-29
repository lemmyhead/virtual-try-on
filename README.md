# Ring Try-On

A prototype web app that lets a shopper see a specific ring, at a roughly
correct real-world size, composited onto a photo of their own hand — before
they buy. See [`PRD.md`](../PRD.md) in the repo root for the full spec this
was built against.

> **Disclaimer shown to users in-app:** *"This is an AI-generated
> visualization to help you gauge fit and style. Actual product may vary."*
> This is a decision aid, not a certified sizing tool.

## How it works

```
product page (URL or .html)  →  parser.py    → ProductSpec (name, price, dimensions, images)
hand photo + reference object →  calibrate.py → mm-per-pixel scale (or an estimated fallback)
ProductSpec + scale + finger  →  prompt_builder.py → text prompt
prompt + images                →  generate.py  → Gemini image edit → composited image
composited image               →  qa_check.py  → pass/fail + retry (up to 3 attempts total)
```

All of the above is orchestrated by `backend/pipeline.py::run_tryon_pipeline`,
which is what `backend/app.py` (FastAPI) calls from `/api/tryon`.

## Requirements

- Python 3.9+ (this repo was built and tested against the macOS system
  Python, 3.9.6 — it's past upstream end-of-life, so if you have Python
  3.10+ available, use it instead and feel free to bump the pins in
  `requirements.txt` to their latest releases).
- A Gemini API key with access to image generation
  ([ai.google.dev](https://ai.google.dev/gemini-api/docs/api-key)).

## Setup

```bash
cd ring-tryon
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then edit .env and set GEMINI_API_KEY
```

> **Note on the `cryptography` pin in `requirements.txt`:** on an Intel Mac,
> newer `cryptography` releases (a transitive dep via `google-genai` →
> `google-auth`) only ship arm64 wheels, which forces a from-source build
> that fails without a Rust toolchain configured for OpenSSL. It's pinned to
> the newest release that still ships an x86_64-compatible wheel. If you're
> on Apple Silicon or Linux, you likely don't need this pin at all.

## Running it

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open **http://localhost:8000/** — the FastAPI server also serves the
static frontend (`frontend/index.html`) directly, so there's nothing else to
start.

### Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key. |
| `MAX_REGENERATIONS_PER_SESSION` | `5` | Regenerations allowed per client IP (plus the initial generation), enforced by an in-memory counter in `app.py`. Resets on server restart — this is a prototype-scale cost control, not a real quota system. |
| `GEMINI_IMAGE_MODEL` | `gemini-3.1-flash-image` | Model used for the actual ring-compositing call. Override if Google ships a newer model — see `backend/config.py` for the rationale. |
| `GEMINI_ANALYSIS_MODEL` | `gemini-3.6-flash` | Model used for the cheaper detection/QA (text+image → JSON) calls. |
| `MAX_GENERATION_ATTEMPTS` | `3` | Total generate+QA attempts before returning the best-scoring one anyway (PRD section 6.4). |

## Testing

```bash
python3 -m pytest tests/ -v
```

- `test_parser.py` runs against the real sample fixture
  (`tests/fixtures/harvest_signet_ring.html`) and checks name/price/
  dimension extraction and that favicon/base64-placeholder junk never
  leaks into `product_images`.
- `test_calibrate.py` unit-tests the pixel math (`compute_mm_per_pixel`,
  `compute_target_pixel_sizes`) with hardcoded bounding boxes — no network.
- `test_pipeline.py` runs the full Stage A→B→C→D orchestration with
  `generate.run_analysis` / `generate.generate_tryon_image` monkeypatched at
  the network boundary — including a case that fails QA on attempt 1 and
  passes on attempt 2, to exercise the retry path, and a case with no
  parseable product dimensions, to exercise the missing-dimension fallback.

All of the above run without a real `GEMINI_API_KEY` or network access.

**Not covered by the automated suite** (per PRD section 10, "Manual/live
test"): an actual end-to-end run against the live Gemini API with a real
hand photo. Before calling this done for real usage, run one manually
through the UI with your own `GEMINI_API_KEY` set, a real hand+card/quarter
photo, and the provided fixture (or a real product URL).

## Known limitations (carried over from the PRD, section 11)

- **Parser fragility**: extraction is tuned against the one provided
  Stone Fruit fixture. It's written to degrade gracefully (return
  `None`/empty rather than guess) against other retailers' markup, but
  hasn't been validated against a second real retailer.
- **Fit accuracy ceiling**: this estimates fit for purchase-decision
  purposes; it does not certify physical accuracy.
- **No persistent storage**: hand photos are processed in-memory for the
  request and never written to disk or a database. There's no user-account
  system in v1.
- **Rate limiting is per-process, in-memory, IP-keyed** — fine for a
  prototype, not a substitute for real auth/quota if this goes further.
- **HEIC uploads** depend on the optional `pillow-heif` package building
  successfully on your platform; jpg/png always work regardless.
