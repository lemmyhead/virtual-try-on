# Learnings: using an image-generation LLM (Gemini) for virtual ring try-on

**Context:** proof-of-concept build of a virtual ring try-on tool (product photo + hand
photo + reference object → composited image via Gemini). This document summarizes the
concrete limitations we hit with the image-LLM approach itself, independent of the rest
of the app's plumbing.

The one-line version: **generative image models are not a measurement or rendering
tool — every constraint you give one (size, placement, "don't change X") is a request
it can partially or fully ignore, not an enforced rule.** Everything below is a
specific instance of that.

---

## 1. Sizing/measurement is fundamentally unreliable in a pure prompt-based approach

This was our core, recurring issue: rings consistently rendered **inflated** relative
to their real-world dimensions, despite us computing correct target measurements
server-side.

- **No hard size enforcement exists.** There's no mechanism to make an image model
  obey an exact pixel or mm target — you can only ask, in text, and hope. Telling it
  "band width ≈ 30px" gives it no way to actually check what it's drawing against that
  number.
- **Strong learned bias toward "hero shot" jewelry photography.** Independent of any
  instruction, the model's training distribution favors dramatic, oversized,
  glamour-style product-photo proportions for rings. This bias actively fights literal
  sizing instructions, not just fails to help with them.
- **Missing data doesn't produce a smaller/safer result — it produces a confident,
  usually-oversized guess.** We didn't have a reliable "ring height" (how far it rises
  off the finger) for most product pages. The model didn't render that dimension
  conservatively when it was absent from the prompt — it invented one, and it tended
  to invent a tall, chunky, domed profile. **Absence of a constraint is not neutral.**
- **What helped (but didn't fully solve it): anchoring to something visible in the same
  photo, instead of an abstract number.** Telling the model "the band is about 4% of
  the credit card's edge — check it against the card that's right there in the photo"
  worked better than a bare pixel count, because it's a comparison the model can
  visually verify rather than an abstract unit it has to trust blindly. This is a real,
  usable mitigation — but it's still a request, with no guarantee, just better odds.
- **Open question for next iteration:** if prompt-based anchoring hits a ceiling, the
  likely next step is to stop asking the model to size the ring at all — deterministically
  crop/resize/composite the ring with real image processing (exact by construction),
  and use the model only for what it's actually reliable at (lighting, shadow,
  perspective blending), not for measurement.

## 2. Detection/calibration calls are estimates, not measurements

To calibrate real-world scale, we asked Gemini to return a bounding box for a
reference object (credit card / quarter) and the target finger, with a self-reported
confidence score.

- This is a **probabilistic estimate from a general multimodal model**, not a
  calibrated computer-vision measurement. It can and does fail, especially when the
  reference object is angled, cropped, small in frame, or not in the same focal plane.
- **The model's "confidence" is itself just another generated number** — not a real
  statistical confidence value. Treating it as ground truth would be a mistake; we
  built a hard floor (reject below a threshold) rather than trusting it at face value.
- **Two independent unreliable steps compound.** Overall sizing accuracy is roughly
  `P(detection succeeds) × P(generation then honors the computed target)`. Both have to
  work. We saw failures attributable to each independently.
- Failures here are **silent unless you explicitly design for them** — we had to build
  a fallback path (estimate from an average finger width) and a warning flag surfaced
  to the end user, rather than letting a failed detection quietly produce a
  confidently-wrong result.

## 3. No output is guaranteed correct — and the verification step isn't fully reliable either

- Every instruction we gave the generation call (correct finger, never thumb/pinky,
  preserve the hand exactly, no extra/missing fingers, no floating ring) is a request
  the model can partially violate, individually or in combination.
- This forced a **separate QA pass** (a second, independent model call) plus a retry
  loop just to catch obviously bad outputs — there's no way to get reliability from the
  generation call alone.
- **QA is not ground truth.** It's the same class of model, self-reporting structured
  JSON about its own kind of output. It can misjudge, and its response has to be
  defensively parsed (malformed JSON, missing/unexpected fields) rather than trusted.
- **Retries reduce but don't eliminate bad outputs.** Worst case, after N attempts you
  still ship "the best of N bad attempts" with a caveat flag — there's no attempt count
  that guarantees a good result. Each retry is also a real cost: 2-3 model calls per
  attempt (detect, generate, QA), so reliability is bought with real spend, not free.

## 4. Infrastructure failures can look identical to "the model did a bad job" if you don't separate them

- An invalid/misconfigured API key surfaces as an ordinary-looking error from the
  provider's API. If your error handling doesn't explicitly distinguish "the request
  never properly reached a working model" from "the model tried and produced something
  wrong," it's easy to misdiagnose a config problem as a model-quality problem (or vice
  versa) and waste time iterating on the wrong thing.
- Worth building this distinction into the pipeline from day one, not after the fact.

## 5. Multi-image input has real, non-obvious behavioral rules

- When you supply multiple reference images with different aspect ratios, Gemini
  adopts the aspect ratio of the **last** image given. Get the ordering backwards and
  you silently get output shaped like the product photo instead of the user's selfie —
  not a crash, just quietly wrong framing.
- This kind of detail comes from reading current docs/changelog, not from the model
  itself or from intuition — worth budgeting explicit time to find these gotchas rather
  than assuming standard behavior.

## 6. Model versioning churn is a real maintenance burden

- The specific image-generation model family we used has renamed/re-versioned multiple
  times within about a year. Model IDs need to be treated as **volatile config**,
  checked against current docs at build time, not hardcoded constants carried over from
  training-data assumptions or a prior project.

---

## Bottom line for the team

Treat an image-generation LLM as a **capable-but-unsupervised collaborator**, not a
rendering engine: it's good at photorealism, style-matching, and lighting/blend work,
and unreliable at anything requiring exact numeric compliance (size, count, precise
placement) unless that thing is also visually checkable against something already in
the frame. Any product requirement with a hard correctness bar (sizing accuracy,
exact counts, precise placement) needs either (a) a verify-and-retry loop budgeted into
cost/latency from the start, (b) the hard constraint enforced deterministically outside
the model (e.g. real image compositing) with the model used only for finishing touches,
or (c) an explicit acknowledgment to stakeholders that the output is an approximation,
not a guarantee.
