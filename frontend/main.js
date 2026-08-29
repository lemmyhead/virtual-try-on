// Plain JS, no build step -- see PRD section 8.

const state = {
  productUrl: null,
  productHtml: null, // raw text of an uploaded .html file, resent to /api/tryon
  productSpecJson: null, // set when the product came from direct photo upload (no page to re-parse)
  regenCount: 0,
};

const $ = (id) => document.getElementById(id);

function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }

// ---------------------------------------------------------------------------
// Step 1: parse product
// ---------------------------------------------------------------------------

$("parse-btn").addEventListener("click", async () => {
  const urlInput = $("product-url").value.trim();
  const fileInput = $("product-file").files[0];
  const imageFiles = Array.from($("product-images").files || []);
  hide($("parse-error"));

  if (!urlInput && !fileInput && imageFiles.length === 0) {
    $("parse-error").textContent =
      "Enter a product URL, upload an .html file, or upload product photo(s).";
    show($("parse-error"));
    return;
  }

  const formData = new FormData();
  // Priority mirrors the UI's left-to-right order: an .html file or URL
  // gives us real page data to parse, so prefer those over a bare photo
  // upload (which can only ever produce name/images, no price/dimensions).
  if (fileInput) {
    state.productHtml = await fileInput.text();
    state.productUrl = null;
    state.productSpecJson = null;
    formData.append("file", fileInput);
  } else if (urlInput) {
    state.productUrl = urlInput;
    state.productHtml = null;
    state.productSpecJson = null;
    formData.append("url", urlInput);
  } else {
    state.productUrl = null;
    state.productHtml = null;
    state.productSpecJson = null; // filled in below once we have the parsed spec back
    imageFiles.forEach((f) => formData.append("images", f));
    const nameInput = $("product-name-input").value.trim();
    if (nameInput) formData.append("product_name", nameInput);
  }

  $("parse-btn").disabled = true;
  try {
    const resp = await fetch("/api/parse-product", { method: "POST", body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }
    const spec = await resp.json();
    // Photo-upload path: there's no page to re-fetch/re-parse later, so hang
    // onto the parsed spec itself and resend it as-is to /api/tryon.
    if (!fileInput && !urlInput) {
      state.productSpecJson = JSON.stringify(spec);
    }
    renderProduct(spec);
    show($("step-hand"));
    show($("step-finger"));
    show($("step-generate"));
  } catch (err) {
    $("parse-error").textContent = err.message;
    show($("parse-error"));
  } finally {
    $("parse-btn").disabled = false;
  }
});

function renderProduct(spec) {
  show($("product-result"));
  $("product-name").textContent = spec.name || "Unknown product";
  $("product-price").textContent = spec.price
    ? `${spec.currency || ""} ${spec.price}`.trim()
    : "Price not found";

  const dimsEl = $("product-dimensions");
  dimsEl.innerHTML = "";
  if (spec.dimensions) {
    for (const [label, value] of Object.entries(spec.dimensions)) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      dimsEl.append(dt, dd);
    }
    hide($("dimensions-warning"));
  } else {
    show($("dimensions-warning"));
  }

  const thumbsEl = $("product-thumbs");
  thumbsEl.innerHTML = "";
  (spec.product_images || []).forEach((url) => {
    const img = document.createElement("img");
    img.src = url;
    img.alt = spec.name || "product photo";
    thumbsEl.appendChild(img);
  });
}

// ---------------------------------------------------------------------------
// Step 4: generate / regenerate
// ---------------------------------------------------------------------------

$("generate-btn").addEventListener("click", () => runTryon());
$("regenerate-btn").addEventListener("click", () => runTryon());

async function runTryon() {
  hide($("generate-error"));
  hide($("regenerate-error"));

  const handFile = $("hand-photo").files[0];
  if (!handFile) {
    $("generate-error").textContent = "Upload a hand photo first (step 2).";
    show($("generate-error"));
    return;
  }
  if (!state.productUrl && !state.productHtml && !state.productSpecJson) {
    $("generate-error").textContent = "Parse a product first (step 1).";
    show($("generate-error"));
    return;
  }

  const formData = new FormData();
  formData.append("hand_photo", handFile);
  formData.append("reference_object", getRadioValue("reference_object"));
  formData.append("target_finger", getRadioValue("target_finger"));
  formData.append("hand_side", getRadioValue("hand_side"));
  if (state.productUrl) formData.append("product_url", state.productUrl);
  if (state.productHtml) formData.append("product_html", state.productHtml);
  if (state.productSpecJson) formData.append("product_spec_json", state.productSpecJson);

  $("generate-btn").disabled = true;
  $("regenerate-btn").disabled = true;
  show($("loading"));
  hide($("step-result"));

  try {
    const resp = await fetch("/api/tryon", { method: "POST", body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }
    const result = await resp.json();
    renderResult(result);
  } catch (err) {
    const target = state.regenCount > 0 ? $("regenerate-error") : $("generate-error");
    target.textContent = err.message;
    show(target);
  } finally {
    hide($("loading"));
    $("generate-btn").disabled = false;
    $("regenerate-btn").disabled = false;
  }
}

function renderResult(result) {
  state.regenCount += 1;
  show($("step-result"));
  $("result-image").src = `data:image/jpeg;base64,${result.image_base64}`;

  if (result.qa_passed) {
    hide($("qa-warning"));
  } else {
    show($("qa-warning"));
  }

  const warningsEl = $("warnings-list");
  warningsEl.innerHTML = "";
  (result.warnings || []).forEach((text) => {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = text;
    warningsEl.appendChild(p);
  });

  $("step-result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}
