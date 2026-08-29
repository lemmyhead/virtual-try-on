from pathlib import Path

from backend.parser import parse_product_page

FIXTURE = Path(__file__).parent / "fixtures" / "harvest_signet_ring.html"


def _load_spec():
    html = FIXTURE.read_text(encoding="utf-8")
    return parse_product_page(html, source_url="https://stonefruit.com/product/harvest-signet-ring")


def test_extracts_name():
    spec = _load_spec()
    assert spec.name == "Harvest Signet Ring"


def test_extracts_dimensions():
    spec = _load_spec()
    assert spec.dimensions is not None
    assert spec.dimensions["Stone Size"] == "8x6mm"
    assert spec.dimensions["Band Width"] == "3mm"


def test_extracts_price():
    spec = _load_spec()
    assert spec.price == "195.00"
    assert spec.currency == "USD"


def test_product_images_non_empty_and_clean():
    spec = _load_spec()
    assert len(spec.product_images) > 0

    # Deduplicated
    assert len(spec.product_images) == len(set(spec.product_images))

    for url in spec.product_images:
        assert not url.startswith("data:"), "base64 placeholder leaked into product_images"
        assert "favicon" not in url.lower()
        assert "apple-icon" not in url.lower()


def test_missing_dimensions_does_not_crash():
    # A page with no JSON-LD and no dimension-shaped description at all.
    html = "<html><head><title>Some Ring</title></head><body><h1>Some Ring</h1></body></html>"
    spec = parse_product_page(html)
    assert spec.dimensions is None
    assert spec.name  # falls back to <title>/<h1>, never crashes


def test_recognizes_ring_height_label_variants():
    from backend.parser import parse_product_page

    html = (
        "<script type=\"application/ld+json\">"
        '{"@type":"Product","name":"Test Ring",'
        '"description":"Band Width: 3mm \\n Total Height: 5mm"}'
        "</script>"
    )
    spec = parse_product_page(html)
    assert spec.dimensions["Ring Height"] == "5mm"
