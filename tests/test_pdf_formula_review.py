"""Formula recovery from real PDF geometry with stubbed model replies."""

from pathlib import Path
from types import SimpleNamespace

import pymupdf as fitz
import pytest

from mathbank import pdf_formula_review as review
from mathbank.task_manager import TaskCancelled


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("mathbank.pdf_vision_request.PDF_VISION_RETRY_DELAY_SECONDS", 0)

    def fail(*args, **kwargs):
        raise AssertionError("Tests must not contact a model")

    monkeypatch.setattr("requests.sessions.Session.request", fail)


def provider():
    return SimpleNamespace(
        api_key="test-key",
        chat_completions_url="https://example.invalid/v1/chat/completions",
        supports_image_input=True,
        model_name="test-model",
    )


def emit(page, x, y, pieces):
    for text, font, size, rise in pieces:
        page.insert_text((x, y + rise), text, fontname=font, fontsize=size)
        x += fitz.get_text_length(text, fontname=font, fontsize=size)
    return x


def add_native_formula(page, x, y, lead, letter, variable, exponent):
    text = page.insert_text((x, y), lead, fontsize=11)
    cursor = x + fitz.get_text_length(lead, fontsize=11)
    cursor = emit(page, cursor, y, [
        (letter, "tiit", 11, 0),
        ("=", "symb", 11, 0),
        ("-", "symb", 11, 0),
        (variable, "tiit", 11, 0),
    ])
    page.insert_text((cursor, y - 6), exponent, fontsize=8)
    page.insert_text((cursor + 8, y), ", where it is defined.", fontsize=11)
    return text


def add_vector_formula(page, x, y):
    page.draw_rect(fitz.Rect(x, y - 8, x + 5, y - 1), color=None, fill=(0, 0, 0))
    page.draw_rect(fitz.Rect(x + 7, y - 8, x + 12, y - 1), color=None, fill=(0, 0, 0))
    page.draw_line(fitz.Point(x - 2, y + 1), fitz.Point(x + 18, y + 1),
                   color=(0, 0, 0), width=1)


def build_pdf(path: Path) -> None:
    with fitz.open() as document:
        page = document.new_page(width=420, height=560)
        add_native_formula(page, 40, 80, "20. A force is given by ", "F", "kx", "3")

        page.insert_text((60, 150), "19. An arrow of mass", fontsize=11)
        page.insert_text((60, 180), "(A)", fontsize=11)
        add_vector_formula(page, 82, 174)

        page.insert_text((60, 240), "a speed of", fontsize=11)
        add_vector_formula(page, 110, 234)
        page.insert_text((132, 240), "when it collides", fontsize=11)

        page = document.new_page(width=420, height=560)
        add_native_formula(page, 40, 80, "21. A second force is ", "G", "cx", "2")
        document.save(path)


PAGE_ONE = (
    "<!-- MATHBANK_PDF_PAGE:1 -->\n"
    "19. An arrow moves with a speed of when it collides.\n"
    "(A)\n"
    "20. A force is given by F = -kx3, where it is defined."
)
PAGE_TWO = (
    "<!-- MATHBANK_PDF_PAGE:2 -->\n"
    "21. A second force is G = -cx2, where it is defined."
)


def page_images(tmp_path: Path, pdf_path: Path) -> list[str]:
    paths = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            target = tmp_path / f"page{index}.png"
            page.get_pixmap(dpi=110).save(target)
            paths.append(str(target))
    return paths


def test_crop_messages_encodes_rendered_page_pngs(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    with fitz.open(pdf_path) as document:
        blocks = review.detect_formula_blocks(document[0], page_index=0)
    messages = review._crop_messages(page_images(tmp_path, pdf_path)[0], blocks[:2], lambda: None)
    images = [part for part in messages if part["type"] == "image_url"]
    assert len(images) == 2
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in images)


def answer_for(block):
    if block["kind"] == "vector" and block["option"] == "A":
        return r"\dfrac{v_0}{2}"
    if block["kind"] == "vector":
        return r"3\,\mathrm{m/s}"
    if "F" in block["alnum"]:
        return r"F=-kx^{3}"
    return r"G=-cx^{2}"


def test_detects_native_glyphs_and_vector_strokes(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    with fitz.open(pdf_path) as document:
        blocks = review.detect_formula_blocks(document[0], page_index=0)
    kinds = {block["kind"] for block in blocks}
    assert kinds == {"glyph", "vector"}
    glyph = next(block for block in blocks if block["kind"] == "glyph")
    assert glyph["alnum"] == "3Fkx"
    assert glyph["question_number"] == 20
    assert any(block["option"] == "A" and block["kind"] == "vector" for block in blocks)
    inline = next(block for block in blocks if not block["option"] and block["kind"] == "vector")
    assert inline["context_before"].endswith("a speed of")
    assert inline["context_after"].startswith("when it collides")


def test_refines_each_page_without_cross_page_bleed(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    calls = []

    def fake_request(image_path, blocks, **kwargs):
        calls.append(Path(image_path).name)
        return {block["id"]: answer_for(block) for block in blocks}

    monkeypatch.setattr(review, "request_formula_blocks", fake_request)
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), [PAGE_ONE, PAGE_TWO],
        page_images(tmp_path, pdf_path), [0, 1], provider=provider(),
    )
    first, second = result["page_texts"]
    assert "$F=-kx^{3}$" in first
    assert r"(A) $\dfrac{v_0}{2}$" in first
    assert r"a speed of $3\,\mathrm{m/s}$ when it collides" in first
    assert "$G=-cx^{2}$" in second
    assert "G=-cx" not in first
    assert "F=-kx" not in second
    report = result["diagnostics"]
    assert report["status"] == "completed"
    assert report["calls"] == 2
    assert report["accepted"] == report["matched"] > 0
    assert report["rejected"] == 0
    assert sorted(calls) == ["page0.png", "page1.png"]


def test_eligible_pages_leave_other_pages_untouched(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    monkeypatch.setattr(review, "request_formula_blocks",
                        lambda image_path, blocks, **kwargs: {
                            block["id"]: answer_for(block) for block in blocks})
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), [PAGE_ONE, PAGE_TWO],
        page_images(tmp_path, pdf_path), [0, 1], provider=provider(),
        eligible_pages={1},
    )
    assert result["page_texts"][0] == PAGE_ONE
    assert "$G=-cx^{2}$" in result["page_texts"][1]
    assert result["diagnostics"]["calls"] == 1


def test_option_scan_skips_interleaved_neighbour_column(tmp_path):
    markdown = (
        "17. Wheel\n(A)\n(B)\n(C)\n19. Arrow\n(D)\nsticks\n(E)\nshown\n"
        "(A)\n(B)\n(C)\n(D)\n(E) 0\n18. Beam\n(A)\n(B)\n(C)\n(D)\n(E) W"
    )
    q17 = review._option_insertion_span(
        markdown, {"option": "D", "question_number": 17, "kind": "vector"})
    q19 = review._option_insertion_span(
        markdown, {"option": "D", "question_number": 19, "kind": "vector"})
    assert q17 is not None and q19 is not None
    assert q17[0] == markdown.index("(D)\nsticks") + 3
    assert q19[0] == markdown.index("(C)\n(D)\n(E) 0") + 7


def test_fragment_slot_absorbs_detached_fraction_pieces():
    markdown = (
        "21. Which equation applies?\n"
        "*dx* 2\n\n"
        "(A) *kx* = *m*\n*dt²* 2 *dx*\n\n"
        "(B) -= *kx m*\n*dt²*\n\n"
        "(C) -= *km u*\n*du* *dt*"
    )
    block = {"option": "A", "question_number": 21, "kind": "glyph",
             "alnum": "2dxkxm2dt"}
    span = review._option_fragment_span(markdown, block)
    assert span is not None
    replaced = markdown[span[0]:span[1]]
    assert replaced.startswith("*dx* 2")
    assert "(A)" in replaced
    assert replaced.rstrip().endswith("2 *dx*")


def test_later_options_never_absorb_the_previous_option_fragment():
    markdown = (
        "20. A force is given.\n"
        "(A) *U kx* =-\n2 12\n\n"
        "(B) *U kx* = 2\n1\n\n"
        "(C) *U kx* =-4\n4"
    )
    block = {"option": "B", "question_number": 20, "kind": "glyph",
             "alnum": "12Ukx2"}
    span = review._formula_span(markdown, block)
    assert span is not None
    assert markdown[span[0]:span[1]].startswith("(B)")
    assert "2 12" not in markdown[span[0]:span[1]]


def test_option_block_without_its_slot_does_not_fall_back_to_prose_context():
    markdown = "31. Instant, ball 2 is launched upward from the ground."
    block = {
        "option": "B", "question_number": 32, "kind": "glyph", "alnum": "axk",
        "context_before": "launched upward from the", "context_after": "ground",
    }
    assert review._formula_span(markdown, block) is None


def test_validation_rejects_hallucinated_or_malformed_latex():
    glyph = {"alnum": "12mA3", "strict_glyphs": True}
    vector = {"alnum": "", "strict_glyphs": False}
    assert review.validate_formula_latex(r"\dfrac{1}{2}mx^2", glyph)[0] is False
    assert review.validate_formula_latex(r"\dfrac{1}{2", glyph) == (False, "unbalanced")
    assert review.validate_formula_latex(r"\unknown{x}", glyph) == (False, "unknown_command")
    assert review.validate_formula_latex("$x$", glyph)[0] is False
    assert review.validate_formula_latex("x\ny", glyph)[0] is False
    assert review.validate_formula_latex(r"\dfrac{v_0}{2}", vector) == (True, "")


def test_validation_accepts_visible_greek_ell_and_pdf_uv_confusion():
    ell = {"kind": "glyph", "strict_glyphs": True, "alnum": "12mA"}
    alpha = {"kind": "glyph", "strict_glyphs": True, "alnum": "12Uxkxax2"}
    speed = {"kind": "glyph", "strict_glyphs": True, "alnum": "4u0"}
    assert review.validate_formula_latex(r"\dfrac{1}{12}m\ell^2", ell) == (True, "")
    assert review.validate_formula_latex(
        r"U(x)=\dfrac{1}{2}kx^2-\alpha x", alpha) == (True, "")
    assert review.validate_formula_latex(r"\dfrac{1}{4}v_0", speed) == (True, "")
    assert review.validate_formula_latex(
        r"\dfrac{1}{12}m x^2", ell) == (False, "missing_glyph")


def test_flat_formula_span_replaces_reordered_glyph_text():
    block = {
        "kind": "glyph", "option": "", "alnum": "12Uxkxax2",
        "context_before": "described by the function", "context_after": ",",
        "text": "12U(x)=kx-ax,2",
    }
    markdown = (
        "32. Suppose that the potential energy can be\n"
        "described by the function $Ux$ ()=-$kx^2 ax$, 2, where constants.\n"
    )
    span = review._formula_span(markdown, block)
    assert span is not None
    assert markdown[span[0]:span[1]].strip() == "$Ux$ ()=-$kx^2 ax$, 2"


def test_trailing_punctuation_is_not_duplicated():
    block = {"context_after": ","}
    assert review._strip_trailing_punctuation(r"U(x)=\alpha x,", block) == r"U(x)=\alpha x"
    assert review._strip_trailing_punctuation(r"U(x)=\alpha x", block) == r"U(x)=\alpha x"


def test_unavailable_provider_keeps_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    unavailable = SimpleNamespace(api_key="", chat_completions_url="", supports_image_input=False)
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), [PAGE_ONE, PAGE_TWO],
        page_images(tmp_path, pdf_path), [0, 1], provider=unavailable,
    )
    assert result["page_texts"] == [PAGE_ONE, PAGE_TWO]
    assert result["diagnostics"]["status"] == "unavailable"


def test_request_failure_keeps_original_page(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    def fail(image_path, blocks, **kwargs):
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(review, "request_formula_blocks", fail)
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), [PAGE_ONE], page_images(tmp_path, pdf_path)[:1],
        [0], provider=provider(),
    )
    assert result["page_texts"] == [PAGE_ONE]
    assert result["diagnostics"]["accepted"] == 0
    assert result["diagnostics"]["pages"][0]["error"] == "RuntimeError"


def test_missing_page_image_reports_without_editing(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), [PAGE_ONE], [], [0], provider=provider(),
    )
    assert result["page_texts"] == [PAGE_ONE]
    assert result["diagnostics"]["pages"][0]["error"] == "missing_page_image"


def test_empty_unanchored_vector_strokes_are_not_formula_blocks(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    blocks = [
        {
            "id": "p1-m1", "kind": "vector", "option": "", "bbox": [20, 20, 30, 30],
            "text": "[矢量公式] 20. A force", "alnum": "",
            "context_before": "", "context_after": "20. A force is given by",
        },
        {
            "id": "p1-m2", "kind": "glyph", "option": "", "bbox": [100, 70, 150, 90],
            "text": "3F=-kx,", "alnum": "3Fkx",
            "context_before": "is given by", "context_after": ", where",
        },
    ]
    monkeypatch.setattr(review, "detect_formula_blocks", lambda page, **kwargs: blocks)
    monkeypatch.setattr(review, "request_formula_blocks",
                        lambda image_path, selected, **kwargs: {
                            block["id"]: r"F=-kx^3" for block in selected})
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(),
        ["20. A force is given by F = -kx3, where it is defined."],
        page_images(tmp_path, pdf_path)[:1],
        [0],
        provider=provider(),
    )
    assert result["diagnostics"]["blocks"] == 1
    assert "[公式待核对]" not in result["page_texts"][0]


def test_cancellation_propagates(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    def cancel():
        raise TaskCancelled("cancelled")

    with pytest.raises(TaskCancelled):
        review.refine_pdf_formulas(
            pdf_path.read_bytes(), [PAGE_ONE], page_images(tmp_path, pdf_path)[:1],
            [0], provider=provider(), check_cancelled=cancel,
        )


def test_adjacent_rejected_slots_get_one_marker(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)
    blocks = [
        {"id": "p1-m1", "bbox": [100, 100, 120, 110], "option": "A",
         "question_number": 19, "kind": "vector", "text": "", "alnum": "",
         "context_before": "(A)", "context_after": ""},
        {"id": "p1-m2", "bbox": [100, 120, 120, 130], "option": "B",
         "question_number": 19, "kind": "vector", "text": "", "alnum": "",
         "context_before": "(B)", "context_after": ""},
    ]
    monkeypatch.setattr(review, "detect_formula_blocks", lambda page, **kwargs: blocks)
    monkeypatch.setattr(review, "request_formula_blocks",
                        lambda image_path, blocks, **kwargs: {
                            block["id"]: None for block in blocks})
    result = review.refine_pdf_formulas(
        pdf_path.read_bytes(), ["(A)\n(B)\n"], page_images(tmp_path, pdf_path)[:1],
        [0], provider=provider(),
    )
    assert result["page_texts"][0].count("[公式待核对]") == 1
