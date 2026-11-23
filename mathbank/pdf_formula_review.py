"""Recover locally linearized formulas without rewriting unproved text.

Native PDF extraction flattens fractions, superscripts and radicals into
linear runs. This module detects those blocks from original glyph geometry,
asks one bounded vision request per page for the blocks that can be located
uniquely in the page text, and replaces a block only after strict validation.
Everything else stays untouched and is reported for review.
"""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
import json
import math
import os
import re
from typing import Callable

from PIL import Image

from mathbank.ai_http import post_chat_completion
from mathbank.ai_json import parse_ai_json
from mathbank.ai_providers import (
    apply_model_thinking_policy, apply_structured_output_policy, resolve_ocr_provider,
)
from mathbank.pdf_vision_request import completion_content, request_pdf_vision
from mathbank.task_manager import TaskCancelled


MAX_BLOCKS_PER_PAGE = 20
MAX_BLOCK_CHARS = 240
MAX_PAGES_PER_TASK = 40
MAX_LATEX_CHARS = 400
_MATH_SYMBOLS = set("=+−-±∓×÷·⋅*/<>≤≥≠≈≡∞√∑∫∂∆∇→←↔⇒⇔∈∉⊂⊆∪∩°′″")
_GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
_MARKUP_TAG = re.compile(r"</?su[pb]>")
_IMAGE_OR_TABLE = re.compile(r"!\[|\\begin\{(?:tabular|array|matrix|cases|aligned)\}")
_COMMANDS = frozenset("""
frac dfrac tfrac cfrac sqrt left right times cdot div pm mp le leq ge geq ne neq approx equiv
sim simeq propto infty pm mp circ degree angle parallel perp vec hat bar overline underline
overrightarrow overleftarrow mathrm mathbf mathit mathsf mathtt text textrm textbf displaystyle
sum prod int iint iiint oint partial nabla Delta delta sigma theta alpha beta gamma lambda mu
omega phi psi rho tau epsilon varepsilon eta kappa xi pi nu chi zeta
alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma
tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sin cos tan cot sec csc arcsin arccos arctan log ln exp lim max min sup inf det gcd
ldots cdots dots vdots ddots cases begin end quad qquad hspace vspace phantom
overline widehat widetilde dot ddot bar vec acute grave breve check tilde
ell hbar
mathrm mathbf mathit mathcal mathbb mathfrak mathsf mathtt
left right big Big bigg Bigg
{ } % & # _ $ , ; : ! \\
""".split())
_COMMAND_GLYPHS = {
    "alpha": "a", "beta": "b", "gamma": "g", "delta": "d", "epsilon": "e",
    "varepsilon": "e", "zeta": "z", "eta": "n", "theta": "t", "iota": "i",
    "kappa": "k", "lambda": "l", "mu": "m", "nu": "n", "xi": "x",
    "omicron": "o", "pi": "p", "rho": "r", "sigma": "s", "tau": "t",
    "upsilon": "u", "phi": "f", "chi": "c", "psi": "p", "omega": "o",
    "Gamma": "G", "Delta": "D", "Theta": "T", "Lambda": "L", "Xi": "X",
    "Pi": "P", "Sigma": "S", "Upsilon": "U", "Phi": "F", "Psi": "P",
    "Omega": "O", "ell": "l", "hbar": "h",
}
_GLYPH_SUBSTITUTIONS = {
    "u": frozenset({"v"}), "v": frozenset({"u"}),
    "A": frozenset({"l"}), "l": frozenset({"A"}),
}


def _font_name(value: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", str(value or "")).removesuffix("-Identity-H")


def _is_math_font(font: str) -> bool:
    name = _font_name(font).lower()
    return any(token in name for token in (
        "cmmi", "cmsy", "cmex", "math", "symbol", "greek", "stix", "xits", "asana", "euclid",
    )) and "helvetica" not in name


def _is_italic(font: str) -> bool:
    name = _font_name(font).lower()
    return "italic" in name or "oblique" in name


def _finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _union(boxes):
    boxes = list(boxes)
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _overlap(first, second) -> float:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))


def _area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _collect_glyphs(page) -> tuple[list[dict], float, tuple]:
    """Return visible glyphs, the dominant body size and the page box."""
    rect = page.rect
    data = page.get_text("rawdict")
    glyphs = []
    seen = set()
    for block in data.get("blocks", []):
        if not isinstance(block, dict) or block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            if not isinstance(line, dict):
                continue
            for span in line.get("spans", []):
                font = str(span.get("font") or "")
                size = span.get("size")
                if not _finite(size) or not 4 <= size <= 48:
                    continue
                for char in span.get("chars", []):
                    value = char.get("c")
                    origin, box = char.get("origin"), char.get("bbox")
                    if not isinstance(value, str) or not value or value.isspace():
                        continue
                    if not origin or not box or not all(_finite(item) for item in (*origin, *box)):
                        continue
                    if box[2] <= box[0] and value not in {"\u0338"}:
                        continue
                    identity = (value, round(float(origin[0]), 1), round(float(origin[1]), 1),
                                round(float(size), 1), _font_name(font))
                    if identity in seen:
                        # Some press PDFs paint the same glyph twice: once as
                        # an outline and once as a text layer. Keep one copy so
                        # context does not turn into "iinniittiiaall".
                        continue
                    seen.add(identity)
                    glyphs.append({"c": value, "font": font, "size": float(size),
                                   "x": float(origin[0]), "y": float(origin[1]),
                                   "bbox": tuple(float(item) for item in box)})
    sizes = Counter(round(glyph["size"], 1) for glyph in glyphs)
    body = max(sizes, key=lambda size: (sizes[size], -abs(size - 11))) if sizes else 11.0
    return glyphs, float(body), (rect.x0, rect.y0, rect.x1, rect.y1)


def _visual_lines(glyphs: list[dict], body: float) -> list[dict]:
    """Merge raw spans that sit on one physical baseline into one visual line."""
    ordered = sorted(glyphs, key=lambda glyph: (glyph["y"], glyph["x"]))
    rows = []
    for glyph in ordered:
        if rows and abs(glyph["y"] - rows[-1]["baseline"]) <= body * 0.45:
            row = rows[-1]
            last_x = max(item["bbox"][2] for item in row["glyphs"])
            if glyph["bbox"][0] - last_x <= body * 3.5:
                row["glyphs"].append(glyph)
                row["baseline"] = sum(item["y"] for item in row["glyphs"]) / len(row["glyphs"])
                continue
        rows.append({"baseline": glyph["y"], "glyphs": [glyph]})
    lines = []
    for row in rows:
        items = sorted(row["glyphs"], key=lambda glyph: glyph["x"])
        text = _line_text(items, body)
        lines.append({"bbox": _union(item["bbox"] for item in items), "glyphs": items,
                      "text": text, "baseline": row["baseline"]})
    return lines


def _line_text(glyphs: list[dict], body: float) -> str:
    """Rebuild visible spacing without relying on discarded whitespace glyphs."""
    parts = []
    previous = None
    for glyph in sorted(glyphs, key=lambda item: item["x"]):
        if previous is not None and glyph["bbox"][0] - previous["bbox"][2] > body * 0.18:
            parts.append(" ")
        parts.append(glyph["c"])
        previous = glyph
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _box_gap(first, second) -> tuple[float, float]:
    return (max(0.0, max(first[0] - second[2], second[0] - first[2])),
            max(0.0, max(first[1] - second[3], second[1] - first[3])))


def _line_context(line: dict, box, body: float) -> tuple[str, str]:
    """Split a line into native text before and after a vector formula."""
    before = [glyph for glyph in line["glyphs"]
              if glyph["bbox"][2] <= box[0] + body * 0.35
              and abs(glyph["y"] - line["baseline"]) <= body * 0.8]
    after = [glyph for glyph in line["glyphs"]
             if glyph["bbox"][0] >= box[2] - body * 0.35
             and abs(glyph["y"] - line["baseline"]) <= body * 0.8]
    # A formula sits inside one text column. Keep only the run that actually
    # touches it so a neighbouring column cannot become part of the anchor.
    if before:
        before.sort(key=lambda glyph: glyph["bbox"][0])
        run = [before[-1]]
        for glyph in reversed(before[:-1]):
            if run[-1]["bbox"][0] - glyph["bbox"][2] > body * 1.6:
                break
            run.append(glyph)
        before = list(reversed(run))
    if after:
        after.sort(key=lambda glyph: glyph["bbox"][0])
        run = [after[0]]
        for glyph in after[1:]:
            if glyph["bbox"][0] - run[-1]["bbox"][2] > body * 1.6:
                break
            run.append(glyph)
        after = run
    return _line_text(before, body), _line_text(after, body)


def _question_number(lines: list[dict], box) -> int | None:
    """Read only a question marker to the left of the formula's own column."""
    best = None
    for line in lines:
        glyphs = sorted(line["glyphs"], key=lambda item: item["x"])
        for index, glyph in enumerate(glyphs):
            if not glyph["c"].isdigit():
                continue
            if index and glyph["bbox"][0] - glyphs[index - 1]["bbox"][2] < 2.0:
                continue
            digits = glyph["c"]
            cursor = index + 1
            while cursor < len(glyphs) and glyphs[cursor]["c"].isdigit() and len(digits) < 3:
                digits += glyphs[cursor]["c"]
                cursor += 1
            if cursor >= len(glyphs) or glyphs[cursor]["c"] not in ".．":
                continue
            if cursor + 1 >= len(glyphs) or not glyphs[cursor + 1]["c"].isalpha():
                continue
            if glyph["bbox"][0] > box[0] + 22:
                continue
            if line["bbox"][1] > box[3] + 2:
                continue
            # Stay in the same text column: the nearest left-hand marker wins,
            # and vertical recency only breaks ties. This prevents a right-
            # column option from borrowing the lower-left question number.
            candidate = (abs(glyph["bbox"][0] - box[0]), -line["bbox"][1], int(digits))
            if best is None or candidate < best:
                best = candidate
    return best[2] if best else None


def _nearest_formula_line(lines: list[dict], box, body: float):
    """Return the best same-line text evidence for a vector formula."""
    candidates = []
    for line in lines:
        line_box = line["bbox"]
        gap = _box_gap(box, line_box)
        if gap[1] > body * 0.9:
            continue
        before, after = _line_context(line, box, body)
        if not before and not after:
            continue
        # The formula must sit inside this line's horizontal reading span, not
        # merely share a row with an unrelated column.
        if box[2] < line_box[0] - body * 2 or box[0] > line_box[2] + body * 2:
            continue
        candidates.append((gap[1], gap[0], -(len(before) + len(after)), line, before, after))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3:]


def _drawing_clusters(page, body: float, page_box) -> list[tuple[tuple, int]]:
    """Cluster small vector strokes into formula-sized regions.

    Long diagram strokes and page frames are excluded. Fraction bars are added
    deliberately: without them a numerator and denominator look like two
    unrelated one-line fragments.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if not drawings or len(drawings) > 4000:
        return []
    rect = (page_box[0], page_box[1], page_box[2], page_box[3])
    page_area = max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1]))
    try:
        image_boxes = [tuple(float(value) for value in item["bbox"])
                       for item in page.get_image_info() if item.get("bbox")]
    except Exception:
        image_boxes = []
    components, bars = [], []
    for drawing in drawings:
        value = drawing.get("rect")
        if value is None:
            continue
        box = (float(value.x0), float(value.y0), float(value.x1), float(value.y1))
        width, height = box[2] - box[0], box[3] - box[1]
        if width < 0.35 and height < 0.35:
            continue
        if width > body * 17 or height > body * 7:
            continue
        if width * height > page_area * 0.03:
            continue
        if any(_overlap(box, image) > _area(box) * 0.55 for image in image_boxes):
            continue
        index = len(components)
        components.append(box)
        if width >= body * 0.35 and height <= 1.6:
            bars.append((index, box))
    if len(components) < 2:
        return []

    parent = list(range(len(components)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    # Glyphs on one visual line are joined only across tiny inter-glyph gaps.
    # The vertical bound is deliberately smaller than normal option spacing.
    for first_index, first in enumerate(components):
        for second_index in range(first_index + 1, len(components)):
            second = components[second_index]
            gap_x, gap_y = _box_gap(first, second)
            if gap_y <= 1.4 and gap_x <= body * 0.5:
                union(first_index, second_index)

    # A drawn fraction bar owns the glyphs immediately above and below it.
    for bar_index, bar in bars:
        bar_width = bar[2] - bar[0]
        for index, box in enumerate(components):
            if index == bar_index:
                continue
            overlap = max(0.0, min(bar[2], box[2]) - max(bar[0], box[0]))
            if overlap < 0.55 * max(1.0, min(bar_width, box[2] - box[0])):
                continue
            if box[3] <= bar[1] + 1.5 and bar[1] - box[3] <= body * 0.9:
                union(bar_index, index)
            elif box[1] >= bar[3] - 1.5 and box[1] - bar[3] <= body * 0.9:
                union(bar_index, index)

    groups = {}
    for index, box in enumerate(components):
        groups.setdefault(find(index), []).append(box)
    staged = []
    for boxes in groups.values():
        staged.append(((min(box[0] for box in boxes), min(box[1] for box in boxes),
                        max(box[2] for box in boxes), max(box[3] for box in boxes)), len(boxes)))
    staged.sort(key=lambda item: (item[0][1], item[0][0]))

    # Coalesce adjacent pieces that belong to the same displayed formula.
    merged = []
    for box, count in staged:
        for index, (existing, existing_count) in enumerate(merged):
            overlap_y = min(box[3], existing[3]) - max(box[1], existing[1])
            minimum_height = min(box[3] - box[1], existing[3] - existing[1])
            gap_x, _ = _box_gap(box, existing)
            if overlap_y >= 0.6 * max(1.0, minimum_height) and gap_x <= body * 2:
                merged[index] = ((min(box[0], existing[0]), min(box[1], existing[1]),
                                  max(box[2], existing[2]), max(box[3], existing[3])),
                                 count + existing_count)
                break
        else:
            merged.append((box, count))
    return [(box, count) for box, count in merged
            if count >= 2 and 5 <= box[2] - box[0] <= body * 20
            and 5 <= box[3] - box[1] <= body * 6.5]


def _vector_formula_blocks(page, glyphs, body, page_box, lines, page_index,
                           max_blocks: int) -> list[dict]:
    width, height = page_box[2] - page_box[0], page_box[3] - page_box[1]
    blocks = []
    for box, count in _drawing_clusters(page, body, page_box):
        if box[1] < 24 or box[3] > height - 24:
            continue
        nearby = _nearest_formula_line(lines, box, body)
        if nearby is None:
            continue
        line, before, after = nearby
        option_match = re.search(r"\(([A-H])\)\s*$", before)
        option = option_match.group(1) if option_match else ""
        question = _question_number(lines, box)
        clipped = (before[-70:], after[:70])
        if not clipped[0] and not clipped[1]:
            continue
        # A bare run of one or two strokes near a page header or figure label
        # is not enough evidence to send a crop for formula recovery.
        if not option and len(re.sub(r"\W+", "", before + after)) < 2:
            continue
        reasons = ["矢量公式"]
        if box[3] - box[1] > body * 1.2:
            reasons.append("分式或纵向叠层")
        if option:
            reasons.append("选项公式")
        blocks.append({
            "id": "", "bbox": [round(box[0] / width * 1000, 4), round(box[1] / height * 1000, 4),
                               round(box[2] / width * 1000, 4), round(box[3] / height * 1000, 4)],
            "text": ((before[-50:] + " [矢量公式] " + after[:50]).strip()),
            "alnum": "", "option": option, "question_number": question,
            "context_before": clipped[0], "context_after": clipped[1],
            "reasons": reasons, "kind": "vector", "strict_glyphs": False,
            "score": count + (5 if option else 0) + (4 if box[3] - box[1] > body * 1.2 else 0),
            "page_index": page_index,
        })
    blocks.sort(key=lambda item: (-item["score"], item["bbox"][1], item["bbox"][0]))
    return blocks[:max_blocks]


def _horizontal_rules(page) -> list[tuple]:
    rules = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            box = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            width, height = box[2] - box[0], box[3] - box[1]
            if 4 <= width <= 220 and height <= 1.6:
                rules.append(box)
    except Exception:
        return []
    return rules


def _is_seed(glyph: dict, body: float) -> bool:
    value = glyph["c"]
    return (value in _MATH_SYMBOLS or value in _GREEK or _is_math_font(glyph["font"])
            or (_is_italic(glyph["font"]) and value.isalpha())
            or glyph["size"] <= body * 0.82 and (value.isalnum() or value in "+-−"))


def _formula_char(glyph: dict, body: float) -> bool:
    return (_is_seed(glyph, body) or glyph["c"].isdigit()
            or glyph["c"] in ".,/+-−=")


def _glyphs_join(first: dict, second: dict, body: float) -> bool:
    """Two glyphs belong to one local formula run when their geometry agrees."""
    one, two = first["bbox"], second["bbox"]
    horizontal_gap = max(0.0, max(one[0] - two[2], two[0] - one[2]))
    if abs(first["y"] - second["y"]) <= body * 0.45:
        first_formula, second_formula = _formula_char(first, body), _formula_char(second, body)
        if first_formula and second_formula:
            return horizontal_gap <= body * 1.2
        if first_formula != second_formula:
            plain = second if first_formula else first
            return horizontal_gap <= body * 0.16 and (plain["c"].isalnum() or plain["c"] in "()[]{}")
        return False
    vertical_overlap = min(one[3], two[3]) - max(one[1], two[1])
    if vertical_overlap <= 0:
        return False
    return (horizontal_gap <= body * 0.65
            and _formula_char(first, body) and _formula_char(second, body))


def _formula_runs(glyphs: list[dict], body: float) -> list[list[dict]]:
    """Group glyphs into formula runs without depending on seed order."""
    parent = list(range(len(glyphs)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a

    ordered = sorted(range(len(glyphs)), key=lambda index: glyphs[index]["y"])
    for position, left in enumerate(ordered):
        first = glyphs[left]
        for right in ordered[position + 1:]:
            second = glyphs[right]
            if second["y"] - first["y"] > body * 1.6:
                break
            if _glyphs_join(first, second, body):
                union(left, right)
    groups = {}
    for index in range(len(glyphs)):
        groups.setdefault(find(index), []).append(glyphs[index])
    return list(groups.values())


def _has_option_marker(run: list[dict], glyphs: list[dict], body: float) -> bool:
    box = _union(item["bbox"] for item in run)
    baseline = sum(item["y"] for item in run) / len(run)
    left = [glyph for glyph in glyphs
            if abs(glyph["y"] - baseline) <= body * 0.45 and glyph["bbox"][2] <= box[2] + 1]
    text = "".join(glyph["c"] for glyph in sorted(left, key=lambda item: item["x"]))[-24:]
    return bool(re.search(r"\([A-H]\)", text))


def _stacked_fraction_runs(runs: list[list[dict]], glyphs: list[dict], body: float) -> list[list[dict]]:
    """Join a numerator/denominator pair that has no drawn fraction bar."""
    def text(run):
        return "".join(glyph["c"] for glyph in sorted(run, key=lambda item: (item["y"], item["x"])))

    short = [run for run in runs
             if 1 <= len(text(run).strip()) <= 4 and any(char.isdigit() for char in text(run))]
    stacked = []
    for index, first in enumerate(short):
        for second in short[index + 1:]:
            one, two = _union(item["bbox"] for item in first), _union(item["bbox"] for item in second)
            widths = (one[2] - one[0], two[2] - two[0])
            if min(widths) <= 0 or max(widths) / min(widths) > 1.7:
                continue
            if max(widths) > body * 1.25:
                continue
            vertical_gap = max(0.0, max(one[1] - two[3], two[1] - one[3]))
            if vertical_gap > body * 0.35:
                continue
            if abs((one[0] + one[2]) / 2 - (two[0] + two[2]) / 2) > body * 0.12:
                continue
            horizontal_overlap = min(one[2], two[2]) - max(one[0], two[0])
            if horizontal_overlap / min(widths) < 0.8:
                continue
            if _has_option_marker(first, glyphs, body) and _has_option_marker(second, glyphs, body):
                continue
            combined = _union([one, two])
            neighbours = []
            for other in runs:
                if other is first or other is second:
                    continue
                box = _union(item["bbox"] for item in other)
                horizontal_gap = max(0.0, max(combined[0] - box[2], box[0] - combined[2]))
                if horizontal_gap > body * 0.6:
                    continue
                if min(combined[3], box[3]) - max(combined[1], box[1]) <= 0:
                    continue
                neighbours.append(other)
            if not neighbours:
                continue
            stacked.append([*first, *second, *(glyph for run in neighbours for glyph in run)])
    return stacked


def _option_label(run: list[dict], glyphs: list[dict], body: float) -> str:
    """Read a preceding (A)-(H) marker on the same baseline, if present."""
    box = _union(item["bbox"] for item in run)
    baseline = sum(item["y"] for item in run) / len(run)
    left = [glyph for glyph in glyphs
            if abs(glyph["y"] - baseline) <= body * 0.45 and glyph["bbox"][2] <= box[0] + 1]
    if not left:
        return ""
    text = "".join(glyph["c"] for glyph in sorted(left, key=lambda item: item["x"]))[-24:]
    match = re.search(r"\(([A-H])\)\s*$", text)
    return match.group(1) if match else ""


def detect_formula_blocks(page, *, page_index: int = 0, max_blocks: int = MAX_BLOCKS_PER_PAGE) -> list[dict]:
    """Detect locally linearized formula candidates from original geometry."""
    if getattr(page, "rotation", 0) != 0:
        return []
    glyphs, body, page_box = _collect_glyphs(page)
    if not glyphs or not body:
        return []
    lines = _visual_lines(glyphs, body)
    rules = _horizontal_rules(page)
    all_runs = _formula_runs(glyphs, body)
    runs = [run for run in all_runs if any(_is_seed(glyph, body) for glyph in run)]
    runs.extend(_stacked_fraction_runs(all_runs, glyphs, body))
    width, height = page_box[2] - page_box[0], page_box[3] - page_box[1]
    candidates = []
    for run in runs:
        box = _union(item["bbox"] for item in run)
        if _area(box) / max(1.0, width * height) > 0.25:
            continue
        ordered = sorted(run, key=lambda glyph: (glyph["y"], glyph["x"]))
        text = "".join(glyph["c"] for glyph in ordered)
        if not 3 <= len(text) <= min(MAX_BLOCK_CHARS, 80):
            continue
        alnum = [glyph["c"] for glyph in ordered if glyph["c"].isalnum()]
        if len(alnum) < 3:
            continue
        formula_ratio = sum(_formula_char(glyph, body) for glyph in run) / max(1, len(run))
        if formula_ratio < 0.25:
            continue
        scripts = sum(glyph["size"] <= body * 0.82 and glyph["c"].isalnum() for glyph in run)
        math_fonts = sum(_is_math_font(glyph["font"]) for glyph in run)
        symbols = sum(glyph["c"] in _MATH_SYMBOLS for glyph in run)
        digits = sum(glyph["c"].isdigit() for glyph in run)
        if not digits and not math_fonts:
            continue
        if any(ord(char) < 32 for char in text):
            continue
        stacked = len({round(glyph["y"], 1) for glyph in run}) > 1 and any(
            abs(glyph["y"] - max(item["y"] for item in run)) > body * 0.5 for glyph in run)
        rule_hit = any(_overlap(box, rule) > 0 for rule in rules)
        strong = bool(stacked or rule_hit or (math_fonts and digits)
                      or (scripts and math_fonts + symbols) or symbols >= 2)
        if not strong:
            continue
        reasons = []
        if scripts:
            reasons.append("上下标")
        if math_fonts:
            reasons.append("数学符号字体")
        if symbols:
            reasons.append("运算符号")
        if stacked or rule_hit:
            reasons.append("分式或纵向叠层")
        nearby = _nearest_formula_line(lines, box, body)
        context_before, context_after = ((nearby[1], nearby[2]) if nearby is not None else ("", ""))
        candidates.append({"bbox": box, "glyphs": run, "reasons": reasons,
                           "option": _option_label(run, glyphs, body),
                           "question_number": _question_number(lines, box),
                           "context_before": context_before[-70:],
                           "context_after": context_after[:70],
                           "score": scripts * 2 + math_fonts * 2 + symbols * 2 + (4 if stacked or rule_hit else 0)})
    # Merge runs that describe the same physical formula.
    merged = []
    for candidate in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        for existing in merged:
            if _overlap(existing["bbox"], candidate["bbox"]) / max(
                    1.0, min(_area(existing["bbox"]), _area(candidate["bbox"]))) >= 0.55:
                existing["bbox"] = _union([existing["bbox"], candidate["bbox"]])
                existing["glyphs"] = list({id(glyph): glyph for glyph in
                                           [*existing["glyphs"], *candidate["glyphs"]]}.values())
                existing["reasons"] = list(dict.fromkeys([*existing["reasons"], *candidate["reasons"]]))
                existing["option"] = existing["option"] or candidate["option"]
                existing["question_number"] = existing.get("question_number") or candidate.get("question_number")
                existing["context_before"] = existing.get("context_before") or candidate.get("context_before", "")
                existing["context_after"] = existing.get("context_after") or candidate.get("context_after", "")
                existing["score"] = max(existing["score"], candidate["score"])
                break
        else:
            merged.append(dict(candidate))
    blocks = []
    for candidate in merged:
        box = candidate["bbox"]
        ordered = sorted(candidate["glyphs"], key=lambda glyph: (glyph["y"], glyph["x"]))
        text = "".join(glyph["c"] for glyph in ordered)
        if not 3 <= len(text) <= min(MAX_BLOCK_CHARS, 80):
            continue
        alnum = "".join(glyph["c"] for glyph in ordered if glyph["c"].isalnum())
        if len(alnum) < 3:
            continue
        blocks.append({
            "id": "", "bbox": [round(box[0] / width * 1000, 4), round(box[1] / height * 1000, 4),
                               round(box[2] / width * 1000, 4), round(box[3] / height * 1000, 4)],
            "text": text, "alnum": alnum, "option": candidate["option"],
            "question_number": candidate.get("question_number"),
            "context_before": candidate.get("context_before", ""),
            "context_after": candidate.get("context_after", ""),
            "kind": "glyph", "strict_glyphs": True,
            "reasons": candidate["reasons"] or ["公式结构"],
            "score": candidate["score"], "page_index": page_index,
        })
    vector_blocks = _vector_formula_blocks(page, glyphs, body, page_box, lines, page_index,
                                           max_blocks=max_blocks)
    for vector in vector_blocks:
        vector_box = vector["bbox"]
        if any(_overlap(vector_box, block["bbox"]) / max(
                1.0, min(_area(vector_box), _area(block["bbox"]))) >= 0.6 for block in blocks):
            continue
        blocks.append(vector)
    blocks.sort(key=lambda item: (-item["score"], item["bbox"][1], item["bbox"][0]))
    blocks = blocks[:max_blocks]
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, block in enumerate(blocks, 1):
        block["id"] = f"p{page_index + 1}-m{index}"
    return blocks


def _normalised_text(markdown: str) -> tuple[list[str], list[int]]:
    chars, positions = [], []
    index = 0
    while index < len(markdown):
        tag = _MARKUP_TAG.match(markdown, index)
        if tag:
            index = tag.end()
            continue
        char = markdown[index]
        if char.isalnum():
            chars.append(char)
            positions.append(index)
        index += 1
    return chars, positions


def _letters(value: str) -> Counter:
    return Counter(char for char in value if char.isalpha())


def _trailing_subset(value: str, block: dict) -> tuple[int, str] | None:
    """Return a short trailing token that clearly belongs to this formula."""
    text = value.rstrip()
    if not text:
        return None
    split = max(text.rfind(" "), text.rfind("\t"))
    token = text[split + 1:] if split >= 0 else text
    if not 1 <= len(token) <= 8:
        return None
    remaining = Counter(block["alnum"])
    for char in token:
        if not char.isalnum():
            return None
        if remaining[char] <= 0:
            return None
        remaining[char] -= 1
    start = len(value[:len(text)]) - len(token)
    return start, token


def _option_math_span(markdown: str, block: dict) -> tuple[int, int, str] | None:
    """Locate one (A)-(H) option, including a detached fraction component."""
    label = block.get("option")
    if not label:
        return None
    expected = _letters(block["alnum"])
    total = len(block["alnum"])
    pattern = re.compile(rf"(?m)^[ \t]*\({re.escape(label)}\)[ \t]*")
    start = _option_scan_start(markdown, block)
    heading = re.search(r"(?m)^[ \t]*\d{1,3}[.．][ \t]+\S", markdown[start:])
    end = start + heading.start() if heading else len(markdown)

    def find_in(range_start, range_end):
        spans = []
        for match in pattern.finditer(markdown, range_start, range_end):
            option_end = range_end
            for following in re.finditer(r"(?m)^[ \t]*\([A-H]\)[ \t]*",
                                         markdown[match.end():range_end]):
                option_end = match.end() + following.start()
                break
            while option_end > match.end():
                token_end = option_end
                while token_end > match.end() and markdown[token_end - 1].isspace():
                    token_end -= 1
                token_start = max(
                    match.end(),
                    max(markdown.rfind(" ", match.end(), token_end),
                        markdown.rfind("\n", match.end(), token_end),
                        markdown.rfind("\t", match.end(), token_end)) + 1,
                )
                token = markdown[token_start:token_end]
                token_letters = _letters(_MARKUP_TAG.sub("", token).replace("*", ""))
                if not token_letters:
                    option_end = token_start
                    continue
                if _glyph_counters_compatible(expected, token_letters, allow_missing=True):
                    option_end = token_end
                    break
                option_end = token_start
            if option_end - match.end() > max(80, total * 10):
                continue
            value = markdown[match.end():option_end]
            if "$" in value or _IMAGE_OR_TABLE.search(value):
                continue
            alnum = [char for char in _MARKUP_TAG.sub("", value) if char.isalnum()]
            letters = _letters("".join(alnum))
            if (not letters and expected) or len(alnum) > total + 8 or not _glyph_counters_compatible(
                    expected, letters, allow_missing=True):
                continue
            left = match.start()
            line_start = markdown.rfind("\n", 0, left) + 1
            previous_end = line_start - 1
            previous_start = markdown.rfind("\n", 0, previous_end) + 1
            previous = (markdown[previous_start:previous_end]
                        if previous_end >= previous_start else "")
            if label == "A" and not re.match(r"\s*\([A-H]\)", previous):
                trailing = _trailing_subset(previous, block)
                if trailing is not None:
                    left = previous_start + trailing[0]
            right = option_end
            while right > match.end() and markdown[right - 1].isspace():
                right -= 1
            spans.append((_glyph_score(expected, letters), left, right, f"({label}) "))
        if not spans:
            return None
        spans.sort(reverse=True)
        if len(spans) > 1 and spans[0][0] == spans[1][0]:
            return None
        return spans[0][1:]

    found = find_in(start, end)
    if found is not None:
        return found
    if start or end < len(markdown):
        return find_in(0, len(markdown))
    return None


def _expand_emphasis(markdown: str, left: int, right: int) -> tuple[int, int]:
    """Keep an adjacent italic delimiter from being orphaned by a rewrite."""
    if markdown[left:right].count("*") % 2:
        if left > 0 and markdown[left - 1] == "*":
            left -= 1
        elif right < len(markdown) and markdown[right] == "*":
            right += 1
    return left, right


def _expand_markup(markdown: str, left: int, right: int) -> tuple[int, int]:
    """Consume adjacent script tags so no empty <sup>/<sub> is left behind."""
    while markdown[right:].lower().startswith(("</sup>", "</sub>")):
        right += 6
    while markdown[:left].lower().endswith(("<sup>", "<sub>")):
        left -= 5
    return left, right


def _unique_markdown_span(markdown: str, block: dict) -> tuple[int, int, str] | None:
    """Find one alphanumeric window that can only be this block."""
    option_span = _option_math_span(markdown, block)
    if option_span is not None:
        return (*_expand_markup(markdown, *_expand_emphasis(markdown, *option_span[:2])),
                option_span[2])
    expected = Counter(block["alnum"])
    total = sum(expected.values())
    if total < 3 or total > len(markdown):
        return None
    chars, positions = _normalised_text(markdown)
    if total > len(chars):
        return None
    window = Counter(chars[:total])
    matches = []
    for start in range(len(chars) - total + 1):
        if start:
            window[chars[start - 1]] -= 1
            if window[chars[start - 1]] == 0:
                del window[chars[start - 1]]
            window[chars[start + total - 1]] += 1
        if window == expected:
            left, right = positions[start], positions[start + total - 1] + 1
            original = markdown[left:right]
            if ("$" in original or _IMAGE_OR_TABLE.search(original)
                    or right - left > max(40, total * 6)):
                continue
            matches.append((left, right))
            if len(matches) > 8:
                return None
    if len(matches) == 1:
        return (*_expand_markup(markdown, *_expand_emphasis(markdown, *matches[0])), "")
    if not matches:
        return None
    # Adjacent options can make one multiset occur twice; the nearest printed
    # option marker before the window disambiguates it.
    preferred = []
    for left, right in matches:
        head = markdown[max(0, left - 300):left]
        markers = list(re.finditer(r"\(([A-H])\)", head))
        if markers and markers[-1].group(1) == block.get("option"):
            preferred.append((left, right))
    return ((*_expand_markup(markdown, *_expand_emphasis(markdown, *preferred[0])), "")
            if len(preferred) == 1 else None)


_OPTION_MARKER = re.compile(r"(?m)^[ \t]*\(([A-H])\)[ \t]*")
_FIRST_OPTION = re.compile(r"(?m)^[ \t]*\(A\)[ \t]*")


def _option_scan_start(markdown: str, block: dict) -> int:
    """Start at this question's first option, skipping an interleaved column.

    pdf-inspector orders text by physical row. In two-column pages the next
    question's lines can therefore appear between this question's (A) and (D)
    markers. Anchoring on the first (A) keeps the later slots attached to the
    right question.
    """
    number = block.get("question_number")
    if not isinstance(number, int) or number <= 0:
        return 0
    heading = re.search(rf"(?m)^[ \t]*{number}[.．][ \t]+\S", markdown)
    if not heading:
        return 0
    first = _FIRST_OPTION.search(markdown, heading.end())
    return first.start() if first else heading.end()


def _option_insertion_span(markdown: str, block: dict) -> tuple[int, int, str, str] | None:
    """Find the visual option slot when a vector formula left no text behind."""
    label = block.get("option")
    if not label:
        return None
    start = _option_scan_start(markdown, block)
    for match in _OPTION_MARKER.finditer(markdown, start):
        if match.group(1) != label:
            continue
        line_end = markdown.find("\n", match.end())
        if line_end < 0:
            line_end = len(markdown)
        rest = markdown[match.end():line_end].strip()
        if not rest:
            return line_end, line_end, " ", ""
        if "[公式待核对]" in rest:
            return None
        if "$" in rest:
            return None
        if block.get("kind") == "vector":
            # A nonempty slot may already be a valid native/OCR formula.
            continue
    return None


def _option_fragment_span(markdown: str, block: dict) -> tuple[int, int, str] | None:
    """Replace a flattened option whose native text lost glyphs (e.g. ℓ)."""
    label = block.get("option")
    expected = Counter(block.get("alnum") or "")
    if not label or not expected:
        return None
    start = _option_scan_start(markdown, block)
    for match in _OPTION_MARKER.finditer(markdown, start):
        if match.group(1) != label:
            continue
        following = _OPTION_MARKER.search(markdown, match.end())
        heading = re.search(r"(?m)^[ \t]*\d{1,3}[.．][ \t]+\S", markdown[match.end():])
        blank = re.search(r"\n[ \t]*\n", markdown[match.end():])
        stops = [match.end() + value.start() for value in (heading, blank) if value]
        if following:
            stops.append(following.start())
        end = min(stops) if stops else len(markdown)
        value = markdown[match.end():end]
        if "$" in value or _IMAGE_OR_TABLE.search(value) or len(value) > 80:
            continue
        letters = _letters(value)
        alnum = [char for char in value if char.isalnum()]
        if not letters:
            continue
        plain = _MARKUP_TAG.sub("", value).replace("*", "")
        if not _glyph_counters_compatible(
                _letters(block.get("alnum") or ""), _letters(plain),
                allow_missing=True, extra_slack=2):
            continue
        if re.search(r"[A-Za-z]{3,}", plain):
            continue
        left = match.start()
        line_start = markdown.rfind("\n", 0, left) + 1
        if label == "A" and line_start > 0:
            previous_end = line_start - 1
            while previous_end > 0:
                previous_start = markdown.rfind("\n", 0, previous_end) + 1
                previous = markdown[previous_start:previous_end]
                if previous.strip():
                    break
                previous_end = previous_start - 1
            else:
                previous = ""
            stripped = _MARKUP_TAG.sub("", previous).replace("*", "").strip()
            if (stripped and not _OPTION_MARKER.match(previous)
                    and not re.search(r"[A-Za-z]{3,}", stripped)
                    and len(stripped) <= 12):
                remaining = Counter(expected)
                for char in stripped:
                    if char.isalnum():
                        if remaining[char] <= 0:
                            break
                        remaining[char] -= 1
                else:
                    left = previous_start
        right = end
        while right > match.end() and markdown[right - 1].isspace():
            right -= 1
        return (*_expand_markup(markdown, left, right), f"({label}) ")
    return None


def _normalised_with_positions(text: str) -> tuple[str, list[int]]:
    chars, positions = [], []
    pending_space = False
    index = 0
    while index < len(text):
        tag = _MARKUP_TAG.match(text, index)
        if tag:
            index = tag.end()
            continue
        char = text[index]
        if char == "*":
            index += 1
            continue
        if char.isspace():
            pending_space = bool(chars)
            index += 1
            continue
        if pending_space:
            chars.append(" ")
            positions.append(index)
            pending_space = False
        chars.append(char)
        positions.append(index)
        index += 1
    return "".join(chars), positions


def _normalised_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _visible_formula_alnum(value: str) -> str:
    """Count letters a LaTeX expression would visibly print."""
    produced = []
    for match in re.finditer(r"\\([A-Za-z]+)|[A-Za-z0-9]", value):
        command = match.group(1)
        produced.append(_COMMAND_GLYPHS.get(command, "") if command else match.group(0))
    return "".join(produced)


def _glyph_counters_compatible(
    required: Counter, produced: Counter, *, allow_missing: bool = False,
    extra_slack: int = 0,
) -> bool:
    # Fraction bars and superscripts make the text layer repeat digits (for
    # example 1/12 can appear as "12"), so digit counts are not stable enough
    # to prove a hallucination. Letters remain the meaningful evidence.
    required_letters = Counter(char for char in required if char.isalpha())
    produced_letters = Counter(char for char in produced if char.isalpha())
    common = required_letters & produced_letters
    missing = required_letters - common
    extra = produced_letters - common
    for char in list(missing):
        for replacement in _GLYPH_SUBSTITUTIONS.get(char, ()):
            count = min(missing[char], extra.get(replacement, 0))
            if count:
                missing[char] -= count
                extra[replacement] -= count
    return ((allow_missing or not any(count > 0 for count in missing.values()))
            and sum(count for count in extra.values() if count > 0) <= extra_slack)


def _glyph_score(required: Counter, produced: Counter) -> int:
    required_letters = Counter(char for char in required if char.isalpha())
    produced_letters = Counter(char for char in produced if char.isalpha())
    score = sum((required_letters & produced_letters).values())
    remaining = required_letters - produced_letters
    available = produced_letters - required_letters
    for char in list(remaining):
        for replacement in _GLYPH_SUBSTITUTIONS.get(char, ()):
            count = min(remaining[char], available.get(replacement, 0))
            if count:
                score += count
                remaining[char] -= count
                available[replacement] -= count
    return score


def _strip_trailing_punctuation(latex: str, block: dict) -> str:
    value = latex.strip()
    after = str(block.get("context_after") or "").lstrip()
    if value and after and after[0] in ",.;:" and value.endswith(after[0]):
        value = value[:-1].rstrip()
    return value


def _context_insertion_span(markdown: str, block: dict) -> tuple[int, int, str, str] | None:
    """Insert a visually recovered inline formula between unique text anchors."""
    before = _normalised_fragment(block.get("context_before"))[-60:]
    after = _normalised_fragment(block.get("context_after"))[:60]
    if not before and not after:
        return None
    source, positions = _normalised_with_positions(markdown)
    if not source:
        return None

    def occurrences(value):
        found = []
        start = 0
        while value:
            index = source.find(value, start)
            if index < 0:
                break
            found.append(index)
            if len(found) > 8:
                return None
            start = index + 1
        return found

    before_hits = occurrences(before) if before else []
    after_hits = occurrences(after) if after else []
    if before and before_hits is None or after and after_hits is None:
        return None
    pairs = []
    if before and after:
        for before_start in before_hits:
            before_end = before_start + len(before)
            for after_start in after_hits:
                if before_end <= after_start <= before_end + 160:
                    pairs.append((after_start - before_end, before_start, before_end, after_start))
        if not pairs:
            return None
        pairs.sort()
        # A repeated phrase must have one clearly nearest mate. Two equally
        # close anchors mean the page text cannot identify the formula.
        if len(pairs) > 1 and pairs[0][0] == pairs[1][0]:
            return None
        _, before_start, before_end, after_start = pairs[0]
    elif before:
        pairs = [(value, value + len(before), None) for value in before_hits]
    else:
        pairs = [(None, None, value) for value in after_hits]
    if not before and after or before and not after:
        if len(pairs) != 1:
            return None
        before_start, before_end, after_start = pairs[0]
    if before_start is not None and before:
        left = positions[before_end - 1] + 1
    elif after_start is not None:
        left = positions[after_start]
    else:
        return None
    if after_start is not None and after:
        right = positions[after_start]
    else:
        right = left
    gap = markdown[left:right]
    if gap and not gap.isspace():
        return None
    prefix = "" if left > 0 and markdown[left - 1].isspace() else " "
    suffix = ("" if right >= len(markdown) or markdown[right].isspace()
              or markdown[right] in ",.;:!?)]}" else " ")
    return left, right, prefix, suffix


def _context_replacement_span(markdown: str, block: dict) -> tuple[int, int, str, str] | None:
    """Replace a flattened formula whose glyphs were reordered by extraction."""
    before = _normalised_fragment(block.get("context_before"))[-60:]
    after = _normalised_fragment(block.get("context_after"))[:60]
    expected = Counter(block.get("alnum") or "")
    if not before or not after or not expected:
        return None
    source, positions = _normalised_with_positions(markdown)
    if not source:
        return None

    def occurrences(value):
        found = []
        start = 0
        while value:
            index = source.find(value, start)
            if index < 0:
                break
            found.append(index)
            if len(found) > 8:
                return None
            start = index + 1
        return found

    before_hits = occurrences(before)
    after_hits = occurrences(after)
    if before_hits is None or after_hits is None:
        return None
    pairs = []
    for before_start in before_hits:
        before_end = before_start + len(before)
        for after_start in after_hits:
            if before_end <= after_start <= before_end + 240:
                pairs.append((after_start - before_end, before_start, before_end, after_start))
    if not pairs:
        return None
    pairs.sort()
    if len(pairs) > 1 and pairs[0][0] == pairs[1][0]:
        return None
    _, _, before_end, after_start = pairs[0]
    left = positions[before_end - 1] + 1
    right = positions[after_start]
    tail = re.search(r"([,.;:])\s*([0-9]{1,4})$", str(block.get("text") or ""))
    if tail and right < len(markdown) and markdown[right] == tail.group(1):
        cursor = right + 1
        while cursor < len(markdown) and markdown[cursor].isspace():
            cursor += 1
        if markdown.startswith(tail.group(2), cursor):
            right = cursor + len(tail.group(2))
    gap = markdown[left:right]
    if not gap.strip() or len(gap) > 240 or _IMAGE_OR_TABLE.search(gap):
        return None
    if re.search(r"[A-Za-z]{3,}", _MARKUP_TAG.sub("", gap)):
        return None
    observed = Counter(char for char in gap if char.isalnum())
    if not _glyph_counters_compatible(expected, observed):
        return None
    prefix = "" if left > 0 and markdown[left - 1].isspace() else " "
    suffix = ("" if right >= len(markdown) or markdown[right].isspace()
              or markdown[right] in ",.;:!?)]}" else " ")
    return left, right, prefix, suffix


def _formula_span(markdown: str, block: dict):
    if block.get("kind") == "vector":
        return _option_insertion_span(markdown, block) or _context_insertion_span(markdown, block)
    if block.get("option"):
        return (_option_fragment_span(markdown, block)
                or _option_math_span(markdown, block))
    span = _unique_markdown_span(markdown, block)
    if span is not None:
        return span
    return _context_insertion_span(markdown, block) or _context_replacement_span(markdown, block)


def _balanced(value: str, left: str, right: str) -> bool:
    depth = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\":
            index += 2
            continue
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return depth == 0


def validate_formula_latex(latex, block: dict) -> tuple[bool, str]:
    """Validate one model reply before it may replace original text."""
    if not isinstance(latex, str) or not 1 <= len(latex.strip()) <= MAX_LATEX_CHARS:
        return False, "empty"
    value = latex.strip()
    if "$" in value or "\ufffd" in value or "<" in value or ">" in value:
        return False, "delimiter_or_replacement_character"
    if "\n" in value or "\r" in value:
        return False, "multi_line"
    if not re.search(r"[A-Za-z0-9]", value):
        return False, "no_visible_token"
    for left, right in (("{", "}"), ("(", ")"), ("[", "]")):
        if not _balanced(value, left, right):
            return False, "unbalanced"
    if len(re.findall(r"\\left\b", value)) != len(re.findall(r"\\right\b", value)):
        return False, "unbalanced_left_right"
    environments = re.findall(r"\\(begin|end)\{([^{}]+)\}", value)
    stack = []
    for kind, name in environments:
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False, "unbalanced_environment"
    if stack:
        return False, "unbalanced_environment"
    for command in re.findall(r"\\([A-Za-z]+|[^A-Za-z])", value):
        if command not in _COMMANDS:
            return False, "unknown_command"
    # LaTeX command names must not satisfy glyph coverage; a hallucinated
    # \dfrac cannot stand in for a missing d of the original figure.
    if block.get("strict_glyphs", True):
        produced = Counter(_visible_formula_alnum(value))
        required = Counter(block.get("alnum") or "")
        if any(char.isdigit() for char in required) and not any(
                char.isdigit() for char in produced):
            return False, "missing_numeric"
        if not _glyph_counters_compatible(required, produced):
            return False, "missing_glyph"
    return True, ""


def _crop_messages(image_path: str, blocks: list[dict], check_cancelled: Callable[[], None]) -> list[dict]:
    messages = []
    try:
        with Image.open(image_path) as source:
            source.load()
            width, height = source.size
            for block in blocks:
                check_cancelled()
                box = block["bbox"]
                pad_x, pad_y = 0.006, 0.008
                extent = ((box[0] / 1000 - pad_x) * width, (box[1] / 1000 - pad_y) * height,
                          (box[2] / 1000 + pad_x) * width, (box[3] / 1000 + pad_y) * height)
                extent = (max(0, extent[0]), max(0, extent[1]),
                          min(width, extent[2]), min(height, extent[3]))
                size = (max(1, math.ceil(extent[2] - extent[0])),
                        max(1, math.ceil(extent[3] - extent[1])))
                with source.transform(size, Image.Transform.EXTENT, extent, Image.Resampling.BICUBIC) as crop:
                    with BytesIO() as buffer:
                        crop.save(buffer, format="PNG")
                        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                messages.extend([
                    {"type": "text", "text": "block_id=" + block["id"]},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
                ])
    except TaskCancelled:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取或裁剪公式块预览图。") from exc
    return messages


def _parse_formula_reply(raw: str, blocks: list[dict]) -> dict[str, str | None]:
    value = parse_ai_json(raw)
    if not isinstance(value, dict) or not isinstance(value.get("blocks"), list):
        raise ValueError("公式块识别结果格式无效。")
    expected = {block["id"] for block in blocks}
    answers = {}
    for item in value["blocks"]:
        if not isinstance(item, dict) or set(item) != {"id", "latex"}:
            raise ValueError("公式块识别结果字段无效。")
        identifier = item.get("id")
        if identifier not in expected or identifier in answers:
            raise ValueError("公式块识别结果引用未知或重复的块。")
        latex = item.get("latex")
        if latex is not None and not isinstance(latex, str):
            raise ValueError("公式块识别结果不是文本。")
        answers[identifier] = latex
    if set(answers) != expected:
        raise ValueError("公式块识别结果不完整。")
    return answers


def request_formula_blocks(
    image_path: str, blocks: list[dict], *, page_number: int, provider,
    post=post_chat_completion, check_cancelled: Callable[[], None] = lambda: None,
    report_attempt: Callable[[dict], None] = lambda _event: None,
) -> dict[str, str | None]:
    """Ask one page-level request for at most ten block-only LaTeX strings."""
    content = [{"type": "text", "text": (
        "下面是从同一页 PDF 裁出的公式块，按 block_id 对应。逐块只转录实际可见字形为一个 LaTeX 数学表达式，"
        "不要写 $ 或 \\( \\)，不要解释、编号或补全看不到的内容。保留字母大小写、数字、上下标、分式、根号、"
        "单位和向量箭头；原页的 u/v 是不同字母，不要按题意替换；α 写成 \\alpha，ℓ 写成 \\ell；"
        "分式用 \\dfrac；不要附带句末标点；看不清或无法确定时 latex 返回 null。"
        '只返回 JSON 对象：{"blocks":[{"id":"p1-m1","latex":"..."}]}，每个给定 block_id 恰好一次。'
    )}]
    content.extend(_crop_messages(image_path, blocks, check_cancelled))
    payload = {"model": provider.model_name, "messages": [{"role": "user", "content": content}],
               "max_tokens": 3000, "stream": False}
    payload = apply_model_thinking_policy(payload, provider=provider, task="ocr")
    payload = apply_structured_output_policy(payload, provider=provider)

    def validate(body):
        raw = completion_content(body, "公式块识别", max_chars=40000, allow_missing_finish=True)
        return _parse_formula_reply(raw, blocks)

    return request_pdf_vision(
        provider, payload, post=post, validate=validate, label="公式块识别",
        stage="formula_blocks", page_index=page_number - 1,
        check_cancelled=check_cancelled, report_attempt=report_attempt,
    )


def refine_pdf_formulas(
    file_bytes: bytes, page_texts: list[str], page_images: list[str], page_indices: list[int], *,
    provider=None, post=post_chat_completion, check_cancelled: Callable[[], None] = lambda: None,
    report_attempt: Callable[[dict], None] = lambda _event: None,
    eligible_pages: set[int] | None = None,
    max_pages: int = MAX_PAGES_PER_TASK,
) -> dict:
    """Refine native pages only; OCR pages already come from a visual transcript."""
    report = {"status": "skipped", "pages_with_blocks": 0, "blocks": 0, "matched": 0,
              "accepted": 0, "rejected": 0, "unmatched": 0, "calls": 0,
              "usage": {}, "pages": []}
    if not isinstance(file_bytes, (bytes, bytearray)) or not page_texts:
        return {"page_texts": list(page_texts), "diagnostics": report}
    provider = provider or resolve_ocr_provider(os.getenv("OCR_PREFER_ENGINE", "siliconflow"))
    if not provider.api_key or not provider.chat_completions_url or not provider.supports_image_input:
        report["status"] = "unavailable"
        return {"page_texts": list(page_texts), "diagnostics": report}
    import pymupdf as fitz

    texts = list(page_texts)
    usage = Counter()

    def capture_attempt(event):
        report_attempt(event)
        value = event.get("usage") if isinstance(event, dict) else None
        if isinstance(value, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if _finite(value.get(key)):
                    usage[key] += int(value[key])

    requested_pages = 0
    with fitz.open(stream=bytes(file_bytes), filetype="pdf") as document:
        for local_index, page_index in enumerate(page_indices):
            if requested_pages >= max_pages:
                break
            check_cancelled()
            if not (0 <= page_index < len(document)) or local_index >= len(texts):
                continue
            if eligible_pages is not None and page_index not in eligible_pages:
                continue
            text = str(texts[local_index] or "")
            if not text.strip():
                continue
            blocks = detect_formula_blocks(document[page_index], page_index=page_index)
            # Empty vector-only candidates are usually diagram strokes or
            # underlines, not recoverable formulas. Keeping them would insert
            # review markers into ordinary prose.
            blocks = [
                block for block in blocks
                if block.get("kind") != "vector" or block.get("alnum")
                or block.get("context_before")
            ]
            if not blocks:
                continue
            report["pages_with_blocks"] += 1
            report["blocks"] += len(blocks)
            matched = [(block, _formula_span(text, block)) for block in blocks]
            located = [(block, span) for block, span in matched if span is not None]
            report["unmatched"] += len(blocks) - len(located)
            for block, span in matched:
                if span is None:
                    report.setdefault("unmatched_blocks", []).append({
                        "page_number": page_index + 1, "block_id": block["id"],
                        "option": block["option"], "text": block["text"][:80],
                    })
            def conflicting(first, second):
                first_start, first_end = first[0], first[1]
                second_start, second_end = second[0], second[1]
                if first_start < second_end and second_start < first_end:
                    return True
                return ((first_start == first_end and second_start <= first_start <= second_end)
                        or (second_start == second_end and first_start <= second_start <= first_end))

            conflicted = set()
            for first_index in range(len(located)):
                for second_index in range(first_index + 1, len(located)):
                    if conflicting(located[first_index][1], located[second_index][1]):
                        conflicted.update((first_index, second_index))
            if conflicted:
                for index in sorted(conflicted):
                    block = located[index][0]
                    report.setdefault("unmatched_blocks", []).append({
                        "page_number": page_index + 1, "block_id": block["id"],
                        "option": block["option"], "text": block["text"][:80],
                        "reason": "overlapping_span",
                    })
                report["unmatched"] += len(conflicted)
                located = [
                    item for index, item in enumerate(located) if index not in conflicted
                ]
            if not located:
                report["pages"].append({"page_number": page_index + 1, "blocks": len(blocks),
                                        "matched": 0, "accepted": 0, "rejected": 0})
                continue
            report["matched"] += len(located)
            requested_pages += 1
            if local_index >= len(page_images) or not page_images[local_index]:
                report["rejected"] += len(located)
                report["pages"].append({"page_number": page_index + 1, "blocks": len(blocks),
                                        "matched": len(located), "accepted": 0,
                                        "rejected": len(located), "error": "missing_page_image"})
                continue
            try:
                answers = request_formula_blocks(
                    page_images[local_index], [block for block, _ in located],
                    page_number=page_index + 1, provider=provider, post=post,
                    check_cancelled=check_cancelled, report_attempt=capture_attempt,
                )
            except TaskCancelled:
                raise
            except Exception as exc:
                report["pages"].append({"page_number": page_index + 1, "blocks": len(blocks),
                                        "matched": len(located), "accepted": 0, "rejected": 0,
                                        "error": type(exc).__name__})
                continue
            report["calls"] += 1
            edits = []
            accepted = rejected = 0
            for block, span in located:
                latex = answers.get(block["id"])
                ok, reason = validate_formula_latex(latex, block)
                start, end, prefix = span[:3]
                suffix = span[3] if len(span) > 3 else ""
                if ok:
                    cleaned = _strip_trailing_punctuation(latex, block)
                    edits.append((start, end, prefix + f"${cleaned}$" + suffix))
                    accepted += 1
                else:
                    marker_gap = text[max(0, start - 20):min(len(text), end + 20)]
                    pending = [edit[0] for edit in edits if "[公式待核对]" in edit[2]]
                    if ("[公式待核对]" not in marker_gap
                            and not any(abs(start - position) <= 24 for position in pending)):
                        edits.append((end, end, " [公式待核对]"))
                    rejected += 1
                    report.setdefault("rejected_reasons", []).append(
                        {"page_number": page_index + 1, "block_id": block["id"], "reason": reason})
            for start, end, replacement in sorted(edits, reverse=True):
                text = text[:start] + replacement + text[end:]
            texts[local_index] = text
            report["accepted"] += accepted
            report["rejected"] += rejected
            report["pages"].append({"page_number": page_index + 1, "blocks": len(blocks),
                                    "matched": len(located), "accepted": accepted, "rejected": rejected})
    report["status"] = "completed" if requested_pages else "no_blocks"
    report["usage"] = dict(usage)
    return {"page_texts": texts, "diagnostics": report}
