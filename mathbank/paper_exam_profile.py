"""Course-aware paper templates and exam-structure checks.

Mathematics keeps its historical templates.  Physics papers use the bundled
exam profiles instead of the 19-question Gaokao layout, and AP Physics C
papers must satisfy the 2027 two-section contract.
"""

from __future__ import annotations

from mathbank.course_curriculum import (
    CourseCurriculumError, curriculum_catalog,
)
from mathbank.physics_groups import PhysicsGroupError, load_physics_groups


PAPER_TYPES = frozenset({"exam", "quiz", "exam_19", "physics_exam", "ap_exam"})
CHOICE_TYPES = frozenset({"single_choice", "multi_choice"})


class PaperStructureError(ValueError):
    """A paper template or its section structure cannot be exported."""


def _course_kind(course):
    if course["subject"] == "math":
        return "math"
    if course["id"] == "cn_hs_physics":
        return "cn_physics"
    if course["id"] in {"ap_physics_c_mechanics", "ap_physics_c_em"}:
        return "ap_physics"
    return "physics"


def _section_totals(items):
    choice_count = choice_score = written_count = written_score = 0
    for item in items:
        question_type = str(item.get("question_type") or "single_choice")
        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        if question_type in CHOICE_TYPES:
            choice_count += 1
            choice_score += score
        else:
            written_count += 1
            written_score += score
    return {
        "choice_count": choice_count,
        "choice_score": choice_score,
        "written_count": written_count,
        "written_score": written_score,
    }


def _province_group(metadata):
    group_id = (metadata or {}).get("province_group_id")
    if group_id in (None, ""):
        return None
    try:
        groups = load_physics_groups()["groups"]
    except PhysicsGroupError as exc:
        raise PaperStructureError(str(exc)) from exc
    group = next((row for row in groups if row["id"] == group_id), None)
    if group is None:
        raise PaperStructureError("所选物理省份组不存在，请重新载入省份组列表")
    return group


def validate_paper_structure(
    course, curriculum_id, paper_type, items, metadata=None,
):
    """Validate one paper and return its stored, JSON-serializable profile."""

    kind = _course_kind(course)
    if paper_type not in PAPER_TYPES:
        raise PaperStructureError("不支持的试卷模板。")
    if kind == "math":
        if paper_type in {"physics_exam", "ap_exam"}:
            raise PaperStructureError("数学试卷不能使用物理或 AP 卷模板。")
        return {"kind": "math", "paper_type": paper_type}
    if paper_type == "exam_19":
        raise PaperStructureError("物理试卷不能使用 19 题高考数学模板。")

    if kind == "cn_physics":
        if paper_type == "ap_exam":
            raise PaperStructureError("国内物理试卷不能使用 AP 卷模板。")
        group = _province_group(metadata)
        if group is not None:
            if group.get("curriculum_id") != curriculum_id:
                raise PaperStructureError("省份组与当前物理教材目录不一致。")
            return {
                "kind": kind,
                "paper_type": paper_type,
                "province_group_id": group["id"],
                "exam_profile_id": group.get("exam_profile_id"),
            }
        return {"kind": kind, "paper_type": paper_type}

    if paper_type != "ap_exam":
        raise PaperStructureError("AP Physics C 试卷必须使用 AP 2027 卷模板。")

    try:
        catalog = curriculum_catalog(course, curriculum_id)
    except CourseCurriculumError as exc:
        raise PaperStructureError(str(exc)) from exc
    exam = catalog.get("exam") or {}
    required_choice = exam.get("multiple_choice")
    required_written = exam.get("free_response")
    if type(required_choice) is not int or type(required_written) is not int:
        raise PaperStructureError("AP 课程目录缺少 2027 考试结构，无法校验试卷。")
    totals = _section_totals(items)
    if (totals["choice_count"] != required_choice
            or totals["written_count"] != required_written):
        raise PaperStructureError(
            f"AP 2027 试卷须为 {required_choice} 道选择题 + {required_written} 道自由作答题，"
            f"当前为 {totals['choice_count']} + {totals['written_count']}。"
        )
    if totals["choice_score"] <= 0 or totals["written_score"] <= 0:
        raise PaperStructureError("AP 2027 选择题与自由作答题的分值必须大于 0。")
    if totals["choice_score"] != totals["written_score"]:
        raise PaperStructureError(
            "AP 2027 选择题与自由作答题各占全卷 50%，"
            f"当前两部分分别为 {totals['choice_score']} 分和 {totals['written_score']} 分。"
        )
    return {
        "kind": kind,
        "paper_type": paper_type,
        "exam_year": exam.get("year"),
        "multiple_choice": required_choice,
        "multiple_choice_minutes": exam.get("multiple_choice_minutes"),
        "free_response": required_written,
        "free_response_minutes": exam.get("free_response_minutes"),
        "section_scores": {
            "multiple_choice": totals["choice_score"],
            "free_response": totals["written_score"],
        },
    }
