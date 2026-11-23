"""Course-aware curriculum trees shared by question, import and paper flows.

Mathematics keeps its six editable textbook trees.  Bundled physics courses
are converted from their official catalog resources into the same three-level
shape so callers do not need a second location protocol.
"""

from __future__ import annotations

from copy import deepcopy
import re

from mathbank.courses import CourseScopeError, get_course
from mathbank.curriculums import load_curriculum, normalize_version_code
from mathbank.physics_curriculums import (
    PhysicsCurriculumError, load_physics_curriculum,
)


_ISBN_SUFFIX = re.compile(r"\s*[（(]\s*ISBN[^）)]*[）)]\s*\Z", re.IGNORECASE)


class CourseCurriculumError(ValueError):
    """A course curriculum cannot service the requested question location."""


def _display_title(value):
    return _ISBN_SUFFIX.sub("", str(value or "")).strip()


def _official_physics_curriculums(course):
    return [
        item for item in course["curriculums"]
        if item.get("verification") == "official" and item.get("resource")
    ]


def default_curriculum_id(course, math_version=None):
    """Return the course's default writable curriculum identifier."""

    if course["subject"] == "math":
        return normalize_version_code(math_version or "A")
    items = _official_physics_curriculums(course)
    if not items:
        raise CourseCurriculumError("该课程尚无可用于题目录入的正式目录")
    return items[0]["id"]


def resolve_curriculum_id(course, requested=None, *, math_version=None):
    """Validate an explicit curriculum identifier or select the course default."""

    if course["subject"] == "math":
        if requested in (None, ""):
            return default_curriculum_id(course, math_version)
        try:
            return normalize_version_code(requested)
        except ValueError as exc:
            raise CourseCurriculumError(str(exc)) from exc
    available = {item["id"] for item in course["curriculums"]}
    writable = {item["id"] for item in _official_physics_curriculums(course)}
    if requested in (None, ""):
        return default_curriculum_id(course)
    if requested not in available:
        raise CourseCurriculumError("课程与教材或框架不匹配")
    if requested not in writable:
        raise CourseCurriculumError("该课程目录尚未核验")
    return requested


def curriculum_tree(course, curriculum_id):
    """Return an isolated ``{book: {chapter: [section]}}`` tree."""

    curriculum_id = resolve_curriculum_id(course, curriculum_id)
    if course["subject"] == "math":
        return load_curriculum(curriculum_id)
    try:
        catalog = load_physics_curriculum(curriculum_id)
    except PhysicsCurriculumError as exc:
        raise CourseCurriculumError(str(exc)) from exc
    if catalog["course_id"] != course["id"]:
        raise CourseCurriculumError("课程目录与课程归属不一致")
    tree = {}
    if catalog["structure"] == "books":
        for book in catalog["books"]:
            chapters = {}
            for chapter in book["chapters"]:
                chapters[chapter["title"]] = [
                    section["title"] for section in chapter["sections"]
                ]
            tree[_display_title(book["title"])] = chapters
    else:
        for unit in catalog["units"]:
            tree[_display_title(unit["title"])] = {
                topic["title"]: [] for topic in unit["topics"]
            }
    return tree


def curriculum_catalog(course, curriculum_id):
    """Return the complete bundled catalog for a physics curriculum."""

    curriculum_id = resolve_curriculum_id(course, curriculum_id)
    if course["subject"] == "math":
        return {
            "curriculum_id": curriculum_id,
            "structure": "books",
            "tree": load_curriculum(curriculum_id),
            "exam": None,
        }
    try:
        catalog = load_physics_curriculum(curriculum_id)
    except PhysicsCurriculumError as exc:
        raise CourseCurriculumError(str(exc)) from exc
    if catalog["course_id"] != course["id"]:
        raise CourseCurriculumError("课程目录与课程归属不一致")
    return {
        "curriculum_id": curriculum_id,
        "structure": catalog["structure"],
        "tree": curriculum_tree(course, curriculum_id),
        "exam": deepcopy(catalog["exam"]),
    }


def validate_question_location(
    tree, compulsory="", chapter="", knowledge="", *, field_prefix="分类",
    strict=True,
):
    """Reject locations outside the selected course curriculum.

    Empty values remain valid so imported drafts may be saved before manual
    classification.  When ``strict`` is true every non-empty component must be
    an exact tree node; mathematics keeps its historical free-text categories.
    """

    compulsory = str(compulsory or "").strip()
    chapter = str(chapter or "").strip()
    knowledge = str(knowledge or "").strip()
    if not strict:
        if not compulsory:
            return ("", "", "")
        return (compulsory, chapter, knowledge)
    if not compulsory:
        if chapter or knowledge:
            raise CourseCurriculumError(f"{field_prefix}缺少学段或单元")
        return ("", "", "")
    if compulsory not in tree:
        raise CourseCurriculumError(f"{field_prefix}不属于当前课程目录")
    chapters = tree[compulsory]
    if not chapter:
        if knowledge:
            raise CourseCurriculumError(f"{field_prefix}缺少章节或主题")
        return (compulsory, "", "")
    if chapter not in chapters:
        raise CourseCurriculumError(f"{field_prefix}不属于当前课程章节")
    sections = chapters[chapter]
    if not knowledge:
        return (compulsory, chapter, "")
    if knowledge not in sections:
        raise CourseCurriculumError(f"{field_prefix}不属于当前课程小节")
    return (compulsory, chapter, knowledge)


def course_from_id(course_id):
    try:
        return get_course(course_id)
    except CourseScopeError as exc:
        raise CourseCurriculumError(str(exc)) from exc


def question_curriculum_id(
    course, requested=None, *, math_version=None, require_available=True,
):
    """Resolve the stored ``version_code`` for a question's course scope."""

    if require_available and course.get("status") != "available":
        raise CourseCurriculumError("该课程尚未开放题目录入、导入、AI 或组卷")
    return resolve_curriculum_id(
        course, requested, math_version=math_version,
    )
