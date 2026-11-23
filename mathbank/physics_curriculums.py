"""Read-only physics curriculum resources.

The files under ``resources/physics`` are bundled catalog snapshots.  Loading
them never enables question writes or changes the active mathematics textbook.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re

from mathbank.paths import PHYSICS_CURRICULUMS_DIR


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RESOURCE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\.json\Z")
_SOURCE_FIELDS = {"publisher", "document", "url", "version", "verified_on"}
_EXAM_REQUIRED_FIELDS = {
    "multiple_choice", "multiple_choice_minutes",
    "free_response", "free_response_minutes",
}
_EXAM_OPTIONAL_FIELDS = {
    "year", "calculator_allowed", "science_practices", "free_response_types",
}
_SKILL_ID = re.compile(r"[1-9]\d*\.[A-Z]\Z")


class PhysicsCurriculumError(ValueError):
    """A bundled physics catalog is missing, malformed, or inconsistent."""


def _text(value, field, *, maximum=500):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PhysicsCurriculumError(f"物理课程目录字段无效: {field}")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise PhysicsCurriculumError(f"物理课程目录标识无效: {field}")
    return value


def _unique(seen, value, field):
    if value in seen:
        raise PhysicsCurriculumError(f"物理课程目录标识重复: {field}")
    seen.add(value)


def _validate_source(source):
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise PhysicsCurriculumError("物理课程目录来源字段无效")
    for field in _SOURCE_FIELDS:
        _text(source[field], f"source.{field}")
    if not source["url"].startswith(("https://", "http://")):
        raise PhysicsCurriculumError("物理课程目录来源链接无效")
    return deepcopy(source)


def _validate_topic(topic, seen):
    if not isinstance(topic, dict) or set(topic) != {"id", "title"}:
        raise PhysicsCurriculumError("物理课程主题字段无效")
    topic_id = _text(topic["id"], "topic.id", maximum=32)
    if not re.fullmatch(r"\d+\.\d+[a-z]?", topic_id):
        raise PhysicsCurriculumError("物理课程主题编号无效")
    _unique(seen, topic_id, "topic.id")
    _text(topic["title"], "topic.title", maximum=300)
    return {"id": topic_id, "title": topic["title"]}


def _validate_units(units):
    if not isinstance(units, list) or not units:
        raise PhysicsCurriculumError("物理课程单元列表不能为空")
    unit_ids, topic_ids, numbers = set(), set(), set()
    result = []
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {
            "id", "number", "title", "weight", "topics",
        }:
            raise PhysicsCurriculumError("物理课程单元字段无效")
        unit_id = _identifier(unit["id"], "unit.id")
        _unique(unit_ids, unit_id, "unit.id")
        number = unit["number"]
        if type(number) is not int or number < 1 or number in numbers:
            raise PhysicsCurriculumError("物理课程单元编号无效或重复")
        numbers.add(number)
        _text(unit["title"], "unit.title", maximum=300)
        _text(unit["weight"], "unit.weight", maximum=32)
        topics = unit["topics"]
        if not isinstance(topics, list) or not topics:
            raise PhysicsCurriculumError("物理课程单元主题不能为空")
        result.append({
            "id": unit_id,
            "number": number,
            "title": unit["title"],
            "weight": unit["weight"],
            "topics": [_validate_topic(topic, topic_ids) for topic in topics],
        })
    return result


def _validate_books(books):
    if not isinstance(books, list) or not books:
        raise PhysicsCurriculumError("物理教材册次列表不能为空")
    book_ids = set()
    result = []
    for book in books:
        if not isinstance(book, dict) or set(book) != {
            "id", "title", "chapters",
        }:
            raise PhysicsCurriculumError("物理教材册次字段无效")
        book_id = _identifier(book["id"], "book.id")
        _unique(book_ids, book_id, "book.id")
        _text(book["title"], "book.title", maximum=300)
        chapters = book["chapters"]
        if not isinstance(chapters, list) or not chapters:
            raise PhysicsCurriculumError("物理教材章节不能为空")
        chapter_ids, section_ids = set(), set()
        normalized_chapters = []
        for chapter in chapters:
            if not isinstance(chapter, dict) or set(chapter) != {
                "id", "title", "sections",
            }:
                raise PhysicsCurriculumError("物理教材章节字段无效")
            chapter_id = _text(chapter["id"], "chapter.id", maximum=32)
            if not re.fullmatch(r"\d+", chapter_id):
                raise PhysicsCurriculumError("物理教材章节编号无效")
            _unique(chapter_ids, chapter_id, "chapter.id")
            _text(chapter["title"], "chapter.title", maximum=300)
            sections = chapter["sections"]
            if not isinstance(sections, list) or not sections:
                raise PhysicsCurriculumError("物理教材章节小节不能为空")
            normalized_sections = []
            for section in sections:
                if not isinstance(section, dict) or set(section) != {
                    "id", "title",
                }:
                    raise PhysicsCurriculumError("物理教材小节字段无效")
                section_id = _text(section["id"], "section.id", maximum=32)
                if not re.fullmatch(r"\d+\.\d+", section_id):
                    raise PhysicsCurriculumError("物理教材小节编号无效")
                _unique(section_ids, section_id, "section.id")
                _text(section["title"], "section.title", maximum=300)
                normalized_sections.append({
                    "id": section_id, "title": section["title"],
                })
            normalized_chapters.append({
                "id": chapter_id,
                "title": chapter["title"],
                "sections": normalized_sections,
            })
        result.append({
            "id": book_id,
            "title": book["title"],
            "chapters": normalized_chapters,
        })
    return result


def _validate_science_practices(practices):
    if not isinstance(practices, list) or not practices:
        raise PhysicsCurriculumError("物理科学实践列表不能为空")
    practice_ids, skill_ids = set(), set()
    result = []
    for practice in practices:
        if not isinstance(practice, dict) or set(practice) != {
            "id", "title", "description", "skills",
        }:
            raise PhysicsCurriculumError("物理科学实践字段无效")
        practice_id = practice["id"]
        if type(practice_id) is not int or practice_id < 1:
            raise PhysicsCurriculumError("物理科学实践编号无效")
        _unique(practice_ids, practice_id, "science_practice.id")
        _text(practice["title"], "science_practice.title", maximum=200)
        _text(practice["description"], "science_practice.description")
        skills = practice["skills"]
        if not isinstance(skills, list) or not skills:
            raise PhysicsCurriculumError("物理科学实践技能不能为空")
        normalized_skills = []
        for skill in skills:
            if not isinstance(skill, dict) or set(skill) != {"id", "title"}:
                raise PhysicsCurriculumError("物理科学实践技能字段无效")
            skill_id = _text(skill["id"], "skill.id", maximum=16)
            if not _SKILL_ID.fullmatch(skill_id) or not skill_id.startswith(
                f"{practice_id}."
            ):
                raise PhysicsCurriculumError("物理科学实践技能编号无效")
            _unique(skill_ids, skill_id, "skill.id")
            _text(skill["title"], "skill.title", maximum=300)
            normalized_skills.append({"id": skill_id, "title": skill["title"]})
        result.append({
            "id": practice_id,
            "title": practice["title"],
            "description": practice["description"],
            "skills": normalized_skills,
        })
    return result


def _validate_free_response_types(types):
    if not isinstance(types, list) or not types:
        raise PhysicsCurriculumError("物理自由作答题型列表不能为空")
    type_ids = set()
    result = []
    for item in types:
        if not isinstance(item, dict) or set(item) != {"id", "title"}:
            raise PhysicsCurriculumError("物理自由作答题型字段无效")
        type_id = item["id"]
        if type(type_id) is not int or type_id < 1:
            raise PhysicsCurriculumError("物理自由作答题型编号无效")
        _unique(type_ids, type_id, "free_response_type.id")
        _text(item["title"], "free_response_type.title", maximum=200)
        result.append({"id": type_id, "title": item["title"]})
    return result


def _validate_exam(exam):
    if (not isinstance(exam, dict)
            or not _EXAM_REQUIRED_FIELDS <= set(exam)
            or not set(exam) <= _EXAM_REQUIRED_FIELDS | _EXAM_OPTIONAL_FIELDS):
        raise PhysicsCurriculumError("物理考试配置字段无效")
    for field in _EXAM_REQUIRED_FIELDS:
        if type(exam[field]) is not int or exam[field] < 1:
            raise PhysicsCurriculumError(f"物理考试配置无效: {field}")
    result = {field: exam[field] for field in _EXAM_REQUIRED_FIELDS}
    if "year" in exam:
        year = exam["year"]
        if type(year) is not int or not 2000 <= year <= 2100:
            raise PhysicsCurriculumError("物理考试年份无效")
        result["year"] = year
    if "calculator_allowed" in exam:
        if type(exam["calculator_allowed"]) is not bool:
            raise PhysicsCurriculumError("物理考试计算器配置无效")
        result["calculator_allowed"] = exam["calculator_allowed"]
    if "science_practices" in exam:
        result["science_practices"] = _validate_science_practices(
            exam["science_practices"]
        )
    if "free_response_types" in exam:
        result["free_response_types"] = _validate_free_response_types(
            exam["free_response_types"]
        )
    return result


def validate_physics_curriculum(data, *, expected_curriculum_id=None,
                                expected_course_id=None):
    """Validate one bundled catalog and return an isolated normalized copy."""

    if not isinstance(data, dict) or set(data) != {
        "schema_version", "curriculum_id", "course_id", "title",
        "verification", "source", "structure", "units", "books", "exam",
    }:
        raise PhysicsCurriculumError("物理课程目录字段无效")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise PhysicsCurriculumError("物理课程目录版本无效")
    curriculum_id = _identifier(data["curriculum_id"], "curriculum_id")
    course_id = _identifier(data["course_id"], "course_id")
    if expected_curriculum_id is not None and curriculum_id != expected_curriculum_id:
        raise PhysicsCurriculumError("物理课程目录标识与资源文件不一致")
    if expected_course_id is not None and course_id != expected_course_id:
        raise PhysicsCurriculumError("物理课程目录与课程归属不一致")
    _text(data["title"], "title", maximum=300)
    if data["verification"] not in {"official", "secondary"}:
        raise PhysicsCurriculumError("物理课程目录核验状态无效")
    source = _validate_source(data["source"])
    if data["structure"] not in {"units", "books"}:
        raise PhysicsCurriculumError("物理课程目录结构类型无效")
    units = data["units"]
    books = data["books"]
    exam = data["exam"]
    if data["structure"] == "units":
        if books != [] or not isinstance(units, list) or not units or exam is None:
            raise PhysicsCurriculumError("物理课程单元目录结构无效")
        normalized_units = _validate_units(units)
        normalized_books = []
        normalized_exam = _validate_exam(exam)
    else:
        if units != [] or not isinstance(books, list) or not books or exam is not None:
            raise PhysicsCurriculumError("物理教材目录结构无效")
        normalized_units = []
        normalized_books = _validate_books(books)
        normalized_exam = None
    return {
        "schema_version": 1,
        "curriculum_id": curriculum_id,
        "course_id": course_id,
        "title": data["title"],
        "verification": data["verification"],
        "source": source,
        "structure": data["structure"],
        "units": normalized_units,
        "books": normalized_books,
        "exam": normalized_exam,
    }


def _curriculum_path(curriculum_id):
    if not isinstance(curriculum_id, str) or not _RESOURCE_NAME.fullmatch(
        f"{curriculum_id}.json"
    ):
        raise PhysicsCurriculumError("物理课程资源标识无效")
    path = PHYSICS_CURRICULUMS_DIR / f"{curriculum_id}.json"
    if not path.is_file():
        raise PhysicsCurriculumError("物理课程资源不存在")
    return path


@lru_cache(maxsize=32)
def _load_cached(path_text):
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicsCurriculumError("物理课程资源无法读取") from exc
    return validate_physics_curriculum(
        data, expected_curriculum_id=path.stem,
    )


def load_physics_curriculum(curriculum_id):
    """Load one bundled physics catalog by stable curriculum ID."""

    path = _curriculum_path(curriculum_id)
    return deepcopy(_load_cached(str(path)))


def physics_curriculum_catalog():
    """Return metadata for every valid bundled physics catalog."""

    if not PHYSICS_CURRICULUMS_DIR.is_dir():
        raise PhysicsCurriculumError("物理课程资源目录不存在")
    items = []
    for path in sorted(PHYSICS_CURRICULUMS_DIR.glob("*.json")):
        if not _RESOURCE_NAME.fullmatch(path.name):
            raise PhysicsCurriculumError("物理课程资源文件名无效")
        catalog = load_physics_curriculum(path.stem)
        if catalog["structure"] == "units":
            count = sum(len(unit["topics"]) for unit in catalog["units"])
            unit_count = len(catalog["units"])
            book_count = 0
        else:
            count = sum(len(chapter["sections"])
                        for book in catalog["books"]
                        for chapter in book["chapters"])
            unit_count = 0
            book_count = len(catalog["books"])
        items.append({
            "curriculum_id": catalog["curriculum_id"],
            "course_id": catalog["course_id"],
            "title": catalog["title"],
            "verification": catalog["verification"],
            "structure": catalog["structure"],
            "unit_count": unit_count,
            "book_count": book_count,
            "topic_count": count,
        })
    return {"schema_version": 1, "curriculums": items}
