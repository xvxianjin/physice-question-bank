"""Stable course identities and explicit course scopes.

Mathematics keeps its legacy curriculum codes. Bundled physics courses use
their official catalog identifiers; every executable flow must resolve one
course explicitly and must not silently fall back to mathematics.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from mathbank.curriculums import CURRICULUM_NAMES, load_curriculum, normalize_version_code
from mathbank.paths import COURSES_FILE, PHYSICS_CURRICULUMS_DIR


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SCOPE_FIELDS = frozenset({"subject", "course_id", "curriculum_id", "revision"})
MATH_COURSE_ID = "cn_hs_math"


class CourseScopeError(ValueError):
    """Malformed, unknown or inconsistent scope."""


class StaleCourseScope(CourseScopeError):
    """A caller holds a scope from another registry revision."""


class CourseNotAvailable(CourseScopeError):
    """A registered course is not yet supported by the runtime."""


@dataclass(frozen=True)
class CourseScope:
    subject: str
    course_id: str
    curriculum_id: str
    revision: str

    def to_dict(self):
        return asdict(self)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CourseScopeError("课程资源含重复字段")
        value[key] = item
    return value


def _validate_registry(raw):
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "courses"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or not isinstance(raw["courses"], list) or not raw["courses"]):
        raise CourseScopeError("课程资源格式或版本无效")
    ids, curriculum_ids = set(), set(CURRICULUM_NAMES)
    result = []
    for entry in raw["courses"]:
        fields = {"id", "subject", "name", "curriculum_provider", "status", "curriculums"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise CourseScopeError("课程资源字段无效")
        code = entry["id"]
        if not isinstance(code, str) or not _IDENTIFIER.fullmatch(code) or code in ids:
            raise CourseScopeError("课程标识无效或重复")
        ids.add(code)
        if (entry["subject"] not in ("math", "physics")
                or not isinstance(entry["name"], str) or not entry["name"].strip()
                or len(entry["name"]) > 200
                or not isinstance(entry["curriculums"], list)):
            raise CourseScopeError("课程属性无效")
        if entry["curriculum_provider"] == "legacy_math":
            if (code != "cn_hs_math" or entry["subject"] != "math"
                    or entry["status"] != "available" or entry["curriculums"]):
                raise CourseScopeError("数学兼容课程定义无效")
            curriculums = [
                {"id": code, "name": name, "verification": "existing"}
                for code, name in CURRICULUM_NAMES.items()
            ]
        elif entry["curriculum_provider"] == "bundled_physics":
            # Physics writes are enabled only after every executable path has
            # been wired; editing JSON alone cannot bypass that boundary.
            if (entry["subject"] != "physics"
                    or entry["status"] not in {"available", "planned"}
                    or not entry["curriculums"]):
                raise CourseScopeError("物理课程定义无效")
            curriculums = deepcopy(entry["curriculums"])
            for curriculum in curriculums:
                if not isinstance(curriculum, dict) or set(curriculum) not in (
                    {"id", "name", "verification"},
                    {"id", "name", "verification", "resource"},
                ):
                    raise CourseScopeError("课程框架字段无效")
                curriculum_id = curriculum["id"]
                verification = curriculum["verification"]
                resource = curriculum.get("resource")
                if (not isinstance(curriculum_id, str) or not _IDENTIFIER.fullmatch(curriculum_id)
                        or curriculum_id in curriculum_ids
                        or not isinstance(curriculum["name"], str) or not curriculum["name"].strip()
                        or len(curriculum["name"]) > 200
                        or verification not in {"pending", "official"}
                        or (verification == "pending" and "resource" in curriculum)
                        or (verification == "official" and (
                            not isinstance(resource, str)
                            or not re.fullmatch(r"physics/[a-z][a-z0-9_]{0,63}\.json", resource)
                            or resource != f"physics/{curriculum_id}.json"
                        ))):
                    raise CourseScopeError("课程框架标识、名称或核验状态无效")
                curriculum_ids.add(curriculum_id)
            if entry["status"] == "available" and not any(
                item.get("verification") == "official" and item.get("resource")
                for item in curriculums
            ):
                raise CourseScopeError("已开放物理课程缺少正式目录")
        else:
            raise CourseScopeError("课程目录提供器无效")
        result.append({
            **deepcopy(entry),
            "curriculums": curriculums,
            "capabilities": {
                "read": True,
                "write": entry["status"] == "available",
                "import": entry["status"] == "available",
                "ai": entry["status"] == "available",
                "paper": entry["status"] == "available",
                "share": entry["status"] == "available",
            },
        })
    if "cn_hs_math" not in ids:
        raise CourseScopeError("课程资源缺少数学兼容入口")
    return result


@lru_cache(maxsize=1)
def _registry():
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    courses = _validate_registry(raw)
    from mathbank.physics_curriculums import (
        PhysicsCurriculumError, validate_physics_curriculum,
    )
    from mathbank.physics_groups import PhysicsGroupError, load_physics_groups
    physics_curriculums = {}
    for course in courses:
        for curriculum in course["curriculums"]:
            resource = curriculum.get("resource")
            if not resource:
                continue
            path = PHYSICS_CURRICULUMS_DIR / Path(resource).name
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CourseScopeError("课程目录资源无法读取") from exc
            try:
                physics_curriculums[resource] = validate_physics_curriculum(
                    data, expected_curriculum_id=Path(resource).stem,
                    expected_course_id=course["id"],
                )
            except PhysicsCurriculumError as exc:
                raise CourseScopeError("课程目录资源与课程归属不一致") from exc
    try:
        physics_groups = load_physics_groups()
    except PhysicsGroupError as exc:
        raise CourseScopeError("物理省份组资源无法读取") from exc
    # Include bundled math contents, not file mtimes, so snapshots survive a
    # restart but detect changed directory definitions. No local metadata read.
    digest_source = {"courses": courses, "math_curriculums": {
        code: load_curriculum(code) for code in CURRICULUM_NAMES
    }, "physics_curriculums": physics_curriculums,
        "physics_groups": physics_groups}
    revision = hashlib.sha256(json.dumps(
        digest_source, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return courses, revision


def course_catalog():
    courses, revision = _registry()
    return {"schema_version": 1, "revision": revision, "courses": deepcopy(courses)}


def get_course(course_id):
    if not isinstance(course_id, str):
        raise CourseScopeError("课程标识必须是文本")
    for course in _registry()[0]:
        if course["id"] == course_id:
            return deepcopy(course)
    raise CourseScopeError("未知课程标识")


def normalize_course_id(value=None):
    """Legacy callers keep mathematics; explicit values must be valid IDs."""

    if value is None or value == "":
        return MATH_COURSE_ID
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CourseScopeError("课程标识无效")
    return value


def require_available_course(course_id=None):
    """Return a course only when its persistence and runtime are enabled."""

    course = get_course(normalize_course_id(course_id))
    if course["status"] != "available":
        raise CourseNotAvailable("该课程尚未开放题目录入、导入、AI 或组卷")
    return course


def resolve_course_scope(value, *, require_available=False):
    """Validate the complete request scope, without any implicit math default."""
    if not isinstance(value, dict) or set(value) != _SCOPE_FIELDS:
        raise CourseScopeError("课程范围须包含 subject、course_id、curriculum_id、revision")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise CourseScopeError("课程范围字段须为非空文本")
    course = get_course(value["course_id"])
    if value["subject"] != course["subject"]:
        raise CourseScopeError("学科与课程不匹配")
    if value["curriculum_id"] not in {item["id"] for item in course["curriculums"]}:
        raise CourseScopeError("课程与教材或框架不匹配")
    if value["revision"] != _registry()[1]:
        raise StaleCourseScope("课程资源版本已变化，请重新读取课程列表")
    if require_available and course["status"] != "available":
        raise CourseNotAvailable("该课程尚未开放题目录入、导入、AI 或组卷")
    return CourseScope(**value)


def legacy_math_scope(version="A"):
    """Explicit adapter for legacy math clients; never used for physics."""
    return resolve_course_scope({
        "subject": "math", "course_id": "cn_hs_math",
        "curriculum_id": normalize_version_code(version), "revision": _registry()[1],
    }, require_available=True)


def curriculum_snapshot(scope):
    """Return bundled definitions, not editable machine-local metadata."""
    if not isinstance(scope, CourseScope):
        raise CourseScopeError("需要已校验的课程范围")
    scope = resolve_course_scope(scope.to_dict(), require_available=True)
    if scope.subject == "math":
        curriculum = load_curriculum(scope.curriculum_id)
    else:
        from mathbank.course_curriculum import CourseCurriculumError, curriculum_tree
        try:
            curriculum = curriculum_tree(get_course(scope.course_id), scope.curriculum_id)
        except CourseCurriculumError as exc:
            raise CourseScopeError(str(exc)) from exc
    return {"scope": scope.to_dict(), "origin": "bundled",
            "curriculum": curriculum}


def course_capabilities(course):
    """Describe executable capabilities without implying a UI-only feature."""

    available = course.get("status") == "available"
    return {
        "read": True,
        "write": available,
        "import": available,
        "ai": available,
        "paper": available,
        "share": available,
    }
