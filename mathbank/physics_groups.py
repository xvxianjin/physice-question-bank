"""Read-only Chinese high-school physics province groups.

A group is the smallest unit for one exam authority and its paper structure.
All groups share the bundled PEP (2019) six-book catalog; edition differences
are not duplicated. Provinces that share the same exam authority and paper
structure belong to one group. Two groups may reuse one ``exam_profile_id``
when their structures match but their exam authorities differ.

``pending`` groups have no verified resources. ``provisional`` groups record
the latest public paper structure from media analysis or compiled real-paper
statistics. ``official`` groups require an official exam-authority source.
The course stays planned until the remaining write, AI, paper and sharing
paths are wired.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re

from mathbank.paths import PHYSICS_GROUPS_FILE


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?\Z")
_SOURCE_FIELDS = {"document", "url", "verified_on", "kind"}
_SOURCE_KINDS = {"official", "media", "compiled"}
_STANDARD_FIELDS = {
    "id", "title", "publisher", "published_on", "source_url", "verified_on",
}
_GROUP_REQUIRED_FIELDS = {"id", "name", "status", "provinces"}
_GROUP_OPTIONAL_FIELDS = {"curriculum_id", "exam_profile_id", "source"}
TEXTBOOK_CURRICULUM_ID = "cn_physics_renjiao_2019"
SCHEMA_VERSION = 2

PROVINCES = {
    "beijing": "北京",
    "tianjin": "天津",
    "hebei": "河北",
    "shanxi": "山西",
    "neimenggu": "内蒙古",
    "liaoning": "辽宁",
    "jilin": "吉林",
    "heilongjiang": "黑龙江",
    "shanghai": "上海",
    "jiangsu": "江苏",
    "zhejiang": "浙江",
    "anhui": "安徽",
    "fujian": "福建",
    "jiangxi": "江西",
    "shandong": "山东",
    "henan": "河南",
    "hubei": "湖北",
    "hunan": "湖南",
    "guangdong": "广东",
    "guangxi": "广西",
    "hainan": "海南",
    "chongqing": "重庆",
    "sichuan": "四川",
    "guizhou": "贵州",
    "yunnan": "云南",
    "xizang": "西藏",
    "shaanxi": "陕西",
    "gansu": "甘肃",
    "qinghai": "青海",
    "ningxia": "宁夏",
    "xinjiang": "新疆",
}


class PhysicsGroupError(ValueError):
    """A bundled province-group resource is missing, malformed, or inconsistent."""


def _text(value, field, *, maximum=500):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PhysicsGroupError(f"物理省份组字段无效: {field}")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise PhysicsGroupError(f"物理省份组标识无效: {field}")
    return value


def _unique(seen, value, field):
    if value in seen:
        raise PhysicsGroupError(f"物理省份组标识重复: {field}")
    seen.add(value)


def _validate_source(source, status):
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise PhysicsGroupError("物理省份组来源字段无效")
    for field in _SOURCE_FIELDS:
        _text(source[field], f"source.{field}")
    if not source["url"].startswith(("https://", "http://")):
        raise PhysicsGroupError("物理省份组来源链接无效")
    if source["kind"] not in _SOURCE_KINDS:
        raise PhysicsGroupError("物理省份组来源类型无效")
    if status == "official" and source["kind"] != "official":
        raise PhysicsGroupError("已核验省份组必须使用官方来源")
    if status == "provisional" and source["kind"] == "official":
        raise PhysicsGroupError("临时基线省份组不能标记为官方来源")
    return deepcopy(source)


def _validate_standard(standard):
    if not isinstance(standard, dict) or set(standard) != _STANDARD_FIELDS:
        raise PhysicsGroupError("物理课程标准字段无效")
    _identifier(standard["id"], "standard.id")
    for field in ("title", "publisher"):
        _text(standard[field], f"standard.{field}")
    for field in ("published_on", "verified_on"):
        _text(standard[field], f"standard.{field}", maximum=10)
        if not _DATE.fullmatch(standard[field]):
            raise PhysicsGroupError(f"物理课程标准日期无效: {field}")
    if not standard["source_url"].startswith(("https://", "http://")):
        raise PhysicsGroupError("物理课程标准来源链接无效")
    return deepcopy(standard)


def _validate_group(group, seen_ids, seen_provinces):
    if not isinstance(group, dict):
        raise PhysicsGroupError("物理省份组条目无效")
    fields = set(group)
    if (not _GROUP_REQUIRED_FIELDS <= fields
            or not fields <= _GROUP_REQUIRED_FIELDS | _GROUP_OPTIONAL_FIELDS):
        raise PhysicsGroupError("物理省份组字段无效")
    group_id = _identifier(group["id"], "group.id")
    _unique(seen_ids, group_id, "group.id")
    _text(group["name"], "group.name", maximum=200)
    status = group["status"]
    if status not in {"pending", "provisional", "official"}:
        raise PhysicsGroupError("物理省份组状态无效")
    provinces = group["provinces"]
    if not isinstance(provinces, list) or not provinces:
        raise PhysicsGroupError("物理省份组成员不能为空")
    normalized_provinces = []
    for code in provinces:
        if not isinstance(code, str) or code not in PROVINCES:
            raise PhysicsGroupError("物理省份组成员无效")
        _unique(seen_provinces, code, "province")
        normalized_provinces.append(code)
    curriculum_id = group.get("curriculum_id")
    exam_profile_id = group.get("exam_profile_id")
    source = group.get("source")
    if status == "pending":
        if curriculum_id is not None or exam_profile_id is not None or source is not None:
            raise PhysicsGroupError("待核验省份组不得声明已核验资源")
        normalized_curriculum = normalized_exam = normalized_source = None
    else:
        if curriculum_id is None:
            raise PhysicsGroupError("已核验省份组缺少教材目录")
        if exam_profile_id is None:
            raise PhysicsGroupError("已核验省份组缺少考试方案")
        if source is None:
            raise PhysicsGroupError("已核验省份组缺少来源")
        normalized_curriculum = _identifier(curriculum_id, "group.curriculum_id")
        if normalized_curriculum != TEXTBOOK_CURRICULUM_ID:
            raise PhysicsGroupError("省份组教材目录必须使用统一的人教版目录")
        normalized_exam = _identifier(exam_profile_id, "group.exam_profile_id")
        normalized_source = _validate_source(source, status)
    return {
        "id": group_id,
        "name": group["name"],
        "status": status,
        "provinces": normalized_provinces,
        "curriculum_id": normalized_curriculum,
        "exam_profile_id": normalized_exam,
        "source": normalized_source,
    }


def validate_physics_groups(data):
    """Validate one bundled province-group resource and return an isolated copy."""

    if (not isinstance(data, dict)
            or set(data) != {"schema_version", "standard", "groups"}):
        raise PhysicsGroupError("物理省份组资源字段无效")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise PhysicsGroupError("物理省份组资源版本无效")
    standard = _validate_standard(data["standard"])
    groups = data["groups"]
    if not isinstance(groups, list):
        raise PhysicsGroupError("物理省份组列表无效")
    seen_ids, seen_provinces = set(), set()
    normalized_groups = [
        _validate_group(group, seen_ids, seen_provinces) for group in groups
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "standard": standard,
        "groups": normalized_groups,
    }


@lru_cache(maxsize=1)
def _load_cached(path_text):
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicsGroupError("物理省份组资源无法读取") from exc
    return validate_physics_groups(data)


def load_physics_groups():
    """Load the bundled province groups by fixed project path."""

    if not PHYSICS_GROUPS_FILE.is_file():
        raise PhysicsGroupError("物理省份组资源不存在")
    return deepcopy(_load_cached(str(PHYSICS_GROUPS_FILE)))


def physics_group_catalog():
    """Return a compact summary without enabling physics writes."""

    catalog = load_physics_groups()
    groups = [
        {
            "id": group["id"],
            "name": group["name"],
            "status": group["status"],
            "provinces": group["provinces"],
            "curriculum_id": group["curriculum_id"],
            "exam_profile_id": group["exam_profile_id"],
            "source_kind": group["source"]["kind"] if group["source"] else None,
        }
        for group in catalog["groups"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "standard": catalog["standard"],
        "group_count": len(groups),
        "province_count": sum(len(group["provinces"]) for group in groups),
        "groups": groups,
    }
