"""Bundled Chinese physics province groups stay strict and read-only."""

import json

import pytest

from mathbank.physics_groups import (
    PROVINCES, PhysicsGroupError, load_physics_groups,
    physics_group_catalog, validate_physics_groups,
)


def _standard():
    return {
        "id": "cn_physics_2017_2025",
        "title": "普通高中物理课程标准日常修订版（2017年版2025年修订）",
        "publisher": "中华人民共和国教育部",
        "published_on": "2025-10",
        "source_url": "https://www.ictr.edu.cn/news/gao/detail/5448.html",
        "verified_on": "2026-11-22",
    }


def _source(kind="media"):
    return {
        "document": "2026 年某省物理卷面结构",
        "url": "https://example.com/exam",
        "verified_on": "2026-11-22",
        "kind": kind,
    }


def _group(**overrides):
    group = {
        "id": "new_curriculum_1",
        "name": "新课标 I 卷组",
        "status": "provisional",
        "provinces": ["beijing", "tianjin"],
        "curriculum_id": "cn_physics_renjiao_2019",
        "exam_profile_id": "new_curriculum_1_2026",
        "source": _source("media"),
    }
    group.update(overrides)
    return group


def _official_group(**overrides):
    values = {"status": "official", "source": _source("official")}
    values.update(overrides)
    return _group(**values)


def _resource(groups=None):
    return {
        "schema_version": 2,
        "standard": _standard(),
        "groups": [] if groups is None else groups,
    }


def test_bundled_groups_record_latest_public_baselines():
    catalog = load_physics_groups()
    assert catalog["schema_version"] == 2
    assert catalog["standard"]["id"] == "cn_physics_2017_2025"
    assert "2025" in catalog["standard"]["title"]
    assert {group["id"] for group in catalog["groups"]} == {
        "group_northwest_2026",
        "group_northeast_2026",
        "group_shandong_2026",
        "group_shanghai_2026",
    }
    assert all(
        group["curriculum_id"] == "cn_physics_renjiao_2019"
        for group in catalog["groups"]
    )
    assert {group["status"] for group in catalog["groups"]} == {
        "provisional", "official",
    }
    summary = physics_group_catalog()
    assert summary["schema_version"] == 2
    assert summary["group_count"] == 4
    assert summary["province_count"] == 10


def test_official_group_accepts_known_provinces():
    result = validate_physics_groups(_resource([_official_group()]))
    assert result["groups"] == [{
        "id": "new_curriculum_1",
        "name": "新课标 I 卷组",
        "status": "official",
        "provinces": ["beijing", "tianjin"],
        "curriculum_id": "cn_physics_renjiao_2019",
        "exam_profile_id": "new_curriculum_1_2026",
        "source": {
            "document": "2026 年某省物理卷面结构",
            "url": "https://example.com/exam",
            "verified_on": "2026-11-22",
            "kind": "official",
        },
    }]


def test_provisional_group_records_latest_public_structure():
    result = validate_physics_groups(_resource([_group()]))
    assert result["groups"][0]["status"] == "provisional"
    assert result["groups"][0]["source"]["kind"] == "media"


def test_source_kind_must_match_status():
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([
            _group(status="official", source=_source("media")),
        ]))
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([
            _group(status="provisional", source=_source("official")),
        ]))
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([
            _group(source={**_source(), "kind": "blog"}),
        ]))


def test_unknown_and_duplicate_provinces_are_rejected():
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([_group(provinces=["atlantis"])]))
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([_group(provinces=["beijing", "beijing"])]))
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([
            _group(),
            _group(id="group_2", provinces=["beijing"]),
        ]))


def test_pending_group_cannot_claim_verified_resources():
    pending = _group(
        status="pending", curriculum_id=None,
        exam_profile_id=None, source=None,
    )
    result = validate_physics_groups(_resource([pending]))
    assert result["groups"][0]["curriculum_id"] is None
    assert result["groups"][0]["exam_profile_id"] is None
    assert result["groups"][0]["source"] is None
    for field in ("curriculum_id", "exam_profile_id", "source"):
        with pytest.raises(PhysicsGroupError):
            validate_physics_groups(_resource([
                _group(status="pending", **{field: _group()[field]}),
            ]))


def test_verified_group_requires_curriculum_exam_and_source():
    for factory in (_group, _official_group):
        for field in ("curriculum_id", "exam_profile_id", "source"):
            broken = factory()
            broken.pop(field)
            with pytest.raises(PhysicsGroupError):
                validate_physics_groups(_resource([broken]))


def test_groups_must_use_the_unified_renjiao_catalog():
    for factory in (_group, _official_group):
        with pytest.raises(PhysicsGroupError):
            validate_physics_groups(_resource([
                factory(curriculum_id="cn_physics_other"),
            ]))


def test_standard_requires_http_url_and_date():
    for field, value in (
        ("source_url", "ftp://example.com/standard"),
        ("published_on", "October 2025"),
        ("verified_on", "2026/11/22"),
    ):
        standard = _standard()
        standard[field] = value
        with pytest.raises(PhysicsGroupError):
            validate_physics_groups({
                "schema_version": 2, "standard": standard, "groups": [],
            })


def test_schema_version_must_be_two():
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups({
            "schema_version": 1, "standard": _standard(), "groups": [],
        })


def test_group_ids_and_fields_are_strict():
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([
            _group(),
            _group(provinces=["hebei"]),
        ]))
    broken = _group()
    broken["unexpected"] = True
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([broken]))
    with pytest.raises(PhysicsGroupError):
        validate_physics_groups(_resource([_group(status="reviewed")]))


def test_province_catalog_is_complete():
    assert len(PROVINCES) == 31
    assert len(set(PROVINCES.values())) == 31
    assert {"beijing", "tianjin", "shanghai", "chongqing"} <= set(PROVINCES)
    assert {"neimenggu", "guangxi", "xizang", "ningxia", "xinjiang"} <= set(PROVINCES)
    assert "shanxi" in PROVINCES and "shaanxi" in PROVINCES


def test_loaded_groups_are_detached():
    first = load_physics_groups()
    first["standard"]["title"] = "changed"
    second = load_physics_groups()
    assert second["standard"]["title"] != "changed"


def test_group_catalog_counts_provinces(tmp_path, monkeypatch):
    import mathbank.physics_groups as module

    path = tmp_path / "groups.json"
    path.write_text(json.dumps(_resource([
        _group(),
        _group(id="group_2", provinces=["hebei"]),
    ]), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(module, "PHYSICS_GROUPS_FILE", path)
    module._load_cached.cache_clear()
    try:
        summary = module.physics_group_catalog()
        assert summary["group_count"] == 2
        assert summary["province_count"] == 3
        assert summary["groups"][0]["source_kind"] == "media"
    finally:
        module._load_cached.cache_clear()


def test_course_revision_includes_group_content(tmp_path, monkeypatch):
    import mathbank.courses as courses_module
    import mathbank.physics_groups as groups_module

    path = tmp_path / "groups.json"
    path.write_text(json.dumps(_resource([_group()]), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(groups_module, "PHYSICS_GROUPS_FILE", path)
    groups_module._load_cached.cache_clear()
    courses_module._registry.cache_clear()
    try:
        before = courses_module.course_catalog()["revision"]
        data = json.loads(path.read_text(encoding="utf-8"))
        data["groups"][0]["name"] = "changed"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        groups_module._load_cached.cache_clear()
        courses_module._registry.cache_clear()
        assert courses_module.course_catalog()["revision"] != before
    finally:
        groups_module._load_cached.cache_clear()
        courses_module._registry.cache_clear()
