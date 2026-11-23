"""Pure course registry and HTTP contract: no app startup or real data."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from mathbank import courses
from mathbank.course_api import router
from mathbank.curriculums import CURRICULUM_NAMES, build_default_metadata, load_curriculum
from mathbank.paths import COURSES_FILE


def scope_for(course_id="cn_hs_math", curriculum_id="A", subject="math"):
    return {"subject": subject, "course_id": course_id, "curriculum_id": curriculum_id,
            "revision": courses.course_catalog()["revision"]}


def test_catalog_keeps_math_single_source_and_physics_available():
    catalog = courses.course_catalog()
    by_id = {course["id"]: course for course in catalog["courses"]}
    assert set(by_id) == {"cn_hs_math", "cn_hs_physics", "ap_physics_c_mechanics", "ap_physics_c_em"}
    assert {item["id"]: item["name"] for item in by_id["cn_hs_math"]["curriculums"]} == CURRICULUM_NAMES
    assert len(catalog["revision"]) == 64
    for course in by_id.values():
        assert course["status"] == "available"
        assert course["capabilities"] == {
            "read": True, "write": True, "import": True,
            "ai": True, "paper": True, "share": True,
        }
        if course["subject"] == "physics":
            assert all(item["verification"] in {"pending", "official"}
                       for item in course["curriculums"])
    assert by_id["ap_physics_c_mechanics"]["curriculums"][0] == {
        "id": "ap_c_mechanics",
        "name": "AP C 力学课程框架（2027 CED）",
        "verification": "official",
        "resource": "physics/ap_c_mechanics.json",
    }


@pytest.mark.parametrize("version", tuple(CURRICULUM_NAMES))
def test_legacy_math_adapter_preserves_resources_and_metadata(version):
    before = build_default_metadata(version)
    scope = courses.legacy_math_scope(version.lower())
    snapshot = courses.curriculum_snapshot(scope)
    assert snapshot["scope"]["curriculum_id"] == version
    assert snapshot["origin"] == "bundled"
    assert snapshot["curriculum"] == load_curriculum(version)
    assert build_default_metadata(version) == before
    with pytest.raises(FrozenInstanceError):
        scope.subject = "physics"


@pytest.mark.parametrize("payload", [
    None, [], {}, {"course_id": "cn_hs_math"},
    {"subject": "math", "course_id": "cn_hs_math", "curriculum_id": "A", "revision": None},
    {"subject": "math", "course_id": ["cn_hs_math"], "curriculum_id": "A", "revision": "x"},
])
def test_scope_requires_complete_typed_fields(payload):
    with pytest.raises(courses.CourseScopeError):
        courses.resolve_course_scope(payload)


@pytest.mark.parametrize("change", [
    {"subject": "physics"}, {"course_id": "unknown"},
    {"curriculum_id": "ap_c_mechanics"}, {"curriculum_id": "../A"},
    {"course_id": "CN_HS_MATH"}, {"unexpected": True},
])
def test_scope_rejects_mismatch_instead_of_math_fallback(change):
    with pytest.raises(courses.CourseScopeError):
        courses.resolve_course_scope({**scope_for(), **change})


def test_stale_scope_and_forged_dataclass_cannot_be_used():
    value = {**scope_for(), "revision": "old"}
    with pytest.raises(courses.StaleCourseScope):
        courses.resolve_course_scope(value)
    with pytest.raises(courses.StaleCourseScope):
        courses.curriculum_snapshot(courses.CourseScope(**value))


@pytest.mark.parametrize("course_id,curriculum_id", [
    ("cn_hs_physics", "cn_physics_renjiao_2019"),
    ("ap_physics_c_mechanics", "ap_c_mechanics"),
    ("ap_physics_c_em", "ap_c_em"),
])
def test_physics_identity_resolves_and_snapshots(course_id, curriculum_id):
    value = scope_for(course_id, curriculum_id, "physics")
    scope = courses.resolve_course_scope(value)
    assert scope.subject == "physics"
    resolved = courses.resolve_course_scope(value, require_available=True)
    snapshot = courses.curriculum_snapshot(resolved)
    assert snapshot["origin"] == "bundled"
    assert snapshot["curriculum"]


def test_pending_physics_curriculum_cannot_be_snapshotted():
    value = scope_for("cn_hs_physics", "cn_physics_core", "physics")
    courses.resolve_course_scope(value, require_available=True)
    with pytest.raises(courses.CourseScopeError) as excinfo:
        courses.curriculum_snapshot(courses.resolve_course_scope(value))
    assert not isinstance(excinfo.value, courses.CourseNotAvailable)


def test_catalog_and_snapshots_are_independent_for_concurrent_clients():
    catalog = courses.course_catalog()
    catalog["courses"][0]["curriculums"].clear()
    assert courses.course_catalog()["courses"][0]["curriculums"]
    versions = list(CURRICULUM_NAMES) * 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(lambda version: courses.curriculum_snapshot(
            courses.legacy_math_scope(version)), versions))
    for version, snapshot in zip(versions, snapshots):
        assert snapshot["curriculum"] == load_curriculum(version)
        snapshot["curriculum"].clear()
    assert courses.curriculum_snapshot(courses.legacy_math_scope())["curriculum"]


def test_resource_revision_tracks_definitions_not_mtimes(tmp_path, monkeypatch):
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    original = courses.course_catalog()["revision"]
    resource = tmp_path / "courses.json"
    resource.write_text(json.dumps(raw, ensure_ascii=False, indent=4), encoding="utf-8")
    monkeypatch.setattr(courses, "COURSES_FILE", resource)
    try:
        courses._registry.cache_clear()
        assert courses.course_catalog()["revision"] == original
        raw["courses"][1]["name"] += " revised"
        resource.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        courses._registry.cache_clear()
        assert courses.course_catalog()["revision"] != original
    finally:
        courses._registry.cache_clear()


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(schema_version=True),
    lambda raw: raw["courses"].append(deepcopy(raw["courses"][0])),
    lambda raw: raw["courses"][1].update(status="unknown"),
    lambda raw: raw["courses"][1].update(curriculum_provider="legacy_math"),
    lambda raw: raw["courses"][2]["curriculums"][0].update(id="cn_physics_core"),
    lambda raw: raw["courses"][1]["curriculums"][0].update(verification="verified"),
    lambda raw: raw["courses"].pop(0),
])
def test_invalid_registry_cannot_enable_physics_or_drop_math(mutation):
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    mutation(raw)
    with pytest.raises(courses.CourseScopeError):
        courses._validate_registry(raw)


def test_duplicate_resource_keys_fail():
    with pytest.raises(courses.CourseScopeError):
        json.loads('{"schema_version":1,"schema_version":1}', object_pairs_hook=courses._unique_object)


def test_http_scope_contract_is_read_only():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        catalog = client.get("/api/courses")
        assert catalog.status_code == 200
        assert client.get("/api/courses/unknown").status_code == 404
        for version in CURRICULUM_NAMES:
            response = client.post("/api/courses/resolve", json=scope_for(curriculum_id=version))
            assert response.status_code == 200
            assert response.json()["curriculum"] == load_curriculum(version)
        assert client.post("/api/courses/resolve", json=scope_for(
            "ap_physics_c_em", "ap_c_em", "physics")).status_code == 200
        assert client.post("/api/courses/resolve", json={**scope_for(), "revision": "old"}).status_code == 409
        assert client.post("/api/courses/resolve", json={**scope_for(), "subject": "physics"}).status_code == 400
        assert client.post("/api/courses/resolve", json={"course_id": "cn_hs_math"}).status_code == 400
        groups = client.get("/api/courses/cn_hs_physics/groups")
        assert groups.status_code == 200
        assert groups.json()["group_count"] == 4
        assert groups.json()["province_count"] == 10
        assert client.get("/api/courses/ap_physics_c_mechanics/groups").status_code == 404
        assert client.get("/api/courses").json() == catalog.json()
