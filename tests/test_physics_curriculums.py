"""Bundled AP Physics C 2027 catalogs remain read-only and internally valid."""

from copy import deepcopy
import json
import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mathbank import courses
from mathbank.course_api import router
from mathbank.paths import COURSES_FILE, PHYSICS_CURRICULUMS_DIR
from mathbank.physics_curriculums import (
    PhysicsCurriculumError, load_physics_curriculum,
    physics_curriculum_catalog, validate_physics_curriculum,
)


def test_ap_mechanics_catalog_matches_the_2027_ced():
    catalog = load_physics_curriculum("ap_c_mechanics")
    assert catalog["course_id"] == "ap_physics_c_mechanics"
    assert catalog["structure"] == "units"
    assert [unit["number"] for unit in catalog["units"]] == list(range(1, 8))
    assert sum(len(unit["topics"]) for unit in catalog["units"]) == 41
    exam = catalog["exam"]
    assert exam["year"] == 2027
    assert exam["multiple_choice"] == 42
    assert exam["multiple_choice_minutes"] == 85
    assert exam["free_response"] == 4
    assert exam["free_response_minutes"] == 95
    assert exam["calculator_allowed"] is True
    assert [practice["id"] for practice in exam["science_practices"]] == [1, 2, 3]
    assert sum(len(practice["skills"]) for practice in exam["science_practices"]) == 10
    assert [item["title"] for item in exam["free_response_types"]] == [
        "Mathematical Routines",
        "Translation Between Representations",
        "Experimental Design and Analysis",
        "Qualitative/Quantitative Translation",
    ]
    assert catalog["units"][0]["topics"][0] == {
        "id": "1.1", "title": "Scalars and Vectors",
    }


def test_ap_em_catalog_matches_the_2027_ced():
    catalog = load_physics_curriculum("ap_c_em")
    assert catalog["course_id"] == "ap_physics_c_em"
    assert [unit["number"] for unit in catalog["units"]] == list(range(8, 14))
    assert sum(len(unit["topics"]) for unit in catalog["units"]) == 31
    assert catalog["exam"]["year"] == 2027
    assert len(catalog["exam"]["science_practices"]) == 3
    assert len(catalog["exam"]["free_response_types"]) == 4
    assert catalog["units"][-1]["topics"][-1] == {
        "id": "13.6",
        "title": "Circuits with Capacitors and Inductors (LC Circuits)",
    }


def test_renjiao_catalog_has_six_books_and_125_sections():
    catalog = load_physics_curriculum("cn_physics_renjiao_2019")
    assert catalog["course_id"] == "cn_hs_physics"
    assert catalog["structure"] == "books"
    assert catalog["exam"] is None
    assert len(catalog["books"]) == 6
    assert catalog["books"][0]["title"].startswith("必修第一册")
    assert catalog["books"][-1]["title"].startswith("选择性必修第三册")
    assert sum(
        len(chapter["sections"])
        for book in catalog["books"]
        for chapter in book["chapters"]
    ) == 125
    assert catalog["books"][0]["chapters"][0]["sections"][0] == {
        "id": "1.1", "title": "质点 参考系",
    }


def test_physics_catalog_is_read_only_and_detached():
    first = load_physics_curriculum("ap_c_mechanics")
    first["units"].clear()
    second = load_physics_curriculum("ap_c_mechanics")
    assert len(second["units"]) == 7
    summary = physics_curriculum_catalog()
    assert {(item["course_id"], item["unit_count"], item["book_count"],
             item["topic_count"]) for item in summary["curriculums"]} == {
        ("ap_physics_c_mechanics", 7, 0, 41),
        ("ap_physics_c_em", 6, 0, 31),
        ("cn_hs_physics", 0, 6, 125),
    }


@pytest.mark.parametrize("curriculum_id", [
    "", "../ap_c_mechanics", "ap_c_mechanics_2027", "AP_C_MECHANICS",
])
def test_unknown_or_unsafe_curriculum_ids_are_rejected(curriculum_id):
    with pytest.raises(PhysicsCurriculumError):
        load_physics_curriculum(curriculum_id)


def test_duplicate_topic_ids_and_bad_course_ownership_are_rejected():
    catalog = load_physics_curriculum("ap_c_mechanics")
    duplicate = deepcopy(catalog)
    duplicate["units"][1]["topics"][0]["id"] = "1.1"
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(duplicate)
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(catalog, expected_course_id="ap_physics_c_em")


def test_exam_science_practices_and_skills_are_validated():
    catalog = load_physics_curriculum("ap_c_mechanics")
    duplicate_skill = deepcopy(catalog)
    duplicate_skill["exam"]["science_practices"][1]["skills"][0]["id"] = "1.A"
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(duplicate_skill)
    wrong_prefix = deepcopy(catalog)
    wrong_prefix["exam"]["science_practices"][0]["skills"][0]["id"] = "2.A"
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(wrong_prefix)
    unknown_field = deepcopy(catalog)
    unknown_field["exam"]["science_practices"][0]["unexpected"] = True
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(unknown_field)
    bad_frq = deepcopy(catalog)
    bad_frq["exam"]["free_response_types"][1]["id"] = 1
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(bad_frq)
    bad_year = deepcopy(catalog)
    bad_year["exam"]["year"] = "2027"
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(bad_year)
    bad_calculator = deepcopy(catalog)
    bad_calculator["exam"]["calculator_allowed"] = "yes"
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(bad_calculator)


def test_books_catalog_stays_compatible_without_ap_exam_fields():
    data = {
        "schema_version": 1,
        "curriculum_id": "cn_physics_core",
        "course_id": "cn_hs_physics",
        "title": "Test physics textbook catalog",
        "verification": "secondary",
        "source": {
            "publisher": "Test publisher",
            "document": "Test textbook catalog",
            "url": "https://example.com/test-catalog",
            "version": "Test edition",
            "verified_on": "2026-11-21",
        },
        "structure": "books",
        "units": [],
        "books": [{
            "id": "book_1",
            "title": "Test book",
            "chapters": [{
                "id": "1",
                "title": "Test chapter",
                "sections": [{"id": "1.1", "title": "Test section"}],
            }],
        }],
        "exam": None,
    }
    result = validate_physics_curriculum(data)
    assert result["structure"] == "books"
    assert result["exam"] is None
    assert result["units"] == []
    with_exam = deepcopy(data)
    with_exam["exam"] = {"multiple_choice": 42}
    with pytest.raises(PhysicsCurriculumError):
        validate_physics_curriculum(with_exam)


def test_official_catalogs_are_registered_and_course_is_available():
    catalog = courses.course_catalog()
    by_id = {course["id"]: course for course in catalog["courses"]}
    assert by_id["ap_physics_c_mechanics"]["status"] == "available"
    assert by_id["ap_physics_c_mechanics"]["curriculums"][0]["verification"] == "official"
    assert load_physics_curriculum("ap_c_mechanics")["course_id"] == "ap_physics_c_mechanics"
    assert load_physics_curriculum("ap_c_em")["course_id"] == "ap_physics_c_em"
    assert by_id["cn_hs_physics"]["curriculums"][1]["id"] == "cn_physics_renjiao_2019"
    assert by_id["cn_hs_physics"]["curriculums"][1]["verification"] == "official"
    assert load_physics_curriculum("cn_physics_renjiao_2019")["course_id"] == "cn_hs_physics"
    assert courses.require_available_course(
        "ap_physics_c_mechanics"
    )["status"] == "available"


def test_official_status_requires_a_safe_resource_path():
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    broken = deepcopy(raw)
    broken["courses"][2]["curriculums"][0].pop("resource")
    with pytest.raises(courses.CourseScopeError):
        courses._validate_registry(broken)
    broken = deepcopy(raw)
    broken["courses"][2]["curriculums"][0]["resource"] = "../ap_c_mechanics.json"
    with pytest.raises(courses.CourseScopeError):
        courses._validate_registry(broken)
    broken = deepcopy(raw)
    broken["courses"][2]["curriculums"][0]["resource"] = "physics/ap_c_em.json"
    with pytest.raises(courses.CourseScopeError):
        courses._validate_registry(broken)


def test_physics_catalog_http_reports_writable_catalogs():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get(
            "/api/courses/ap_physics_c_mechanics/curriculums/ap_c_mechanics"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["writable"] is True
        assert payload["course_status"] == "available"
        assert len(payload["units"]) == 7
        assert payload["exam"]["year"] == 2027
        assert len(payload["exam"]["science_practices"]) == 3
        assert len(payload["exam"]["free_response_types"]) == 4
        assert client.get(
            "/api/courses/ap_physics_c_em/curriculums/ap_c_em"
        ).json()["exam"]["multiple_choice"] == 42
        renjiao = client.get(
            "/api/courses/cn_hs_physics/curriculums/cn_physics_renjiao_2019"
        )
        assert renjiao.status_code == 200
        assert renjiao.json()["writable"] is True
        assert renjiao.json()["course_status"] == "available"
        assert len(renjiao.json()["books"]) == 6
        assert client.get(
            "/api/courses/cn_hs_physics/curriculums/cn_physics_core"
        ).status_code == 409
        assert client.get(
            "/api/courses/ap_physics_c_mechanics/curriculums/unknown"
        ).status_code == 404
        assert client.get(
            "/api/courses/unknown/curriculums/ap_c_mechanics"
        ).status_code == 404


def test_catalog_revision_includes_ap_resource_content(tmp_path, monkeypatch):
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    resource_dir = tmp_path / "physics"
    resource_dir.mkdir()
    for name in (
        "ap_c_mechanics.json", "ap_c_em.json",
        "cn_physics_renjiao_2019.json",
    ):
        shutil.copy2(PHYSICS_CURRICULUMS_DIR / name, resource_dir / name)
    courses_file = tmp_path / "courses.json"
    courses_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(courses, "COURSES_FILE", courses_file)
    monkeypatch.setattr(courses, "PHYSICS_CURRICULUMS_DIR", resource_dir)
    courses._registry.cache_clear()
    try:
        before = courses.course_catalog()["revision"]
        resource = resource_dir / "ap_c_mechanics.json"
        data = json.loads(resource.read_text(encoding="utf-8"))
        data["source"]["verified_on"] = "2026-11-22"
        resource.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        courses._registry.cache_clear()
        assert courses.course_catalog()["revision"] != before
    finally:
        courses._registry.cache_clear()


def test_registry_rejects_catalog_course_mismatch(tmp_path, monkeypatch):
    raw = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    resource_dir = tmp_path / "physics"
    resource_dir.mkdir()
    for name in (
        "ap_c_mechanics.json", "ap_c_em.json",
        "cn_physics_renjiao_2019.json",
    ):
        shutil.copy2(PHYSICS_CURRICULUMS_DIR / name, resource_dir / name)
    mechanics = resource_dir / "ap_c_mechanics.json"
    data = json.loads(mechanics.read_text(encoding="utf-8"))
    data["course_id"] = "ap_physics_c_em"
    mechanics.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    courses_file = tmp_path / "courses.json"
    courses_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(courses, "COURSES_FILE", courses_file)
    monkeypatch.setattr(courses, "PHYSICS_CURRICULUMS_DIR", resource_dir)
    courses._registry.cache_clear()
    try:
        with pytest.raises(courses.CourseScopeError):
            courses.course_catalog()
    finally:
        courses._registry.cache_clear()
