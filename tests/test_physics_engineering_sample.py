"""M3/M4 engineering sample and course-isolated paper contracts."""

import json
from pathlib import Path

import pytest

from mathbank.course_curriculum import (
    curriculum_tree, resolve_curriculum_id, validate_question_location,
)
from mathbank.courses import get_course
from mathbank.database import Question
from main import LOCAL_TOKEN


FIXTURE = Path(__file__).parent / "fixtures" / "physics_engineering_sample.json"


def _sample():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_engineering_sample_has_required_scope_and_coverage():
    payload = _sample()
    entries = payload["entries"]
    assert len(entries) == 120
    assert {course: sum(item["course_id"] == course for item in entries)
            for course in payload["counts"]} == payload["counts"]

    for entry in entries:
        course = get_course(entry["course_id"])
        resolved = resolve_curriculum_id(course, entry["curriculum_id"])
        tree = curriculum_tree(course, resolved)
        validate_question_location(
            tree, entry["compulsory"], entry["chapter"], entry["knowledge"],
            strict=course["subject"] == "physics",
        )
        assert entry["question_type"] in {
            "single_choice", "multi_choice", "fill_in_blank", "detailed_answer",
        }
        assert "formula" in entry["coverage"]
        assert entry["content"].strip()
        assert entry["answer_markdown"].strip()

    flags = {name: sum(name in item["coverage"] for item in entries)
             for name in ("formula", "subquestions", "units", "table", "figure")}
    assert flags["formula"] == 120
    assert flags["subquestions"] >= 60
    assert flags["units"] >= 40
    assert flags["table"] >= 20
    assert flags["figure"] >= 20


def test_sample_questions_save_and_stay_inside_their_course(client, db_session):
    headers = {"X-Local-Token": LOCAL_TOKEN}
    saved = {}
    for course_id in (
        "cn_hs_math", "cn_hs_physics",
        "ap_physics_c_mechanics", "ap_physics_c_em",
    ):
        entry = next(
            item for item in _sample()["entries"]
            if item["course_id"] == course_id and "figure" not in item["coverage"]
        )
        response = client.post("/api/questions", headers=headers, data={
            "content": entry["content"],
            "question_type": entry["question_type"],
            "category_compulsory": entry["compulsory"],
            "category_chapter": entry["chapter"],
            "category_knowledge": entry["knowledge"],
            "difficulty": "medium",
            "source": entry["source"],
            "answer_markdown": entry["answer_markdown"],
            "review": "",
            "course_id": entry["course_id"],
            "curriculum_id": entry["curriculum_id"],
            "image_paths": "[]",
        })
        assert response.status_code == 200, response.text
        saved[course_id] = response.json()["question"]["id"]

    for course_id, question_id in saved.items():
        listing = client.get("/api/questions", params={"course_id": course_id})
        assert listing.status_code == 200
        assert [item["id"] for item in listing.json()] == [question_id]
        detail = client.get(
            f"/api/questions/{question_id}", params={"course_id": course_id},
        )
        assert detail.status_code == 200
        assert detail.json()["course_id"] == course_id

    wrong = client.get(
        f"/api/questions/{saved['cn_hs_physics']}",
        params={"course_id": "cn_hs_math"},
    )
    assert wrong.status_code == 404


def test_cross_course_mutation_association_and_stats_are_isolated(client):
    headers = {"X-Local-Token": LOCAL_TOKEN}

    def create(course_id, curriculum_id, compulsory, chapter, knowledge):
        response = client.post("/api/questions", headers=headers, data={
            "content": f"{course_id} 隔离测试",
            "question_type": "single_choice",
            "category_compulsory": compulsory,
            "category_chapter": chapter,
            "category_knowledge": knowledge,
            "difficulty": "medium",
            "source": "隔离测试",
            "answer_markdown": "答案",
            "review": "",
            "course_id": course_id,
            "curriculum_id": curriculum_id,
            "image_paths": "[]",
        })
        assert response.status_code == 200, response.text
        return response.json()["question"]["id"]

    math_id = create("cn_hs_math", "A", "必修一", "第一章", "")
    physics_id = create(
        "cn_hs_physics", "cn_physics_renjiao_2019",
        "必修第一册", "运动的描述", "质点 参考系",
    )

    update = client.put(
        f"/api/questions/{physics_id}",
        headers=headers,
        data={
            "course_id": "cn_hs_math",
            "content": "越权修改",
            "question_type": "single_choice",
            "difficulty": "medium",
        },
        params={"course_id": "cn_hs_math"},
    )
    assert update.status_code == 409
    assert client.delete(
        f"/api/questions/{physics_id}", headers=headers,
        params={"course_id": "cn_hs_math"},
    ).status_code == 404
    assert client.post(
        f"/api/questions/{physics_id}/associate",
        headers=headers,
        data={"target_id": math_id, "course_id": "cn_hs_physics"},
    ).status_code == 400

    math_stats = client.get("/api/stats", params={"course_id": "cn_hs_math"})
    physics_stats = client.get("/api/stats", params={"course_id": "cn_hs_physics"})
    assert math_stats.status_code == physics_stats.status_code == 200
    assert math_stats.json()["total_count"] == 1
    assert physics_stats.json()["total_count"] == 1


def _add_questions(db_session, course_id, subject, count, question_type):
    rows = []
    for index in range(count):
        row = Question(
            content=f"{course_id} {question_type} {index}",
            question_type=question_type,
            course_id=course_id,
            subject=subject,
        )
        db_session.add(row)
        rows.append(row)
    db_session.commit()
    return [row.id for row in rows]


def _save_paper(client, course_id, curriculum_id, paper_type, questions):
    response = client.post("/api/paper/save", json={
        "title": f"{course_id} {paper_type}",
        "paper_type": paper_type,
        "course_id": course_id,
        "curriculum_id": curriculum_id,
        "metadata": (
            {"province_group_id": "group_shanghai_2026"}
            if course_id == "cn_hs_physics" else {}
        ),
        "questions": questions,
    }, headers={"X-Local-Token": LOCAL_TOKEN})
    return response


@pytest.mark.parametrize("course_id,curriculum_id", [
    ("ap_physics_c_mechanics", "ap_c_mechanics"),
    ("ap_physics_c_em", "ap_c_em"),
])
def test_two_ap_2027_papers_per_course(client, db_session, course_id, curriculum_id):
    for paper_index in range(2):
        choice = _add_questions(
            db_session, course_id, "physics", 42, "single_choice",
        )
        written = _add_questions(
            db_session, course_id, "physics", 4, "detailed_answer",
        )
        questions = (
            [{"id": qid, "score": 4} for qid in choice]
            + [{"id": qid, "score": 42} for qid in written]
        )
        response = _save_paper(
            client, course_id, curriculum_id, "ap_exam", questions,
        )
        assert response.status_code == 200, response.text

        invalid = _save_paper(
            client, course_id, curriculum_id, "ap_exam",
            questions[:-1],
        )
        assert invalid.status_code == 400
        assert "42" in invalid.json()["message"]

    listed = client.get("/api/papers", params={"course_id": course_id})
    assert listed.status_code == 200
    assert len(listed.json()["data"]) == 2


def test_two_domestic_physics_papers_and_math_isolation(client, db_session):
    for paper_index in range(2):
        rows = (
            _add_questions(db_session, "cn_hs_physics", "physics", 8, "single_choice")
            + _add_questions(db_session, "cn_hs_physics", "physics", 4, "multi_choice")
            + _add_questions(db_session, "cn_hs_physics", "physics", 3, "detailed_answer")
        )
        response = _save_paper(
            client, "cn_hs_physics", "cn_physics_renjiao_2019",
            "physics_exam", [{"id": qid, "score": 5} for qid in rows],
        )
        assert response.status_code == 200, response.text

    assert client.get(
        "/api/papers", params={"course_id": "cn_hs_physics"},
    ).json()["data"]
    assert client.get(
        "/api/papers", params={"course_id": "cn_hs_math"},
    ).json()["data"] == []
