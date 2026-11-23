"""Course ownership persists without changing legacy mathematics behavior."""

import sqlite3

import pytest
from sqlalchemy import create_engine

from mathbank import database as database_module
from mathbank import db_migrations
from mathbank.database import KnowledgePointCurriculum, Question, QuestionCurriculum


COURSE_COLUMN_INDEXES = (
    "ix_questions_subject",
    "ix_questions_course_id",
    "ix_question_curriculums_course_id",
    "ix_knowledge_point_curriculums_course_id",
)


def _create_v16_database(path):
    engine = create_engine(f"sqlite:///{path}")
    database_module.Base.metadata.create_all(bind=engine)
    engine.dispose()
    with sqlite3.connect(path) as connection:
        for index in COURSE_COLUMN_INDEXES:
            connection.execute(f'DROP INDEX IF EXISTS "{index}"')
        connection.execute("ALTER TABLE question_curriculums DROP COLUMN course_id")
        connection.execute("ALTER TABLE knowledge_point_curriculums DROP COLUMN course_id")
        connection.execute("ALTER TABLE questions DROP COLUMN course_id")
        connection.execute("ALTER TABLE questions DROP COLUMN subject")
        connection.execute("PRAGMA user_version=16")
        connection.execute(
            "INSERT INTO questions (id, content, image_paths) "
            "VALUES (1, '旧数学题', '[]')"
        )
        connection.execute(
            "INSERT INTO question_curriculums "
            "(id, question_id, version_code, compulsory, chapter, knowledge) "
            "VALUES (1, 1, 'A', '必修一', '第一章', '1.1')"
        )
        connection.execute(
            "INSERT INTO knowledge_points "
            "(code, name, original_name, source, aliases) "
            "VALUES ('p1', '旧知识点', '旧知识点', 'local', '[]')"
        )
        connection.execute(
            "INSERT INTO knowledge_point_curriculums "
            "(point_code, version_code, compulsory, chapter, knowledge, "
            "reference_section, level) "
            "VALUES ('p1', 'A', '必修一', '第一章', '1.1', '', 'leaf')"
        )


def _columns(path, table):
    with sqlite3.connect(path) as connection:
        return {
            row[1]: row
            for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        }


def test_version_seventeen_backfills_all_course_ownership(tmp_path, monkeypatch):
    database_path = tmp_path / "v16.db"
    _create_v16_database(database_path)
    monkeypatch.setattr(
        db_migrations, "SCHEMA_SNAPSHOT_DIR", tmp_path / "schema_snapshots"
    )
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        result = db_migrations.migrate_database(engine)
    finally:
        engine.dispose()

    assert result["from_version"] == 16
    assert result["to_version"] == 17 == db_migrations.LATEST_SCHEMA_VERSION
    assert result["backup"]
    assert result["added_questions_subject"] == 1
    assert result["added_questions_course_id"] == 1
    assert result["added_question_curriculums_course_id"] == 1
    assert result["added_knowledge_point_curriculums_course_id"] == 1

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 17
        assert connection.execute(
            "SELECT subject, course_id FROM questions WHERE id=1"
        ).fetchone() == ("math", "cn_hs_math")
        assert connection.execute(
            "SELECT course_id FROM question_curriculums WHERE id=1"
        ).fetchone() == ("cn_hs_math",)
        assert connection.execute(
            "SELECT course_id FROM knowledge_point_curriculums WHERE point_code='p1'"
        ).fetchone() == ("cn_hs_math",)
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert set(COURSE_COLUMN_INDEXES).issubset(indexes)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    assert _columns(database_path, "questions")["subject"][3] == 1
    assert _columns(database_path, "questions")["course_id"][3] == 1


def test_version_seventeen_migration_is_idempotent(tmp_path, monkeypatch):
    database_path = tmp_path / "v16.db"
    _create_v16_database(database_path)
    monkeypatch.setattr(
        db_migrations, "SCHEMA_SNAPSHOT_DIR", tmp_path / "schema_snapshots"
    )
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        db_migrations.migrate_database(engine)
        second = db_migrations.migrate_database(engine)
    finally:
        engine.dispose()
    assert second["from_version"] == 17
    assert second["to_version"] == 17
    assert second["backup"] is None


def test_course_migration_rolls_back_columns_and_version(tmp_path, monkeypatch):
    database_path = tmp_path / "v16.db"
    _create_v16_database(database_path)
    monkeypatch.setattr(
        db_migrations, "SCHEMA_SNAPSHOT_DIR", tmp_path / "schema_snapshots"
    )

    def fail(_connection):
        raise RuntimeError("forced course validation failure")

    monkeypatch.setattr(db_migrations, "_validate_course_schema", fail)
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        with pytest.raises(RuntimeError, match="forced course validation failure"):
            db_migrations.migrate_database(engine)
    finally:
        engine.dispose()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 16
        assert "subject" not in _columns(database_path, "questions")
        assert "course_id" not in _columns(database_path, "question_curriculums")


def _question_payload(**overrides):
    payload = {
        "content": "测试题干",
        "question_type": "single_choice",
        "category_compulsory": "必修一",
        "category_chapter": "第一章",
        "category_knowledge": "1.1",
        "difficulty": "medium",
        "source": "",
        "answer_markdown": "",
        "review": "",
        "tikz_code": "",
        "figure_align": "right",
        "figure_align_custom": "false",
        "figure_size": "auto",
        "tags": "",
        "related_question_id": "",
        "image_paths": "[]",
    }
    payload.update(overrides)
    return payload


_PHYSICS_LOCATION = {
    "course_id": "cn_hs_physics",
    "curriculum_id": "cn_physics_renjiao_2019",
    "category_compulsory": "必修第一册",
    "category_chapter": "运动的描述",
    "category_knowledge": "质点 参考系",
}


def test_question_api_defaults_to_math_and_accepts_physics_course(client, db_session):
    from main import LOCAL_TOKEN

    headers = {"X-Local-Token": LOCAL_TOKEN}
    created = client.post("/api/questions", data=_question_payload(), headers=headers)
    assert created.status_code == 200
    question = created.json()["question"]
    assert question["subject"] == "math"
    assert question["course_id"] == "cn_hs_math"

    row = db_session.get(Question, question["id"])
    assert (row.subject, row.course_id) == ("math", "cn_hs_math")
    mapping = db_session.query(QuestionCurriculum).filter_by(
        question_id=row.id
    ).one()
    assert mapping.course_id == "cn_hs_math"

    physics = client.post(
        "/api/questions",
        data=_question_payload(**_PHYSICS_LOCATION),
        headers=headers,
    )
    assert physics.status_code == 200
    physics_question = physics.json()["question"]
    assert physics_question["subject"] == "physics"
    assert physics_question["course_id"] == "cn_hs_physics"
    physics_row = db_session.get(Question, physics_question["id"])
    assert (physics_row.subject, physics_row.course_id) == (
        "physics", "cn_hs_physics",
    )
    physics_mapping = db_session.query(QuestionCurriculum).filter_by(
        question_id=physics_row.id
    ).one()
    assert physics_mapping.course_id == "cn_hs_physics"
    assert physics_mapping.version_code == "cn_physics_renjiao_2019"

    rejected = client.post(
        "/api/questions",
        data=_question_payload(
            course_id="cn_hs_physics",
            curriculum_id="cn_physics_renjiao_2019",
        ),
        headers=headers,
    )
    assert rejected.status_code == 400
    assert db_session.query(Question).count() == 2

    unknown = client.post(
        "/api/questions",
        data=_question_payload(course_id="unknown_course"),
        headers=headers,
    )
    assert unknown.status_code == 400


def test_question_reads_are_scoped_to_the_requested_course(client, db_session):
    from main import LOCAL_TOKEN

    headers = {"X-Local-Token": LOCAL_TOKEN}
    created = client.post("/api/questions", data=_question_payload(), headers=headers)
    question_id = created.json()["question"]["id"]
    created_physics = client.post(
        "/api/questions",
        data=_question_payload(**_PHYSICS_LOCATION),
        headers=headers,
    )
    assert created_physics.status_code == 200
    physics_id = created_physics.json()["question"]["id"]

    listed = client.get("/api/questions").json()
    assert [item["id"] for item in listed] == [question_id]
    physics_listed = client.get(
        "/api/questions?course_id=cn_hs_physics"
    ).json()
    assert [item["id"] for item in physics_listed] == [physics_id]
    assert client.get("/api/questions?course_id=unknown").status_code == 400

    assert client.get(f"/api/questions/{question_id}?course_id=cn_hs_math").status_code == 200
    assert client.get(f"/api/questions/{physics_id}").status_code == 404
    assert client.get(f"/api/questions/{physics_id}?course_id=cn_hs_math").status_code == 404
    assert client.get(
        f"/api/questions/{physics_id}?course_id=cn_hs_physics"
    ).status_code == 200

    stats = client.get("/api/stats").json()
    assert stats["total_count"] == 1
    physics_stats = client.get("/api/stats?course_id=cn_hs_physics").json()
    assert physics_stats["total_count"] == 1

    moved = client.put(
        f"/api/questions/{question_id}",
        data=_question_payload(**_PHYSICS_LOCATION),
        headers=headers,
    )
    assert moved.status_code == 409
    db_session.expire_all()
    assert db_session.get(Question, question_id).course_id == "cn_hs_math"


def test_exchange_serialization_preserves_course_ownership(client, db_session):
    from main import LOCAL_TOKEN
    from mathbank.sync_helper import serialize_exchange_questions

    headers = {"X-Local-Token": LOCAL_TOKEN}
    created = client.post(
        "/api/questions",
        data=_question_payload(course_id="cn_hs_math"),
        headers=headers,
    )
    question = db_session.get(Question, created.json()["question"]["id"])
    records, mappings = serialize_exchange_questions(db_session, [question], "A")
    assert records[0]["subject"] == "math"
    assert records[0]["course_id"] == "cn_hs_math"
    assert mappings[question.id][0]["course_id"] == "cn_hs_math"
