"""AP 2027 and physics paper templates are enforced before export."""

import pytest
import zipfile
from io import BytesIO

from mathbank.courses import get_course
from mathbank.paper_exam_profile import PaperStructureError, validate_paper_structure
from mathbank.paper_helper import build_latex_document
from mathbank.question_types import PAPER_TYPE_ORDER, paper_type_order
from mathbank.word_export_helper import build_word_document


def _items(choice=42, written=4, choice_score=10, written_score=105):
    return (
        [{"question_type": "single_choice", "score": choice_score}] * choice
        + [{"question_type": "detailed_answer", "score": written_score}] * written
    )


@pytest.mark.parametrize("course_id,curriculum_id", [
    ("ap_physics_c_mechanics", "ap_c_mechanics"),
    ("ap_physics_c_em", "ap_c_em"),
])
def test_ap_2027_requires_42_choice_and_4_free_response(course_id, curriculum_id):
    course = get_course(course_id)
    profile = validate_paper_structure(
        course, curriculum_id, "ap_exam", _items(),
    )
    assert profile["exam_year"] == 2027
    assert profile["multiple_choice"] == 42
    assert profile["free_response"] == 4
    assert profile["section_scores"] == {
        "multiple_choice": 420, "free_response": 420,
    }

    with pytest.raises(PaperStructureError, match="42 道选择题"):
        validate_paper_structure(
            course, curriculum_id, "ap_exam", _items(choice=41),
        )
    with pytest.raises(PaperStructureError, match="50%"):
        validate_paper_structure(
            course, curriculum_id, "ap_exam",
            _items(choice_score=1, written_score=9),
        )
    with pytest.raises(PaperStructureError, match="大于 0"):
        validate_paper_structure(
            course, curriculum_id, "ap_exam",
            _items(choice_score=0, written_score=0),
        )
    with pytest.raises(PaperStructureError, match="AP 2027"):
        validate_paper_structure(
            course, curriculum_id, "exam", _items(),
        )


def test_domestic_physics_uses_group_and_rejects_math_exam_template():
    course = get_course("cn_hs_physics")
    profile = validate_paper_structure(
        course, "cn_physics_renjiao_2019", "physics_exam",
        [{"question_type": "single_choice", "score": 4}],
    )
    assert profile["kind"] == "cn_physics"

    grouped = validate_paper_structure(
        course, "cn_physics_renjiao_2019", "physics_exam",
        [{"question_type": "single_choice", "score": 4}],
        {"province_group_id": "group_shanghai_2026"},
    )
    assert grouped["province_group_id"] == "group_shanghai_2026"
    assert grouped["exam_profile_id"] == "cn_physics_shanghai_2026"

    with pytest.raises(PaperStructureError, match="19 题"):
        validate_paper_structure(
            course, "cn_physics_renjiao_2019", "exam_19",
            [{"question_type": "single_choice", "score": 4}],
        )
    with pytest.raises(PaperStructureError, match="不存在"):
        validate_paper_structure(
            course, "cn_physics_renjiao_2019", "physics_exam",
            [{"question_type": "single_choice", "score": 4}],
            {"province_group_id": "missing_group"},
        )


def test_physics_paper_latex_has_two_ap_sections():
    questions = [
        {
            "question": {
                "id": 1, "content": "选择题题干", "question_type": "single_choice",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 1,
        },
        {
            "question": {
                "id": 2, "content": "自由作答题干",
                "question_type": "detailed_answer",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 10,
        },
    ]
    source = build_latex_document(
        "AP 测试卷", "", "ap_exam", questions, subject="物理",
    )
    assert "Section I: Multiple Choice" in source
    assert "Section II: Free Response" in source
    assert r"\subject{物理}" in source
    assert "180 分钟" in source


def test_physics_word_export_has_two_ap_sections():
    questions = [
        {
            "question": {
                "id": 1, "content": "选择题题干", "question_type": "single_choice",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 1,
        },
        {
            "question": {
                "id": 2, "content": "自由作答题干",
                "question_type": "detailed_answer",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 10,
        },
    ]
    document, diagnostics = build_word_document(
        "AP 测试卷", "", "ap_exam", questions, subject="物理",
    )
    with zipfile.ZipFile(BytesIO(document)) as package:
        xml = package.read("word/document.xml").decode("utf-8")
    assert "Section I: Multiple Choice" in xml
    assert "Section II: Free Response" in xml
    assert diagnostics["answer_card_omitted"] is False


def test_ap_exam_ignores_saved_section_order_that_reverses_sections():
    reversed_order = ["detailed_answer", "single_choice"]
    assert paper_type_order(
        reversed_order, "ap_exam", [], reversed_order,
    ) == list(PAPER_TYPE_ORDER)

    questions = [
        {
            "question": {
                "id": 2, "content": "自由作答题干",
                "question_type": "detailed_answer",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 10,
        },
        {
            "question": {
                "id": 1, "content": "选择题题干",
                "question_type": "single_choice",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 1,
        },
    ]
    source = build_latex_document(
        "AP 测试卷", "", "ap_exam", questions, subject="物理",
        section_order=reversed_order,
    )
    assert source.index("Section I: Multiple Choice") < source.index(
        "Section II: Free Response"
    )


def test_ap_exam_keeps_custom_written_type_in_section_two():
    questions = [
        {
            "question": {
                "id": 1, "content": "选择题题干", "question_type": "single_choice",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 1,
        },
        {
            "question": {
                "id": 2, "content": "实验分析题干",
                "question_type": "experiment_analysis",
                "course_id": "ap_physics_c_mechanics",
            },
            "score": 9,
        },
    ]
    source = build_latex_document(
        "AP 测试卷", "", "ap_exam", questions, subject="物理",
        section_order=["experiment_analysis", "single_choice"],
    )
    assert "实验分析题干" in source
    assert source.index("Section I: Multiple Choice") < source.index(
        "Section II: Free Response"
    )
    assert source.index("Section II: Free Response") < source.index("实验分析题干")
