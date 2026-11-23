"""Course discovery and validation with no Session, initialization or writes."""

from fastapi import APIRouter, HTTPException

from mathbank.courses import (
    CourseNotAvailable, CourseScopeError, StaleCourseScope, course_capabilities,
    course_catalog, curriculum_snapshot, get_course, resolve_course_scope,
)
from mathbank.course_curriculum import curriculum_tree
from mathbank.physics_curriculums import (
    PhysicsCurriculumError, load_physics_curriculum,
)
from mathbank.physics_groups import (
    PhysicsGroupError, physics_group_catalog,
)


router = APIRouter(prefix="/api/courses", tags=["courses"])


@router.get("")
def list_courses():
    return course_catalog()


@router.post("/resolve")
def resolve_scope(payload: dict):
    """Validate capability too: a planned identity cannot execute math flows."""
    try:
        scope = resolve_course_scope(payload, require_available=True)
        return curriculum_snapshot(scope)
    except (StaleCourseScope, CourseNotAvailable) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CourseScopeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{course_id}")
def course_details(course_id: str):
    try:
        course = get_course(course_id)
        return {"revision": course_catalog()["revision"], **course}
    except CourseScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{course_id}/curriculums/{curriculum_id}")
def curriculum_details(course_id: str, curriculum_id: str):
    """Read a bundled catalog without enabling writes for a planned course."""

    try:
        course = get_course(course_id)
    except CourseScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    curriculum = next(
        (item for item in course["curriculums"]
         if item["id"] == curriculum_id),
        None,
    )
    if curriculum is None:
        raise HTTPException(status_code=404, detail="未找到对应的课程目录")
    if curriculum.get("verification") != "official":
        raise HTTPException(status_code=409, detail="该课程目录尚未核验")
    try:
        catalog = load_physics_curriculum(curriculum_id)
    except PhysicsCurriculumError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if catalog["course_id"] != course_id:
        raise HTTPException(status_code=409, detail="课程目录与课程归属不一致")
    capabilities = course_capabilities(course)
    return {
        "course_id": course_id,
        "curriculum_id": curriculum_id,
        "course_status": course["status"],
        "capabilities": capabilities,
        "writable": capabilities["write"],
        "tree": curriculum_tree(course, curriculum_id),
        **catalog,
    }


@router.get("/{course_id}/groups")
def course_group_details(course_id: str):
    """Return read-only exam-structure groups for the domestic physics course."""

    try:
        course = get_course(course_id)
    except CourseScopeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if course["id"] != "cn_hs_physics":
        raise HTTPException(status_code=404, detail="该课程没有省份组资源")
    try:
        payload = physics_group_catalog()
    except PhysicsGroupError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "course_id": course["id"],
        "course_status": course["status"],
        **payload,
    }
