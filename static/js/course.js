/**
 * course.js - course switch and curriculum scope for every HTTP flow.
 * Loaded after api.js and before the feature scripts. No build step.
 */
(function () {
    'use strict';

    const COURSE_KEY = 'mathbank_active_course';
    const CURRICULUM_KEY_PREFIX = 'mathbank_active_curriculum:';
    const listeners = new Set();
    let catalogPromise = null;
    let catalog = null;
    let active = null;
    let activationSequence = 0;

    function storageGet(key) {
        try { return localStorage.getItem(key); } catch (error) { return null; }
    }

    function storageSet(key, value) {
        try { localStorage.setItem(key, value); } catch (error) { /* offline-safe */ }
    }

    function courseById(courseId) {
        return (catalog && catalog.courses || []).find(item => item.id === courseId) || null;
    }

    function officialCurriculum(course) {
        return (course.curriculums || []).find(item =>
            item.verification === 'official' && item.resource) || null;
    }

    function curriculumById(course, curriculumId) {
        return (course.curriculums || []).find(item => item.id === curriculumId) || null;
    }

    async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let payload = null;
        try { payload = await response.json(); } catch (error) { payload = null; }
        if (!response.ok) {
            const message = payload && (payload.detail || payload.message)
                ? (payload.detail || payload.message) : `HTTP ${response.status}`;
            throw new Error(message);
        }
        return payload;
    }

    async function loadCatalog() {
        const payload = await fetchJson('/api/courses', { cache: 'no-store' });
        if (!payload || !Array.isArray(payload.courses) || payload.courses.length === 0) {
            throw new Error('课程列表为空');
        }
        catalog = payload;
        return payload;
    }

    function ready() {
        if (!catalogPromise) {
            catalogPromise = loadCatalog().catch(error => {
                catalogPromise = null;
                throw error;
            });
        }
        return catalogPromise;
    }

    async function loadMathCurriculum() {
        const meta = await fetchJson('/api/config/metadata', { cache: 'no-store' });
        if (!meta || !meta.curriculum || typeof meta.curriculum !== 'object') {
            throw new Error('数学教材目录不可用');
        }
        return {
            curriculumId: meta.curriculum_version || 'A',
            tree: meta.curriculum,
            curriculumName: '数学教材',
            exam: null,
            status: 'available',
        };
    }

    async function loadPhysicsCurriculum(course, requestedId) {
        const stored = storageGet(CURRICULUM_KEY_PREFIX + course.id);
        let item = curriculumById(course, requestedId)
            || curriculumById(course, stored)
            || officialCurriculum(course);
        if (!item || item.verification !== 'official' || !item.resource) {
            throw new Error('该课程没有已核验的可用目录');
        }
        const payload = await fetchJson(
            `/api/courses/${encodeURIComponent(course.id)}/curriculums/${encodeURIComponent(item.id)}`,
            { cache: 'no-store' },
        );
        return {
            curriculumId: item.id,
            tree: payload.tree || {},
            curriculumName: item.name,
            exam: payload.exam || null,
            status: payload.course_status || course.status,
        };
    }

    function applyDocumentState(state) {
        document.body.dataset.courseId = state.courseId;
        document.body.dataset.courseSubject = state.subject;
        document.body.dataset.courseWritable = state.writable ? 'true' : 'false';
        window.activeCurriculumVersion = state.curriculumId;
        window.editorCurriculumVersion = state.curriculumId;
        if (window.systemMetadata && typeof window.systemMetadata === 'object') {
            window.systemMetadata.curriculum = state.tree;
        }
        window.categoryTree = state.tree;
    }

    function notify(state) {
        listeners.forEach(listener => {
            try { listener(state); } catch (error) { console.warn('course listener failed', error); }
        });
        document.dispatchEvent(new CustomEvent('mathbank:course-changed', { detail: state }));
    }

    function renderSwitcher() {
        const host = document.getElementById('appCourseSwitcher');
        const select = document.getElementById('appCourseSelect');
        if (!host || !select || !catalog || !active) return;
        select.innerHTML = '';
        catalog.courses.forEach(course => {
            const option = document.createElement('option');
            option.value = course.id;
            option.textContent = course.id === 'ap_physics_c_mechanics'
                ? 'AP C 力学'
                : course.id === 'ap_physics_c_em'
                    ? 'AP C 电磁学'
                    : course.name;
            select.appendChild(option);
        });
        select.value = active.courseId;
        const status = document.getElementById('appCourseStatus');
        if (status) {
            const writable = active.writable;
            status.textContent = writable ? '已开放' : '只读';
            status.className = 'app-course-status ' + (writable ? 'is-open' : 'is-readonly');
        }
        host.title = `${active.courseName} · ${active.curriculumName}`;
    }

    async function activate(courseId, options = {}) {
        const sequence = ++activationSequence;
        await ready();
        const course = courseById(courseId) || courseById('cn_hs_math');
        if (!course) throw new Error('未找到课程');
        const requestedCurriculum = options.curriculumId || null;
        let curriculumState;
        if (course.subject === 'math') {
            curriculumState = await loadMathCurriculum();
        } else {
            curriculumState = await loadPhysicsCurriculum(course, requestedCurriculum);
        }
        if (sequence !== activationSequence) return active;
        const capabilities = course.capabilities || {};
        active = Object.freeze({
            courseId: course.id,
            subject: course.subject,
            courseName: course.name,
            curriculumId: curriculumState.curriculumId,
            curriculumName: curriculumState.curriculumName,
            tree: curriculumState.tree,
            exam: curriculumState.exam,
            status: course.status,
            writable: Boolean(capabilities.write && course.status === 'available'),
            capabilities: Object.freeze({ ...capabilities }),
            revision: catalog.revision,
        });
        applyDocumentState(active);
        storageSet(COURSE_KEY, active.courseId);
        storageSet(CURRICULUM_KEY_PREFIX + active.courseId, active.curriculumId);
        if (options.persist !== false) {
            try { history.replaceState(null, '', window.location.pathname + window.location.search); } catch (error) { /* no-op */ }
        }
        renderSwitcher();
        notify(active);
        return active;
    }

    function params() {
        const result = { course_id: active ? active.courseId : 'cn_hs_math' };
        const curriculumId = active && active.subject === 'math'
            ? (window.activeCurriculumVersion || active.curriculumId)
            : (active && active.curriculumId);
        if (curriculumId) result.curriculum_id = curriculumId;
        return result;
    }

    function appendTo(target) {
        const values = params();
        Object.keys(values).forEach(key => {
            if (target && typeof target.set === 'function') target.set(key, values[key]);
            else if (target && typeof target.append === 'function') target.append(key, values[key]);
        });
        return target;
    }

    function bodyData(data) {
        if (data instanceof FormData) {
            if (!data.has('course_id')) data.set('course_id', params().course_id);
            if (!data.has('curriculum_id')) data.set('curriculum_id', params().curriculum_id);
            return data;
        }
        if (data && typeof data === 'object') {
            return { course_id: params().course_id, curriculum_id: params().curriculum_id, ...data };
        }
        return data;
    }

    function withQuery(url) {
        const parsed = new URL(url, window.location.origin);
        appendTo(parsed.searchParams);
        return parsed.pathname + parsed.search + parsed.hash;
    }

    function jsonBody(data) {
        return JSON.stringify(bodyData(data));
    }

    async function refreshViews() {
        if (typeof window.loadCategories === 'function') {
            await window.loadCategories();
        }
        const workspace = document.body.dataset.workspace || 'dashboard';
        if (workspace === 'bank' && typeof window.loadQuestions === 'function') {
            window.loadQuestions();
        } else if (workspace === 'paper' && typeof window.renderPaperWorkspace === 'function') {
            window.renderPaperWorkspace();
        } else if (typeof window.loadDashboardData === 'function') {
            window.loadDashboardData();
        }
    }

    async function switchCourse(courseId) {
        const previous = active;
        if (previous && previous.courseId !== courseId
                && window.resetEditorForCourseChange
                && typeof window.resetEditorForCourseChange.isDirty === 'function'
                && window.resetEditorForCourseChange.isDirty()) {
            const proceed = window.confirm('切换课程会清空当前编辑区，确定继续吗？');
            if (!proceed) {
                renderSwitcher();
                return;
            }
        }
        try {
            await activate(courseId);
            if (previous && previous.courseId !== courseId
                    && typeof window.resetEditorForCourseChange === 'function') {
                window.resetEditorForCourseChange();
            }
            await refreshViews();
            if (window.showToast && previous && previous.courseId !== courseId) {
                window.showToast(`已切换到${window.MathBankCourse.active.courseName}`, 'info');
            }
        } catch (error) {
            if (window.showToast) window.showToast(error.message || '课程切换失败', 'error');
            renderSwitcher();
        }
    }

    function installSwitcher() {
        if (document.getElementById('appCourseSwitcher')) return;
        const brand = document.querySelector('.app-nav-brand');
        if (!brand || !brand.parentNode) return;
        const host = document.createElement('div');
        host.id = 'appCourseSwitcher';
        host.className = 'app-course-switcher';
        host.innerHTML = `
            <label for="appCourseSelect" class="app-course-label">当前课程</label>
            <select id="appCourseSelect" class="app-course-select" aria-label="切换课程"></select>
            <span id="appCourseStatus" class="app-course-status is-open">已开放</span>
        `;
        brand.insertAdjacentElement('afterend', host);
        host.querySelector('#appCourseSelect').addEventListener('change', event => {
            switchCourse(event.target.value);
        });
    }

    async function init() {
        installSwitcher();
        try {
            await ready();
            const stored = storageGet(COURSE_KEY);
            const initial = courseById(stored) ? stored : 'cn_hs_math';
            await activate(initial, { persist: false });
            await refreshViews();
        } catch (error) {
            console.error('课程初始化失败:', error);
            const status = document.getElementById('appCourseStatus');
            if (status) {
                status.textContent = '课程不可用';
                status.className = 'app-course-status is-readonly';
            }
        }
    }

    window.MathBankCourse = {
        get active() { return active; },
        get catalog() { return catalog; },
        ready,
        params,
        appendTo,
        bodyData,
        withQuery,
        jsonBody,
        switchCourse,
        refreshViews,
        onChange(listener) {
            if (typeof listener === 'function') listeners.add(listener);
            return () => listeners.delete(listener);
        },
        isMath() { return Boolean(active && active.subject === 'math'); },
        isWritable() { return Boolean(active && active.writable); },
        questionLabel() {
            if (!active) return '数学题';
            if (active.subject === 'math') return '数学题';
            if (active.courseId === 'ap_physics_c_mechanics') return 'AP C 力学题';
            if (active.courseId === 'ap_physics_c_em') return 'AP C 电磁学题';
            return `${active.courseName}题`;
        },
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
