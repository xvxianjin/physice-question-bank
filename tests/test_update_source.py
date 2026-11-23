"""Update-source contract tests; no main import or live service required."""
import ast
import re
from pathlib import Path
import shutil
import subprocess

import pytest

from mathbank import GITHUB_REPO, UPSTREAM_REPO
from mathbank.update_source import get_update_source


ROOT = Path(__file__).resolve().parents[1]


def test_unconfigured_fork_keeps_attribution_without_updates(monkeypatch):
    assert UPSTREAM_REPO == "JudgePeach/math-question-bank"
    assert GITHUB_REPO == ""
    monkeypatch.setenv("MATHBANK_UPDATE_REPO", "explicit/fork")
    assert get_update_source({})["enabled"] is False
    assert get_update_source()["repo"] == "explicit/fork"


@pytest.mark.parametrize("value", [
    "", " ", "owner", "/repo", "owner/", "../repo", "owner/..",
    "owner/.", "-owner/repo", "owner-/repo", "ow--ner/repo",
    "owner/repo/extra", "https://github.com/owner/repo",
    "owner/repo?token=secret", "owner/repo#fragment",
    "owner/repo\n", " owner/repo", "owner/repo ", "用户/repo",
    "owner\\repo", "owner/%2e%2e", "a" * 40 + "/repo",
    "owner/" + "r" * 101, None, 123,
])
def test_invalid_source_never_exposes_input(value):
    source = get_update_source({"MATHBANK_UPDATE_REPO": value})
    assert source["enabled"] is False
    assert source["repo"] == source["release_url"] == ""
    assert source["reason"]
    expected = get_update_source({"MATHBANK_UPDATE_REPO": "invalid/secret?token=hidden"})
    if value:
        assert source["reason"] == expected["reason"]


@pytest.mark.parametrize("value", [
    "owner/repo", "A-1/repo_name.v2", "a/r", "a" * 39 + "/" + "r" * 100,
])
def test_explicit_valid_source(value):
    assert get_update_source({"MATHBANK_UPDATE_REPO": value}) == {
        "enabled": True, "repo": value,
        "release_url": f"https://github.com/{value}/releases/latest",
        "reason": "",
    }


@pytest.mark.parametrize("repo", ["", "explicit/fork", UPSTREAM_REPO])
def test_launcher_modern_identity_does_not_depend_on_release_repo(repo):
    source = (ROOT / "scripts/local_launcher.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "existing_service_ready")
    version = {"repo": repo, "project_instance_id": "a" * 64,
               "server_instance_id": "server"}
    health = {**version, "ready": True}
    namespace = {
        "read_json": lambda route: version if route == "/api/version" else health,
        "REPOSITORY": UPSTREAM_REPO, "ACTIVE_ROOT": ROOT,
        "project_instance_id": lambda root: "a" * 64,
        "UNVERIFIED_PORTS": set(), "PORT": 8000, "re": re,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<launcher>", "exec"), namespace)
    assert namespace["existing_service_ready"]() is True
    version["project_instance_id"] = "b" * 64
    assert namespace["existing_service_ready"]() is False


def test_actual_js_disabled_state_and_failed_preview():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend functions")
    source = (ROOT / "static/js/api.js").read_text(encoding="utf-8")
    update = source[source.index("        // Check if user is on Mac or Windows"):
                    source.index("        // Populate Categories in Editor Selects")]
    assert "JudgePeach/math-question-bank/releases" not in update
    assert "demoData" not in update
    harness = r"""
const assert = require('node:assert/strict');
global.window = global;
const nodes = new Map();
global.document = {getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {
        innerText: '', classList: {add(){}, remove(){}},
        querySelector() { return null; },
        removeAttribute(key) { delete this[key]; },
        setAttribute(key, value) { this[key] = value; }
    });
    return nodes.get(id);
}};
global.navigator = {userAgent: ''};
global.localStorage = {getItem(){ return null; }, setItem(){}, removeItem(){}};
global.showToast = () => {};
global.MathBankModal = {close(){}, open(){}};
let latestReleaseCache = null;
const IGNORED_UPDATE_KEY = 'ignored', LAST_CHECK_TIME_KEY = 'last';
"""
    checks = r"""
(async () => {
    global.fetch = async () => ({json: async () => ({
        update_enabled: false, update_reason: 'disabled', has_update: false
    })});
    await checkAppUpdateManually(true);
    assert.equal(nodes.get('aboutUpdateStatusText').innerText, 'disabled');
    assert.equal(nodes.get('aboutReleaseLink').href, undefined);
    assert.equal(nodes.get('btnViewOnGithub')['aria-disabled'], 'true');
    refreshReleaseLinks({update_enabled: true, release_url: 'https://github.com/explicit/fork/releases/latest'});
    assert.equal(nodes.get('aboutReleaseLink').href, 'https://github.com/explicit/fork/releases/latest');
    assert.equal(nodes.get('btnViewOnGithub')['aria-disabled'], undefined);
    refreshReleaseLinks({update_enabled: false, release_url: 'https://github.com/explicit/fork/releases/latest'});
    assert.equal(nodes.get('aboutReleaseLink').href, undefined);
    assert.equal(nodes.get('aboutReleaseLink').tabindex, '-1');
    refreshReleaseLinks({update_enabled: true, release_url: 'javascript:alert(1)'});
    assert.equal(nodes.get('aboutReleaseLink').href, undefined);
    openUpdateModal({update_enabled: false, has_update: true});
    assert.equal(nodes.has('updateModalTitle'), false);
    latestReleaseCache = null;
    global.fetch = async () => { throw new Error('offline'); };
    previewUpdateModalDemo();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(latestReleaseCache, null);
    assert.match(nodes.get('aboutUpdateStatusText').innerText, /下载不可用/);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", harness + update + checks],
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_html_release_links_are_inert_before_configuration():
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.items = []

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.items.append(dict(attrs))

    parser = Links()
    parser.feed((ROOT / 'static/index.html').read_text(encoding='utf-8'))
    for identifier in ('aboutReleaseLink', 'btnViewOnGithub'):
        links = [item for item in parser.items if item.get('id') == identifier]
        assert len(links) == 1
        assert 'href' not in links[0]
        assert links[0]['aria-disabled'] == 'true'
        assert links[0]['tabindex'] == '-1'
    assert not any('JudgePeach/math-question-bank/releases' in item.get('href', '') for item in parser.items)
