"""Resolve an explicitly configured update source without network or Git access."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping


def get_update_source(environ: Mapping[str, str] | None = None) -> dict:
    """Return safe public metadata; never include an invalid configuration value."""
    value = (os.environ if environ is None else environ).get("MATHBANK_UPDATE_REPO", "")
    if not value:
        return {
            "enabled": False, "repo": "", "release_url": "",
            "reason": "未配置更新源（MATHBANK_UPDATE_REPO）。",
        }
    # GitHub owner: 1–39 ASCII characters, no edge or consecutive hyphens.
    # Repository: 1–100 ASCII characters; reject path segments and URL syntax.
    if (not isinstance(value, str)
            or re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9_.-]{1,100}",
                value,
            ) is None
            or "--" in value.split("/")[0]
            or value.split("/")[-1] in {".", ".."}):
        return {
            "enabled": False, "repo": "", "release_url": "",
            "reason": "更新源配置无效，MATHBANK_UPDATE_REPO 必须为合法的 owner/repo。",
        }
    return {
        "enabled": True, "repo": value,
        "release_url": f"https://github.com/{value}/releases/latest",
        "reason": "",
    }
