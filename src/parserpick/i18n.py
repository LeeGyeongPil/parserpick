"""UI language selection and message lookup.

Precedence: --lang > PARSERPICK_LANG > LC_ALL > LC_MESSAGES > LANG > English.
Only "ko" and "en" exist; any value starting with "ko" selects Korean, anything else English.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

SUPPORTED = ("en", "ko")
DEFAULT = "en"
_LOCALES = Path(__file__).parent / "locales"


def normalize(value: str | None) -> str | None:
    """Map a raw value such as "ko_KR.UTF-8" to "ko"/"en"; None when the value is empty/unset."""
    if not value:
        return None
    lowered = value.strip().lower()
    if not lowered or lowered in ("c", "posix"):
        return None
    return "ko" if lowered.startswith("ko") else "en"


def resolve_lang(cli_value: str | None, environ: Mapping[str, str]) -> str:
    """Pick the UI language following the documented precedence."""
    for candidate in (
        cli_value,
        environ.get("PARSERPICK_LANG"),
        environ.get("LC_ALL"),
        environ.get("LC_MESSAGES"),
        environ.get("LANG"),
    ):
        chosen = normalize(candidate)
        if chosen:
            return chosen
    return DEFAULT


@lru_cache(maxsize=None)
def catalog(lang: str) -> dict[str, str]:
    """Load the message table for a language (cached)."""
    path = _LOCALES / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(lang: str, key: str, **params: object) -> str:
    """Translate `key`; missing keys fall back to English, then to the key itself."""
    table = catalog(lang)
    text = table.get(key)
    if text is None:
        text = catalog(DEFAULT).get(key, key)
    try:
        return text.format(**params) if params else text
    except (KeyError, IndexError):
        return text
