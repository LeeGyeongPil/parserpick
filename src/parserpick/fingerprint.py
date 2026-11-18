"""Format registry (data/formats.toml) and marker fingerprinting of text."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import meta

DATA_FILE = Path(__file__).parent / "data" / "formats.toml"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Parse the format table once."""
    with DATA_FILE.open("rb") as handle:
        return tomllib.load(handle)


def families() -> dict[str, dict[str, Any]]:
    return load_registry()["families"]


def find_markers(text: str, *, strong_only: bool = False) -> list[str]:
    """Return ids of all registry markers present in `text` (registry order)."""
    found: list[str] = []
    for marker in load_registry()["markers"]:
        if strong_only and not marker.get("strong", False):
            continue
        if marker.get("regex"):
            hit = re.search(marker["text"], text) is not None
        else:
            hit = marker["text"] in text
        if hit:
            found.append(marker["id"])
    return found


def marker_text(marker_id: str) -> str:
    for marker in load_registry()["markers"]:
        if marker["id"] == marker_id:
            return marker["text"]
    raise KeyError(marker_id)


# Ordered rules: the first rule whose marker ids are all present wins.
# `absent` lists markers that must NOT be present for the rule to apply.
_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("openai", ("harmony_channel",), ()),
    ("minimax", ("minimax_tool_call",), ()),
    ("kimi_k2", ("kimi_section",), ()),
    ("deepseek_v3", ("deepseek_calls_begin", "deepseek_v3_function_sep"), ()),
    ("deepseek_v3", ("deepseek_calls_begin", "json_fence"), ()),  # V3/R1 wrap arguments in ```json
    ("deepseek_v31", ("deepseek_calls_begin",), ()),
    ("mistral", ("mistral_tool_calls",), ()),
    ("granite", ("granite_tool_call",), ()),
    ("glm45", ("tool_call_tag", "arg_key"), ()),
    ("qwen3_xml", ("tool_call_tag", "xml_function", "xml_parameter"), ()),
    ("hermes", ("tool_call_tag",), ()),
    ("pythonic", ("pythonic_list",), ("python_tag",)),
    ("llama3_json", ("python_tag",), ()),
    ("llama3_json", ("json_parameters",), ()),
]

# Families whose rules overlap enough that two of them present means "ambiguous".
_DISTINCT_STRONG = ("harmony_channel", "minimax_tool_call", "kimi_section", "deepseek_calls_begin",
                    "mistral_tool_calls", "granite_tool_call")


def classify(marker_ids: list[str]) -> tuple[str | None, list[str]]:
    """Map marker ids to (family, candidates). family is None when nothing matches.

    `candidates` lists other families that also match, in priority order; it is used
    to report ambiguity when unrelated strong markers appear together.
    """
    present = set(marker_ids)
    matches: list[str] = []
    for family, needed, absent in _RULES:
        if all(m in present for m in needed) and not any(m in present for m in absent):
            if family not in matches:
                matches.append(family)
    if not matches:
        return None, []
    return matches[0], matches[1:]


def is_ambiguous(marker_ids: list[str]) -> bool:
    """True when markers of two unrelated strong families are both present."""
    return sum(1 for m in _DISTINCT_STRONG if m in marker_ids) > 1


@dataclass
class Fingerprint:
    """What we learned from a template and the tokenizer's added tokens."""

    markers: list[str] = field(default_factory=list)  # all marker ids found
    template_markers: list[str] = field(default_factory=list)
    token_markers: list[str] = field(default_factory=list)
    uses_tools: bool = True
    think_capable: bool = False
    family: str | None = None
    alternatives: list[str] = field(default_factory=list)
    ambiguous: bool = False


def template_uses_tools(template: str) -> bool:
    """True when the Jinja template reads the `tools` variable (parsed, not a substring test)."""
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    try:
        ast = env.parse(template)
    except jinja2.TemplateSyntaxError:
        return "tools" in template
    return "tools" in meta.find_undeclared_variables(ast)


def fingerprint(templates: list[str], tokens: list[str], tools_template: str | None = None) -> Fingerprint:
    """Fingerprint all template texts plus added tokens.

    `tools_template` is the template used to decide whether `tools` is rendered
    (the tool-use one when the checkpoint ships several named templates).
    """
    joined = "\n".join(templates)
    from_template = find_markers(joined)
    from_tokens = find_markers("\n".join(tokens))
    markers = list(dict.fromkeys(from_template + from_tokens))
    family, alternatives = classify(markers)
    check_text = tools_template if tools_template is not None else (templates[0] if templates else "")
    return Fingerprint(
        markers=markers,
        template_markers=from_template,
        token_markers=[m for m in from_tokens if m not in from_template],
        uses_tools=template_uses_tools(check_text) if check_text else False,
        think_capable="think_open" in markers or "think_close" in markers,
        family=family,
        alternatives=alternatives,
        ambiguous=is_ambiguous(markers),
    )
