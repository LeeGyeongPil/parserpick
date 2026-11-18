"""Human-readable text and stable JSON for inspect, replay and check-template results."""

from __future__ import annotations

import json
from typing import Any

from .checks import CHECK_TIME_LIMIT, CheckResult
from .i18n import t
from .issues import Issue
from .recommend import Recommendation
from .replay.diagnose import ReplayResult

MAX_TEXT = 600


def localized_detail(reason: str, params: dict[str, Any], fallback: str, lang: str) -> str:
    """The sentence for a reason code (`detail.<reason>`), or `fallback` (an external message as is)."""
    return t(lang, f"detail.{reason}", **params) if reason else fallback


def format_label(family: str, lang: str) -> str:
    """Human description of a tool-call family, e.g. 'Kimi K2 section tokens' / 'Kimi K2 구역 토큰'."""
    return t(lang, f"format.{family}.label")


def issue_message(issue: Issue, lang: str) -> str:
    params = dict(issue.params)
    if params.get("family") and "label" in params:
        # params.label 은 --json 용 언어 중립 영어. 화면 문장에는 고른 언어의 설명을 넣는다.
        params["label"] = format_label(params["family"], lang)
    if issue.code == "parser-mismatch":
        # 엔진별 이름이 비어 있으면(문서에 없음) 빈칸 대신 "문서화된 이름 없음" 을 보여 준다. --json params 는 그대로 "".
        for engine in ("vllm", "sglang"):
            params[engine] = params.get(engine) or t(lang, "lbl.no_documented_name")
    if params.get("reason"):
        # params 의 detail 은 언어 중립 영어 원문(--json 용). 화면에는 이유 코드로 번역한 문장을 넣는다.
        params["detail"] = localized_detail(params["reason"], params, params.get("detail", ""), lang)
    return t(lang, issue.message_key, **params)


def issue_dict(issue: Issue, lang: str) -> dict[str, Any]:
    """`code` and `params` never depend on the UI language; only `message` does."""
    return {"code": issue.code, "severity": issue.severity, "message": issue_message(issue, lang),
            "params": issue.params}


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "..."


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix


def _is_mock(label: str) -> bool:
    return label.startswith("fixture:")


# --- inspect ------------------------------------------------------------------------------

def inspect_dict(rec: Recommendation, lang: str) -> dict[str, Any]:
    return {
        "source": rec.source,
        "mock": _is_mock(rec.source),
        "engine": rec.engine,
        "engine_version": rec.engine_version,
        "family": rec.family,
        "tool_parser": rec.tool_parser,
        "reasoning_parser": rec.reasoning_parser,
        "flags": rec.flags,
        "flags_line": rec.flags_line,
        "markers": rec.markers,
        "alternatives": rec.alternatives,
        "warnings": [issue_dict(w, lang) for w in rec.warnings],
        "notes": [issue_dict(n, lang) for n in rec.notes],
    }


def inspect_text(rec: Recommendation, lang: str) -> str:
    lines = [f"parserpick inspect: {rec.source}   ({t(lang, 'lbl.engine')}: {rec.engine}"
             + (f" {rec.engine_version}" if rec.engine_version else "") + ")"]
    if _is_mock(rec.source):
        lines.append(f"[{t(lang, 'mock.fixture_banner')}]")
    lines.append("")
    if rec.family:
        # 계열 이름(hermes 등)은 식별자라 그대로, 괄호 안 설명은 메시지 표에서 언어별로 가져온다
        lines.append(f"{t(lang, 'lbl.format')}: {rec.family}  ({format_label(rec.family, lang)})")
    else:
        lines.append(f"{t(lang, 'lbl.format')}: {t(lang, 'lbl.unknown')}")
    evidence = [f"{m['text']} [{t(lang, 'lbl.where.' + m['where'])}]"
                for m in rec.markers if m["id"] not in ("think_open", "think_close")]
    lines.append(f"{t(lang, 'lbl.evidence')}: " + (", ".join(evidence) if evidence else t(lang, "lbl.none")))
    lines.append(f"{t(lang, 'lbl.reasoning_parser')}: {rec.reasoning_parser or t(lang, 'lbl.none')}")
    if rec.flags:
        lines += ["", f"{t(lang, 'lbl.serve_with')}:", f"  {rec.flags_line}"]
    for title, items in (("lbl.warnings", rec.warnings), ("lbl.notes", rec.notes)):
        if items:
            lines += ["", f"{t(lang, title)}:"]
            for item in items:
                lines.append(f"  [{t(lang, 'sev.' + item.severity)}] {item.code}: {issue_message(item, lang)}")
    return "\n".join(lines)


# --- replay -------------------------------------------------------------------------------

def replay_dict(result: ReplayResult, lang: str) -> dict[str, Any]:
    return {
        "format": result.format,
        "reasoning_rule": result.reasoning_rule,
        "reasoning": result.reasoning,
        "content": result.content,
        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in result.tool_calls],
        "diagnostics": [issue_dict(d, lang) for d in result.diagnostics],
    }


def replay_text(result: ReplayResult, lang: str, engine: str = "vllm") -> str:
    lines = [f"parserpick replay   ({t(lang, 'lbl.pipeline')}: reasoning={result.reasoning_rule} -> "
             f"tool={result.format}, {t(lang, 'lbl.engine')}: {engine})", ""]
    lines.append(f"{t(lang, 'lbl.reasoning')}:")
    lines.append(_indent(_clip(result.reasoning)) if result.reasoning else f"    {t(lang, 'lbl.empty')}")
    lines.append(f"{t(lang, 'lbl.content')}:")
    lines.append(_indent(_clip(result.content)) if result.content else f"    {t(lang, 'lbl.empty')}")
    lines.append(f"{t(lang, 'lbl.tool_calls')}: {len(result.tool_calls)}")
    for call in result.tool_calls:
        lines.append(f"    - {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")
    if result.diagnostics:
        lines += ["", f"{t(lang, 'lbl.diagnostics')}:"]
        for diag in result.diagnostics:
            lines.append(f"  [{t(lang, 'sev.' + diag.severity)}] {diag.code}: {issue_message(diag, lang)}")
    lines += ["", t(lang, "replay.approx")]
    return "\n".join(lines)


# --- check-template -----------------------------------------------------------------------

def check_message(result: CheckResult, lang: str) -> str:
    if not result.key:
        return ""
    params: dict[str, Any] = {"seconds": f"{CHECK_TIME_LIMIT:g}", **result.params}
    detail = localized_detail(result.reason, params, result.detail, lang)
    return t(lang, result.key, **{**params, "error_type": result.error_type, "detail": detail})


def checks_dict(label: str, template_name: str, results: list[CheckResult], lang: str) -> dict[str, Any]:
    return {
        "source": label,
        "mock": _is_mock(label),
        "template": template_name,
        "checks": [{"id": r.id, "status": r.status, "error_type": r.error_type, "detail": r.detail,
                    "reason": r.reason, "message": check_message(r, lang)} for r in results],
    }


def checks_text(label: str, template_name: str, results: list[CheckResult], lang: str) -> str:
    lines = [f"parserpick check-template: {label}   ({t(lang, 'lbl.template')}: {template_name})"]
    if _is_mock(label):
        lines.append(f"[{t(lang, 'mock.fixture_banner')}]")
    lines.append("")
    for r in results:
        lines.append(f"  {r.status:<4}  {r.id:<17} {check_message(r, lang)}")
    return "\n".join(lines)
