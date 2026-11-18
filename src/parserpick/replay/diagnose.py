"""Replay a capture through the engine order (reasoning split, then tool extraction) and explain leaks."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fingerprint import classify, families, find_markers
from ..issues import ERROR, INFO, WARNING, Issue
from .formats import EXTRACTORS, CallError, ToolCall, call_error
from .reasoning import split_reasoning

_COMPATIBLE = {frozenset({"deepseek_v3", "deepseek_v31"})}


@dataclass
class ReplayResult:
    format: str
    reasoning_rule: str
    reasoning: str
    content: str
    tool_calls: list[ToolCall]
    diagnostics: list[Issue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(d.severity == ERROR for d in self.diagnostics)


def _call_issue(error: CallError) -> Issue:
    """Diagnostic for a call that could not be extracted.

    params keep the English `detail` (language independent, for --json) plus `reason` and its
    values, from which the report builds the localized sentence (see report.issue_message).
    """
    params = {"detail": error.message, "position": error.position}
    if error.reason:
        params.update(reason=error.reason, **error.params)
    return Issue(error.code, ERROR, params=params)


def _same_group(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) in _COMPATIBLE


def replay(text: str, family: str, reasoning_rule: str = "none", engine: str = "vllm") -> ReplayResult:
    """Run the pipeline for `family` (a key of EXTRACTORS) and diagnose the outcome."""
    if family not in EXTRACTORS:
        raise ValueError(f"no replay rule for family {family!r}")
    reasoning, content = split_reasoning(text, reasoning_rule)
    extraction = EXTRACTORS[family](content)
    result = ReplayResult(family, reasoning_rule, reasoning, extraction.rest, extraction.calls)
    diagnostics = result.diagnostics

    for error in extraction.errors:
        diagnostics.append(_call_issue(error))

    leaked = classify(find_markers(reasoning, strong_only=True))[0] if reasoning else None
    if leaked:
        severity = WARNING if extraction.calls else ERROR
        diagnostics.append(Issue("call-inside-reasoning", severity, params={
            "detected": leaked, "reasoning_rule": reasoning_rule}))

    rest = extraction.rest
    if rest and not extraction.errors:
        detected = classify(find_markers(rest, strong_only=True))[0]
        spec = families()[family]
        opens = rest.count(spec["open"]) if spec.get("open") else 0
        closes = rest.count(spec["close"]) if spec.get("close") else 0
        if opens and spec.get("close") and closes < opens:
            # 고른 형식의 여는 표지가 닫히지 않았다 = 잘린 출력. 잘린 조각은 표지가 모자라 다른
            # 계열(예: <tool_call> 만 남은 qwen3_xml → hermes)로 보이므로 파서를 바꾸라고 하지 않는다.
            diagnostics.append(_call_issue(call_error(
                "unclosed-call", "unmatched-marker", open=spec["open"], close=spec["close"])))
        elif detected and not _same_group(detected, family):
            other = families()[detected]
            # 제안은 이 엔진에 문서화된 이름만. 다른 엔진 이름으로 대신 채우면(예: SGLang 에 granite)
            # 엔진이 모르는 값이라 기동을 거부하므로, 이름이 없으면 플래그 없이 "이 엔진에는 파서 없음"을 알린다.
            suggest = other.get(engine, "")
            diagnostics.append(Issue("parser-mismatch", ERROR,
                                     key="issue.parser-mismatch" if suggest else "issue.parser-mismatch.noname",
                                     params={
                "chosen": family, "detected": detected, "engine": engine, "suggest": suggest,
                "vllm": other.get("vllm", ""), "sglang": other.get("sglang", "")}))
        elif opens:
            diagnostics.append(_call_issue(call_error("broken-arguments", "unparsed-body")))

    if extraction.mixed_text and extraction.calls:
        diagnostics.append(Issue("text-mixed-pythonic", WARNING))

    if not result.tool_calls and not diagnostics:
        diagnostics.append(Issue("no-call", INFO))
    return result
