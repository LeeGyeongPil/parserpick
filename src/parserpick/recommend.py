"""Turn a fingerprint into engine-specific serving flags plus warnings."""

from __future__ import annotations

from dataclasses import dataclass, field

import jinja2

from .render import make_environment
from .fingerprint import Fingerprint, families, fingerprint, load_registry, marker_text
from .i18n import t
from .issues import ERROR, INFO, WARNING, Issue
from .source import ModelSource

ENGINES = ("vllm", "sglang")


@dataclass
class Recommendation:
    source: str
    engine: str
    engine_version: str | None
    family: str | None
    reasoning_parser: str | None
    tool_parser: str | None
    flags: list[str]
    markers: list[dict[str, str]]
    warnings: list[Issue] = field(default_factory=list)
    notes: list[Issue] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    fingerprint: Fingerprint | None = None

    @property
    def flags_line(self) -> str:
        return " ".join(self.flags)

    @property
    def has_error(self) -> bool:
        return any(w.severity == ERROR for w in self.warnings)


def engine_name(family: str, engine: str) -> str:
    """Tool-call parser name for the engine; empty string when the docs list none."""
    return families()[family].get(engine, "")


def tool_call_flags(engine: str, tool_name: str) -> list[str]:
    """Tool-call flags in a combination the engine accepts at startup (vLLM 0.34.0, SGLang 0.5.23).

    vLLM loads the parser named by --tool-call-parser only when --enable-auto-tool-choice is set,
    and `vllm serve` exits with "--enable-auto-tool-choice requires --tool-call-parser" when the
    switch comes without a parser. So the two go together, or neither is printed.
    SGLang has no --enable-auto-tool-choice; --tool-call-parser alone turns parsing on.
    """
    if not tool_name:
        # 엔진 문서에 파서 이름이 없으면 도구 호출 플래그를 하나도 내지 않는다(붙여 넣으면 기동 실패하는 줄 방지).
        return []
    if engine == "vllm":
        return ["--enable-auto-tool-choice", "--tool-call-parser", tool_name]
    return ["--tool-call-parser", tool_name]


def resolve_family(name: str) -> str | None:
    """Map any known parser name or alias (vLLM, SGLang, old names) to a family id."""
    lowered = name.strip().lower()
    for family, spec in families().items():
        names = {family, spec.get("vllm", ""), spec.get("sglang", ""), *spec.get("aliases", [])}
        if lowered in {n.lower() for n in names if n}:
            return family
    return None


def _minor(version: str) -> str:
    return ".".join(version.strip().lstrip("v").split(".")[:2])


def _marker_rows(fp: Fingerprint) -> list[dict[str, str]]:
    rows = []
    for marker_id in fp.markers:
        where = "template" if marker_id in fp.template_markers else "tokens"
        rows.append({"id": marker_id, "text": marker_text(marker_id), "where": where})
    return rows


def recommend(source: ModelSource, engine: str = "vllm", engine_version: str | None = None) -> Recommendation:
    """Judge the format from template content and special tokens (never from the model name)."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    registry = load_registry()
    fp = fingerprint(list(source.templates.values()), source.added_tokens, tools_template=source.tool_template)
    rec = Recommendation(
        source=source.label, engine=engine, engine_version=engine_version, family=fp.family,
        reasoning_parser=None, tool_parser=None, flags=[], markers=_marker_rows(fp),
        alternatives=fp.alternatives, fingerprint=fp,
    )

    for name, text in source.templates.items():
        try:
            make_environment().parse(text)
        except jinja2.TemplateSyntaxError as exc:
            # exc.message 는 Jinja 의 외부 메시지, "(N행)" 틀은 메시지 표 detail.jinja-syntax 에서 번역한다.
            where = {"jinja_error": exc.message, "line": exc.lineno}
            rec.warnings.append(Issue("template-syntax-error", WARNING, params={
                "name": name, "detail": t("en", "detail.jinja-syntax", **where), "reason": "jinja-syntax",
                **where}))
            break

    if engine_version and _minor(engine_version) != _minor(registry["meta"]["engines"][engine]):
        rec.warnings.append(Issue("unverified-engine-version", WARNING, params={
            "engine": engine, "version": engine_version, "verified": registry["meta"]["engines"][engine],
            "url": registry["meta"]["docs"][engine],
        }))

    if fp.family is None:
        rec.warnings.append(Issue("unknown-format", ERROR))
        return rec
    spec = families()[fp.family]

    if fp.ambiguous:
        rec.warnings.append(Issue("ambiguous", WARNING, params={"markers": ", ".join(fp.markers)}))

    template_hint = spec.get(f"chat_template_{engine}", "")

    use_hf = source.has_response_template and engine == "vllm"
    if source.has_response_template:
        rec.warnings.append(Issue("response-template-present", INFO,
                                  key="issue.response-template-present" if engine == "vllm"
                                  else "issue.response-template-present.sglang"))
    tool_name = "hf" if use_hf else engine_name(fp.family, engine)
    # 도구 호출 플래그는 파서 이름이 정해진 뒤에 한 곳(tool_call_flags)에서만 만든다.
    flags: list[str] = tool_call_flags(engine, tool_name)
    if tool_name:
        rec.tool_parser = tool_name
    else:
        # vLLM 문장은 --enable-auto-tool-choice 까지 뺀 이유와 그 결과(도구 요청 거부)를 따로 설명한다.
        rec.warnings.append(Issue("no-engine-parser", WARNING,
                                  key="issue.no-engine-parser.vllm" if engine == "vllm" else "issue.no-engine-parser",
                                  params={
            # label 은 --json 에 남는 언어 중립 영어 설명. 화면 문장은 report.issue_message 가 언어별로 바꾼다.
            "engine": engine, "family": fp.family, "label": t("en", f"format.{fp.family}.label")}))

    reasoning = ""
    if use_hf:
        reasoning = "hf"
    elif spec.get("reasoning_always") or (fp.think_capable and spec.get("reasoning_rule")):
        reasoning = spec.get(f"reasoning_{engine}", "")
        if not reasoning:
            rec.warnings.append(Issue("no-engine-parser", WARNING, key="issue.no-engine-reasoning", params={
                "engine": engine, "family": fp.family}))
    if reasoning:
        flags += ["--reasoning-parser", reasoning]
        rec.reasoning_parser = reasoning

    if not fp.uses_tools:
        rec.warnings.append(Issue(
            "tools-not-rendered", WARNING,
            key="issue.tools-not-rendered" if template_hint else "issue.tools-not-rendered.nohint",
            params={"hint": template_hint},
        ))
        if template_hint:
            flags += ["--chat-template", template_hint]
    elif source.named and "tool_use" in source.templates:
        # 이름 붙은 템플릿은 --chat-template 으로 고르지 않는다: 두 엔진 모두 그 값을 파일 경로로 읽어
        # `tool_use` 같은 이름을 주면 기동을 거부한다(vLLM validate_chat_template, SGLang template_manager).
        if engine == "sglang":
            # SGLang 은 이름 목록의 첫 템플릿을 쓰므로 도구용 템플릿을 이름으로 지정해야 한다.
            rec.warnings.append(Issue("named-template", WARNING, key="issue.named-template.sglang",
                                      params={"name": "tool_use"}))
            flags += ["--hf-chat-template-name", "tool_use"]
        else:
            # vLLM 은 도구가 있는 요청에서 transformers 규칙대로 tool_use 템플릿을 스스로 고른다(플래그 불필요).
            rec.warnings.append(Issue("named-template", INFO, params={"name": "tool_use"}))
    rec.flags = flags

    old = spec.get("old_names", {})
    if engine == "vllm" and old.get("vllm") and not use_hf:
        rec.notes.append(Issue("old-name", INFO, params={
            "old": old["vllm"], "new": tool_name, "url": old.get("vllm_source", "")}))
    if spec.get("note_key"):
        rec.notes.append(Issue("reasoning-name-unlisted", INFO, key=spec["note_key"],
                               params={"name": spec.get("reasoning_vllm", "")}))
    # Always keep the same-model alias from the other engine visible for people porting configs.
    other = "sglang" if engine == "vllm" else "vllm"
    other_name = engine_name(fp.family, other)
    if other_name and other_name != tool_name:
        # 다른 엔진에 그대로 붙여 넣을 수 있게 그 엔진의 조합 규칙(vLLM 은 --enable-auto-tool-choice 포함)으로 보여 준다.
        rec.notes.append(Issue("other-engine-name", INFO, params={
            "engine": other, "name": other_name, "flags": " ".join(tool_call_flags(other, other_name))}))
    return rec
