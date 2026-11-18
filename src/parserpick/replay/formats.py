"""Tool-call extractors, one per output format.

Written from the documented output formats (clean room). They approximate what the serving
engines do; they are not byte-for-byte copies of any engine's parser.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..i18n import t


@dataclass
class ToolCall:
    name: str
    arguments: Any


@dataclass
class CallError:
    """Why a call could not be extracted.

    `message` is a language-independent English detail (kept in --json output). When `reason`
    is set, the CLI shows the localized sentence `detail.<reason>` filled with `params` instead;
    an empty `reason` means `message` is an external parser message (Python's ast) shown as is.
    """

    code: str  # "broken-arguments" | "unclosed-call"
    message: str
    position: int | None = None
    reason: str = ""
    params: dict[str, Any] = field(default_factory=dict)


def call_error(code: str, reason: str, position: int | None = None, **params: Any) -> CallError:
    """A CallError whose sentence lives in the message table under `detail.<reason>`.

    The English sentence is stored as the language-independent `message`; the CLI shows the
    sentence in the chosen language. No sentence is written in code.
    """
    return CallError(code, t("en", f"detail.{reason}", **params), position, reason, params)


class _BadPythonic(ValueError):
    """The pythonic call list has a shape we reject; `reason` names a `detail.<reason>` message."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Extraction:
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[CallError] = field(default_factory=list)
    rest: str = ""  # text that is not part of a recognized call
    mixed_text: bool = False  # pythonic: prose around the call list


def json_problem(text: str, exc: json.JSONDecodeError) -> CallError:
    """A JSON error at the very end of the text means the output was cut off."""
    truncated = exc.pos >= len(text.rstrip()) or exc.msg.startswith("Unterminated")
    code = "unclosed-call" if truncated else "broken-arguments"
    # exc.msg 는 파이썬 json 의 외부 메시지라 원문 그대로, 행·열 틀은 메시지 표에서 번역한다.
    return call_error(code, "json-error", exc.pos, json_error=exc.msg, line=exc.lineno, column=exc.colno,
                      char=exc.pos)


def _arguments(value: Any) -> Any:
    """Arguments may be an object or a JSON string; keep objects, decode strings when possible."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return {} if value is None else value


def _from_object(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    return ToolCall(obj["name"], _arguments(obj.get("arguments", obj.get("parameters"))))


def _coerce(value: str) -> Any:
    """XML-style formats carry strings; decode obvious JSON scalars/containers."""
    text = value.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?|true|false|null|\{.*\}|\[.*\]", text, re.S):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


# --- <tool_call> ... </tool_call> formats (hermes, qwen3_xml, glm45) ----------------------
_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"


def _split_calls(content: str) -> tuple[list[str], str]:
    """Split text into call bodies (between the markers) and the leftover text.

    Works in one pass over the text. A regex with a lazy `.*?` retried from every opening marker
    is quadratic when a looping output repeats the opening marker and never closes it. An opening
    marker without a closing marker stays in the leftover text.
    """
    parts = content.split(_CALL_CLOSE)
    bodies: list[str] = []
    rest: list[str] = []
    for part in parts[:-1]:
        start = part.find(_CALL_OPEN)
        if start < 0:
            rest.append(part + _CALL_CLOSE)
            continue
        rest.append(part[:start])
        bodies.append(part[start + len(_CALL_OPEN):])
    rest.append(parts[-1])
    return bodies, "".join(rest)


# --- hermes -------------------------------------------------------------------------------
_HERMES_BODY = re.compile(r"\s*(\{.*\})\s*", re.S)


def extract_hermes(content: str) -> Extraction:
    out = Extraction()
    bodies, rest = _split_calls(content)
    for raw in bodies:
        matched = _HERMES_BODY.fullmatch(raw)
        if not matched:
            rest += _CALL_OPEN + raw + _CALL_CLOSE
            continue
        body = matched.group(1)
        try:
            call = _from_object(json.loads(body))
        except json.JSONDecodeError as exc:
            out.errors.append(json_problem(body, exc))
            continue
        if call is None:
            out.errors.append(call_error("broken-arguments", "no-name"))
        else:
            out.calls.append(call)
    out.rest = rest.strip()
    return out


# --- qwen3_xml ----------------------------------------------------------------------------
_QWEN_BODY = re.compile(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", re.S)
_QWEN_PARAM_OPEN = re.compile(r"<parameter=([^>\s]+)>")


def _qwen_params(body: str) -> list[tuple[str, str]]:
    """Linear scan: split on the closing tag so unclosed repeats cannot cause quadratic regex work."""
    pairs = []
    for segment in body.split("</parameter>")[:-1]:
        opened = _QWEN_PARAM_OPEN.search(segment)
        if opened:
            value = segment[opened.end():]
            value = value[1:] if value.startswith("\n") else value
            pairs.append((opened.group(1), value[:-1] if value.endswith("\n") else value))
    return pairs


def extract_qwen3_xml(content: str) -> Extraction:
    out = Extraction()
    bodies, rest = _split_calls(content)
    for raw in bodies:
        matched = _QWEN_BODY.fullmatch(raw)
        if not matched:
            rest += _CALL_OPEN + raw + _CALL_CLOSE
            continue
        args = {key: _coerce(value) for key, value in _qwen_params(matched.group(2))}
        out.calls.append(ToolCall(matched.group(1), args))
    out.rest = rest.strip()
    return out


# --- glm45 --------------------------------------------------------------------------------
_GLM_BODY = re.compile(r"\s*([A-Za-z_][\w.\-]*)(?=[\s<])(.*)", re.S)


def _glm_args(body: str) -> list[tuple[str, str]]:
    """Linear scan over <arg_key>..</arg_key><arg_value>..</arg_value> pairs."""
    pairs = []
    for segment in body.split("</arg_value>")[:-1]:
        key_start = segment.find("<arg_key>")
        key_end = segment.find("</arg_key>", key_start + 1) if key_start >= 0 else -1
        value_start = segment.find("<arg_value>", key_end) if key_end >= 0 else -1
        if value_start >= 0:
            pairs.append((segment[key_start + 9:key_end], segment[value_start + 11:]))
    return pairs


def extract_glm45(content: str) -> Extraction:
    out = Extraction()
    bodies, rest = _split_calls(content)
    for raw in bodies:
        matched = _GLM_BODY.fullmatch(raw)
        if not matched:
            rest += _CALL_OPEN + raw + _CALL_CLOSE
            continue
        args = {key.strip(): _coerce(value) for key, value in _glm_args(matched.group(2))}
        out.calls.append(ToolCall(matched.group(1), args))
    out.rest = rest.strip()
    return out


# --- llama3_json --------------------------------------------------------------------------
PYTHON_TAG = "<|python_tag|>"


def extract_llama3_json(content: str) -> Extraction:
    out = Extraction(rest=content.strip())
    text = content.strip()
    if text.startswith(PYTHON_TAG):
        text = text[len(PYTHON_TAG):].lstrip()
    elif not text.startswith("{"):
        return out
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ";"):
            index += 1
        if index >= len(text) or text[index] != "{":
            break
        try:
            obj, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            out.errors.append(json_problem(text, exc))
            out.rest = ""
            return out
        call = _from_object(obj)
        if call is None:
            return Extraction(rest=content.strip())  # a JSON answer, not a call
        out.calls.append(call)
    out.rest = text[index:].strip()
    return out


# --- pythonic -----------------------------------------------------------------------------
_PY_START = re.compile(r"\[\s*[A-Za-z_][\w.]*\(")


def _list_end(text: str, start: int) -> int | None:
    """Index just after the bracket that closes the list opened at `start`, or None."""
    depth, quote, escaped = 0, "", False
    for pos in range(start, len(text)):
        char = text[pos]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
            if depth == 0:
                return pos + 1
    return None


def extract_pythonic(content: str) -> Extraction:
    match = _PY_START.search(content)
    if not match:
        return Extraction(rest=content.strip())
    end = _list_end(content, match.start())
    if end is None:
        return Extraction(rest="", errors=[call_error("unclosed-call", "list-unclosed", len(content.rstrip()))])
    snippet = content[match.start():end]
    out = Extraction()
    try:
        tree = ast.parse(snippet, mode="eval").body
        if not isinstance(tree, ast.List):
            raise _BadPythonic("not-a-list")
        for element in tree.elts:
            if not isinstance(element, ast.Call) or element.args or not isinstance(
                    element.func, (ast.Name, ast.Attribute)):
                raise _BadPythonic("bad-pythonic-item")
            name = ast.unparse(element.func)
            out.calls.append(ToolCall(name, {kw.arg: ast.literal_eval(kw.value) for kw in element.keywords
                                             if kw.arg}))
    except _BadPythonic as exc:
        out.calls.clear()
        out.errors.append(call_error("broken-arguments", exc.reason))
        return out
    except (SyntaxError, ValueError) as exc:
        # 파이썬 ast 가 낸 외부 메시지: 번역하지 않고 그대로 보여 준다.
        position = getattr(exc, "offset", None)
        out.calls.clear()
        out.errors.append(CallError("broken-arguments", f"{type(exc).__name__}: {exc}", position))
        return out
    out.rest = (content[:match.start()] + content[end:]).strip()
    out.mixed_text = bool(out.rest)
    return out


# --- mistral ------------------------------------------------------------------------------
MISTRAL_MARK = "[TOOL_CALLS]"


def extract_mistral(content: str) -> Extraction:
    head, marker, tail = content.partition(MISTRAL_MARK)
    out = Extraction(rest=content.strip())
    if not marker:
        return out
    out.rest = head.strip()
    body = tail.lstrip()
    try:
        data, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as exc:
        out.errors.append(json_problem(body, exc))
        return out
    items = data if isinstance(data, list) else [data]
    for item in items:
        call = _from_object(item)
        if call is None:
            out.errors.append(call_error("broken-arguments", "no-name"))
        else:
            out.calls.append(call)
    return out


# --- deepseek (v3 and v3.1) ---------------------------------------------------------------
_DS_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALLS_END = "<｜tool▁calls▁end｜>"
_DS_CALL = re.compile(r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>", re.S)
_DS_SEP = "<｜tool▁sep｜>"
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def extract_deepseek(content: str) -> Extraction:
    out = Extraction()
    for match in _DS_CALL.finditer(content):
        head, sep, tail = match.group(1).partition(_DS_SEP)
        if not sep:
            out.errors.append(call_error("broken-arguments", "missing-separator"))
            continue
        if head.strip() == "function":  # V3/R1 layout: function<sep>name\n```json\nargs\n```
            name, _, raw_args = tail.partition("\n")
        else:  # V3.1 layout: name<sep>args
            name, raw_args = head, tail
        raw_args = raw_args.strip()
        fenced = _FENCE.match(raw_args)
        if fenced:
            raw_args = fenced.group(1)
        try:
            out.calls.append(ToolCall(name.strip(), json.loads(raw_args)))
        except json.JSONDecodeError as exc:
            out.errors.append(json_problem(raw_args, exc))
    rest = _DS_CALL.sub("", content).replace(_DS_CALLS_END, "")
    # wrapper opened but its calls were all consumed: drop the wrapper, keep unclosed call text
    if _DS_CALLS_BEGIN in rest and "<｜tool▁call▁begin｜>" not in rest:
        rest = rest.replace(_DS_CALLS_BEGIN, "")
    out.rest = rest.strip()
    return out


# --- kimi_k2 ------------------------------------------------------------------------------
_KIMI = re.compile(r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>", re.S)
_KIMI_WRAP = ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>")


def extract_kimi_k2(content: str) -> Extraction:
    out = Extraction()
    for match in _KIMI.finditer(content):
        ident = match.group(1).strip()
        name = re.sub(r":\d+$", "", ident)
        name = name[len("functions."):] if name.startswith("functions.") else name
        raw_args = match.group(2).strip()
        try:
            out.calls.append(ToolCall(name, json.loads(raw_args)))
        except json.JSONDecodeError as exc:
            out.errors.append(json_problem(raw_args, exc))
    rest = _KIMI.sub("", content)
    if "<|tool_call_begin|>" not in rest:
        for wrapper in _KIMI_WRAP:
            rest = rest.replace(wrapper, "")
    out.rest = rest.strip()
    return out


EXTRACTORS: dict[str, Callable[[str], Extraction]] = {
    "hermes": extract_hermes,
    "qwen3_xml": extract_qwen3_xml,
    "glm45": extract_glm45,
    "llama3_json": extract_llama3_json,
    "pythonic": extract_pythonic,
    "mistral": extract_mistral,
    "deepseek_v3": extract_deepseek,
    "deepseek_v31": extract_deepseek,
    "kimi_k2": extract_kimi_k2,
}
