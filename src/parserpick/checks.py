"""check-template: five sandboxed render tests of the message shapes tool-calling clients send.

The five checks run in a separate Python process with one hard wall-clock limit for the whole
run (CHECK_TIME_LIMIT). A template that loops forever without printing anything cannot be
interrupted from inside Jinja, but a process can always be killed.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .i18n import t
from .render import MAX_SECONDS, Clock, RenderResult, render_template

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
CHECK_IDS = ("tools-rendered", "dict-arguments", "string-arguments", "tool-call-id", "tool-role")

# Hard limit for one check-template run, including the ~0.2 s it takes to start the render process.
# Inside the process all renders share a MAX_SECONDS (5 s) budget, so a range() loop stops there;
# anything that ignores that budget is killed at CHECK_TIME_LIMIT.
CHECK_TIME_LIMIT = MAX_SECONDS + 2.0

# Errors that say "this template cannot be rendered at all", whatever messages we feed it.
# A check that fails with one of these reports a plain render failure instead of blaming its own
# input (e.g. "call IDs are rejected" for a template that exceeds a size limit everywhere).
WHOLE_TEMPLATE_ERRORS = frozenset({
    "RenderLimit", "RecursionError", "MemoryError", "SecurityError", "TemplateSyntaxError",
    "RenderProcessError",
})

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
SHORT_ID = "abc123XYZ"  # 9 alphanumeric characters: accepted by templates that enforce an ID shape
OPENAI_ID = "call_" + "a1B2c3D4e5F6g7H8i9J0k1L2"  # what OpenAI-style clients send (29 characters)


@dataclass
class CheckResult:
    id: str
    status: str
    error_type: str = ""
    detail: str = ""  # exception text or a short note; language independent
    key: str = ""  # i18n key explaining the outcome
    reason: str = ""  # stable code for a limit/process outcome (message key `detail.<reason>`)
    params: dict[str, Any] = field(default_factory=dict)  # values for the localized messages


def _conversation(arguments: Any, call_id: str, with_tool_result: bool = False) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Weather in Seoul?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "get_weather", "arguments": copy.deepcopy(arguments)},
        }]},
    ]
    if with_tool_result:
        messages.append({"role": "tool", "tool_call_id": call_id, "name": "get_weather",
                         "content": json.dumps({"temp": 3})})
    return messages


def _failed(check_id: str, result: RenderResult, key: str, status: str = FAIL) -> CheckResult:
    """A failed render. Whole-template errors override the check's own explanation."""
    if result.reason == "time-limit":
        # 공유 시간 예산을 다 씀: 이 검사는 "멈춤", 뒤 검사는 iter_checks 가 "실행 안 함"으로 채운다.
        key, status = "check.time_budget", FAIL
    elif result.error_type in WHOLE_TEMPLATE_ERRORS:
        # 템플릿 전체가 렌더링되지 않는 오류: 검사별 원인(ID 거부 등)을 대지 않는다.
        key, status = "check.render_failed", FAIL
    return CheckResult(check_id, status, result.error_type, result.error, key, result.reason,
                       dict(result.params))


Render = Callable[[list[dict[str, Any]]], RenderResult]


# --- the five checks (each takes a render function and returns one result) -----------------

def _check_tools_rendered(render: Render) -> CheckResult:
    """1) Are the tool definitions visible to the model at all?"""
    rendered = render([{"role": "user", "content": "Weather in Seoul?"}])
    if not rendered.ok:
        return _failed("tools-rendered", rendered, "check.render_failed")
    if "get_weather" in rendered.text:
        return CheckResult("tools-rendered", PASS, key="check.tools_rendered.pass")
    return CheckResult("tools-rendered", FAIL, key="check.tools_rendered.fail")


def _check_dict_arguments(render: Render) -> CheckResult:
    """2) Past tool calls with arguments as a dict (how engines usually hand them over)."""
    rendered = render(_conversation({"city": "Seoul"}, SHORT_ID))
    if not rendered.ok:
        return _failed("dict-arguments", rendered, "check.render_failed")
    if "Seoul" not in rendered.text:
        return CheckResult("dict-arguments", WARN, key="check.arguments_dropped")
    return CheckResult("dict-arguments", PASS, key="check.dict_arguments.pass")


def _check_string_arguments(render: Render) -> CheckResult:
    """3) Arguments as a JSON string (the OpenAI wire format): rendered? double-encoded?"""
    rendered = render(_conversation(json.dumps({"city": "Seoul"}), SHORT_ID))
    if not rendered.ok:
        # 문자열 인자에서만 깨지면 WARN: 엔진이 인자를 dict 로 바꿔 넘길 수 있다.
        # 템플릿 전체가 깨지는 오류는 _failed 가 FAIL render_failed 로 바꾼다.
        return _failed("string-arguments", rendered, "check.string_arguments.error", WARN)
    if '{\\"' in rendered.text or '\\"city\\"' in rendered.text:
        return CheckResult("string-arguments", WARN, key="check.string_arguments.double_encoded")
    if "Seoul" not in rendered.text:
        return CheckResult("string-arguments", WARN, key="check.arguments_dropped")
    return CheckResult("string-arguments", PASS, key="check.string_arguments.pass")


def _check_tool_call_id(render: Render) -> CheckResult:
    """4) OpenAI-style call IDs (call_ + 24 characters).

    A dict-arguments failure is reported by check 2, so retry with string arguments first.
    """
    rendered = render(_conversation({"city": "Seoul"}, OPENAI_ID, with_tool_result=True))
    if not rendered.ok:
        rendered = render(_conversation(json.dumps({"city": "Seoul"}), OPENAI_ID, with_tool_result=True))
    if not rendered.ok:
        return _failed("tool-call-id", rendered, "check.tool_call_id.fail")
    return CheckResult("tool-call-id", PASS, key="check.tool_call_id.pass")


def _check_tool_role(render: Render) -> CheckResult:
    """5) A tool-role message after a call (retry with string arguments, like check 4)."""
    rendered = render(_conversation({"city": "Seoul"}, SHORT_ID, with_tool_result=True))
    if not rendered.ok:
        rendered = render(_conversation(json.dumps({"city": "Seoul"}), SHORT_ID, with_tool_result=True))
    if not rendered.ok:
        return _failed("tool-role", rendered, "check.render_failed")
    if "temp" not in rendered.text:
        return CheckResult("tool-role", FAIL, key="check.tool_role.fail")
    return CheckResult("tool-role", PASS, key="check.tool_role.pass")


_CHECKS: tuple[Callable[[Render], CheckResult], ...] = (
    _check_tools_rendered, _check_dict_arguments, _check_string_arguments, _check_tool_call_id,
    _check_tool_role,
)


def _limit_result(check_id: str, error_type: str, key: str, reason: str, **params: Any) -> CheckResult:
    """A FAIL whose detail is the message-table sentence `detail.<reason>` (English kept for --json)."""
    return CheckResult(check_id, FAIL, error_type, t("en", f"detail.{reason}", **params), key, reason, params)


def _not_run(check_id: str, seconds: float) -> CheckResult:
    return _limit_result(check_id, "RenderLimit", "check.not_run", "not-run", seconds=f"{seconds:g}")


def iter_checks(template: str, clock: Clock | None = None) -> Iterator[CheckResult]:
    """Run the five checks in this process, yielding each result as soon as it is known.

    All renders share one `clock`, so the time budget covers the whole run, not each check.
    Once the budget is used up, the remaining checks are reported as "not run" (the same shape
    as when the render process is killed). Used inside the render process; call `run_checks`
    for templates you do not trust.
    """
    clock = clock or Clock()

    def render(messages: list[dict[str, Any]]) -> RenderResult:
        return render_template(template, clock, messages=messages, tools=TOOLS)

    for index, check in enumerate(_CHECKS):
        result = check(render)
        yield result
        if result.reason == "time-limit":
            for later_id in CHECK_IDS[index + 1:]:
                yield _not_run(later_id, clock.seconds)
            return


# --- the render process -----------------------------------------------------------------------

_LINE_PREFIX = "PARSERPICK_CHECK "

# Code the child Python runs. `-I` (isolated mode) keeps the current folder, PYTHONPATH and user
# site-packages out of sys.path, so a json.py or jinja2.py planted in a downloaded model folder
# is never imported. The parent's import path is passed in explicitly instead.
_CHILD_CODE = (
    "import json, sys\n"
    "job = json.loads(sys.stdin.buffer.read())\n"
    "sys.path[:0] = job['path']\n"
    "from parserpick.checks import serve_job\n"
    "serve_job(job)\n"
)


def serve_job(job: dict[str, Any]) -> None:
    """Child side: run the checks and print one JSON line per finished check, flushing each.

    Flushing line by line means the parent keeps every finished result even if it has to kill
    this process in the middle of a later check.
    """
    for result in iter_checks(job["template"], Clock(job["seconds"])):
        sys.stdout.write(_LINE_PREFIX + json.dumps(asdict(result)) + "\n")
        sys.stdout.flush()


def _import_path() -> list[str]:
    """Absolute sys.path entries the child needs to import parserpick and jinja2 (never the cwd)."""
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = ""
    package_root = str(Path(__file__).resolve().parent.parent)  # the folder that contains parserpick/
    paths = [package_root]
    for entry in sys.path:
        if entry and os.path.isabs(entry) and entry != cwd and entry not in paths:
            paths.append(entry)
    return paths


def run_checks(template: str) -> list[CheckResult]:
    """Run all five checks in a separate process, stopped after CHECK_TIME_LIMIT seconds in total."""
    job = json.dumps({"path": _import_path(), "template": template, "seconds": MAX_SECONDS}).encode()
    try:
        process = subprocess.Popen([sys.executable, "-I", "-c", _CHILD_CODE],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return [CheckResult(c, FAIL, "RenderProcessError", str(exc), "check.render_failed") for c in CHECK_IDS]

    timed_out = False
    try:
        try:
            out, err = process.communicate(job, timeout=CHECK_TIME_LIMIT)
        except subprocess.TimeoutExpired:
            # 벽시계 제한: 출력 없이 도는 루프도 여기서 끝난다. kill 뒤 communicate 로 남은 출력과
            # 종료 상태를 거둬 좀비 프로세스를 남기지 않는다.
            timed_out = True
            process.kill()
            out, err = process.communicate()
    finally:
        if process.poll() is None:  # Ctrl-C or another error while waiting: never leave it running
            process.kill()
            process.wait()

    # 끝난 검사는 자식이 한 줄씩 보냈으니 그대로 살린다.
    results: list[CheckResult] = []
    for line in out.decode("utf-8", "replace").splitlines():
        if line.startswith(_LINE_PREFIX) and len(results) < len(CHECK_IDS):
            results.append(CheckResult(**json.loads(line[len(_LINE_PREFIX):])))

    # 끝나지 못한 검사: 돌던 하나는 "멈춤", 그 뒤는 "실행 안 함"(시간 예산을 이미 다 씀).
    running = len(results)
    stderr_tail = (err.decode("utf-8", "replace").strip().splitlines() or [""])[-1]
    for index in range(running, len(CHECK_IDS)):
        check_id = CHECK_IDS[index]
        if timed_out and index == running:
            results.append(_limit_result(check_id, "RenderLimit", "check.time_limit", "killed",
                                         seconds=f"{CHECK_TIME_LIMIT:g}"))
        elif timed_out:
            results.append(_not_run(check_id, CHECK_TIME_LIMIT))
        elif stderr_tail:
            # 자식 파이썬이 남긴 마지막 줄(외부 메시지)을 그대로 보여 준다.
            results.append(CheckResult(check_id, FAIL, "RenderProcessError", stderr_tail, "check.render_failed"))
        else:
            results.append(_limit_result(check_id, "RenderProcessError", "check.render_failed", "process-exit",
                                         code=process.returncode))
    return results


def has_fail(results: list[CheckResult]) -> bool:
    return any(r.status == FAIL for r in results)
