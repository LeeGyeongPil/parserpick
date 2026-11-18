"""Sandboxed chat-template rendering with Hugging Face compatible globals.

Templates come from model repos we do not trust. This module gives two in-process guards:
the immutable Jinja sandbox (no attribute tricks, no file access) and cheap limits (output size,
huge `*`/`**`, a deadline checked by range() and between output chunks).

The in-process deadline cannot stop every endless loop: a loop over a long string that prints
nothing never returns control to us. The hard wall-clock limit therefore lives one level up:
`checks.run_checks` renders in a separate Python process and kills it when time is up.
Call `run_checks` (not `render_template`) for templates you do not trust.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import jinja2  # noqa: F401  (imported so a broken install fails here, not mid-render)
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .i18n import t

MAX_OUTPUT_CHARS = 1_000_000
MAX_SECONDS = 5.0  # in-process deadline (shared by all renders of one check-template run)
MAX_SEQUENCE = 100_000  # largest string/list a template may build with * or **
MAX_EXPONENT = 1000  # largest integer exponent allowed with **


class TemplateRaised(Exception):
    """The template called raise_exception(...)."""


class RenderLimit(Exception):
    """Rendering exceeded a limit.

    `reason` is a stable code (time-limit, repeat-limit, exponent-limit, lipsum-unavailable,
    output-limit) and `params` its values. The sentence lives in the message table under
    `detail.<reason>`: the CLI shows it in the chosen language, and the English one fills
    `str(exc)`, the language-independent `detail` field of --json output.
    """

    def __init__(self, reason: str, **params: Any) -> None:
        self.reason = reason
        self.params = params
        super().__init__(t("en", f"detail.{reason}", **params))


@dataclass
class RenderResult:
    ok: bool
    text: str = ""
    error_type: str = ""
    error: str = ""  # exception text (language independent)
    reason: str = ""  # RenderLimit reason code; empty for other errors
    params: dict[str, Any] = field(default_factory=dict)


class Clock:
    """A wall-clock deadline. One clock can be shared by several renders (one overall budget)."""

    def __init__(self, seconds: float = MAX_SECONDS) -> None:
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise RenderLimit("time-limit", seconds=f"{self.seconds:g}")


class _LimitedRange:
    """range() replacement that checks the deadline on creation and on every item.

    Loops that print nothing never reach the output check, so range() checks the clock itself.
    (Loops over strings or lists are not covered here; the separate render process covers them.)
    """

    def __init__(self, clock: Clock, *args: int) -> None:
        clock.check()
        self._clock = clock
        self._range = range(*args)

    def __iter__(self):  # noqa: ANN204
        for value in self._range:
            self._clock.check()
            yield value

    def __reversed__(self):  # noqa: ANN204
        for value in reversed(self._range):
            self._clock.check()
            yield value

    def __len__(self) -> int:
        return len(self._range)

    def __getitem__(self, index: Any) -> Any:
        return self._range[index]


class _Environment(ImmutableSandboxedEnvironment):
    """Immutable sandbox that also refuses huge repeated strings/lists and huge powers."""

    intercepted_binops = frozenset({"*", "**"})

    def call_binop(self, context: Any, operator: str, left: Any, right: Any) -> Any:
        if operator == "*":
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (str, list, tuple)) and isinstance(count, int) and len(seq) * count > MAX_SEQUENCE:
                    raise RenderLimit("repeat-limit", limit=MAX_SEQUENCE)
        if operator == "**" and isinstance(right, int) and isinstance(left, int) and abs(right) > MAX_EXPONENT:
            raise RenderLimit("exponent-limit", limit=MAX_EXPONENT)
        return super().call_binop(context, operator, left, right)


def make_environment(clock: Clock | None = None) -> ImmutableSandboxedEnvironment:
    clock = clock or Clock()
    env = _Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.globals["range"] = lambda *args: _LimitedRange(clock, *args)

    def lipsum(*_args: Any, **_kwargs: Any) -> str:
        raise RenderLimit("lipsum-unavailable")

    env.globals["lipsum"] = lipsum

    def tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
               separators: Any = None, sort_keys: bool = False) -> str:
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message: str) -> None:
        raise TemplateRaised(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "19 Nov 2026"
    return env


def render_template(template: str, clock: Clock | None = None, **variables: Any) -> RenderResult:
    """Render in this process with the sandbox. Never raises; failures come back as ok=False.

    Pass a shared `clock` to give several renders one overall time budget; without one, this
    render gets its own MAX_SECONDS budget starting after the template is compiled.
    """
    own_clock = clock is None
    clock = clock or Clock()
    env = make_environment(clock)
    variables.setdefault("bos_token", "<s>")
    variables.setdefault("eos_token", "</s>")
    variables.setdefault("add_generation_prompt", True)
    try:
        clock.check()  # a shared budget may already be used up by an earlier render
        compiled = env.from_string(template)
        if own_clock:
            clock.deadline = time.monotonic() + clock.seconds
        parts: list[str] = []
        size = 0
        for chunk in compiled.generate(**variables):
            parts.append(chunk)
            size += len(chunk)
            if size > MAX_OUTPUT_CHARS:
                raise RenderLimit("output-limit", limit=MAX_OUTPUT_CHARS)
            clock.check()
        return RenderResult(True, "".join(parts))
    except RenderLimit as exc:
        # 이유 코드와 값을 함께 돌려줘 화면에서 언어별 문장으로 바꿀 수 있게 한다.
        return RenderResult(False, error_type="RenderLimit", error=str(exc), reason=exc.reason,
                            params=dict(exc.params))
    except Exception as exc:  # noqa: BLE001 - every template failure is a result, not a crash
        return RenderResult(False, error_type=type(exc).__name__, error=str(exc))
