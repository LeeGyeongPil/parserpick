"""Mock-mode demo: runs the real code paths on built-in synthetic fixtures, offline.

Every step calls the same functions the real commands use (load_source, recommend, replay,
run_checks). Nothing here prints canned output.
"""

from __future__ import annotations

from typing import Any

from . import report
from .capture import load_capture
from .checks import has_fail, run_checks
from .i18n import t
from .recommend import recommend
from .replay import replay, resolve_format, settings_from_recommendation
from .source import load_source


def run_demo(lang: str, as_json: bool = False) -> int:
    """Run all demo steps. Always returns 0: intended failures are labelled as expected."""
    steps: list[dict[str, Any]] = []
    total = 9

    def show(title_key: str, command: str, text: str, data: dict[str, Any], expected_failure: bool) -> None:
        number = len(steps) + 1
        steps.append({"step": number, "command": command, "expected_failure": expected_failure, "result": data})
        if as_json:
            return
        print(f"=== {t(lang, 'demo.step', n=number, total=total, title=t(lang, title_key))}")
        print(f"$ {command}")
        print(text)
        if expected_failure:
            print(t(lang, "demo.expected"))
        print()

    if not as_json:
        print(t(lang, "demo.title"))
        print()

    for title, name, engine in (
        ("demo.cmd.inspect_coder_vllm", "qwen3-coder", "vllm"),
        ("demo.cmd.inspect_coder_sglang", "qwen3-coder", "sglang"),
        ("demo.cmd.inspect_deepseek", "deepseek-v31", "vllm"),
    ):
        rec = recommend(load_source(f"fixture:{name}"), engine)
        show(title, f"parserpick inspect fixture:{name} --engine {engine}",
             report.inspect_text(rec, lang), report.inspect_dict(rec, lang), False)

    qwen3 = recommend(load_source("fixture:qwen3"))
    qwen3_family, qwen3_rule = settings_from_recommendation(qwen3)
    replays = (
        ("demo.cmd.replay_a", "a_think_then_call", "--model fixture:qwen3", qwen3_family, qwen3_rule, False),
        ("demo.cmd.replay_b", "b_call_in_think", "--model fixture:qwen3", qwen3_family, qwen3_rule, True),
        ("demo.cmd.replay_c", "c_qwen3_xml", "--format hermes", resolve_format("hermes"), "none", True),
        ("demo.cmd.replay_d", "c_qwen3_xml", "--format qwen3_xml", resolve_format("qwen3_xml"), "none", False),
        ("demo.cmd.replay_e", "e_broken_json", "--format hermes", resolve_format("hermes"), "none", True),
    )
    for title, capture, options, family, rule, expected in replays:
        result = replay(load_capture(f"fixture:{capture}"), family, rule)
        show(title, f"parserpick replay fixture:{capture} {options}",
             report.replay_text(result, lang), report.replay_dict(result, lang), expected)

    source = load_source("fixture:mistral")
    results = run_checks(source.tool_template)
    show("demo.cmd.check_mistral", "parserpick check-template fixture:mistral",
         report.checks_text(source.label, source.tool_template_name, results, lang),
         report.checks_dict(source.label, source.tool_template_name, results, lang), has_fail(results))

    if as_json:
        print(report.dumps({"mock": True, "steps": steps}))
    else:
        print(t(lang, "demo.done"))
    return 0
