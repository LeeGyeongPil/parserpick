"""AC-6: sandboxed render checks."""

from parserpick.checks import FAIL, PASS, WARN, has_fail, run_checks
from parserpick.source import load_source


def _status(name):
    results = run_checks(load_source(f"fixture:{name}").tool_template)
    return {r.id: r for r in results}


def test_fixture_checks():
    mistral = _status("mistral")
    assert mistral["tool-call-id"].status == FAIL
    assert "length 9" in mistral["tool-call-id"].detail
    assert mistral["string-arguments"].status == WARN
    assert mistral["string-arguments"].key == "check.string_arguments.double_encoded"
    assert mistral["dict-arguments"].status == PASS

    deepseek = _status("deepseek-v31")
    assert deepseek["dict-arguments"].status == FAIL
    assert deepseek["dict-arguments"].error_type == "TypeError"
    assert deepseek["tools-rendered"].status == FAIL

    qwen3 = _status("qwen3")
    assert [r.status for r in qwen3.values()] == [PASS] * 5
    assert len(qwen3) == 5


def test_every_fixture_runs_without_crashing():
    from parserpick.source import fixture_names

    for name in fixture_names():
        results = run_checks(load_source(f"fixture:{name}").tool_template)
        assert len(results) == 5


UNSAFE = "{{ ''.__class__.__mro__[1].__subclasses__() }} tools: {{ tools }}"


def test_sandbox_blocks_unsafe_template():
    results = run_checks(UNSAFE)
    assert has_fail(results)
    first = results[0]
    assert first.status == FAIL
    assert first.error_type == "SecurityError"
    assert [r.status for r in results] == [FAIL] * 5


def test_unsafe_template_does_not_execute(tmp_path):
    marker = tmp_path / "pwned"
    template = ("{% for c in ''.__class__.__mro__[1].__subclasses__() %}"
                f"{{{{ c('{marker}', 'w').write('x') if c.__name__ == 'TextIOWrapper' else '' }}}}{{% endfor %}}")
    run_checks(template)
    assert not marker.exists()


def test_syntax_error_is_a_fail_not_a_crash():
    results = run_checks("{% if %}")
    assert results[0].status == FAIL and results[0].error_type == "TemplateSyntaxError"


def test_infinite_output_is_cut_off():
    results = run_checks("{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}")
    assert results[0].status == FAIL


def test_silent_loop_hits_the_time_limit():
    """Q-3: a loop that prints nothing must still stop at the render time limit."""
    import time

    from parserpick.render import render_template

    template = ("{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}"
                "<tool_call>{{ tools }}")
    start = time.monotonic()
    result = render_template(template, tools=[])
    assert time.monotonic() - start < 6.5
    assert not result.ok and result.error_type == "RenderLimit"


STRING_LOOP = ('{% set s = "x"*99999 %}{% for a in s %}{% for b in s %}{% endfor %}{% endfor %}'
               "<tool_call>{{ tools }}")


def test_silent_string_loop_is_killed_within_the_limit(monkeypatch):
    """QA round 2 Q-1: a silent loop over a long string never returns control to Jinja, so only
    the separate render process with a wall-clock limit can stop it. The killed process must
    not be left running or as a zombie."""
    import subprocess
    import time

    import parserpick.checks as checks

    started = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(checks.subprocess, "Popen", recording_popen)
    start = time.monotonic()
    results = run_checks(STRING_LOOP)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0 and elapsed < checks.CHECK_TIME_LIMIT + 2.0
    assert [r.status for r in results] == [FAIL] * 5
    assert {r.error_type for r in results} == {"RenderLimit"}
    assert results[0].key == "check.time_limit"
    assert [r.key for r in results[1:]] == ["check.not_run"] * 4

    run_checks(load_source("fixture:qwen3").tool_template)  # a normal run is reaped too
    assert len(started) == 2
    assert all(p.returncode is not None for p in started)


def test_silent_range_loop_shares_one_budget():
    """Round 1 Q-3 follow-up: five checks no longer take 5 s each; the whole run has one budget."""
    import time

    from parserpick.checks import CHECK_TIME_LIMIT

    template = ("{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}"
                "<tool_call>{{ tools }}")
    start = time.monotonic()
    results = run_checks(template)
    assert time.monotonic() - start < CHECK_TIME_LIMIT + 2.0
    assert [r.status for r in results] == [FAIL] * 5
    assert {r.error_type for r in results} == {"RenderLimit"}
    # QA round 3 Q-2c: same shape as the killed case - one "stopped", the rest "not run"
    assert results[0].key == "check.time_budget" and results[0].reason == "time-limit"
    assert [r.key for r in results[1:]] == ["check.not_run"] * 4
    assert {r.params["seconds"] for r in results} == {"5"}


def test_finished_checks_survive_a_later_kill():
    """Results the render process already sent are kept when a later check has to be killed."""
    # 문자열 인자(검사 3)에서만 끝없이 돈다: 검사 1·2 는 정상 결과가 남아야 한다.
    template = ('{{ tools }}{% for m in messages %}{% for c in m.tool_calls or [] %}'
                '{% if c.function.arguments is string %}{% set s = "x"*99999 %}'
                '{% for a in s %}{% for b in s %}{% endfor %}{% endfor %}{% endif %}'
                '{{ c.function.arguments }}{% endfor %}{% endfor %}')
    results = run_checks(template)
    assert [r.status for r in results[:2]] == [PASS, PASS]
    assert results[2].key == "check.time_limit"
    assert [r.key for r in results[3:]] == ["check.not_run"] * 2


def test_render_process_crash_is_a_fail_not_a_crash(monkeypatch):
    import parserpick.checks as checks

    monkeypatch.setattr(checks, "_CHILD_CODE", "import sys\nsys.stderr.write('boom\\n')\nraise SystemExit(5)\n")
    results = run_checks("{{ tools }}")
    assert [r.status for r in results] == [FAIL] * 5
    assert {r.error_type for r in results} == {"RenderProcessError"}
    assert results[0].detail == "boom"


def test_time_limit_messages_name_the_limit():
    from parserpick.checks import CHECK_TIME_LIMIT, CheckResult
    from parserpick.report import check_message

    limit = f"{CHECK_TIME_LIMIT:g}"
    for key in ("check.time_limit", "check.not_run"):
        result = CheckResult("tools-rendered", FAIL, "RenderLimit", "", key)
        assert limit in check_message(result, "en") and limit in check_message(result, "ko")
        assert "초" in check_message(result, "ko")
    budget = CheckResult("tools-rendered", FAIL, "RenderLimit", "", "check.time_budget", "time-limit",
                         {"seconds": "5"})
    assert "5-second" in check_message(budget, "en") and "5초" in check_message(budget, "ko")


def test_planted_module_in_current_folder_is_not_imported(tmp_path, monkeypatch):
    """The render process runs Python in isolated mode: a json.py in the model folder is ignored."""
    marker = tmp_path / "planted-ran"
    (tmp_path / "json.py").write_text(f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(3)\n")
    monkeypatch.chdir(tmp_path)
    results = run_checks(load_source("fixture:qwen3").tool_template)
    assert [r.status for r in results] == [PASS] * 5
    assert not marker.exists()


def test_huge_repeated_string_is_refused():
    from parserpick.render import render_template

    result = render_template("{{ 'x' * 100000000 }}")
    assert not result.ok and result.error_type == "RenderLimit"
    assert result.reason == "repeat-limit" and result.params == {"limit": 100_000}
    assert render_template("{{ 'ab' * 3 }}{% for i in range(3) %}{{ i }}{% endfor %}").text == "ababab012"


def test_whole_template_failures_do_not_blame_one_check():
    """QA round 3 Q-2: a template that cannot render at all gives FAIL 'render failed' for every
    check - never 'call IDs are rejected' or a string-arguments WARN with engine advice."""
    for template in ('{% set s = "x"*200000 %}{{ s }}',  # size limit everywhere
                     "{% macro r(n) %}{{ r(n+1) }}{% endmacro %}{{ r(0) }}<tool_call>{{ tools }}"):  # recursion
        results = run_checks(template)
        assert [r.status for r in results] == [FAIL] * 5
        assert {r.key for r in results} == {"check.render_failed"}, results


def test_input_specific_failures_keep_their_own_explanation():
    """Errors caused by one message shape still name that shape."""
    mistral = _status("mistral")
    assert mistral["tool-call-id"].error_type == "TemplateRaised"
    assert mistral["tool-call-id"].key == "check.tool_call_id.fail"
    # a template that breaks only on JSON-string arguments: WARN with the engine hint
    only_strings = "{{ tools }}{% for m in messages %}{% for c in m.tool_calls or [] %}{{ c.function.arguments.city }}{{ c.function.arguments.items() }}{% endfor %}{% endfor %}"
    results = {r.id: r for r in run_checks(only_strings)}
    assert results["string-arguments"].status == WARN
    assert results["string-arguments"].key == "check.string_arguments.error"
