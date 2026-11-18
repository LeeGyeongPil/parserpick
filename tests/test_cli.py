"""AC-7, AC-9: JSON schema, exit codes, demo."""

import json
import re
import subprocess
import sys
import time

import pytest

from parserpick.cli import main


def _json(capsys, argv):
    code = main([*argv, "--json"], environ={})
    return code, json.loads(capsys.readouterr().out)


def test_json_schema(capsys):
    code, data = _json(capsys, ["inspect", "fixture:qwen3-coder"])
    assert code == 0
    assert {"family", "engine", "flags", "markers", "warnings"} <= set(data)
    assert data["family"] == "qwen3_xml" and data["engine"] == "vllm"
    assert "--tool-call-parser" in data["flags"] and "qwen3_xml" in data["flags"]
    assert all("code" in w for w in data["warnings"])
    assert {m["id"] for m in data["markers"]} >= {"tool_call_tag", "xml_function", "xml_parameter"}
    assert data["mock"] is True

    code, data = _json(capsys, ["replay", "fixture:b_call_in_think", "--model", "fixture:qwen3"])
    assert code == 1
    assert {"reasoning", "content", "tool_calls", "diagnostics"} <= set(data)
    assert data["tool_calls"] == [] and data["diagnostics"][0]["code"] == "call-inside-reasoning"

    code, data = _json(capsys, ["replay", "fixture:a_think_then_call", "--model", "fixture:qwen3"])
    assert code == 0 and data["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Seoul"}}]

    code, data = _json(capsys, ["check-template", "fixture:mistral"])
    assert code == 1
    assert [c["id"] for c in data["checks"]] == [
        "tools-rendered", "dict-arguments", "string-arguments", "tool-call-id", "tool-role"]
    assert {c["status"] for c in data["checks"]} <= {"PASS", "WARN", "FAIL"}


def test_codes_do_not_depend_on_language(capsys):
    cases = [["inspect", "fixture:deepseek-v31"], ["replay", "fixture:c_qwen3_xml", "--format", "hermes"],
             ["check-template", "fixture:mistral"]]
    for argv in cases:
        _, en = _json(capsys, [*argv, "--lang", "en"])
        _, ko = _json(capsys, [*argv, "--lang", "ko"])
        key = "checks" if "checks" in en else ("diagnostics" if "diagnostics" in en else "warnings")
        pick = (lambda d: [(c["id"], c["status"]) for c in d["checks"]]) if key == "checks" else (
            lambda d: [c["code"] for c in d[key]])
        assert pick(en) == pick(ko)
        assert en != ko  # messages differ


def test_missing_file_exit_2(capsys, tmp_path):
    assert main(["replay", str(tmp_path / "nope.txt"), "--format", "hermes"], environ={}) == 2
    assert "nope.txt" in capsys.readouterr().err
    assert main(["inspect", str(tmp_path / "no-such-dir")], environ={}) == 2
    assert main(["check-template", str(tmp_path)], environ={}) == 2  # a folder without a template


def test_usage_errors_exit_2(capsys):
    assert main(["replay", "fixture:c_qwen3_xml"], environ={}) == 2  # neither --format nor --model
    assert main(["replay", "fixture:c_qwen3_xml", "--format", "nonsense"], environ={}) == 2
    assert main(["replay", "fixture:c_qwen3_xml", "--format", "granite"], environ={}) == 2  # no rule yet
    assert main(["replay", "fixture:c_qwen3_xml", "--format", "hermes", "--reasoning", "x"], environ={}) == 2
    with pytest.raises(SystemExit) as exc:
        main(["inspect"], environ={})
    assert exc.value.code == 2


def test_check_template_on_endless_template_ends_in_time(capsys, tmp_path):
    """QA round 2 Q-1 reproduction: a folder whose template loops silently over a long string."""
    (tmp_path / "chat_template.jinja").write_text(
        '{% set s = "x"*99999 %}{% for a in s %}{% for b in s %}{% endfor %}{% endfor %}<tool_call>{{ tools }}',
        encoding="utf-8")
    start = time.monotonic()
    code = main(["--lang", "en", "check-template", str(tmp_path)], environ={})
    assert time.monotonic() - start < 10.0
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("FAIL") == 5
    assert "limit for all checks" in out


def test_argparse_errors_follow_the_language(capsys):
    """QA round 2 Q-4: argparse's own error text is localized too, and English stays English."""
    import argparse

    original = argparse._
    for argv, korean in (
        (["--lang", "ko", "inspect"], "필수 인자가 없습니다: SOURCE"),
        (["--lang", "ko", "inspect", "fixture:qwen3", "--engine", "tgi"], "잘못된 값"),
        (["--lang", "ko", "inspect", "fixture:qwen3", "--nope"], "알 수 없는 인자: --nope"),
        (["--lang", "ko", "replay", "x", "--format"], "값 하나가 필요합니다"),
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv, environ={})
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert korean in err, err
        assert "사용법:" in err and "오류:" in err
        for english in ("error:", "required", "invalid choice", "unrecognized", "expected one"):
            assert english not in err, err
    assert argparse._ is original  # the gettext swap is undone after parsing

    with pytest.raises(SystemExit):
        main(["--lang", "en", "inspect"], environ={})
    err = capsys.readouterr().err
    assert "error: the following arguments are required: SOURCE" in err
    assert not re.search("[가-힣]", err)


def test_unknown_format_inspect_exits_1(capsys):
    assert main(["inspect", "fixture:plain"], environ={}) == 1
    assert "unknown-format" in capsys.readouterr().out


def test_lang_flag_accepted_after_subcommand(capsys):
    assert main(["inspect", "fixture:qwen3", "--lang", "ko"], environ={}) == 0
    assert main(["--lang", "ko", "inspect", "fixture:qwen3"], environ={}) == 0


def test_demo_exits_zero_and_shows_the_lessons(capsys):
    started = time.monotonic()
    assert main(["demo", "--lang", "en"], environ={}) == 0
    out = capsys.readouterr().out
    assert time.monotonic() - started < 5
    assert "--tool-call-parser qwen3_xml" in out
    assert "call-inside-reasoning" in out
    assert "parser-mismatch" in out and "broken-arguments" in out
    assert out.count("(expected:") == 4
    assert "mock" in out.lower()


def test_demo_json(capsys):
    assert main(["demo", "--json"], environ={}) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mock"] is True and len(data["steps"]) == 9


def test_python_dash_m_entry_point():
    done = subprocess.run([sys.executable, "-m", "parserpick", "demo", "--lang", "en"],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0 and "call-inside-reasoning" in done.stdout


def test_real_folder_source_end_to_end(capsys, tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}<tool_call>x</tool_call>{% endif %}")
    code, data = _json(capsys, ["inspect", str(tmp_path)])
    assert code == 0 and data["family"] == "hermes" and data["mock"] is False
