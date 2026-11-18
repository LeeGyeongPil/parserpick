"""AC-1, AC-2, AC-3, AC-10: fingerprint -> family -> engine flags and template warnings."""

import json
import tomllib
from pathlib import Path

import pytest

from parserpick.cli import main
from parserpick.i18n import t
from parserpick.recommend import recommend
from parserpick.replay.diagnose import replay
from parserpick.report import issue_message
from parserpick.source import fixture_names, load_source

# fixture name -> (tool family, vLLM reasoning parser or None)
EXPECTED = {
    "qwen3": ("hermes", "qwen3"),
    "qwen3-coder": ("qwen3_xml", None),  # has <tool_call> but is NOT hermes
    "hermes3": ("hermes", None),
    "glm45": ("glm45", "glm45"),  # has <tool_call> but is NOT hermes
    "llama3": ("llama3_json", None),
    "pythonic": ("pythonic", None),
    "deepseek-r1": ("deepseek_v3", "deepseek_r1"),
    "deepseek-v31": ("deepseek_v31", "deepseek_v3"),
    "kimi-k2": ("kimi_k2", None),
    "gpt-oss": ("openai", "openai_gptoss"),
    "granite": ("granite", None),
    "mistral": ("mistral", None),
    "minimax": ("minimax", "minimax_m2_append_think"),
}


def _rec(name, engine="vllm", version=None):
    return recommend(load_source(f"fixture:{name}"), engine, version)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_families(name):
    family, reasoning = EXPECTED[name]
    rec = _rec(name)
    assert rec.family == family
    assert rec.reasoning_parser == reasoning


def test_fixture_families_cover_thirteen():
    assert len(EXPECTED) == 13


# (family fixture, vLLM tool parser, SGLang tool parser)
NAME_PAIRS = [
    ("qwen3-coder", "qwen3_xml", "qwen3_coder"),
    ("qwen3", "hermes", "qwen"),
    ("glm45", "glm45", "glm"),
    ("deepseek-v31", "deepseek_v31", "deepseekv31"),
    ("gpt-oss", "openai", "gpt-oss"),
]


@pytest.mark.parametrize(("name", "vllm_name", "sglang_name"), NAME_PAIRS)
def test_engine_name_mapping(name, vllm_name, sglang_name):
    vllm = _rec(name, "vllm")
    assert f"--enable-auto-tool-choice --tool-call-parser {vllm_name}" in vllm.flags_line
    sglang = _rec(name, "sglang")
    assert f"--tool-call-parser {sglang_name}" in sglang.flags_line
    assert "--enable-auto-tool-choice" not in sglang.flags_line


def test_unverified_version_warns():
    verified = _rec("qwen3-coder", "vllm", "0.34.2")
    assert "unverified-engine-version" not in [w.code for w in verified.warnings]
    older = _rec("qwen3-coder", "vllm", "0.31.0")
    codes = [w.code for w in older.warnings]
    assert "unverified-engine-version" in codes
    warning = next(w for w in older.warnings if w.code == "unverified-engine-version")
    assert warning.params["url"].startswith("https://")
    old = [n for n in older.notes if n.code == "old-name"]
    assert old and old[0].params["old"] == "qwen3_coder"
    assert old[0].params["url"].startswith("https://github.com/vllm-project/vllm/issues/")


def test_glm_old_name_note_has_source():
    note = next(n for n in _rec("glm45").notes if n.code == "old-name")
    assert note.params["old"] == "glm4_moe"
    assert "discuss.vllm.ai" in note.params["url"]


def _codes(name):
    rec = _rec(name)
    return {w.code for w in rec.warnings}


def test_template_warnings():
    every = {name: _codes(name) for name in fixture_names()}

    def only(code):
        return {name for name, codes in every.items() if code in codes}

    assert only("tools-not-rendered") == {"deepseek-r1", "deepseek-v31"}
    assert only("named-template") == {"hermes3"}
    assert only("response-template-present") == {"hf-response"}
    assert only("unknown-format") == {"plain"}

    # hints and flags
    assert "--chat-template examples/tool_chat_template_deepseekv31.jinja" in _rec("deepseek-v31").flags_line
    # Named templates: never `--chat-template tool_use` (both engines read that value as a file path and
    # refuse to start). vLLM picks tool_use by itself for tool requests; SGLang needs it named.
    hermes3_vllm = _rec("hermes3")
    assert hermes3_vllm.flags_line == "--enable-auto-tool-choice --tool-call-parser hermes"
    assert next(w for w in hermes3_vllm.warnings if w.code == "named-template").severity == "info"
    hermes3_sglang = _rec("hermes3", "sglang")
    assert hermes3_sglang.flags_line == "--tool-call-parser qwen --hf-chat-template-name tool_use"
    assert next(w for w in hermes3_sglang.warnings if w.code == "named-template").severity == "warning"
    hf = _rec("hf-response")
    assert "--tool-call-parser hf" in hf.flags_line and "--reasoning-parser hf" in hf.flags_line
    sglang_hf = recommend(load_source("fixture:hf-response"), "sglang")
    assert sglang_hf.flags_line == "--tool-call-parser qwen --reasoning-parser qwen3"  # no hf parser in SGLang
    assert _rec("plain").flags == []


def test_decision_ignores_model_name(tmp_path):
    """A renamed checkpoint is still recognized: only template content matters."""
    src = load_source("fixture:qwen3-coder")
    src.label = "totally-unrelated/my-finetune"
    assert recommend(src).family == "qwen3_xml"


def test_unsupported_engine_name_is_omitted_not_invented():
    rec = recommend(load_source("fixture:granite"), "sglang")
    assert "--tool-call-parser" not in rec.flags_line
    assert "no-engine-parser" in [w.code for w in rec.warnings]


# --- AC-10: every printed flag line is one the engine accepts at startup -----------------------
#
# tests/data/engine_parsers.toml holds the registered parser names and the flag rules of the
# verified engine versions (vLLM 0.34.0, SGLang 0.5.23), with source URLs and line numbers.

SNAPSHOT = tomllib.loads((Path(__file__).parent / "data" / "engine_parsers.toml").read_text(encoding="utf-8"))
NO_TOOL_PARSER = [("minimax", "vllm"), ("minimax", "sglang"), ("granite", "sglang")]


def assert_engine_accepts(engine: str, tokens: list[str]) -> None:
    """Fail when `engine` would refuse to start with this flag line (rules from the snapshot)."""
    rules = SNAPSHOT[engine]
    values: dict[str, str | None] = {}
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        assert flag not in values, f"{flag} given twice: {tokens}"
        if flag in rules["switch_flags"]:
            values[flag] = None
            i += 1
            continue
        assert flag in rules["value_flags"], f"{engine} has no flag {flag}: {tokens}"
        assert i + 1 < len(tokens) and not tokens[i + 1].startswith("--"), f"{flag} needs a value: {tokens}"
        values[flag] = tokens[i + 1]
        i += 2

    has_auto = "--enable-auto-tool-choice" in values
    has_parser = "--tool-call-parser" in values
    if engine == "vllm":
        # (a) "Error: --enable-auto-tool-choice requires --tool-call-parser"; and the parser is only
        # loaded when the switch is on, so a parser without the switch would silently do nothing.
        assert has_auto == has_parser, f"vLLM needs both or neither: {tokens}"
    else:
        # (b) SGLang has no such flag
        assert not has_auto, f"SGLang has no --enable-auto-tool-choice: {tokens}"
    # (c) only names registered in the verified engine version
    if has_parser:
        assert values["--tool-call-parser"] in rules["tool"], f"{engine} has no tool parser {values}"
    if "--reasoning-parser" in values:
        assert values["--reasoning-parser"] in rules["reasoning"], f"{engine} has no reasoning parser {values}"
    # (d) --chat-template is read as a file path (or literal Jinja); a bare name such as tool_use is refused
    if "--chat-template" in values:
        assert values["--chat-template"].endswith(".jinja"), f"--chat-template must be a file: {tokens}"


def test_startup_rule_cases_cover_all_fixtures():
    assert len(fixture_names()) == 15


@pytest.mark.parametrize(("name", "engine"), [(n, e) for n in fixture_names() for e in ("vllm", "sglang")])
def test_flags_pass_engine_startup_rules(name, engine):
    rec = _rec(name, engine)
    assert_engine_accepts(engine, rec.flags)
    # the note that shows the same format on the other engine is a paste line too
    for note in rec.notes:
        if note.code == "other-engine-name":
            assert_engine_accepts(note.params["engine"], note.params["flags"].split())
    if rec.tool_parser is None and rec.family is not None:
        assert "no-engine-parser" in [w.code for w in rec.warnings]


@pytest.mark.parametrize("lang", ["en", "ko"])
@pytest.mark.parametrize(("name", "engine"), NO_TOOL_PARSER)
def test_no_tool_parser_omits_auto_tool_choice(capsys, name, engine, lang):
    """qa-fresh F-1: no documented tool parser -> no tool-call flags at all, a warning, exit code 0."""
    code = main(["inspect", f"fixture:{name}", "--engine", engine, "--json", "--lang", lang], environ={})
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert "--tool-call-parser" not in data["flags"]
    assert "--enable-auto-tool-choice" not in data["flags"]
    warning = next(w for w in data["warnings"] if w["code"] == "no-engine-parser")
    assert warning["severity"] == "warning"
    if engine == "vllm":
        # the sentence says why the switch is gone, in both languages
        assert "--enable-auto-tool-choice" in warning["message"]
    if (name, engine) == ("minimax", "vllm"):
        assert data["flags_line"] == "--reasoning-parser minimax_m2_append_think"

    # the human-readable "Serve with" block shows the same line (or is absent when there is nothing to paste)
    code = main(["inspect", f"fixture:{name}", "--engine", engine, "--lang", lang], environ={})
    text = capsys.readouterr().out
    assert code == 0
    serve_with = t(lang, "lbl.serve_with") + ":"
    if data["flags"]:
        lines = text.splitlines()
        assert lines[lines.index(serve_with) + 1] == "  " + data["flags_line"]
    else:
        assert serve_with not in text


# Tool-call text of formats that some engine has no parser for, left in content by the hermes parser.
FOREIGN_CALLS = {
    "minimax": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="city">Seoul</parameter>\n'
               "</invoke>\n</minimax:tool_call>",
    "granite": '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Seoul"}}]',
}


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("detected", sorted(FOREIGN_CALLS))
def test_parser_mismatch_suggests_only_names_the_engine_has(engine, detected):
    """replay's parser-mismatch advice is a flag too: never an empty value or the other engine's name."""
    result = replay(FOREIGN_CALLS[detected], "hermes", "none", engine)
    issue = next(d for d in result.diagnostics if d.code == "parser-mismatch")
    assert issue.params["detected"] == detected
    suggest = issue.params["suggest"]
    if suggest:
        assert_engine_accepts(engine, (["--enable-auto-tool-choice"] if engine == "vllm" else [])
                              + ["--tool-call-parser", suggest])
    for lang in ("en", "ko"):
        message = issue_message(issue, lang)
        if not suggest:
            assert "--tool-call-parser" not in message
            assert t(lang, "lbl.no_documented_name") in message  # instead of an empty "SGLang: )"
    expected = {("granite", "vllm"): "granite"}.get((detected, engine), "")
    assert suggest == expected


def test_syntax_error_template_warns(tmp_path):
    """Q-7: a template that cannot even be parsed is reported by inspect."""
    from parserpick.source import read_local

    (tmp_path / "chat_template.jinja").write_text("{% for x in %}<tool_call>{{ tools }}", encoding="utf-8")
    rec = recommend(read_local(tmp_path))
    assert "template-syntax-error" in [w.code for w in rec.warnings]
