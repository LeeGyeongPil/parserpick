"""AC-8: bilingual UI."""

import json
import re

import pytest

from parserpick import i18n
from parserpick.cli import main

HANGUL = re.compile("[가-힣]")


def test_keys_match():
    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    assert set(en) ^ set(ko) == set()
    assert all(en[k].strip() and ko[k].strip() for k in en)


def test_placeholders_match():
    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    field = re.compile(r"\{(\w+)\}")
    for key in en:
        assert set(field.findall(en[key])) == set(field.findall(ko[key])), key


def test_every_issue_code_has_a_message():
    import tomllib  # noqa: F401  (registry sanity)

    en = i18n.catalog("en")
    for key in ("issue.call-inside-reasoning", "issue.parser-mismatch", "issue.broken-arguments",
                "issue.unclosed-call", "issue.no-call", "issue.text-mixed-pythonic"):
        assert key in en


def test_precedence():
    env = {"PARSERPICK_LANG": "ko", "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    assert i18n.resolve_lang("en", env) == "en"  # --lang beats everything
    assert i18n.resolve_lang(None, env) == "ko"  # PARSERPICK_LANG beats LC_ALL/LANG
    assert i18n.resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "en", "LANG": "en"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_MESSAGES": "ko_KR", "LANG": "en_US"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"
    assert i18n.resolve_lang(None, {"LANG": "C"}) == "en"
    assert i18n.resolve_lang(None, {}) == "en"


def _run(capsys, argv, env=None):
    code = main(argv, environ=env or {})
    return code, capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["inspect", "fixture:qwen3-coder"],
    ["inspect", "fixture:deepseek-v31"],
    ["replay", "fixture:b_call_in_think", "--model", "fixture:qwen3"],
    ["replay", "fixture:c_qwen3_xml", "--format", "hermes"],
    ["check-template", "fixture:mistral"],
    ["demo"],
])
def test_cli_output_language(capsys, argv):
    _, ko = _run(capsys, [*argv, "--lang", "ko"])
    _, en = _run(capsys, [*argv, "--lang", "en"])
    assert HANGUL.search(ko)
    assert not HANGUL.search(en)


def test_help_is_localized(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "ko", "--help"], environ={})
    assert exc.value.code == 0
    assert HANGUL.search(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        main(["--lang", "en", "inspect", "--help"], environ={})
    assert not HANGUL.search(capsys.readouterr().out)


def test_env_var_selects_language(capsys):
    _, out = _run(capsys, ["inspect", "fixture:qwen3"], {"PARSERPICK_LANG": "ko"})
    assert HANGUL.search(out)
    _, out = _run(capsys, ["inspect", "fixture:qwen3", "--lang", "en"], {"PARSERPICK_LANG": "ko"})
    assert not HANGUL.search(out)


def test_errors_are_localized(capsys):
    code = main(["inspect", "/definitely/not/here", "--lang", "ko"], environ={})
    assert code == 2
    assert HANGUL.search(capsys.readouterr().err)


@pytest.mark.parametrize("argv", [["--help"], ["replay", "--help"], ["inspect", "--help"],
                                  ["check-template", "--help"], ["demo", "--help"]])
def test_korean_help_has_no_english_argparse_text(argv, capsys):
    """Q-5: argparse's built-in English strings must be localized too."""
    with pytest.raises(SystemExit):
        main(["--lang", "ko", *argv], environ={})
    out = capsys.readouterr().out
    for english in ("usage:", "show this help", "options:", "positional arguments:", "version number"):
        assert english not in out
    assert "사용법: parserpick" in out and "사용법: 사용법" not in out


# --- QA round 3 Q-1: no sentence written by this project may stay English under --lang ko ---------
#
# Allowed in Korean output: external messages (Python json/ast, Jinja, a template's own
# raise_exception text), input data (captured text, file paths, parser and fixture names) and
# command echoes. Everything parserpick writes itself must come from ko.json.

# Messages from Python and Jinja that the test inputs below are known to trigger.
EXTERNAL_MESSAGES = (
    "Expecting property name enclosed in double quotes", "Unterminated string starting at",
    "Invalid control character at", "Expecting value", "maximum recursion depth exceeded",
    "Expected an expression, got 'end of statement block'", "can only concatenate str",
    "Tool call IDs should be alphanumeric strings with length 9",
)
# Phrases parserpick used to write in English (QA round 3 repro and its siblings).
OLD_ENGLISH = (
    "has no matching", "took longer", "repeating a sequence", "could not be parsed", "render process",
    "exponent too large", "is not available", "output exceeded", "not run:", "stopped after",
    "no string 'name'", "never closed", "not a list", "each item must", "separator token",
    "line 1 column", "(char ", " at char ", "(line ", "are rejected", "Fine if your engine",
)
_ENGLISH_RUN = re.compile(r"[A-Za-z][A-Za-z'_.\-]*(?:[ ,]+[A-Za-z][A-Za-z'_.\-]*){3,}")
_PYTHON_ERROR = re.compile(r"\b(?!RenderLimit\b|RenderProcessError\b)[A-Z]\w*(?:Error|Raised): [^\n]*")
# OS errno text such as "[Errno 49] Can't assign requested address." -- only the OS phrase (up to its period or line end);
# project English after it stays visible to the check
_OS_ERROR = re.compile(r"\[Errno -?\d+\] [^.\n]*\.?")


def _english_catalog_fragments() -> set[str]:
    """Literal pieces (3+ words) of English messages that the Korean table does not also contain."""
    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    pieces = set()
    for key, text in en.items():
        for piece in re.split(r"\{\w+\}|%\(\w+\)[rs]|%[rs]", text):
            piece = piece.strip(" .,:;()'\n")
            if len(piece.split()) >= 3 and piece not in ko[key]:
                pieces.add(piece)
    return pieces


def _project_english(text: str) -> list[str]:
    """English word runs left after removing what is allowed to stay English."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # 명령 반향·제목 줄(파서·픽스처 이름)과 4칸 들여쓴 데이터 블록(reasoning/content/tool_calls)은 원문 데이터
        if stripped.startswith(("parserpick ", "$ parserpick", "사용법: ", "parserpick:")) or line.startswith("    "):
            continue
        lines.append(line)
    rest = "\n".join(lines)
    for message in EXTERNAL_MESSAGES:
        rest = rest.replace(message, "")
    rest = _OS_ERROR.sub("", rest)  # OS 가 준 errno 설명은 OS·환경마다 달라지는 원문
    rest = _PYTHON_ERROR.sub("", rest)  # 파이썬·Jinja 예외 원문 (우리 RenderLimit 은 제외)
    rest = re.sub(r"[\w.\-]+(?:, [\w.\-<>/]+){2,}", "", rest)  # 사용 가능한 이름 목록
    return _ENGLISH_RUN.findall(rest)


def test_os_error_mask_is_narrow():
    """The guard hides OS errno phrases only, not project English that follows them."""
    ok = "네트워크 오류 [Errno 61] Connection refused. 다시 시도하세요"
    assert _project_english(ok) == []
    assert _project_english("[Errno 49] Can't assign requested address.") == []
    bad = "[Errno 61] Connection refused. Could not reach the server right now"
    assert _project_english(bad) != []


def _cases(tmp_path):
    """Every subcommand on representative inputs, each limit/diagnostic reason, and error paths."""
    files = {
        "g.txt": "ok\n<tool_call>get_weather\n",  # QA repro 1: glm45 cut off -> unclosed-call
        "cut_qwen.txt": "<tool_call>\n<function=get_weather>\n<parameter=ci",
        "noname.txt": '<tool_call>{"x": 1}</tool_call>',
        "body.txt": "<tool_call>[1, 2]</tool_call>",
        "notlist.txt": "[get_weather(city='Seoul')][0]",
        "item.txt": "[get_weather('Seoul')]",
        "openlist.txt": "[get_weather(city='Seoul')",
        "nosep.txt": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "mistral_noname.txt": '[TOOL_CALLS][{"x": 1}]',
        "shape.json": json.dumps({"choices": [{}]}),
        "granite.txt": '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Seoul"}}]',  # no SGLang name
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    templates = {
        "repeat": '{% set s = "x"*200000 %}{{ s }}',  # QA repro 2
        "recursion": "{% macro r(n) %}{{ r(n+1) }}{% endmacro %}{{ r(0) }}<tool_call>{{ tools }}",
        "exponent": "{{ 2 ** 5000 }}<tool_call>{{ tools }}",
        "lipsum": "{{ lipsum() }}<tool_call>{{ tools }}",
        "output": "{% for i in range(2000) %}{{ 'x' * 1000 }}{% endfor %}<tool_call>{{ tools }}",
        "syntax": "{% if %}<tool_call>{{ tools }}",
        "unsafe": "{{ ''.__class__.__mro__[1].__subclasses__() }}<tool_call>{{ tools }}",
    }
    for name, text in templates.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "chat_template.jinja").write_text(text, encoding="utf-8")
    (tmp_path / "badjson").mkdir()
    (tmp_path / "badjson" / "tokenizer_config.json").write_text("{not json", encoding="utf-8")
    t = str(tmp_path)
    return [
        ["inspect", "fixture:qwen3-coder"], ["inspect", "fixture:deepseek-v31"], ["inspect", "fixture:hermes3"],
        ["inspect", "fixture:hf-response"], ["inspect", "fixture:plain"], ["inspect", "fixture:minimax"],
        # v2 플래그 조합 규칙 문구(엔진별 no-engine-parser·이름 붙은 템플릿·response_template)
        ["inspect", "fixture:minimax", "--engine", "sglang"], ["inspect", "fixture:granite", "--engine", "sglang"],
        ["inspect", "fixture:hermes3", "--engine", "sglang"], ["inspect", "fixture:hf-response", "--engine", "sglang"],
        ["replay", f"{t}/granite.txt", "--format", "hermes", "--engine", "sglang"],
        ["replay", f"{t}/granite.txt", "--format", "hermes"],
        ["inspect", "fixture:gpt-oss", "--engine", "sglang", "--engine-version", "0.1.0"],
        ["inspect", f"{t}/syntax"],
        ["replay", "fixture:b_call_in_think", "--model", "fixture:qwen3"],
        ["replay", "fixture:c_qwen3_xml", "--format", "hermes"],
        ["replay", "fixture:e_broken_json", "--format", "hermes"],
        ["replay", "fixture:cut_llama3_json", "--format", "llama3_json"],
        ["replay", "fixture:cut_pythonic", "--format", "pythonic"],
        ["replay", "fixture:cut_mistral", "--format", "mistral"],
        ["replay", "fixture:ok_pythonic", "--format", "hermes"],
        ["replay", f"{t}/g.txt", "--format", "glm45"],
        ["replay", f"{t}/cut_qwen.txt", "--format", "qwen3_xml"],
        ["replay", f"{t}/noname.txt", "--format", "hermes"],
        ["replay", f"{t}/body.txt", "--format", "hermes"],
        ["replay", f"{t}/notlist.txt", "--format", "pythonic"],
        ["replay", f"{t}/item.txt", "--format", "pythonic"],
        ["replay", f"{t}/openlist.txt", "--format", "pythonic"],
        ["replay", f"{t}/nosep.txt", "--format", "deepseek_v31"],
        ["replay", f"{t}/mistral_noname.txt", "--format", "mistral"],
        ["check-template", "fixture:mistral"], ["check-template", "fixture:deepseek-v31"],
        *[["check-template", f"{t}/{name}"] for name in templates],
        ["demo"],
        # 오류 경로
        ["inspect", f"{t}/badjson"], ["inspect", f"{t}/no-such-dir"], ["inspect", "fixture:nope"],
        ["inspect", "hf:not-a-repo"], ["inspect", "hf:org/model"],
        ["replay", f"{t}/missing.txt", "--format", "hermes"], ["replay", f"{t}/g.txt"],
        ["replay", f"{t}/g.txt", "--format", "nope"], ["replay", f"{t}/g.txt", "--format", "minimax"],
        ["replay", f"{t}/g.txt", "--format", "hermes", "--reasoning", "nope"],
        ["replay", f"{t}/g.txt", "--model", "fixture:plain"],
        ["replay", f"{t}/shape.json", "--format", "hermes"],
        ["inspect"], ["inspect", "fixture:qwen3", "--engine", "tgi"], ["inspect", "fixture:qwen3", "--nope"],
    ]


def _run_ko(argv):
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            main([*argv, "--lang", "ko"], environ={"HF_ENDPOINT": "http://127.0.0.1:9"})
        except SystemExit:
            pass
    return out.getvalue() + err.getvalue()


def test_every_reason_code_in_the_source_has_messages():
    """Each `detail.<reason>` the code can produce exists in both tables (some reasons are hard to
    trigger from the CLI, e.g. a pythonic call text that is not a list)."""
    from pathlib import Path

    import parserpick

    patterns = (r'call_error\(\s*"?[\w-]+"?,\s*"([\w-]+)"', r'RenderLimit\("([\w-]+)"', r'_BadPythonic\("([\w-]+)"',
                r'_limit_result\([^)]*?"check\.[\w.]+",\s*"([\w-]+)"', r'"reason": "([\w-]+)"')
    source = "\n".join(p.read_text(encoding="utf-8") for p in Path(parserpick.__file__).parent.rglob("*.py"))
    reasons = {match for pattern in patterns for match in re.findall(pattern, source)}
    assert {"time-limit", "repeat-limit", "unmatched-marker", "unparsed-body", "json-error", "not-a-list",
            "process-exit", "killed", "jinja-syntax"} <= reasons
    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    for reason in reasons:
        assert f"detail.{reason}" in en and f"detail.{reason}" in ko, reason


def test_korean_output_has_no_project_english(tmp_path):
    """QA round 3 Q-1: --lang ko output keeps only external messages and data in English."""
    fragments = _english_catalog_fragments()
    for argv in _cases(tmp_path):
        text = _run_ko(argv)
        where = " ".join(argv)
        assert HANGUL.search(text), where
        assert "detail." not in text, where  # 메시지 표에 없는 이유 코드가 키 그대로 새지 않았나
        for phrase in OLD_ENGLISH:
            assert phrase not in text, (where, phrase, text)
        for piece in fragments:
            assert piece not in text, (where, piece, text)
        assert _project_english(text) == [], (where, text)


def test_korean_output_names_the_real_cause(tmp_path):
    """The two QA reproductions read as Korean sentences with the actual values."""
    _cases(tmp_path)  # writes the input files

    def check(name):
        return _run_ko(["check-template", str(tmp_path / name)])

    replayed = _run_ko(["replay", str(tmp_path / "g.txt"), "--format", "glm45"])
    assert "출력이 도구 호출 도중에 끝났습니다('<tool_call>' 를 닫는 '</tool_call>' 가 없음)" in replayed
    assert check("repeat").count("렌더링 실패: RenderLimit: 문자열·목록을 100000개 넘게 반복해 만들 수 없습니다") == 5
    assert "거듭제곱 지수가 너무 큽니다(1000 초과)" in check("exponent")
    assert "1000000자를 넘었습니다" in check("output")
    assert "lipsum() 은 쓸 수 없습니다" in check("lipsum")
    assert "1행 2열 (문자 위치 1)" in _run_ko(["replay", "fixture:e_broken_json", "--format", "hermes"])


def test_range_loop_budget_reads_in_korean(tmp_path):
    """QA round 3 repro 2 (range loop): one 'stopped' line naming the shared budget, then 'not run'."""
    (tmp_path / "chat_template.jinja").write_text(
        '{% set ns = namespace(s="x") %}{% for a in range(99999) %}{% for b in range(99999) %}'
        '{% set ns.s = ns.s ~ "y" %}{% endfor %}{% endfor %}', encoding="utf-8")
    text = _run_ko(["check-template", str(tmp_path)])
    assert text.count("FAIL") == 5
    assert "멈춤: 모든 검사가 함께 쓰는 렌더링 시간 5초를 다 썼습니다" in text
    assert text.count("실행 안 함") == 4
    assert _project_english(text) == [] and "took longer" not in text


def test_render_process_exit_reads_in_korean(monkeypatch):
    import parserpick.checks as checks
    from parserpick.report import check_message

    monkeypatch.setattr(checks, "_CHILD_CODE", "raise SystemExit(5)\n")
    results = checks.run_checks("{{ tools }}")
    assert {r.reason for r in results} == {"process-exit"}
    assert results[0].detail == "render process exited with code 5"  # --json 은 언어 중립 영어
    message = check_message(results[0], "ko")
    assert "종료 코드 5" in message and "exited" not in message


def test_json_detail_and_params_do_not_depend_on_language(capsys, tmp_path):
    """Only `message` follows --lang; detail, reason and params stay language independent."""
    (tmp_path / "g.txt").write_text("ok\n<tool_call>get_weather\n", encoding="utf-8")
    argv = ["replay", str(tmp_path / "g.txt"), "--format", "glm45", "--json"]
    main([*argv, "--lang", "en"], environ={})
    en = json.loads(capsys.readouterr().out)["diagnostics"][0]
    main([*argv, "--lang", "ko"], environ={})
    ko = json.loads(capsys.readouterr().out)["diagnostics"][0]
    assert en["params"] == ko["params"]
    assert en["params"]["reason"] == "unmatched-marker"
    assert en["params"]["detail"] == "'<tool_call>' has no matching '</tool_call>'"
    assert en["message"] != ko["message"] and HANGUL.search(ko["message"])



# --- QA round 4 Q-1: the family description in '(...)' after the format name is UI text too ---------

def _template_fixtures() -> list[str]:
    from pathlib import Path

    import parserpick

    folder = Path(parserpick.__file__).parent / "fixtures" / "templates"
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def test_every_family_has_a_label_in_both_languages():
    """Each family in formats.toml has `format.<family>.label` in both tables, and the Korean one
    is a Korean description (not the English text copied over)."""
    from parserpick.fingerprint import families

    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    for family in families():
        key = f"format.{family}.label"
        assert key in en and key in ko, key
        assert en[key] != ko[key], key
        assert HANGUL.search(ko[key]), key
    # 표에 있는 설명 키는 모두 실제 계열의 것 (오타·남은 키 없음)
    label_keys = {k for k in en if k.startswith("format.")}
    assert label_keys == {f"format.{family}.label" for family in families()}


def test_formats_toml_holds_no_ui_text():
    """formats.toml is data only: no `label` (moved to the message tables)."""
    from parserpick.fingerprint import families

    for family, spec in families().items():
        assert "label" not in spec, family


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_korean_inspect_never_shows_an_english_family_label(engine):
    """Every built-in template fixture, both engines: no English label text anywhere in --lang ko,
    and the recognized family is described with its Korean label."""
    from parserpick.fingerprint import families

    en, ko = i18n.catalog("en"), i18n.catalog("ko")
    english_labels = {en[f"format.{family}.label"] for family in families()}
    seen = set()
    for fixture in _template_fixtures():
        text = _run_ko(["inspect", f"fixture:{fixture}", "--engine", engine])
        for label in english_labels:
            assert label not in text, (fixture, engine, label)
        for family in families():
            if f": {family}  (" in text:
                assert f": {family}  ({ko[f'format.{family}.label']})" in text, (fixture, family)
                seen.add(family)
    assert len(seen) >= 12  # 형식 표의 12개 계열이 모두 한 번 이상 한국어 설명으로 나왔다


def test_korean_demo_never_shows_an_english_family_label():
    en = i18n.catalog("en")
    text = _run_ko(["demo"])
    for key, label in en.items():
        if key.startswith("format."):
            assert label not in text, label
    assert "Qwen3-Coder 형 XML" in text


def test_no_engine_parser_warning_uses_the_korean_label(capsys):
    """QA repro: minimax on sglang has no documented parser name; the warning sentence is Korean,
    while --json params.label stays the language-independent English description."""
    text = _run_ko(["inspect", "fixture:minimax", "--engine", "sglang"])
    assert "no-engine-parser" in text
    assert "'minimax' 계열(MiniMax 형 XML (<minimax:tool_call>))" in text
    assert "MiniMax <minimax:tool_call> XML" not in text

    params = {}
    for lang in ("en", "ko"):
        main(["inspect", "fixture:minimax", "--engine", "sglang", "--json", "--lang", lang], environ={})
        data = json.loads(capsys.readouterr().out)
        warning = next(w for w in data["warnings"] if w["code"] == "no-engine-parser")
        params[lang] = warning["params"]
    assert params["en"] == params["ko"]
    assert params["en"]["label"] == "MiniMax <minimax:tool_call> XML"


def test_korean_call_inside_reasoning_names_the_thinking_mode():
    """QA round 5 Q-1 (minor): the advice reads '생각(thinking) 모드', not a bare English word."""
    text = _run_ko(["replay", "fixture:b_call_in_think", "--model", "fixture:qwen3"])
    assert "생각(thinking) 모드를 끄거나" in text
    assert "thinking 을" not in text


def test_formats_toml_prose_never_reaches_korean_output(tmp_path):
    """Any multi-word English text kept in formats.toml (model lists, rule descriptions) must not be
    printed under --lang ko. Direct substring check, no word-run heuristic."""
    import tomllib
    from pathlib import Path

    import parserpick

    with (Path(parserpick.__file__).parent / "data" / "formats.toml").open("rb") as handle:
        table = tomllib.load(handle)

    prose: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and " " in value and not value.startswith(("http://", "https://")):
            if len(re.findall(r"[A-Za-z]{2,}", value)) >= 2:
                prose.add(value)

    walk(table)
    assert prose  # 예: 'no reasoning split', 'Mistral-7B-Instruct-v0.3 and compatible'
    argvs = _cases(tmp_path) + [["inspect", f"fixture:{f}", "--engine", e]
                                for f in _template_fixtures() for e in ("vllm", "sglang")]
    for argv in argvs:
        text = _run_ko(argv)
        for value in prose:
            assert value not in text, (argv, value)
