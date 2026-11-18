"""AC-4, AC-5: replay of probe cases A-E and the eight extraction rules."""

import json

import pytest

from parserpick.capture import load_capture, parse_capture_text
from parserpick.recommend import recommend
from parserpick.replay import replay, resolve_format, resolve_reasoning, settings_from_recommendation
from parserpick.source import load_source


def _run(capture, format_name, reasoning="none"):
    return replay(load_capture(f"fixture:{capture}"), resolve_format(format_name), resolve_reasoning(reasoning))


def _codes(result):
    return [d.code for d in result.diagnostics]


def test_probe_cases():
    # A: call after the think block works
    a = _run("a_think_then_call", "hermes", "think")
    assert a.reasoning == "need weather"
    assert [(c.name, c.arguments) for c in a.tool_calls] == [("get_weather", {"city": "Seoul"})]
    assert not a.has_error and _codes(a) == []

    # B: the call lives inside the think block and disappears
    b = _run("b_call_in_think", "hermes", "think")
    assert b.tool_calls == []
    assert "<tool_call>" in b.reasoning
    assert _codes(b) == ["call-inside-reasoning"] and b.has_error

    # C: XML read by the hermes rule stays in content
    c = _run("c_qwen3_xml", "hermes")
    assert c.tool_calls == [] and "<function=get_weather>" in c.content
    assert _codes(c) == ["parser-mismatch"] and c.has_error
    assert c.diagnostics[0].params["suggest"] == "qwen3_xml"

    # D: same XML with the right rule
    d = _run("c_qwen3_xml", "qwen3_xml")
    assert [(x.name, x.arguments) for x in d.tool_calls] == [("get_weather", {"city": "Seoul"})]
    assert d.content == "" and not d.has_error

    # E: single-quoted JSON
    e = _run("e_broken_json", "hermes")
    assert e.tool_calls == [] and _codes(e) == ["broken-arguments"] and e.has_error
    assert e.diagnostics[0].params["position"] == 1
    assert "char 1" in e.diagnostics[0].params["detail"]


def test_sglang_alias_and_old_names_resolve():
    assert resolve_format("qwen3_coder") == "qwen3_xml"
    assert resolve_format("qwen") == "hermes"
    assert resolve_format("llama3") == "llama3_json"
    assert resolve_reasoning("qwen3") == "think"
    assert resolve_reasoning("deepseek-r1") == "think_implicit_start"


FORMATS = ["hermes", "qwen3_xml", "glm45", "llama3_json", "pythonic", "mistral", "deepseek_v31", "kimi_k2"]


@pytest.mark.parametrize("family", FORMATS)
def test_format_extractors(family):
    result = _run(f"ok_{family}", family)
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("get_weather", {"city": "Seoul"})]
    assert result.content == "" and result.diagnostics == []


def test_eight_formats_listed():
    assert len(FORMATS) == 8


@pytest.mark.parametrize("family", FORMATS)
def test_unclosed_call(family):
    result = _run(f"cut_{family}", family)
    assert result.tool_calls == []
    assert [d.code for d in result.diagnostics] == ["unclosed-call"], result.diagnostics
    assert result.has_error


@pytest.mark.parametrize(("family", "raw"), [
    ("qwen3_xml", "<tool_call>\n<function=x>\n"),
    ("qwen3_xml", "<tool_call>\n<function=get_weather>\n<parameter=ci"),
    ("qwen3_xml", "<tool_call>"),
    ("glm45", "ok\n<tool_call>get_weather\n"),
    ("glm45", "<tool_call>get_weather\n<arg_key>ci"),
])
def test_cut_before_first_argument_is_unclosed_not_mismatch(family, raw):
    """QA round 2 Q-2: a cut-off qwen3_xml/glm45 call looks like bare hermes markers, but the
    right answer is unclosed-call, never 'switch to hermes'."""
    result = replay(raw, family)
    assert _codes(result) == ["unclosed-call"], result.diagnostics
    assert result.has_error


def test_complete_other_format_is_still_a_mismatch():
    """The Q-2 fix must keep probe case C: balanced qwen3_xml markers under hermes -> parser-mismatch."""
    raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>"
    assert _codes(replay(raw, "hermes")) == ["parser-mismatch"]
    assert _codes(replay(raw[:-len("</tool_call>")], "hermes")) == ["unclosed-call"]


def test_deepseek_v3_layout_with_json_fence():
    raw = ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
           "```json\n{\"city\": \"Seoul\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    result = replay(raw, "deepseek_v3")
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("get_weather", {"city": "Seoul"})]


def test_two_calls_and_typed_xml_values():
    raw = ("<tool_call>\n<function=a>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>\n"
           "<tool_call>\n<function=b>\n<parameter=s>\nhello\n</parameter>\n</function>\n</tool_call>")
    result = replay(raw, "qwen3_xml")
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("a", {"n": 3}), ("b", {"s": "hello"})]


def test_plain_text_is_no_call_info_not_error():
    result = replay("The weather in Seoul is cold.", "hermes")
    assert _codes(result) == ["no-call"] and not result.has_error


def test_pythonic_text_mix_warns():
    result = replay('Sure! [get_weather(city="Seoul")] done', "pythonic")
    assert len(result.tool_calls) == 1
    assert _codes(result) == ["text-mixed-pythonic"] and not result.has_error


def test_llama3_json_answer_is_not_a_call():
    result = replay('{"answer": 42}', "llama3_json")
    assert result.tool_calls == [] and _codes(result) == ["no-call"]


def test_implicit_think_start():
    result = replay('plan it</think>\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>',
                    "hermes", "think_implicit_start")
    assert result.reasoning == "plan it" and [c.name for c in result.tool_calls] == ["f"]


def test_model_option_derives_settings():
    rec = recommend(load_source("fixture:qwen3"))
    assert settings_from_recommendation(rec) == ("hermes", "think")
    rec = recommend(load_source("fixture:qwen3-coder"))
    assert settings_from_recommendation(rec) == ("qwen3_xml", "none")


def test_completions_json_capture_is_unwrapped():
    payload = json.dumps({"choices": [{"text": "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"}]})
    assert parse_capture_text(payload).startswith("<tool_call>")
    assert parse_capture_text('{"name": "f", "parameters": {}}').startswith('{"name"')  # a raw call, not a response


@pytest.mark.parametrize("fmt, unit", [
    ("hermes", '<tool_call>{"name": '),
    ("qwen3_xml", "<tool_call><function=f><parameter=a>1 "),
    ("glm45", "<tool_call>f<arg_key>a</arg_key><arg_value>"),
])
def test_unclosed_repeats_are_linear(fmt, unit):
    """Q-6: a looping output that repeats the opening marker must not take quadratic time."""
    import time

    from parserpick.replay.formats import EXTRACTORS

    text = unit * (300_000 // len(unit))
    start = time.monotonic()
    EXTRACTORS[fmt](text)
    EXTRACTORS[fmt](text + "</tool_call>")
    assert time.monotonic() - start < 2.0
