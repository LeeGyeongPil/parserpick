"""Marker fingerprinting details beyond the fixture table."""

from parserpick.fingerprint import classify, find_markers, fingerprint, is_ambiguous, template_uses_tools


def test_tools_variable_detected_by_parsing_not_substring():
    assert template_uses_tools("{% if tools %}x{% endif %}")
    assert not template_uses_tools("{# tools are never used here #}{{ messages }}")
    assert not template_uses_tools("{{ 'tools' }}")
    assert template_uses_tools("{% for tool in tools %}{{ tool }}{% endfor %}")


def test_deepseek_r1_style_template_without_literal_function_sep():
    template = ("{{ '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + t.type + '<｜tool▁sep｜>' + t.name"
                " + '\\n```json\\n' }}")
    assert classify(find_markers(template))[0] == "deepseek_v3"
    assert classify(find_markers("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"))[0] == "deepseek_v31"


def test_single_tool_call_tag_is_not_enough_to_claim_hermes():
    assert classify(["tool_call_tag"])[0] == "hermes"
    assert classify(["tool_call_tag", "xml_function", "xml_parameter"])[0] == "qwen3_xml"
    assert classify(["tool_call_tag", "arg_key"])[0] == "glm45"
    # only part of the XML pair: stays hermes rather than guessing
    assert classify(["tool_call_tag", "xml_function"])[0] == "hermes"


def test_llama3_markers_do_not_beat_hermes():
    family, alternatives = classify(["tool_call_tag", "json_parameters", "python_tag"])
    assert family == "hermes" and "llama3_json" in alternatives
    assert not is_ambiguous(["tool_call_tag", "json_parameters", "python_tag"])


def test_unrelated_strong_markers_are_ambiguous():
    assert is_ambiguous(["mistral_tool_calls", "kimi_section"])


def test_pythonic_loses_to_python_tag():
    assert classify(["pythonic_list"])[0] == "pythonic"
    assert classify(["pythonic_list", "python_tag"])[0] == "llama3_json"


def test_token_markers_count_even_when_template_lacks_them():
    fp = fingerprint(["{{ tools }}"], ["<tool_call>"])
    assert fp.family == "hermes" and fp.token_markers == ["tool_call_tag"]
