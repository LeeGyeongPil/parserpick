"""Reasoning split: the first stage of the engine pipeline (reasoning, then tool extraction)."""

from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"
RULES = ("none", "think", "think_implicit_start")


def split_reasoning(text: str, rule: str) -> tuple[str, str]:
    """Return (reasoning, content).

    think:                the output starts with <think>; everything up to </think> is reasoning.
                          No leading <think> means no reasoning.
    think_implicit_start: <think> was already in the prompt; everything before </think> is
                          reasoning. Without </think> the whole output is treated as reasoning.
    """
    if rule == "none":
        return "", text
    if rule == "think":
        body = text.lstrip()
        if not body.startswith(OPEN):
            return "", text
        body = body[len(OPEN):]
        head, sep, tail = body.partition(CLOSE)
        if not sep:
            return head.strip(), ""  # never closed: the engine keeps everything as reasoning
        return head.strip(), tail.strip()
    if rule == "think_implicit_start":
        body = text.lstrip()
        if body.startswith(OPEN):
            body = body[len(OPEN):]
        head, sep, tail = body.partition(CLOSE)
        if not sep:
            return head.strip(), ""
        return head.strip(), tail.strip()
    raise ValueError(f"unknown reasoning rule {rule!r}; choose from {RULES}")
