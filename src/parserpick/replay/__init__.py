"""Offline replay of captured raw model output."""

from __future__ import annotations

from typing import Any

from ..fingerprint import families, load_registry
from ..recommend import Recommendation, resolve_family
from .diagnose import ReplayResult, replay
from .formats import EXTRACTORS
from .reasoning import RULES

__all__ = ["ReplayResult", "ReplaySetupError", "replay", "replayable_families", "resolve_format",
           "resolve_reasoning", "settings_from_recommendation"]


class ReplaySetupError(ValueError):
    """Bad --format/--reasoning/--model. `key` and `params` build the localized message."""

    def __init__(self, message: str, key: str, **params: Any) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


def replayable_families() -> list[str]:
    return [name for name in families() if name in EXTRACTORS]


def resolve_format(name: str) -> str:
    """Accept vLLM/SGLang/alias names; raise ReplaySetupError for unknown or unsupported formats."""
    available = ", ".join(replayable_families())
    family = resolve_family(name)
    if family is None:
        raise ReplaySetupError(f"unknown format '{name}'. Replayable: {available}",
                               "err.unknown_format", name=name, available=available)
    if family not in EXTRACTORS:
        raise ReplaySetupError(f"'{name}' ({family}) has no replay rule yet. Replayable: {available}",
                               "err.no_replay_rule", name=name, available=available)
    return family


def resolve_reasoning(name: str) -> str:
    """Accept a rule name (think, think_implicit_start, none) or an engine parser name."""
    if name in RULES:
        return name
    aliases = load_registry().get("reasoning_aliases", {})
    if name in aliases:
        return aliases[name]
    available = ", ".join([*RULES, *sorted(aliases)])
    raise ReplaySetupError(f"unknown reasoning parser '{name}'. Use one of: {available}",
                           "err.unknown_reasoning", name=name, available=available)


def settings_from_recommendation(rec: Recommendation) -> tuple[str, str]:
    """(family, reasoning rule) implied by an inspect result."""
    if rec.family is None:
        raise ReplaySetupError(f"cannot pick a replay format for {rec.source}: nothing recognized",
                               "err.model_unusable", source=rec.source)
    family = resolve_format(rec.family)
    think = rec.fingerprint is not None and rec.fingerprint.think_capable
    rule = families()[family].get("reasoning_rule", "none") if think else "none"
    return family, rule
