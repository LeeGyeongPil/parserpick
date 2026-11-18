"""Read a captured raw model output: a plain text file or a /v1/completions JSON response."""

from __future__ import annotations

import json
from pathlib import Path

from .source import EXIT_INPUT, SourceError


def parse_capture_text(raw: str) -> str:
    """Return the generated text. JSON responses with choices[0].text are unwrapped."""
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw  # a raw JSON-looking tool call that is not a response: keep as text
        if isinstance(data, dict) and isinstance(data.get("choices"), list):
            choices = data["choices"]
            first = choices[0] if choices else None
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"]
            raise SourceError(
                "the JSON capture has 'choices' but no choices[0].text; pass the /v1/completions "
                "response or save the generated text to a plain file",
                EXIT_INPUT, key="err.capture_shape",
            )
    return raw


CAPTURES = Path(__file__).parent / "fixtures" / "captures"


def fixture_capture_names() -> list[str]:
    return sorted(p.stem for p in CAPTURES.glob("*.txt"))


def load_capture(path: Path | str) -> str:
    """Read and unwrap a capture file; `fixture:<name>` reads a built-in practice capture."""
    if isinstance(path, str) and path.startswith("fixture:"):
        name = path[len("fixture:"):]
        path = CAPTURES / f"{name}.txt"
        if not path.is_file():
            raise SourceError(f"unknown capture fixture '{name}'. Available: {', '.join(fixture_capture_names())}",
                              EXIT_INPUT, key="err.no_capture", path=f"fixture:{name}")
    path = Path(path)
    if not path.is_file():
        raise SourceError(f"{path}: capture file not found", EXIT_INPUT, key="err.no_capture", path=str(path))
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError(f"{path}: not valid UTF-8 text", EXIT_INPUT, key="err.capture_encoding",
                          path=str(path)) from exc
    return parse_capture_text(raw)
