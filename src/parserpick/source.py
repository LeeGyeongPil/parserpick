"""Load a chat template and tokenizer facts from a local folder, hf:<repo>, or fixture:<name>."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures" / "templates"
HF_FILES = ("tokenizer_config.json", "chat_template.jinja", "chat_template.json")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_TIMEOUT = 30.0

EXIT_INPUT = 2
EXIT_DOWNLOAD = 4


class SourceError(Exception):
    """Raised when a source cannot be read. `exit_code` is the CLI exit code to use."""

    def __init__(self, message: str, exit_code: int = EXIT_INPUT, key: str = "", **params: object) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.key = key  # i18n key for the CLI; empty means use the plain message
        self.params = params


@dataclass
class ModelSource:
    """Everything parserpick needs from a model folder."""

    label: str
    templates: dict[str, str] = field(default_factory=dict)  # name -> template text
    added_tokens: list[str] = field(default_factory=list)
    has_response_template: bool = False
    named: bool = False  # chat_template was a list of named templates
    origin: str = ""  # which file the template came from

    @property
    def tool_template_name(self) -> str:
        """Name of the template used for tool-calling checks (tool_use if present)."""
        for name in ("tool_use", "default"):
            if name in self.templates:
                return name
        return next(iter(self.templates))

    @property
    def tool_template(self) -> str:
        return self.templates[self.tool_template_name]


def _templates_from_config_value(value: Any) -> tuple[dict[str, str], bool]:
    """chat_template may be a string or a list of {"name", "template"} entries."""
    if isinstance(value, str):
        return {"default": value}, False
    if isinstance(value, list):
        named = {
            str(entry["name"]): str(entry["template"])
            for entry in value
            if isinstance(entry, dict) and "name" in entry and "template" in entry
        }
        return named, True
    return {}, False


def _added_tokens(config: dict[str, Any]) -> list[str]:
    decoder = config.get("added_tokens_decoder")
    tokens: list[str] = []
    if isinstance(decoder, dict):
        for entry in decoder.values():
            if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                tokens.append(entry["content"])
    return tokens


def build_source(label: str, files: dict[str, str]) -> ModelSource:
    """Assemble a ModelSource from raw file contents (name -> text)."""
    config: dict[str, Any] = {}
    if "tokenizer_config.json" in files:
        try:
            config = json.loads(files["tokenizer_config.json"])
        except json.JSONDecodeError as exc:
            raise SourceError(
                f"{label}: tokenizer_config.json is not valid JSON ({exc.msg} at char {exc.pos})",
                key="err.bad_json", label=label, json_error=exc.msg, char=exc.pos,
            ) from exc
    templates: dict[str, str] = {}
    named = False
    origin = ""
    if "chat_template.jinja" in files:
        templates, origin = {"default": files["chat_template.jinja"]}, "chat_template.jinja"
        # a list in tokenizer_config can still add named templates next to the .jinja file
        if isinstance(config.get("chat_template"), list):
            extra, named = _templates_from_config_value(config["chat_template"])
            for name, text in extra.items():
                templates.setdefault(name, text)
            named = named and len(extra) > 0
    elif "chat_template" in config:
        templates, named = _templates_from_config_value(config["chat_template"])
        origin = "tokenizer_config.json"
    elif "chat_template.json" in files:
        try:
            inner = json.loads(files["chat_template.json"])
        except json.JSONDecodeError as exc:
            raise SourceError(
                f"{label}: chat_template.json is not valid JSON ({exc.msg} at char {exc.pos})",
                key="err.bad_json", label=label, json_error=exc.msg, char=exc.pos,
            ) from exc
        templates, named = _templates_from_config_value(inner.get("chat_template") if isinstance(inner, dict) else None)
        origin = "chat_template.json"
    if not templates:
        raise SourceError(
            f"{label}: no chat template found (looked for chat_template.jinja, tokenizer_config.json "
            "'chat_template', chat_template.json)",
            key="err.no_template", label=label,
        )
    return ModelSource(
        label=label,
        templates=templates,
        added_tokens=_added_tokens(config),
        has_response_template=("response_template" in config) or ("response_schema" in config),
        named=named,
        origin=origin,
    )


def _read_text(candidate: Path) -> str:
    """Read one model file as UTF-8, turning decoding and OS errors into a SourceError."""
    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError(
            f"{candidate}: not valid UTF-8 text. Save the file as UTF-8 or point to the right model folder.",
            key="err.template_encoding", path=str(candidate),
        ) from exc
    except OSError as exc:
        raise SourceError(
            f"{candidate}: cannot be read ({exc.strerror or exc}).",
            key="err.template_unreadable", path=str(candidate), reason=str(exc.strerror or exc),
        ) from exc


def read_local(path: Path, label: str | None = None) -> ModelSource:
    """Read a model folder."""
    if not path.is_dir():
        raise SourceError(f"{path}: not a directory", key="err.not_dir", path=str(path))
    files: dict[str, str] = {}
    for name in HF_FILES:
        candidate = path / name
        if candidate.is_file():
            files[name] = _read_text(candidate)
    return build_source(label or str(path), files)


def fixture_names() -> list[str]:
    return sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())


def read_fixture(name: str) -> ModelSource:
    folder = FIXTURES / name
    if not folder.is_dir():
        raise SourceError(
            f"unknown fixture '{name}'. Available: {', '.join(fixture_names())}",
            key="err.unknown_fixture", name=name, available=", ".join(fixture_names()),
        )
    return read_local(folder, label=f"fixture:{name}")


def _fetch(url: str, token: str | None, timeout: float) -> str | None:
    """GET a text file. Returns None on 404; raises SourceError(exit 4) otherwise."""
    request = urllib.request.Request(url, headers={"User-Agent": "parserpick"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https/http by design
            return response.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError(
            f"GET {url} returned text that is not valid UTF-8.",
            key="err.template_encoding", path=url,
        ) from exc
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        hint = " (gated or private repo? set HF_TOKEN)" if exc.code in (401, 403) else ""
        raise SourceError(
            f"GET {url} failed with HTTP {exc.code}{hint}",
            EXIT_DOWNLOAD, key="err.http", url=url, status=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SourceError(
            f"GET {url} failed: {reason}. Check the network/proxy, HF_ENDPOINT, or pass a local folder.",
            EXIT_DOWNLOAD, key="err.network", url=url, reason=str(reason),
        ) from exc


def read_hf(spec: str, environ: dict[str, str] | None = None, timeout: float = HF_TIMEOUT) -> ModelSource:
    """Fetch templates for `<org>/<repo>[@rev]` with plain HTTPS GETs."""
    env = os.environ if environ is None else environ
    repo, _, rev = spec.partition("@")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise SourceError(
            f"'{spec}' is not a valid Hugging Face repo id; use hf:<org>/<repo>[@revision]",
            key="err.bad_repo", spec=spec,
        )
    endpoint = env.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    token = env.get("HF_TOKEN") or None
    files: dict[str, str] = {}
    for name in HF_FILES:
        url = f"{endpoint}/{repo}/resolve/{rev or 'main'}/{name}"
        text = _fetch(url, token, timeout)
        if text is not None:
            files[name] = text
    if not files:
        url = f"{endpoint}/{repo}/resolve/{rev or 'main'}/tokenizer_config.json"
        raise SourceError(
            f"GET {url} returned HTTP 404 and no other template file exists; check the repo id/revision.",
            EXIT_DOWNLOAD, key="err.http", url=url, status=404,
        )
    return build_source(f"hf:{spec}", files)


def load_source(spec: str, environ: dict[str, str] | None = None) -> ModelSource:
    """Resolve `fixture:<name>`, `hf:<repo>`, or a local folder path."""
    if spec.startswith("fixture:"):
        return read_fixture(spec[len("fixture:"):])
    if spec.startswith("hf:"):
        return read_hf(spec[len("hf:"):], environ)
    return read_local(Path(spec).expanduser())
