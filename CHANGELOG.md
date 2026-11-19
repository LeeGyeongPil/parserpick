# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-11-19

### Fixed
- Tests: the English-output check no longer trips on OS-provided errno messages.

## [0.1.1] - 2026-11-19

### Fixed
- `hf:` sources: a download cut short by the server now exits with code 4 and the network message instead of a traceback.

## [0.1.0] - 2026-11-19

### Added
- `parserpick inspect <source>`: judges the tool-call format from chat-template and special-token
  markers (not the model name) and prints `--tool-call-parser` / `--reasoning-parser` /
  `--chat-template` flags for vLLM or SGLang, with warnings for templates that never render tools,
  named `tool_use` templates, declared `response_template`, unknown formats and unverified engine versions.
- Printed flag lines are combinations the engine accepts at startup, checked in tests against the
  registered parser names and startup checks of vLLM 0.34.0 and SGLang 0.5.23: on vLLM
  `--enable-auto-tool-choice` appears only together with `--tool-call-parser`; a format with no
  documented parser name for the engine (MiniMax on both, Granite on SGLang) gets a `no-engine-parser`
  warning and no tool-call flags; a named `tool_use` template is selected with SGLang's
  `--hf-chat-template-name` (vLLM picks it by itself for requests that send tools) rather than
  `--chat-template tool_use`, which both engines refuse at startup.
- `parserpick replay <capture>`: replays captured raw output through reasoning split then tool
  extraction for nine formats (hermes, qwen3_xml, glm45, llama3_json, pythonic, mistral,
  deepseek_v3/v31, kimi_k2) and explains leaks with `call-inside-reasoning`, `parser-mismatch`,
  `broken-arguments`, `unclosed-call`, `no-call` and `text-mixed-pythonic`.
- `parserpick check-template <source>`: five sandboxed render checks (tools rendered, dict arguments,
  string arguments, OpenAI-style call IDs, tool role) using Jinja's immutable sandbox, rendered in a
  separate process with a shared 5-second budget (7-second hard kill); a template that fails to render
  at all is reported as one rendering failure on every check.
- `parserpick demo`: offline walkthrough on built-in synthetic practice fixtures.
- Sources: local model folders, `hf:<org>/<repo>[@rev]` (plain HTTPS, `HF_TOKEN`, `HF_ENDPOINT`) and
  `fixture:<name>`.
- `--json` output with language-independent codes; `--lang en|ko` (also `PARSERPICK_LANG`, `LC_ALL`,
  `LC_MESSAGES`, `LANG`).
- Format table in `data/formats.toml` with engine parser names verified against vLLM 0.34.0 and
  SGLang 0.5.23 documentation on 2026-11-19.
