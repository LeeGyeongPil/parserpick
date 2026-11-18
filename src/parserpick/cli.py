"""Command line entry point: inspect, replay, check-template, demo."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from . import __version__, report
from .capture import load_capture
from .checks import has_fail, run_checks
from .i18n import SUPPORTED, resolve_lang, t
from .recommend import ENGINES, recommend
from .replay import (ReplaySetupError, replay, resolve_format, resolve_reasoning,
                     settings_from_recommendation)
from .source import SourceError, load_source

EXIT_OK, EXIT_PROBLEM, EXIT_USAGE = 0, 1, 2


def _prescan_lang(argv: Sequence[str]) -> str | None:
    """Find --lang before argparse runs, so --help itself can be localized."""
    for index, token in enumerate(argv):
        if token == "--lang" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--lang="):
            return token.split("=", 1)[1]
    return None


def _formatter_for(lang: str) -> type[argparse.HelpFormatter]:
    """A help formatter whose 'usage:' prefix is localized (argparse hard-codes it in English)."""

    class LocalizedFormatter(argparse.HelpFormatter):
        def add_usage(self, usage, actions, groups, prefix=None):  # type: ignore[no-untyped-def]
            return super().add_usage(usage, actions, groups,
                                     t(lang, "help.usage_prefix") if prefix is None else prefix)

    return LocalizedFormatter


# argparse 가 영어로 박아 둔 오류 문구(gettext msgid) → 메시지 표 키.
# 목록에 없는 문구는 영어 그대로 나온다(파이썬 버전마다 문구가 조금씩 다르다).
_ARGPARSE_KEYS = {
    "%(prog)s: error: %(message)s\n": "argparse.error",
    "argument %(argument_name)s: %(message)s": "argparse.argument",
    "the following arguments are required: %s": "argparse.required",
    "unrecognized arguments: %s": "argparse.unrecognized",
    "invalid choice: %(value)r (choose from %(choices)s)": "argparse.invalid_choice",
    "invalid choice: %(value)r, maybe you meant %(closest)r? (choose from %(choices)s)":
        "argparse.invalid_choice_suggest",
    "invalid %(type)s value: %(value)r": "argparse.invalid_value",
    "expected one argument": "argparse.expected_one",
    "expected at most one argument": "argparse.expected_at_most_one",
    "expected at least one argument": "argparse.expected_at_least_one",
    "expected %s argument": "argparse.expected_n_one",
    "expected %s arguments": "argparse.expected_n",
    "ambiguous option: %(option)s could match %(matches)s": "argparse.ambiguous",
    "ignored explicit argument %r": "argparse.ignored_explicit",
    "not allowed with argument %s": "argparse.not_allowed_with",
    "one of the arguments %s is required": "argparse.one_required",
    "unexpected option string: %s": "argparse.unexpected_option",
    "unknown parser %(parser_name)r (choices: %(choices)s)": "argparse.unknown_parser",
}


@contextmanager
def _argparse_messages(lang: str) -> Iterator[None]:
    """Route argparse's own error messages through the message table while parsing.

    argparse looks its English strings up through module-level gettext functions (`_`, `ngettext`)
    at call time, so swapping those two names for the duration of parse_args() localizes errors
    such as a missing required argument. The originals are always restored.
    """
    original_gettext, original_ngettext = argparse._, argparse.ngettext  # type: ignore[attr-defined]

    def gettext(message: str) -> str:
        key = _ARGPARSE_KEYS.get(message)
        return t(lang, key) if key else original_gettext(message)

    def ngettext(singular: str, plural: str, count: int) -> str:
        key = _ARGPARSE_KEYS.get(singular if count == 1 else plural)
        return t(lang, key) if key else original_ngettext(singular, plural, count)

    argparse._ = gettext  # type: ignore[attr-defined]
    argparse.ngettext = ngettext  # type: ignore[attr-defined]
    try:
        yield
    finally:
        argparse._ = original_gettext  # type: ignore[attr-defined]
        argparse.ngettext = original_ngettext  # type: ignore[attr-defined]


def _localize(parser: argparse.ArgumentParser, lang: str) -> argparse.ArgumentParser:
    """Replace argparse's built-in English group titles and -h text with the message table."""
    parser._positionals.title = t(lang, "help.positionals")  # noqa: SLF001 - no public API for this
    parser._optionals.title = t(lang, "help.options")  # noqa: SLF001
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=t(lang, "help.help"))
    return parser


def build_parser(lang: str) -> argparse.ArgumentParser:
    fmt = _formatter_for(lang)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", choices=SUPPORTED, default=argparse.SUPPRESS, help=t(lang, "help.lang"))
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help=t(lang, "help.json"))

    parser = argparse.ArgumentParser(
        prog="parserpick", description=t(lang, "app.description"), epilog=t(lang, "help.epilog"),
        parents=[common], add_help=False, formatter_class=fmt,
    )
    _localize(parser, lang)
    parser.add_argument("--version", action="version", version=f"parserpick {__version__}",
                        help=t(lang, "help.version"))
    sub = parser.add_subparsers(dest="command", required=True, title=t(lang, "help.commands"),
                                metavar="{inspect,replay,check-template,demo}")

    inspect_p = sub.add_parser("inspect", parents=[common], add_help=False, formatter_class=fmt, help=t(lang, "help.cmd.inspect"),
                               description=t(lang, "help.cmd.inspect"))
    _localize(inspect_p, lang)
    inspect_p.add_argument("source", metavar="SOURCE", help=t(lang, "help.source"))
    inspect_p.add_argument("--engine", choices=ENGINES, default="vllm", help=t(lang, "help.engine"))
    inspect_p.add_argument("--engine-version", metavar="VERSION", help=t(lang, "help.engine_version"))

    replay_p = sub.add_parser("replay", parents=[common], add_help=False, formatter_class=fmt, help=t(lang, "help.cmd.replay"),
                              description=t(lang, "help.cmd.replay"))
    _localize(replay_p, lang)
    replay_p.add_argument("capture", metavar="CAPTURE", help=t(lang, "help.capture"))
    replay_p.add_argument("--format", metavar="NAME", help=t(lang, "help.format"))
    replay_p.add_argument("--reasoning", metavar="NAME", help=t(lang, "help.reasoning"))
    replay_p.add_argument("--model", metavar="SOURCE", help=t(lang, "help.model"))
    replay_p.add_argument("--engine", choices=ENGINES, default="vllm", help=t(lang, "help.engine"))

    check_p = sub.add_parser("check-template", parents=[common], add_help=False, formatter_class=fmt, help=t(lang, "help.cmd.check"),
                             description=t(lang, "help.cmd.check"))
    _localize(check_p, lang)
    check_p.add_argument("source", metavar="SOURCE", help=t(lang, "help.source"))

    demo_p = sub.add_parser("demo", parents=[common], add_help=False, formatter_class=fmt,
                            help=t(lang, "help.cmd.demo"), description=t(lang, "help.cmd.demo"))
    _localize(demo_p, lang)
    return parser


def _fail(lang: str, key: str, code: int, **params: object) -> int:
    print(t(lang, "err.prefix", message=t(lang, key, **params)), file=sys.stderr)
    return code


def _source_error(lang: str, exc: SourceError) -> int:
    message = t(lang, exc.key, **exc.params) if exc.key else str(exc)
    print(t(lang, "err.prefix", message=message), file=sys.stderr)
    return exc.exit_code


def run_inspect(args: argparse.Namespace, lang: str, environ: Mapping[str, str]) -> int:
    source = load_source(args.source, dict(environ))
    rec = recommend(source, args.engine, args.engine_version)
    print(report.dumps(report.inspect_dict(rec, lang)) if getattr(args, "json", False)
          else report.inspect_text(rec, lang))
    return EXIT_PROBLEM if rec.has_error else EXIT_OK


def run_replay(args: argparse.Namespace, lang: str, environ: Mapping[str, str]) -> int:
    if not args.format and not args.model:
        return _fail(lang, "err.need_format", EXIT_USAGE)
    text = load_capture(args.capture)
    family, rule = "", "none"
    try:
        if args.model:
            rec = recommend(load_source(args.model, dict(environ)), args.engine)
            family, rule = settings_from_recommendation(rec)
        if args.format:
            family = resolve_format(args.format)
        if args.reasoning:
            rule = resolve_reasoning(args.reasoning)
    except ReplaySetupError as exc:
        return _fail(lang, exc.key, EXIT_USAGE, **exc.params)
    result = replay(text, family, rule, args.engine)
    print(report.dumps(report.replay_dict(result, lang)) if getattr(args, "json", False)
          else report.replay_text(result, lang, args.engine))
    return EXIT_PROBLEM if result.has_error else EXIT_OK


def run_check(args: argparse.Namespace, lang: str, environ: Mapping[str, str]) -> int:
    source = load_source(args.source, dict(environ))
    results = run_checks(source.tool_template)
    name = source.tool_template_name
    print(report.dumps(report.checks_dict(source.label, name, results, lang)) if getattr(args, "json", False)
          else report.checks_text(source.label, name, results, lang))
    return EXIT_PROBLEM if has_fail(results) else EXIT_OK


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    lang = resolve_lang(_prescan_lang(args_list), env)
    with _argparse_messages(lang):
        parser = build_parser(lang)
        args = parser.parse_args(args_list)
    lang = resolve_lang(getattr(args, "lang", None), env)
    try:
        if args.command == "inspect":
            return run_inspect(args, lang, env)
        if args.command == "replay":
            return run_replay(args, lang, env)
        if args.command == "check-template":
            return run_check(args, lang, env)
        from .demo import run_demo  # local import keeps `--help` fast

        return run_demo(lang, as_json=getattr(args, "json", False))
    except SourceError as exc:
        return _source_error(lang, exc)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
