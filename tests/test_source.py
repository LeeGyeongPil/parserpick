"""AC-10: local folders and hf: sources without real network access."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from parserpick.cli import main
from parserpick.source import SourceError, load_source, read_hf


def test_local_dir_variants(tmp_path):
    # chat_template as a plain string
    a = tmp_path / "a"
    a.mkdir()
    (a / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "STRING {{ tools }}", "added_tokens_decoder": {"1": {"content": "<tool_call>"}}}))
    src = load_source(str(a))
    assert src.templates == {"default": "STRING {{ tools }}"} and not src.named
    assert src.added_tokens == ["<tool_call>"]

    # chat_template as a named list
    b = tmp_path / "b"
    b.mkdir()
    (b / "tokenizer_config.json").write_text(json.dumps({"chat_template": [
        {"name": "default", "template": "D"}, {"name": "tool_use", "template": "T {{ tools }}"}]}))
    src = load_source(str(b))
    assert src.named and src.tool_template_name == "tool_use" and src.tool_template == "T {{ tools }}"

    # chat_template.jinja wins over tokenizer_config
    c = tmp_path / "c"
    c.mkdir()
    (c / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))
    (c / "chat_template.jinja").write_text("NEW {{ tools }}")
    assert load_source(str(c)).templates["default"] == "NEW {{ tools }}"

    # chat_template.json is read when nothing else exists
    d = tmp_path / "d"
    d.mkdir()
    (d / "chat_template.json").write_text(json.dumps({"chat_template": "JSON {{ tools }}"}))
    assert load_source(str(d)).templates["default"] == "JSON {{ tools }}"

    # response_template flag
    e = tmp_path / "e"
    e.mkdir()
    (e / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x", "response_template": {}}))
    assert load_source(str(e)).has_response_template


def test_local_dir_errors(tmp_path):
    with pytest.raises(SourceError) as exc:
        load_source(str(tmp_path))
    assert exc.value.exit_code == 2 and "no chat template" in str(exc.value)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "tokenizer_config.json").write_text("{not json")
    with pytest.raises(SourceError) as exc:
        load_source(str(bad))
    assert exc.value.exit_code == 2


class _Hub(BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []
    files = {
        "/org/model/resolve/main/tokenizer_config.json": json.dumps({
            "chat_template": "{% if tools %}<tool_call>{{ tools }}</tool_call>{% endif %}",
            "added_tokens_decoder": {"1": {"content": "<tool_call>"}}}),
    }

    def do_GET(self):  # noqa: N802
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        body = self.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture
def hub():
    _Hub.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Hub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_hf_fetch_via_local_server(hub):
    src = read_hf("org/model", {"HF_ENDPOINT": hub, "HF_TOKEN": "secret-token"})
    assert src.label == "hf:org/model"
    assert "<tool_call>" in src.templates["default"]
    paths = [p for p, _ in _Hub.seen]
    assert "/org/model/resolve/main/tokenizer_config.json" in paths
    assert all(auth == "Bearer secret-token" for _, auth in _Hub.seen)
    # missing optional files (chat_template.jinja, chat_template.json) are simply skipped
    assert "/org/model/resolve/main/chat_template.jinja" in paths


def test_hf_fetch_via_cli(hub, capsys, monkeypatch):
    code = main(["inspect", "hf:org/model", "--json"], environ={"HF_ENDPOINT": hub})
    assert code == 0
    assert json.loads(capsys.readouterr().out)["family"] == "hermes"


def test_hf_fetch_error_exit_4(hub, capsys):
    with pytest.raises(SourceError) as exc:
        read_hf("org/missing", {"HF_ENDPOINT": hub})
    assert exc.value.exit_code == 4
    assert "404" in str(exc.value) and "/org/missing/resolve/main/" in str(exc.value)
    code = main(["inspect", "hf:org/missing"], environ={"HF_ENDPOINT": hub})
    assert code == 4
    err = capsys.readouterr().err
    assert "404" in err and hub in err


def test_hf_connection_failure_exit_4(capsys):
    # nothing listens on this port: the message names the URL and the reason
    code = main(["inspect", "hf:org/model"], environ={"HF_ENDPOINT": "http://127.0.0.1:9"})
    assert code == 4
    assert "http://127.0.0.1:9/org/model" in capsys.readouterr().err


def test_hf_bad_repo_id():
    with pytest.raises(SourceError) as exc:
        read_hf("just-a-name", {})
    assert exc.value.exit_code == 2


def test_non_utf8_template_exits_2_with_message(tmp_path, capsys):
    """Q-4: undecodable template files are an input error (exit 2), not a traceback."""
    (tmp_path / "chat_template.jinja").write_bytes(b"\xff\xfe\x00bad")
    assert main(["inspect", str(tmp_path)], environ={}) == 2
    err = capsys.readouterr().err
    assert "UTF-8" in err and "chat_template.jinja" in err and "Traceback" not in err
    assert main(["--lang", "ko", "inspect", str(tmp_path)], environ={}) == 2
    assert "UTF-8" in capsys.readouterr().err
    with pytest.raises(SourceError):
        load_source(str(tmp_path))
