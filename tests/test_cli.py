"""Tests for the command-line interface."""

import json
from pathlib import Path

from dictcollision.__main__ import main


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_cli_report(tmp_path: Path, capsys):
    tok_file = _write(tmp_path, "tokens.txt", "the cat sat on the mat " * 30)
    dict_file = _write(tmp_path, "dict.txt", "the\ncat\nsat\non\nmat\ndog\n")

    rc = main(["--tokens", str(tok_file), "--dict", str(dict_file),
               "--n-nulls", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ClassifyResult" in out
    assert "net signal" in out


def test_cli_json(tmp_path: Path, capsys):
    tok_file = _write(tmp_path, "tokens.txt", "ab cd ef gh " * 40)
    dict_file = _write(tmp_path, "dict.txt", "ab\ncd\nef\n")

    rc = main(["--tokens", str(tok_file), "--dict", str(dict_file),
               "--n-nulls", "3", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "result" in data
    assert "net_signal" in data["result"]


def test_cli_baselines(tmp_path: Path, capsys):
    tok_file = _write(tmp_path, "tokens.txt", "ab cd ef " * 50)
    dict_file = _write(tmp_path, "dict.txt", "ab\ncd\n")

    rc = main(["--tokens", str(tok_file), "--dict", str(dict_file),
               "--n-nulls", "3", "--baselines"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "four_category_net" in out


def test_cli_noise_only(tmp_path: Path, capsys):
    tok_file = _write(tmp_path, "tokens.txt", "ab cd " * 20)
    dict_file = _write(tmp_path, "dict.txt", "ab\ncd\nef\n")

    rc = main(["--tokens", str(tok_file), "--dict", str(dict_file),
               "--noise-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Predicted collision rate" in out


def test_cli_missing_token_file(tmp_path: Path, capsys):
    dict_file = _write(tmp_path, "dict.txt", "hello\n")
    rc = main(["--tokens", str(tmp_path / "nope.txt"),
               "--dict", str(dict_file)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "No such file" in err or "error" in err
