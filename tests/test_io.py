"""Tests for file loaders."""

from pathlib import Path

from dictcollision import load_dictionary, load_tokens


def test_load_dictionary_plain(tmp_path: Path):
    p = tmp_path / "dict.txt"
    p.write_text("hello\nworld\n# comment\n\nfoo\n", encoding="utf-8")
    words = load_dictionary(p)
    assert words == {"hello", "world", "foo"}


def test_load_dictionary_hermitdave_format(tmp_path: Path):
    p = tmp_path / "freq.txt"
    p.write_text("the 12345\nand 9876\nof 5432\n", encoding="utf-8")
    words = load_dictionary(p)
    assert words == {"the", "and", "of"}


def test_load_dictionary_csv(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("hello,5\nworld,3\n", encoding="utf-8")
    assert load_dictionary(p) == {"hello", "world"}


def test_load_dictionary_min_length(tmp_path: Path):
    p = tmp_path / "d.txt"
    p.write_text("a\nab\nabc\nabcd\n", encoding="utf-8")
    assert load_dictionary(p, min_length=3) == {"abc", "abcd"}


def test_load_dictionary_max_entries(tmp_path: Path):
    p = tmp_path / "d.txt"
    p.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert load_dictionary(p, max_entries=2, min_length=0) == {"a", "b"}


def test_load_dictionary_lowercase(tmp_path: Path):
    p = tmp_path / "d.txt"
    p.write_text("HELLO\nWorld\n", encoding="utf-8")
    assert load_dictionary(p, lowercase=True) == {"hello", "world"}


def test_load_tokens_whitespace(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text("the quick brown\nfox jumps\n", encoding="utf-8")
    assert load_tokens(p) == ["the", "quick", "brown", "fox", "jumps"]


def test_load_tokens_delimiter(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text("ab.cd.ef\ngh.ij\n", encoding="utf-8")
    assert load_tokens(p, delimiter=".") == ["ab", "cd", "ef", "gh", "ij"]


def test_load_tokens_lowercase(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text("ABC def\n", encoding="utf-8")
    assert load_tokens(p, lowercase=True) == ["abc", "def"]
