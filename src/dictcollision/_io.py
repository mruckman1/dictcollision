"""File loaders for dictionaries and decoded token streams.

Handles the common formats researchers actually use:

    - one-word-per-line (.txt)
    - hermitdave FrequencyWords format:  word<space>count
    - two-column CSV/TSV (word, count) with optional header

Encoding defaults to UTF-8 and can be overridden. Blank lines and lines
starting with '#' are skipped.
"""

from __future__ import annotations

from pathlib import Path


def _looks_like_count(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def load_dictionary(
    path: str | Path,
    encoding: str = "utf-8",
    min_length: int = 1,
    max_entries: int | None = None,
    lowercase: bool = False,
) -> set[str]:
    """Load a dictionary file into a set of words.

    Accepted formats (auto-detected per line):

    - Single token per line: ``hello``
    - Token + count (space or tab separated): ``hello 42``
    - Token + count, comma-separated: ``hello,42``

    Lines that are blank or start with ``#`` are skipped.

    Parameters
    ----------
    path : str or Path
    encoding : str
        File encoding (default "utf-8").
    min_length : int
        Skip entries shorter than this many characters (default 1).
    max_entries : int or None
        Truncate to the first N valid entries. Useful for "slice the
        frequency list to the top N" workflows (see hermitdave's
        ``full.txt`` files).
    lowercase : bool
        Lowercase each entry before adding to the set.

    Returns
    -------
    set[str]
    """
    p = Path(path)
    words: set[str] = set()
    n_added = 0
    for raw in p.read_text(encoding=encoding).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Try comma split first, then whitespace
        if "," in line:
            parts = [s.strip() for s in line.split(",")]
        else:
            parts = line.split()

        if not parts:
            continue

        # If second field looks like a count, take first field
        if len(parts) >= 2 and _looks_like_count(parts[1]):
            w = parts[0]
        else:
            w = parts[0]

        if lowercase:
            w = w.lower()
        if len(w) < min_length:
            continue

        if w not in words:
            words.add(w)
            n_added += 1
            if max_entries is not None and n_added >= max_entries:
                break

    return words


def load_tokens(
    path: str | Path,
    encoding: str = "utf-8",
    delimiter: str | None = None,
    lowercase: bool = False,
) -> list[str]:
    """Load a decoded-token stream from a text file.

    Parameters
    ----------
    path : str or Path
    encoding : str
    delimiter : str or None
        If None, tokens are whitespace-split (default). Pass a literal
        delimiter (e.g. "." for dot-separated cipher codes, "," for
        CSV-like single-row streams) to split on that instead.
    lowercase : bool

    Returns
    -------
    list[str]
    """
    p = Path(path)
    text = p.read_text(encoding=encoding)
    if delimiter is None:
        tokens = text.split()
    else:
        # Treat the entire file as one long delimited stream, but also
        # honor newlines so a multi-line file splits sensibly.
        parts: list[str] = []
        for line in text.splitlines():
            parts.extend(line.split(delimiter))
        tokens = [t.strip() for t in parts if t.strip()]

    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens
