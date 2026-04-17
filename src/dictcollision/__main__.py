"""Command-line interface for dictcollision.

Usage:

    python -m dictcollision --tokens decoded.txt --dict latin_50k.txt
    python -m dictcollision --tokens decoded.txt --dict latin.txt --baselines
    python -m dictcollision --tokens decoded.txt --dict latin.txt --json

Input files:

    --tokens  PATH  whitespace-split text file of decoded tokens
                    (use --delimiter to change)
    --dict    PATH  one-word-per-line or "word count" dictionary file

See `python -m dictcollision --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys

from dictcollision import (
    __version__,
    classify,
    classify_by_length,
    noise_floor,
)
from dictcollision._io import load_dictionary, load_tokens


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dictcollision",
        description=(
            "Calibrate dictionary hit rates: separate real matches from "
            "chance collisions. Takes a token file and a dictionary file, "
            "runs the four-category framework, prints an interpretation."
        ),
    )
    p.add_argument("--tokens", required=True, help="Path to decoded-tokens file.")
    p.add_argument("--dict", required=True, dest="dictionary",
                   help="Path to dictionary file.")
    p.add_argument("--delimiter", default=None,
                   help="Token delimiter (default: whitespace).")
    p.add_argument("--encoding", default="utf-8", help="File encoding.")
    p.add_argument("--lowercase", action="store_true",
                   help="Lowercase both tokens and dictionary before matching.")
    p.add_argument("--min-length", type=int, default=1,
                   help="Skip dictionary entries shorter than this (default 1).")
    p.add_argument("--max-dict", type=int, default=None,
                   help="Truncate dictionary to the first N entries.")
    p.add_argument("--n-nulls", type=int, default=5,
                   help="Null corpora per classify (default 5).")
    p.add_argument("--threshold", type=int, default=2,
                   help="Min nulls a word must appear in to be 'shared' (default 2).")
    p.add_argument("--null-model", choices=["unigram", "bigram", "trigram"],
                   default="bigram",
                   help="N-gram order for the null model (default bigram).")
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--baselines", action="store_true",
                   help="Run all paper baselines and print Table-2-style comparison.")
    p.add_argument("--by-length", action="store_true",
                   help="Print per-length-bucket breakdown (paper Section 10).")
    p.add_argument("--noise-only", action="store_true",
                   help="Only compute the analytical noise floor and exit.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a report.")
    p.add_argument("--version", action="version", version=f"dictcollision {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        tokens = load_tokens(
            args.tokens,
            encoding=args.encoding,
            delimiter=args.delimiter,
            lowercase=args.lowercase,
        )
        dictionary = load_dictionary(
            args.dictionary,
            encoding=args.encoding,
            min_length=args.min_length,
            max_entries=args.max_dict,
            lowercase=args.lowercase,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not tokens:
        print("error: token file is empty", file=sys.stderr)
        return 2
    if not dictionary:
        print("error: dictionary is empty", file=sys.stderr)
        return 2

    if args.noise_only:
        r = noise_floor(tokens, dictionary)
        if args.json:
            print(json.dumps({
                "noise_floor": r,
                "n_tokens": len(tokens),
                "n_dict_entries": len(dictionary),
            }, indent=2))
        else:
            print(f"Predicted collision rate: {r:.1%}")
            print(f"  (n_tokens={len(tokens)}, dict_size={len(dictionary)})")
        return 0

    result = classify(
        tokens,
        dictionary,
        n_nulls=args.n_nulls,
        threshold=args.threshold,
        base_seed=args.seed,
        null_model=args.null_model,
    )

    if args.json:
        out: dict = {
            "input": {
                "n_tokens": len(tokens),
                "n_dict_entries": len(dictionary),
                "null_model": args.null_model,
                "n_nulls": args.n_nulls,
                "threshold": args.threshold,
            },
            "result": result.to_dict(),
        }
        if args.baselines:
            from dictcollision.baselines import all_methods
            out["baselines"] = all_methods(
                tokens, dictionary, n_nulls=args.n_nulls, base_seed=args.seed
            )
        if args.by_length:
            buckets = classify_by_length(
                tokens,
                dictionary,
                n_nulls=args.n_nulls,
                threshold=args.threshold,
                base_seed=args.seed,
                null_model=args.null_model,
            )
            out["by_length"] = [b.to_dict() for b in buckets]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"dictcollision {__version__}")
    print(f"tokens: {args.tokens}  (n={len(tokens)})")
    print(f"dict:   {args.dictionary}  (n={len(dictionary)})")
    print(f"null_model={args.null_model}  n_nulls={args.n_nulls}  threshold={args.threshold}")
    print()
    print(result.summary())

    if args.baselines:
        from dictcollision.baselines import all_methods
        print()
        print("Baseline comparison (paper Table 2 style):")
        methods = all_methods(
            tokens, dictionary, n_nulls=args.n_nulls, base_seed=args.seed
        )
        name_w = max(len(k) for k in methods)
        for name, val in methods.items():
            print(f"  {name:<{name_w}}  {val:>7.1%}")

    if args.by_length:
        buckets = classify_by_length(
            tokens,
            dictionary,
            n_nulls=args.n_nulls,
            threshold=args.threshold,
            base_seed=args.seed,
            null_model=args.null_model,
        )
        print()
        print("By-length breakdown (paper Section 10):")
        print(f"  {'len':>4}  {'n':>6}  {'apparent':>8}  {'net':>7}  {'correction':>10}")
        for b in buckets:
            print(
                f"  {b.length:>4}  {b.n_tokens:>6}  "
                f"{b.apparent_hit_rate:>7.1%}  {b.net_signal:>6.1%}  "
                f"{b.correction:>9.1%}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
