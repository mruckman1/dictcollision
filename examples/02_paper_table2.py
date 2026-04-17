"""Reproduce the shape of Table 2 from Ruckman (2026).

Shows how the six correction methods differ. Builds two scenarios:

    1. A real decoded corpus (long English-like words) against an
       English dictionary. All methods agree.
    2. The same decoded corpus against a large "wrong-language"
       dictionary that happens to share the character set. This is
       where single-word tests (permutation / BH-FDR / BLAST) fail:
       they accept rare chance matches as signal while the
       four-category framework correctly subtracts them.

Run:
    python examples/02_paper_table2.py
"""

from __future__ import annotations

import random

from dictcollision.baselines import all_methods


def main() -> None:
    rng = random.Random(42)

    # A short synthetic "English-ish" corpus.
    vocab = [
        "the", "cat", "dog", "runs", "jumps", "sits", "bird", "fish",
        "tree", "leaf", "rain", "sun", "moon", "star", "wind", "fire",
    ]
    tokens = [rng.choice(vocab) for _ in range(1500)]

    right_dict = set(vocab)  # exactly the words used

    # Wrong-language dictionary: large, random 3- to 5-letter strings from
    # the same alphabet as the tokens. Shares character statistics so
    # chance collisions are non-trivial.
    chars = "abcdefghijklmnopqrstuvwxyz"
    wrong_dict = set()
    while len(wrong_dict) < 5000:
        L = rng.choice([3, 4, 5])
        wrong_dict.add("".join(rng.choice(chars) for _ in range(L)))

    for label, dictionary in [
        ("Correct dictionary", right_dict),
        ("Large wrong-language dictionary (same alphabet)", wrong_dict),
    ]:
        print(f"\n=== {label} (dict size {len(dictionary)}) ===")
        methods = all_methods(tokens, dictionary, n_nulls=5)
        name_w = max(len(k) for k in methods)
        for name, val in methods.items():
            print(f"  {name:<{name_w}}  {val:>7.1%}")

    print(
        "\nInterpretation: on the wrong-language dictionary, apparent_hit_rate\n"
        "and the single-word tests (permutation_test, bh_fdr, blast_evalue)\n"
        "still claim non-trivial signal driven by chance character collisions.\n"
        "four_category_net is the only method that calibrates properly."
    )


if __name__ == "__main__":
    main()
