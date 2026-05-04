"""Calibrate a stochastic substitution-cipher search against a
matched-budget shuffle baseline.

The story: when decoded tokens come from a search over a key space,
absolute net_signal can mislead. The search will find local optima on
short ciphers even when no real linguistic signal exists. The fix is
to give the same search procedure the same matched-budget opportunity
on shuffles of the cipher (multiset-preserving permutations) and ask
whether the real cipher's signal exceeds the shuffle distribution.

This worked example sets up a simple monoalphabetic substitution
cipher and a "search" that decrypts with a known-good key. Two cases:

  Case A — real English plaintext encrypted with a known key. The
  search recovers real text. The shuffle baseline does NOT (decrypting
  a shuffled cipher with the same key produces a permutation of the
  plaintext characters, no word boundaries). Calibration shows a
  strongly positive z-score.

  Case B — random letters used as cipher input. The search produces
  gibberish; shuffles of random gibberish also produce gibberish.
  Calibration correctly reports indistinguishability.

The point is not the cryptanalysis itself but the calibration. Run:

    python examples/04_search_calibrated.py
"""

from __future__ import annotations

import random
import string

from dictcollision import search_calibrated_signal


# ---------------------------------------------------------------------------
# Toy plaintext + dictionary
# ---------------------------------------------------------------------------

PLAINTEXT_PARAGRAPH = (
    "the quick brown fox jumps over the lazy dog and then runs back to the "
    "warm old farmhouse where the cat naps on the rug while the kettle "
    "whistles softly and rain falls on the wide green garden the dog "
    "sleeps under the table and the kitten chases shadows across the floor "
    "while the rain taps gently on the window and the wind moves the "
    "branches of the tall old oak tree above the quiet farmhouse"
)

DICTIONARY = set(
    PLAINTEXT_PARAGRAPH.split()
    + ["jumped", "running", "house", "warmth", "kitten", "raining", "barks"]
)


# ---------------------------------------------------------------------------
# Substitution cipher
# ---------------------------------------------------------------------------

SECRET_KEY_SEED = 11


def make_substitution_alphabet(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def apply_alphabet(symbols: list[str], alphabet: dict[str, str]) -> list[str]:
    return [alphabet.get(s, s) for s in symbols]


def encrypt_text(text: str, key: dict[str, str]) -> list[str]:
    """Return cipher as a flat list of symbols (letters and the literal ' ')."""
    return [key.get(ch, ch) for ch in text]


# ---------------------------------------------------------------------------
# A search procedure: decrypt with a known-good key
# ---------------------------------------------------------------------------


def make_known_key_search(decrypt_alphabet: dict[str, str]):
    """Return a search_fn that decrypts with a fixed alphabet and tokenises."""

    def search(cipher_symbols: list[str]) -> list[str]:
        decoded_chars = apply_alphabet(cipher_symbols, decrypt_alphabet)
        text = "".join(decoded_chars)
        return [tok for tok in text.split() if tok]

    return search


# ---------------------------------------------------------------------------
# Two cases
# ---------------------------------------------------------------------------


def case_a_real_plaintext():
    print("=" * 72)
    print("Case A — real English plaintext under a substitution cipher")
    print("=" * 72)

    encrypt_key = make_substitution_alphabet(seed=SECRET_KEY_SEED)
    decrypt_key = {v: k for k, v in encrypt_key.items()}

    cipher = encrypt_text(PLAINTEXT_PARAGRAPH, encrypt_key)
    search = make_known_key_search(decrypt_key)

    decoded = search(cipher)
    print(f"Decoded sample: {' '.join(decoded[:12])!r} ...")
    print()

    result = search_calibrated_signal(
        cipher_symbols=cipher,
        search_fn=search,
        dictionary=DICTIONARY,
        n_shuffles=20,
        base_seed=42,
    )
    print(result.summary())
    print()
    return result


def case_b_random_cipher():
    print("=" * 72)
    print("Case B — random characters as 'cipher' input (no real signal)")
    print("=" * 72)

    rng = random.Random(99)
    cipher = [
        rng.choice(string.ascii_lowercase) if ch != " " else " "
        for ch in PLAINTEXT_PARAGRAPH
    ]

    decrypt_key = {
        v: k
        for k, v in make_substitution_alphabet(seed=SECRET_KEY_SEED).items()
    }
    search = make_known_key_search(decrypt_key)

    decoded = search(cipher)
    print(f"Decoded sample: {' '.join(decoded[:12])!r} ...")
    print()

    result = search_calibrated_signal(
        cipher_symbols=cipher,
        search_fn=search,
        dictionary=DICTIONARY,
        n_shuffles=20,
        base_seed=42,
    )
    print(result.summary())
    print()
    return result


if __name__ == "__main__":
    a = case_a_real_plaintext()
    b = case_b_random_cipher()
    print("=" * 72)
    print("Takeaway:")
    print(
        f"  Case A z-score = {a.z_score:6.2f}  (real text → search finds signal)\n"
        f"  Case B z-score = {b.z_score:6.2f}  (random input → no signal)"
    )
    print()
    print(
        "The shuffle baseline tells you whether your search procedure has\n"
        "found real linguistic structure or a chance pattern with the same\n"
        "character budget. Always calibrate against shuffles when your\n"
        "decoded tokens came from a key-space search."
    )
    print("=" * 72)
