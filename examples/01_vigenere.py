"""Evaluate a Vigenere candidate key against a Latin dictionary.

Simulates the scenario from Section 9.2 of Ruckman (2026): encrypt a
plaintext with one key, decrypt with a candidate, ask whether the
dictionary hit rate is genuine or chance.

Run:
    python examples/01_vigenere.py
"""

from __future__ import annotations

import random

from dictcollision import classify, noise_floor


def vigenere(text: str, key: str, encrypt: bool = True) -> str:
    out = []
    k = key.lower()
    idx = 0
    for ch in text.lower():
        if ch.isalpha():
            shift = ord(k[idx % len(k)]) - ord("a")
            if not encrypt:
                shift = -shift
            out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
            idx += 1
        else:
            out.append(ch)
    return "".join(out)


# Tiny synthetic Latin-ish corpus.
PLAINTEXT = (
    "gallia est omnis divisa in partes tres quarum unam incolunt belgae "
    "aliam aquitani tertiam qui ipsorum lingua celtae nostra galli "
    "appellantur hi omnes lingua institutis legibus inter se differunt "
) * 20

DICTIONARY = {
    "gallia", "est", "omnis", "divisa", "in", "partes", "tres",
    "quarum", "unam", "incolunt", "belgae", "aliam", "aquitani",
    "tertiam", "qui", "ipsorum", "lingua", "celtae", "nostra", "galli",
    "appellantur", "hi", "omnes", "institutis", "legibus", "inter",
    "se", "differunt", "atque", "inde", "prope", "inquit",
}

CORRECT_KEY = "salve"
WRONG_KEY = "bogus"


def evaluate(label: str, key: str):
    ciphertext = vigenere(PLAINTEXT, CORRECT_KEY, encrypt=True)
    decoded = vigenere(ciphertext, key, encrypt=False)
    tokens = decoded.split()

    predicted = noise_floor(tokens, DICTIONARY)
    result = classify(tokens, DICTIONARY, n_nulls=5)

    print(f"\n=== {label} (key={key!r}) ===")
    print(f"n_tokens={len(tokens)}  predicted_noise={predicted:.1%}")
    print(result.summary())


if __name__ == "__main__":
    random.seed(0)
    evaluate("Correct key", CORRECT_KEY)
    evaluate("Wrong key", WRONG_KEY)
