"""Pick a dictionary using the recommender (paper Section 12).

When you do not know the plaintext language, the recommender ranks
candidate dictionaries by excess hit rate above the predicted noise
floor. The top-ranked dictionary is the one most likely to match the
encoding.

Run:
    python examples/03_dictionary_recommender.py
"""

from __future__ import annotations

from dictcollision import recommend

# Pretend-decoded tokens that happen to be Italian words.
tokens = (
    "il gatto mangia il pesce il cane corre nel parco la donna legge il "
    "libro sotto il sole i bambini giocano con la palla nella piazza"
).split() * 30

candidates: dict[str, set[str]] = {
    "italian_small": {
        "il", "gatto", "mangia", "pesce", "cane", "corre", "nel",
        "parco", "donna", "legge", "libro", "sotto", "sole",
    },
    "italian_large": {
        "il", "gatto", "mangia", "pesce", "cane", "corre", "nel",
        "parco", "donna", "legge", "libro", "sotto", "sole", "bambini",
        "giocano", "con", "palla", "nella", "piazza", "casa", "albero",
        "fiore", "uccello", "montagna", "mare", "fiume", "strada",
    },
    "english": {
        "the", "cat", "eats", "fish", "dog", "runs", "in", "park",
        "woman", "reads", "book", "under", "sun", "children", "play",
    },
    "german": {
        "der", "die", "das", "und", "ist", "in", "auf", "mit", "von",
        "nicht", "ein", "zu", "sich", "nach",
    },
}

print("Ranked by excess (observed - predicted):")
for r in recommend(tokens, candidates, objective="excess"):
    print("  " + r.summary())

print("\nRanked by SNR (observed / predicted):")
for r in recommend(tokens, candidates, objective="snr"):
    print("  " + r.summary())
