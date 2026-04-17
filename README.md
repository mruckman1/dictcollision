# dictcollision

[![PyPI](https://img.shields.io/pypi/v/dictcollision)](https://pypi.org/project/dictcollision/)
[![Python](https://img.shields.io/pypi/pyversions/dictcollision)](https://pypi.org/project/dictcollision/)
[![License](https://img.shields.io/pypi/l/dictcollision)](https://github.com/mruckman1/dictcollision/blob/main/LICENSE)

**Calibrate dictionary hit rates.** Given a list of short strings and a
reference dictionary, separate real matches from chance collisions.

---

## The general problem

You have a stream of short strings and a big reference dictionary. Some
fraction match. How many are **real** matches vs. the dictionary being
large enough that anything would match?

| Domain | "Decoded tokens" | "Dictionary" | "Real signal" means |
|---|---|---|---|
| **Decipherment / cryptanalysis** | candidate plaintext | language wordlist | the decode works |
| **OCR validation** | extracted strings | lexicon | the OCR read correctly |
| **Spell-check eval** | candidate corrections | target vocabulary | the correction fired |
| **Autocomplete ranking** | prefix expansions | vocab | the candidate is meaningful |
| **Password audit** | cracked-string attempts | common-words list | the password was weak, not a random collision |
| **Fuzzy dedup** | near-match candidates | canonical set | they are actually duplicates |
| **RNG / fuzzer QA** | generated strings | wordlist | did your generator accidentally emit real words? |

If the input is short (2–4 chars) and the dictionary is large (10K+),
naive hit-rate metrics are badly inflated by chance. This package fixes
that.

## Install

```bash
pip install dictcollision
pip install "dictcollision[viz]"   # with matplotlib
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dictcollision                 # into a uv project
uv pip install dictcollision         # into the active venv
uv tool install dictcollision        # install the CLI globally
```

## Quick start

```python
from dictcollision import noise_floor, classify, recommend

# 1. One-line prediction: how much of my 43% hit rate is chance?
predicted = noise_floor(decoded_tokens, dictionary)
print(f"Chance alone predicts {predicted:.1%}; observed 43.0%; "
      f"excess {0.43 - predicted:.1%}")

# 2. Full four-category analysis.
result = classify(decoded_tokens, dictionary)
print(result.summary())

# 3. Rank candidate dictionaries when you don't know the language.
ranked = recommend(decoded_tokens,
                   {"latin_10k": latin_words, "german_50k": german_words})
print(ranked[0].name, ranked[0].excess)
```

`result.summary()` prints:

```
ClassifyResult (n=5000 tokens)
  apparent hit rate :  99.6%
  net signal        :  70.1%   <- calibrated metric
  correction        :  29.5%   <- amount subtracted

  signal       70.1%  ████████████████░░░░░░░  real matches
  shared_hit   19.4%  ████░░░░░░░░░░░░░░░░░░░  chance collisions
  anti_signal   0.6%  ░░░░░░░░░░░░░░░░░░░░░░░  phantom matches
  shared_miss   9.9%  ██░░░░░░░░░░░░░░░░░░░░░  non-dict tokens

  Interpretation: strong signal — dictionary is a good fit
```

## Command line

No Python required:

```bash
python -m dictcollision --tokens decoded.txt --dict latin_50k.txt
python -m dictcollision --tokens decoded.txt --dict latin.txt --baselines
python -m dictcollision --tokens decoded.txt --dict latin.txt --json > report.json
```

Supported dictionary formats: one word per line, `word count` (hermitdave
FrequencyWords), or CSV. See `python -m dictcollision --help`.

## Input and output contract

**Input:**

```python
decoded_tokens : list[str]          # e.g. ["the", "cat", "ab", "cd", ...]
                                    # any Unicode; no preprocessing assumed
dictionary     : set[str] | list[str]   # reference words, same encoding
```

**Output (`classify`)** → `ClassifyResult` dataclass:

| Field | Type | Range | Meaning |
|---|---|---|---|
| `net_signal` | float | `[-1, 1]` | **The calibrated metric.** signal − anti_signal |
| `signal` | float | `[0, 1]` | real hits |
| `shared_hit` | float | `[0, 1]` | chance collisions that happen to also be real words |
| `anti_signal` | float | `[0, 1]` | phantom matches (null-only) |
| `shared_miss` | float | `[0, 1]` | non-dictionary tokens |
| `apparent_hit_rate` | float | `[0, 1]` | what a naive evaluator would report |
| `correction` | float | `≥ 0` | apparent − net_signal |
| `signal_words` | `list[str]` |  | types driving real signal |
| `anti_signal_words` | `list[str]` |  | types that inflate chance — inspect to diagnose |
| `n_tokens` | int |  | total count |

### Interpreting `net_signal`

| Range | Meaning |
|---|---|
| `≥ 0.20` | **Strong signal** — dictionary is a good fit |
| `0.05 – 0.20` | **Partial signal** — possibly correct with caveats |
| `≈ 0` | **No signal beyond chance** |
| `< 0` | **Worse than random** — wrong language or wrong decode key |

## The core equation

The predicted noise floor for dictionary $D$ against decoded text with
character distribution $p$ is:

$$\hat{r} \\;=\\; \sum_{w \in D}\\; \prod_{i=1}^{|w|} p(w_i)$$

For every word in the dictionary, multiply together the character
frequencies of your decoded output. Sum. That number is how many of
your tokens would match by accident.

## Four-category framework

| Category | In dictionary? | In real text? | In null corpora? |
|---|---|---|---|
| **Signal** | yes | yes | no |
| **Shared hit** | yes | yes | yes |
| **Anti-signal** | yes | no | yes |
| **Shared miss** | no | — | — |

**Net signal = Signal − Anti-signal** is the calibrated metric.

Null corpora are generated from the decoded text's character bigram
distribution (configurable: unigram / bigram / trigram), preserving
character-pair frequencies and token lengths while destroying word
identity. On wrong-language evaluations the four-category framework
is the only method among six tested that correctly reports signal as
≤ 0.

## Full API

```python
from dictcollision import (
    noise_floor,                  # analytical collision prediction
    classify,                     # four-category classification
    classify_by_length,           # per-length-bucket breakdown
    recommend,                    # rank candidate dictionaries
    null_distribution,            # Monte Carlo null distribution
    bootstrap_ci,                 # bootstrap CI on net_signal
    load_dictionary, load_tokens, # file loaders
)

from dictcollision.baselines import (
    apparent_hit_rate,            # no correction
    subtract_null,                # naive baseline
    permutation_test,             # per-word Poisson test
    bh_fdr,                       # Benjamini-Hochberg
    blast_evalue,                 # BLAST-style E-value
    all_methods,                  # all six in one dict (Table 2 style)
)

from dictcollision.viz import (
    plot_decomposition,           # paper Figure 1
    plot_size_sweep,              # paper Figure 2
    plot_method_comparison,       # paper Figure 5
    plot_length_stratified,       # paper Figure 13
)
```

## Examples

Self-contained scripts in [examples/](examples/):

- [01_vigenere.py](examples/01_vigenere.py) — evaluate a Vigenere candidate key
- [02_paper_table2.py](examples/02_paper_table2.py) — reproduce the six-method comparison
- [03_dictionary_recommender.py](examples/03_dictionary_recommender.py) — pick the right dictionary without knowing the language

## Paper

Methodology, experiments, validation:

Ruckman, M. (2026). *The Dictionary Collision Effect in Computational
Decipherment.* Source, figures, and reproduction code:
<https://github.com/mruckman1/signal-isolation-paper>

## Citation

```bibtex
@article{ruckman2026dictcollision,
  title={The Dictionary Collision Effect in Computational Decipherment},
  author={Ruckman, Matthew},
  year={2026},
  url={https://github.com/mruckman1/signal-isolation-paper}
}
```

## License

MIT
