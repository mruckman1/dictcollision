# Changelog

## 0.2.1 (2026-04-16)

- Docs: add uv install instructions (`uv add`, `uv pip install`, `uv tool install`).
- Housekeeping: expanded `.gitignore` to cover Hypothesis, coverage, IDE,
  and editor artifacts.

## 0.2.0 (2026-04-16)

New features:

- **CLI**: `python -m dictcollision --tokens decoded.txt --dict words.txt`
  with `--baselines`, `--by-length`, `--json`, `--noise-only`,
  `--null-model {unigram,bigram,trigram}`.
- **`dictcollision.baselines`**: the five correction methods tested
  alongside the four-category framework in the paper
  (`apparent_hit_rate`, `subtract_null`, `permutation_test`, `bh_fdr`,
  `blast_evalue`, `all_methods`). Reproduces Table 2 / Figure 5.
- **Trigram and unigram null models**: `classify(..., null_model="trigram")`.
  Section 8 of the paper shows trigram nulls strengthen wrong-language
  detection.
- **Length-stratified classification**: `classify_by_length()` returns
  per-length-bucket correction magnitudes (paper Section 10).
- **Monte Carlo null distribution**: `null_distribution(tokens, dict, n=40)`
  returns the empirical distribution of `net_signal` under resampling.
- **Bootstrap CI**: `bootstrap_ci(tokens, dict, n=200)`.
- **File loaders**: `load_dictionary()` handles plain text, hermitdave
  `word count` format, and CSV. `load_tokens()` supports custom delimiters.
- **`ClassifyResult.summary()`** and `Recommendation.summary()`:
  human-readable bar-chart reports with interpretation of the score.
- **`to_dict()`** methods on all result types for JSON export.
- **Expanded viz**: `plot_size_sweep`, `plot_method_comparison`,
  `plot_length_stratified` replicate paper Figures 2, 5, and 13.
- **Examples directory**: three self-contained scripts demonstrating
  Vigenere evaluation, the Table 2 comparison, and the recommender.
- **Property-based tests** via Hypothesis covering the scoring invariants.

Docs:

- README rewritten with a domain-applicability table, explicit I/O
  contract, an interpretation scale for `net_signal`, and a full API
  index.

No breaking changes to the 0.1.x public API.

## 0.1.1 (2026-04-16)

- Docs: add PyPI badges, LaTeX-formatted collision equation,
  link to the accompanying paper repository. No code changes.

## 0.1.0 (2026-04-16)

- Initial release
- `noise_floor()`: analytical collision rate prediction
- `classify()`: four-category token classification (signal, shared_hit, anti_signal, shared_miss)
- `recommend()`: dictionary recommender ranked by excess or SNR
- `dictcollision.viz`: optional matplotlib visualization helpers
