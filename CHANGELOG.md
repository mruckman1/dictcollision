# Changelog

## 0.3.0 (2026-05-03)

The library now handles both the fixed-decode case (existing API) and
the stochastic-search case (new `search_calibrated_signal`). All
changes are additive; the 0.2.x public API is unchanged.

New features:

- **`search_calibrated_signal()`**: matched-budget shuffle calibration
  for searches over a key space. Takes a `cipher_symbols` sequence and
  a `search_fn` callable, runs the search on the real cipher and on
  `n_shuffles` permutations of the cipher's symbol multiset, returns
  the observed net_signal alongside a z-score and percentile against
  the shuffle distribution. Use this when your decoded tokens were
  produced by SA, hill-climbing, AZdecrypt, etc.
- **`SearchCalibrationResult`**: result type with `summary()` and
  `to_dict()`, exported from the top-level package.
- **`ClassifyResult.signal_word_counts` and `anti_signal_word_counts`**:
  new dicts mapping each signal/anti-signal word type to its real
  (resp. mean null) count. The existing `signal_words` /
  `anti_signal_words` lists remain unchanged. `to_dict()` emits the
  counts in `..._counts_top20` keys for JSON consumers.
- **`ClassifyResult.overfit_score()`**: heuristic concentration of the
  signal-token mass on the top three signal word types. Real text
  typically <0.4; SA-overfit decodes often >0.7.
- **`noise_floor(..., word_weights=...)`**: optional per-word
  weighting for chance-collision contributions (e.g. information
  weighting via `-log(corpus_freq)`). Default `None` preserves the
  existing unweighted formulation.
- **Length-aware interpretation**: `ClassifyResult.summary()` now
  appends a note when `n_tokens < 200` recommending
  `search_calibrated_signal()` for stochastic-search decodes — the
  regime where absolute net_signal can mislead.

Docs and examples:

- README: new "When your decode came from a search" section
  contrasting `null_distribution` (fixed decode) with
  `search_calibrated_signal` (calibrated search procedure).
- New `examples/04_search_calibrated.py`: real-plaintext vs.
  random-input case study showing the same search procedure produces a
  z-score of ~70 on real ciphertext and ~0 on random input.

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
