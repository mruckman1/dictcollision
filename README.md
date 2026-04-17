# dictcollision

[![PyPI](https://img.shields.io/pypi/v/dictcollision)](https://pypi.org/project/dictcollision/)
[![Python](https://img.shields.io/pypi/pyversions/dictcollision)](https://pypi.org/project/dictcollision/)
[![License](https://img.shields.io/pypi/l/dictcollision)](https://github.com/mruckman1/dictcollision/blob/main/LICENSE)

**Your decipherment reports a 43% dictionary hit rate. Is that real?**

`dictcollision` answers this question. When decoded strings are short
(2–4 characters) and dictionaries are large (≥10K words), chance
collisions produce matches at rates that approach genuine decipherment
rates. This package predicts the collision rate and separates real
signal from noise.

## Install

```bash
pip install dictcollision
```

For plotting support:

```bash
pip install dictcollision[viz]
```

## Quick start

### One-line noise floor check

```python
from dictcollision import noise_floor

predicted = noise_floor(decoded_tokens, dictionary)
print(f"Chance collisions alone: {predicted:.1%}")
print(f"Your observed rate:      43.0%")
print(f"Genuine signal:          {0.43 - predicted:.1%}")
```

### Full four-category analysis

```python
from dictcollision import classify

result = classify(decoded_tokens, dictionary)
print(f"Signal:          {result.signal:.1%}")
print(f"Shared hit:      {result.shared_hit:.1%}")
print(f"Anti-signal:     {result.anti_signal:.1%}")
print(f"Net signal:      {result.net_signal:.1%}")
print(f"Apparent rate:   {result.apparent_hit_rate:.1%}")
```

### Rank candidate dictionaries

```python
from dictcollision import recommend

ranked = recommend(
    decoded_tokens,
    {"latin_10k": latin_words, "german_50k": german_words},
    objective="excess",
)
for r in ranked:
    print(f"{r.name}: excess={r.excess:.3f}, snr={r.snr:.1f}")
```

## The core equation

The predicted noise floor for dictionary $D$ against decoded text with
character distribution $p$ is:

$$\hat{r} \\;=\\; \sum_{w \in D}\\; \prod_{i=1}^{|w|} p(w_i)$$

For every word in the dictionary, multiply together the character
frequencies of your decoded output. Sum. That number is how many of
your tokens would match by accident.

## How it works

The four-category framework classifies every decoded token type:

| Category | In dictionary? | In real text? | In null corpora? |
|----------|---------------|---------------|------------------|
| **Signal** | yes | yes | no |
| **Shared hit** | yes | yes | yes |
| **Anti-signal** | yes | no | yes |
| **Shared miss** | no | — | — |

**Net signal = Signal − Anti-signal** is the calibrated metric.

Null corpora are generated from the decoded text's character bigram
distribution, preserving character-pair frequencies and token lengths
while destroying word identity.

## Paper

The methodology, experiments, and validation behind this package are
described in:

Ruckman, M. (2026). *The Dictionary Collision Effect in Computational
Decipherment.* Source, figures, and reproduction code:
https://github.com/mruckman1/signal-isolation-paper

## Citation

If you use this package in your research, please cite:

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
