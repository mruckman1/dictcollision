# Examples

Self-contained scripts that exercise the full API. All use synthetic or
inline data so they run without downloading anything.

| Script | Demonstrates |
|---|---|
| [01_vigenere.py](01_vigenere.py) | Evaluating a Vigenere candidate key against a Latin dictionary — correct vs. wrong key. |
| [02_paper_table2.py](02_paper_table2.py) | Running the six-method comparison from Table 2 of Ruckman (2026). |
| [03_dictionary_recommender.py](03_dictionary_recommender.py) | Picking the best dictionary from a set of candidates without knowing the plaintext language. |

Run any of them:

```bash
python examples/01_vigenere.py
python examples/02_paper_table2.py
python examples/03_dictionary_recommender.py
```
