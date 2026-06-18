# Sequence-Aware Fraud Detection Evaluation Harness

Card fraud is not a property of a single transaction. A $4 online purchase is
unremarkable alone; six of them in ninety seconds from an unrecognised device is
a card-testing attack. The signal lives in the *sequence*, not in any row read
alone.

This project is the apparatus to make that measurable: it generates synthetic
transaction data with known fraud sequences and deliberately ambiguous
legitimate behaviour ("hard negatives"), scores it, and evaluates the scorer
under an explicit **cost model** rather than accuracy — which is meaningless at a
sub-1% fraud rate, where flagging nothing scores ~99.7%.

The deliverable is not a fraud model. It is the harness that tells you whether
*any* fraud model is good enough, where it fails, and at what threshold to run it.

---

## Architecture

Six modules connected by CSV interfaces, each independently runnable and
testable.

```
generate_synthetic.py  -->  transactions.csv  --+--> profile.py  --\
                            (native currency)    |                   +--> features.py --> score.py --> evaluate.py
                       fx_rates.csv  -----------+--> features.py --/
```

| Module | Responsibility |
|---|---|
| `fx.py` | Currency→USD conversion; `fx_rates.csv` writer |
| `generate_synthetic.py` | Labelled synthetic transactions with injected fraud sequences and hard negatives |
| `profile.py` | Per-card USD-normalised baseline |
| `features.py` | Sequence deltas, trailing point-in-time baseline, profile join |
| `scorer.py` | `Scorer` protocol — the interface seam for the ML swap-in |
| `score.py` | `RuleScorer`: five transparent rules + card-level aggregation |
| `evaluate.py` | Threshold sweep, cost models, per-scenario recall, hard-negative analysis |

**Swap contract:** only `score.py` changes when an ML model replaces the rule
baseline; `features.py` and `evaluate.py` are untouched.

---

## Fraud scenarios and hard negatives

Each fraud scenario has a hard-negative twin: legitimate behaviour separated from
fraud by exactly one dimension. The twins are why a single-row threshold fails on
precision.

| Fraud scenario | Fingerprint | Hard-negative twin | Separating dimension |
|---|---|---|---|
| `card_testing` | Many tiny online amounts, minutes apart, new device | — | velocity + amount cluster |
| `account_takeover` | New device + new IP country, escalating amounts | `hard_neg_new_device`: new device, **home** country, normal amounts | country + amount |
| `impossible_travel` | Legit local txn, far-country txn minutes later | `hard_neg_travel`: far country, but **hours** apart | time delta |
| `stolen_spree` | Run of mid/large purchases, unusual categories | `hard_neg_big_ticket`: **one** large legit purchase | run length |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt   # pytest only; pipeline is stdlib
```

## Running the pipeline

```bash
python -m fraud_eval.fx --out fx_rates.csv
python -m fraud_eval.generate_synthetic --cards 1000 --days 30 --out transactions.csv
python -m fraud_eval.profile --in transactions.csv --fx-rates fx_rates.csv --out card_profiles.csv
python -m fraud_eval.features --txns transactions.csv --profiles card_profiles.csv --fx-rates fx_rates.csv --out featured.csv
python -m fraud_eval.score --in featured.csv --agg decaying_sum --row-out scored_rows.csv --card-out scored_cards.csv
python -m fraud_eval.evaluate --rows scored_rows.csv --cards scored_cards.csv
```

Randomness is seeded; the same `--seed` produces byte-identical output.

## Running the tests

```bash
python -m pytest tests/ -q
```

27 tests, one per acceptance criterion in the brief, run entirely in memory.

---

## Sample results (1,000 cards, threshold 0.50)

Figures from one representative run; reproducible with a fixed `--seed`.

Per-scenario recall — reported separately because a blended number hides the spread:

| Scenario | Recall |
|---|---|
| card_testing | 56% |
| account_takeover | 25% |
| impossible_travel | 22% |
| stolen_spree | 56% |

Hard-negative false-positive rate, sequence-aware scorer vs. a naive single-row threshold:

| | False positives | FP rate |
|---|---|---|
| Sequence-aware (this project) | 2 / 66 | **3%** |
| Naive single-row amount threshold | 45 / 66 | **68%** |

The two cost models (fixed 20:1 vs. amount-weighted) prefer different thresholds
(0.50 vs. 0.05) — surfaced as a business decision, not resolved by the harness.

---

## Key design decisions

- **All-rows baseline.** The card profile is built from all transactions, not
  labelled-legitimate ones only, because in production labels don't exist at
  profile-build time. Fraud rows nudge the statistics upward; the trailing
  point-in-time baseline in `features.py` recovers the signal using only prior
  rows, with no look-ahead.
- **Cost-weighted evaluation.** Two configurable cost models; accuracy is never a
  headline metric.
- **Decaying-sum aggregation.** Card scores accumulate with exponential decay
  (default `0.9`) so many small signals (card-testing) can outweigh one isolated
  medium signal, which `max` would underweight.
- **Explainable baseline first.** Every rule decision carries a reason string,
  and the ML swap-in is measured against this baseline on the same harness.

---

## Next: ML scorer swap-in

`fraud_eval/score_ml.py` will implement `Scorer` with a scikit-learn model
(logistic regression or gradient-boosted tree) trained on population-wide
featured rows, dropping into `evaluate.py` unchanged for a direct rules-vs-ML
comparison on recall, precision, and cost.
