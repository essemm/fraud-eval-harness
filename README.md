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

The pipeline is a package of modules connected by CSV interfaces, each
independently runnable and testable.

```
generate_synthetic.py  -->  transactions.csv  --+
                            (native currency)    |
                       fx_rates.csv  ------------+--> profile.py   --> card_profiles.csv --+
                                                 |                                          |
                                                 +--> features.py <-------------------------+
                                                          |
                                                          v
                                            score.py / score_ml.py --> evaluate.py
```

| Module | Responsibility |
|---|---|
| `fx.py` | Currency→USD conversion; `fx_rates.csv` writer |
| `generate_synthetic.py` | Labelled synthetic transactions with injected fraud sequences and hard negatives |
| `profile.py` | Per-card USD-normalised baseline |
| `features.py` | Sequence deltas, trailing point-in-time baseline, profile join |
| `scorer.py` | `Scorer` protocol — the interface seam for the ML swap-in |
| `score.py` | `RuleScorer`: five transparent rules + card-level aggregation |
| `score_ml.py` | `MLScorer`: logistic regression behind the same protocol, trained out-of-sample |
| `evaluate.py` | Threshold sweep, cost models, per-scenario recall, operating-point diagnostics, hard-negative analysis |

**Swap contract:** only the scorer changes when the ML model replaces the rule
baseline. `RuleScorer` and `MLScorer` both satisfy the `Scorer` protocol;
`features.py` and `evaluate.py` are untouched. This is what makes the
rules-vs-ML comparison a fair test.

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

33 tests, one per acceptance criterion in the brief, run entirely in memory.

---

## Rules vs ML: the central finding

The rule baseline and an ML scorer (`score_ml.py`, logistic regression trained
on a different seed from the evaluation data) are compared on the **same**
harness. The result is not "one wins": at each scorer's own cost-minimising
operating point, the two sit at **different points on the precision/recall
frontier**. Which you prefer depends on the cost model — which is exactly the
trade-off the harness is built to surface.

Comparing the two at a single shared threshold (e.g. 0.50) is misleading: a
calibrated ML probability and a hand-tuned rule score do not mean the same thing
at the same number. `evaluate.py` reports diagnostics at each scorer's own
operating point, not a shared one.

Per-scenario recall is reported separately for each attack type — a blended
number hides that a detector may catch sprees and miss card-testing entirely.
Figures below are **mean ± sample standard deviation over 6 independent seeds**
(3,000 cards each, 10% fraud rate, seed pairs (1,101)…(6,106) via
`scripts/run_seed.py` and `fraud_eval/aggregate_runs.py`):

| Scenario | Rules (@ thr 0.05) | ML (@ thr ~0.30) |
|---|---|---|
| Card testing | 0.850 ± 0.042 | 0.951 ± 0.034 |
| Account takeover | 0.853 ± 0.038 | 0.907 ± 0.018 |
| Impossible travel | 0.774 ± 0.055 | 0.871 ± 0.063 |
| Stolen spree | 0.824 ± 0.066 | 0.909 ± 0.037 |

ML achieves higher recall on all four scenarios. That is the cost of running at
a lower threshold (0.05 vs ~0.30): the rules scorer flags a larger fraction of
the card population, accepting more false alarms in exchange for catching more
fraud.

**Hard-negative false-positive rate** — the clearest measure of whether
sequence context earns its keep. Hard-negative cards are by construction the
legitimate accounts most likely to be confused with fraud:

| Approach | FP rate on hard-negative cards |
|---|---|
| Rules (sequence-aware) | **0.625 ± 0.026** |
| Naive single-row amount threshold | 0.743 ± 0.030 |
| ML (sequence-aware) | 0.818 ± 0.046 |

The rules scorer sits 12 points below the naive baseline — it earns its
complexity. The ML scorer sits 7 points above it: at its own operating point,
the model flags more hard-negative cards than a simple amount threshold would.
This is not a failure of the sequence features; it is a consequence of the ML
model's higher recall. The explicit fingerprints in the rules — country-change
within three hours, velocity burst above a count threshold — are designed to
fire on fraud sequences and not on their near-twins. A linear classifier that
cannot express those exact conditions trades hard-negative precision for recall.

The two cost models (fixed 20:1 vs. amount-weighted) prefer different operating
thresholds — surfaced as a business decision, not resolved by the harness.

### Figures

![Cost vs threshold](viz/figures/01_cost_vs_threshold.png)

*Cost vs decision threshold (representative seed 1). Stacked panels because the
two y-scales differ by orders of magnitude. Dashed lines mark each scorer's
cost-minimising threshold.*

![Precision–recall curve](viz/figures/02_precision_recall.png)

*Precision–recall curve (representative seed 1). Each scorer's operating point
is marked at its own cost-minimising threshold, not a shared one.*

![Per-scenario recall](viz/figures/03_per_scenario_recall.png)

*Per-scenario recall at each scorer's own operating point. Mean ± 1 sd over 6
seeds. ML leads on recall across all scenarios; the overlap on impossible travel
is widest because that scenario's recall is most variable across seeds.*

![Hard-negative FP rate](viz/figures/04_hard_negative_fp.png)

*Hard-negative false-positive rate. Lower is better. Rules beats the naive
single-row baseline; ML does not at its operating point.*

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

## Running the multi-seed evaluation

The per-scenario figures above come from a 6-seed run. Each seed pair generates
independent eval and train datasets, scores with both scorers, and writes
per-seed artifacts to `runs/seed_N/`:

```bash
for i in 1 2 3 4 5 6; do
    python scripts/run_seed.py --eval-seed $i --train-seed $((i + 100))
done
python -m fraud_eval.aggregate_runs   # reads runs/seed_*/metrics_*.json
pip install -r viz/requirements-viz.txt
python -m viz.make_plots              # writes viz/figures/*.png
```

`scripts/run_seed.py` uses library calls throughout (no subprocess shelling)
and keeps the cost knobs fixed across all seeds so operating points are
comparable. `fraud_eval/aggregate_runs.py` uses sample standard deviation
(`statistics.stdev`, n−1) and skips absent scenarios rather than zero-filling.

## Running the ML scorer

The ML scorer trains out-of-sample — on data generated with a different seed
from the evaluation set — so it is never evaluated on rows it learned from:

```bash
# generate + feature a training set on one seed, an eval set on another
python -m fraud_eval.generate_synthetic --seed 101 --cards 3000 --out transactions_train.csv
python -m fraud_eval.generate_synthetic --seed 1   --cards 3000 --out transactions_eval.csv
# (profile + features each, producing featured_train.csv and featured_eval.csv)

python -m fraud_eval.score_ml \
    --train-featured featured_train.csv --featured featured_eval.csv \
    --row-out scored_rows_ml.csv --card-out scored_cards_ml.csv

python -m fraud_eval.evaluate --rows scored_rows_ml.csv --cards scored_cards_ml.csv
```

The ML scored output drops into `evaluate.py` unchanged — the same harness, the
same cost models — which is what makes the rules-vs-ML comparison fair.
