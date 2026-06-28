# Handoff - fraud-detection reliability + investigation layer

_Last updated: 2026-06-28. Implementation handoff for running Codex in a
terminal._

## Current status

Repo: `/Users/scott/Projects/fraud-detection`

The existing harness is complete through the earlier four-figure milestone:

- `fraud_eval/` contains the core pipeline: `fx`, `generate_synthetic`,
  `profile`, `features`, `scorer`, `score`, `score_ml`, `evaluate`, and
  `aggregate_runs`.
- `scripts/run_seed.py` runs one out-of-sample eval/train seed pair for both
  rules and ML.
- `runs/seed_*/metrics_*.json`, `runs/seed_*/sweep_*.csv`, and
  `runs/aggregate.json` exist from the 6-seed run.
- `viz/make_plots.py` currently renders four PNGs in `viz/figures/`.
- README reflects the four-figure rules-vs-ML story.
- The old test suite had 36 passing tests.

`PROJECT_BRIEF.md` has since been updated and is now ahead of the
implementation. Treat that brief as the source of truth. The new work is:

1. ML row-probability reliability diagnostics and a fifth plot.
2. A downstream local-LLM investigation-note layer.

Important: the working tree already has an unstaged `PROJECT_BRIEF.md` change.
Do not revert it. Implement against it.

## Existing findings to preserve

Seed pairs are `(eval, train) = (1,101) ... (6,106)`, with 3,000 cards per eval
seed and fraud rate 0.10.

At each scorer's own fixed-ratio operating point, not a shared threshold:

| scenario | Rules @0.05 | ML @~0.30 |
|---|---:|---:|
| card_testing | 0.850 +/- 0.042 | 0.950 +/- 0.036 |
| account_takeover | 0.860 +/- 0.036 | 0.909 +/- 0.019 |
| impossible_travel | 0.774 +/- 0.055 | 0.837 +/- 0.065 |
| stolen_spree | 0.824 +/- 0.066 | 0.907 +/- 0.033 |
| hard-neg seq FP rate | 0.625 +/- 0.026 | 0.807 +/- 0.056 |
| hard-neg naive FP rate | 0.743 +/- 0.030 | 0.743 +/- 0.030 |

The story is a precision/recall frontier trade-off, not simple ML dominance.
ML gets higher recall; rules have materially lower hard-negative false positives.
Keep that framing in README and figure captions.

## SPEC 3 - ML probability reliability

### Goal

Implement `PROJECT_BRIEF.md` sections 8.3, 10/E6-E9, and 10/V1,V6-V8.

This diagnostic answers whether the ML row score can be interpreted as a
probability before card-level aggregation. It must use held-out ML scored rows,
not training rows and not `scored_cards_ml.csv`.

### Required outputs

- `runs/seed_1/scored_rows_ml.csv` or equivalent held-out ML scored-row artifact
  for the representative seed.
- `runs/seed_1/reliability_ml.json`, or another clearly documented path, with:
  - `source`
  - `n_rows`
  - `n_bins`
  - `n_populated_bins`
  - `brier_score`
  - `expected_calibration_error`
  - `bins`
- `viz/figures/05_ml_reliability_diagram.png`

Recommended JSON bin record:

```json
{
  "bin_lower": 0.0,
  "bin_upper": 0.1,
  "count": 123,
  "mean_predicted_probability": 0.034,
  "observed_fraud_rate": 0.041
}
```

Use fixed-width intervals over `[0, 1]`; make the final bin include `1.0`.
Expected calibration error should be weighted by bin count:

```text
sum((bin_count / n_rows) * abs(observed_fraud_rate - mean_predicted_probability))
```

Brier score is mean squared error between row `score` and row `is_fraud`.

### Implementation notes

1. Extend `scripts/run_seed.py` so it can persist scored rows/cards for at least
   the ML scorer. It already has held-out `scored_rows_ml` in memory, so this is
   the least surprising source for the reliability artifact.
2. Prefer writing both ML and rules scored artifacts per seed:
   - `runs/seed_N/scored_rows_rules.csv`
   - `runs/seed_N/scored_cards_rules.csv`
   - `runs/seed_N/scored_rows_ml.csv`
   - `runs/seed_N/scored_cards_ml.csv`
   This also gives the investigation layer clean inputs.
3. Add `viz/reliability.py` as a pure artifact consumer. It should read
   `scored_rows_ml.csv`, compute the reliability JSON, and optionally render a
   text report. Keep it stdlib-only if possible.
4. Update `viz/make_plots.py` to draw the fifth static PNG. It may read
   `reliability_ml.json` by default and optionally compute from scored rows if
   the JSON is absent.
5. Keep plotting dependencies confined to `viz/`. No `fraud_eval/` module should
   import matplotlib or anything from `viz/`.
6. Do not use the word `accuracy` in the reliability file name, plot title, axis
   labels, caption, or README reference. The brief explicitly forbids it here.

Suggested CLI:

```bash
python scripts/run_seed.py --eval-seed 1 --train-seed 101
python -m viz.reliability \
  --rows runs/seed_1/scored_rows_ml.csv \
  --out runs/seed_1/reliability_ml.json \
  --bins 10
python -m viz.make_plots \
  --reliability runs/seed_1/reliability_ml.json \
  --out-dir viz/figures
```

Static PNG acceptance matters first. Updating the optional Plotly interactive
HTML can be a follow-up unless it is cheap and stays clean.

### Tests to add

Create focused tests, probably `tests/test_reliability.py`.

Cover:

- E6: reliability consumes row-level ML scored rows with `score` and `is_fraud`;
  it does not use card-level fields.
- E7: bins include count, mean predicted probability, and observed fraud rate;
  empty bins are skipped or explicitly represented as empty.
- E8: Brier score and ECE are computed correctly on a tiny deterministic input.
- E9/V5: reliability metadata and plot labels do not contain `accuracy`
  case-insensitively.
- V6/V7: plot function writes `05_ml_reliability_diagram.png` and includes a
  diagonal perfect-calibration reference line. If this is hard to assert from
  the image, test the plotting function inputs/metadata and visually inspect the
  generated PNG.
- V8: plotting remains in `viz/`; core `fraud_eval/` modules do not import
  matplotlib.

## SPEC 4 - downstream investigation layer

### Goal

Implement `PROJECT_BRIEF.md` section 13 and acceptance criteria A1-A7.

The investigation layer is a downstream consumer only. It prepares structured
case notes for a human reviewer. It must not decide fraud, change scores, set
thresholds, alter features, or modify core evaluation artifacts.

Recommended package:

```text
investigation/
  __init__.py
  build_cases.py
  investigate.py
  evaluate_notes.py
tests/
  test_investigation.py
```

Keep the package stdlib-only unless there is a strong reason otherwise.

### Case builder

`investigation/build_cases.py` should read scored cards plus scored rows and
emit one JSONL case per selected high-score card.

Recommended selection:

- CLI args:
  - `--rows scored_rows_ml.csv`
  - `--cards scored_cards_ml.csv`
  - `--scorer ml`
  - `--threshold 0.30`
  - `--limit 20`
  - `--top-rows 5`
  - `--out investigation_cases.jsonl`
- Select cards with `card_score >= threshold`, sorted by descending
  `card_score`, limited by `--limit`.
- For each card, include the top suspicious rows by row `score`.

Each JSONL record should separate prompt-safe payload from evaluation-only
labels:

```json
{
  "case_id": "card_000123",
  "prompt_payload": {
    "card_id": "card_000123",
    "card_score": 0.84,
    "scorer": "ml",
    "decision_threshold": 0.30,
    "top_suspicious_rows": [],
    "evidence_facts": []
  },
  "evaluation": {
    "any_fraud": 1,
    "fraud_scenario": "card_testing",
    "has_hard_neg": false
  }
}
```

The `prompt_payload` must not contain ground-truth fields:

- `is_fraud`
- `scenario`
- `any_fraud`
- `fraud_scenario`
- `has_hard_neg`

`evaluation` may contain those fields for rubric checks, but it must never be
sent to the LLM.

Strong recommendation: generate an `evidence_facts` array with exact strings
derived from the case, for example:

- `txn tx_001 scored 0.812 because ml:velocity_1h=8`
- `txn tx_001 had 8 transactions in the trailing 1h`
- `txn tx_001 used a new device`

Then instruct the LLM to copy supporting evidence from `evidence_facts`. This
makes grounding auditable with simple exact-match tests.

### Investigator

`investigation/investigate.py` should consume case JSONL and write one
structured note JSON object per case.

Required note schema:

```json
{
  "card_id": "card_000123",
  "recommended_action": "manual_review",
  "risk_summary": "Short evidence-grounded summary.",
  "supporting_evidence": [
    "Observed fact from the case input"
  ],
  "missing_information": [
    "Information a human reviewer should obtain"
  ],
  "customer_safe_language": "Neutral wording suitable for customer contact.",
  "caveats": [
    "Reasons the case may be a false positive"
  ]
}
```

Allowed `recommended_action` values:

- `no_action`
- `manual_review`
- `step_up_auth`
- `customer_contact`
- `block_or_suspend`

The local-LLM interface should be explicit and testable. A simple command-based
adapter is enough:

```bash
python -m investigation.investigate \
  --cases investigation_cases.jsonl \
  --out investigation_notes.jsonl \
  --llm-command "ollama run llama3.2:3b-instruct"
```

The command adapter can send the prompt on stdin and read stdout. Keep tests away
from any live model by adding a deterministic fake model path, for example:

```bash
python -m investigation.investigate \
  --cases investigation_cases.jsonl \
  --out investigation_notes.jsonl \
  --fake
```

Validation before writing:

- required fields present
- `recommended_action` is in the allowed enum
- list fields are lists of strings
- `card_id` matches the input case
- no forbidden phrases such as `confirmed fraud`
- no customer-accusatory phrasing
- no reference to withheld labels

If the model returns invalid JSON or fails validation, prefer a clear exception
and no partial write over silently accepting unsafe output.

### Note evaluator

`investigation/evaluate_notes.py` should compare cases and notes, then emit
per-case rubric results plus aggregate pass rates.

Recommended output:

```json
{
  "n_cases": 20,
  "aggregate": {
    "grounded_evidence": 0.95,
    "no_forbidden_conclusion": 1.0,
    "valid_action": 1.0,
    "missing_information": 0.9,
    "customer_safe_language": 1.0,
    "hard_negative_caution": 0.8
  },
  "cases": [
    {
      "card_id": "card_000123",
      "grounded_evidence": true,
      "no_forbidden_conclusion": true,
      "valid_action": true,
      "missing_information": true,
      "customer_safe_language": true,
      "hard_negative_caution": true
    }
  ]
}
```

Rubric suggestions:

- `grounded_evidence`: every `supporting_evidence` item is exactly present in
  `prompt_payload.evidence_facts`, or traceable to a row reason/transaction id.
- `no_forbidden_conclusion`: note does not claim confirmed fraud and does not
  accuse the customer.
- `valid_action`: action is in the enum.
- `missing_information`: `missing_information` is non-empty.
- `customer_safe_language`: customer-facing text uses neutral phrasing.
- `hard_negative_caution`: for cases where `evaluation.has_hard_neg` is true,
  the note includes a caveat or review posture and avoids overconfident
  `block_or_suspend`.

### Investigation tests

Create deterministic in-memory fixtures; do not require a live LLM.

Cover:

- A1: `build_cases.py` emits compact case objects and no ground-truth labels in
  `prompt_payload`.
- A2: `investigate.py` writes one validated JSON object per input case.
- A3: invalid `recommended_action` is rejected.
- A4: forbidden conclusions and accusatory language are rejected or fail the
  evaluator.
- A5: evaluator emits per-case rubric booleans and aggregate pass rates.
- A6: tests use fake model or fixtures only.
- A7: investigation layer is downstream only. It should not import or mutate
  `fraud_eval` internals, and should work from scored CSV artifacts.

## README updates after implementation

After SPEC 3:

- Add `05_ml_reliability_diagram.png` to the Figures section.
- Add a short paragraph explaining that reliability is row-level probability
  calibration before card-level aggregation.
- Mention Brier score and expected calibration error from the generated artifact.
- Do not call the card-level decaying-sum score a probability.
- Do not use `accuracy` in the reliability reference.

After SPEC 4:

- Add a short section for the optional investigation layer.
- Emphasize that the local LLM is a constrained summarizer, not the fraud
  detector and not the threshold selector.
- Include the fake/test path and, if present locally, the local model command.
- Keep raw LLM outputs out of README unless they are deterministic and small.

## Verification commands

Use the project venv:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

Reliability smoke path:

```bash
python scripts/run_seed.py --eval-seed 1 --train-seed 101
python -m viz.reliability \
  --rows runs/seed_1/scored_rows_ml.csv \
  --out runs/seed_1/reliability_ml.json \
  --bins 10
python -m viz.make_plots \
  --reliability runs/seed_1/reliability_ml.json \
  --out-dir viz/figures
```

Full figure regeneration after reliability is wired:

```bash
python -m viz.make_plots --out-dir viz/figures
```

Investigation smoke path, after scored artifacts exist:

```bash
python -m investigation.build_cases \
  --rows runs/seed_1/scored_rows_ml.csv \
  --cards runs/seed_1/scored_cards_ml.csv \
  --scorer ml \
  --threshold 0.30 \
  --limit 20 \
  --out runs/seed_1/investigation_cases.jsonl

python -m investigation.investigate \
  --cases runs/seed_1/investigation_cases.jsonl \
  --out runs/seed_1/investigation_notes.jsonl \
  --fake

python -m investigation.evaluate_notes \
  --cases runs/seed_1/investigation_cases.jsonl \
  --notes runs/seed_1/investigation_notes.jsonl \
  --out runs/seed_1/investigation_eval.json
```

## Guardrails for terminal Codex

- Read `PROJECT_BRIEF.md` first. It is the implementation spec.
- Keep edits closely scoped to reliability and investigation.
- Do not refactor the core fraud pipeline while adding downstream consumers.
- Do not add network requirements. The local LLM path must be optional and tests
  must use a fake.
- Preserve the scorer swap contract: `features.py` and `evaluate.py` should not
  know which scorer produced the rows.
- Preserve the visualizer layering: plotting belongs in `viz/`, not
  `fraud_eval/`.
- Be careful with already modified files. Do not revert user changes.
- When updating copy, keep it plain and specific. Avoid hype and avoid presenting
  the LLM as an autonomous decision-maker.
