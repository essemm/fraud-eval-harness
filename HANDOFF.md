# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-18 (session 3). Pick-up note for resuming, incl. a new chat._

## Where we are

Pipeline complete, tested, and **refactored into a package**. Repo at
`/Users/scott/Projects/fraud-detection/`. Layout now:

    fraud_eval/              the package
      __init__.py
      scorer.py              the Scorer protocol (shared interface seam)
      fx.py
      generate_synthetic.py
      profile.py
      features.py
      score.py               RuleScorer (implements Scorer structurally)
      evaluate.py
    tests/                   pytest suite, 27 tests, one per §10 criterion
    requirements.txt         runtime = stdlib only; pytest is the dev dependency
    PROJECT_BRIEF.md, README.md (empty), HANDOFF.md, .gitignore

## Done this session (refactor)

- Created `fraud_eval/` package; moved all six modules in; added __init__.py.
- Extracted the `Scorer` protocol into its own `scorer.py` (the agreed "option
  1" pre-shaping for the future ML scorer). Both RuleScorer and a future
  MLScorer import the contract from this neutral module as equals. RuleScorer
  satisfies it structurally (implements score_row; no explicit inheritance).
- Converted internal imports to relative (`from .fx import ...`,
  `from .scorer import Scorer`). Updated usage docstrings to
  `python -m fraud_eval.<module>`. Updated all test imports to
  `from fraud_eval import ...`.
- Added requirements.txt (pytest only; runtime is stdlib).
- Verified in staging: full suite 27/27 PASS; full pipeline runs end to end via
  `python -m fraud_eval.<module>`. On-disk files confirmed correct by diff.

## TO-DO on Scott's machine (couldn't be done remotely)

1. `python -m pytest tests/ -q` from repo root — confirm 27 pass in place.
2. `git rm OLD_score_DELETE_ME.py` — stale duplicate of the old root-level
   score.py (renamed because the connector can't delete files; the real one is
   now fraud_eval/score.py).
3. Optionally regenerate outputs via the new invocation (see commands below).
4. Commit the refactor as its own commit.

## Running it (new package invocation)

ALWAYS activate the venv first (see below). Then, from the repo root:

    python -m fraud_eval.fx --out fx_rates.csv
    python -m fraud_eval.generate_synthetic --cards 1000 --days 30 --out transactions.csv
    python -m fraud_eval.profile --in transactions.csv --fx-rates fx_rates.csv --out card_profiles.csv
    python -m fraud_eval.features --txns transactions.csv --profiles card_profiles.csv --fx-rates fx_rates.csv --out featured.csv
    python -m fraud_eval.score --in featured.csv --agg decaying_sum --row-out scored_rows.csv --card-out scored_cards.csv
    python -m fraud_eval.evaluate --rows scored_rows.csv --cards scored_cards.csv

## Virtual environment

The venv lives in `.venv/` (gitignored). From inside the project directory:

    source .venv/bin/activate        # activate — prompt then shows (.venv)
    deactivate                       # when done

If `.venv/` ever needs recreating (new machine, deleted folder):

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt

Note: macOS system pip is broken (dead /usr/bin/python shebang) and Homebrew
Python is externally-managed (PEP 668). The venv sidesteps both. Always use
`python3 -m pip` / `python3 -m pytest` rather than bare `pip`/`pytest`.

## Next steps (in order)

1. Finish the 4 to-do items above (verify, git rm, commit).
2. Write the **README** (currently empty) — reflect the package layout and the
   `python -m` commands. The hard-negative twin table (brief §5) and the real
   eval findings lift well into it. Repo name for resume: **fraud-eval-harness**;
   title "Sequence-Aware Fraud Detection Evaluation Harness".
3. **ML scorer swap-in** (the agreed scope, NOT per-card/daily-retrain):
   - A single model trained on the WHOLE population's featured rows (the signal
     lives across cards, not within one). scikit-learn logistic regression or a
     gradient-boosted tree.
   - New file `fraud_eval/score_ml.py`: imports `Scorer` from `.scorer`,
     implements `score_row`, drops into the SAME evaluate.py harness. Adds
     scikit-learn to requirements.
   - The payoff is the A/B: run ML vs RuleScorer through the same harness, same
     cost models, same per-scenario recall + hard-negative test. metrics.json
     was designed to make this comparison clean.
   - Per-card / daily-retrain / rolling-window online learning is explicitly OUT
     of scope (overkill for static synthetic data) but is a GOOD interview
     talking point as "how I'd extend this for concept drift in production."

## Eval findings to reuse (from a 1500-card run, threshold 0.50)

- Per-scenario recall: card_testing 71% / account_takeover 44% /
  impossible_travel 23% / stolen_spree 29% (the spread a blended number hides).
- Hard-negative FP: sequence-aware 14% vs naive single-row 68% — the sequence
  approach earns its keep (~5x fewer false positives).
- Cost models can disagree on the operating point (that's the point). At
  aggressive ratios the cost-min degenerates to "flag all"; evaluate.py detects
  and names that rather than presenting it as real. Default ratio lowered to
  20:1 for an interior optimum.

## Open decisions (brief §11) — resolved

- Amount-weighted FN cost: full amount lost (issuer-liability cap = future work).
- Aggregation: both built; decaying_sum default.
- Trailing window: time-based.
- Minor/open: evaluate.py hardcodes reference threshold 0.50 + target recall
  0.90; could expose as CLI args.

## Key context

- Scott reviews each file before it lands; edits independently between rounds;
  plain, specific, non-hyperbolic language; never the word "shape" (use design/
  structure/layout); verifies claims from output artifacts, not just code.
- Both target labs (Anthropic FDE / OpenAI Applied AI SA) weight evaluation
  methodology heavily — evaluate.py + the test suite + the ML A/B are the most
  interview-relevant artifacts.
- Filesystem connector with write_file enabled (read+write ~/Projects); cannot
  delete files (hence the rename-to-DELETE_ME pattern). GitHub connector tools
  don't surface mid-session — try a fresh chat; public repos readable via web
  fetch with a URL.
