# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-17. Pick-up note for resuming, including in a new chat._

## Where we are

Pipeline modules built and verified end-to-end:
`fx.py` → `generate_synthetic.py` → `profile.py` → `features.py` → `score.py`.
The last module, **`evaluate.py`, is not yet written** — it's the next build.

All on disk at `/Users/scott/Projects/fraud-detection/`. Git repo initialised;
about to commit the basic flat structure as a clean baseline BEFORE refactoring.

## Just finished

- Added a **currency** dimension. `transactions.csv` now carries native `amount`
  + `currency` (ISO-4217); `fx_rates.csv` maps currency→USD; `fx.py` holds the
  shared `to_usd` helper. profile.py and features.py both normalise to USD;
  native amount + currency ride through to the scorer reason string for
  human-readable, native-currency alerts.
- PROJECT_BRIEF.md updated to match (new §4.4 fx_rates contract, currency
  invariants, currency-neutrality requirement §5, new acceptance criteria
  G5/G6/X1/P5/F4). Language edits done: "Non-goals"→"Known limitations",
  "Data contracts"→"Files and interfaces", removed "load-bearing"/"breaking
  change" phrasing, "Why this shape"→"Why this design" (avoid the word "shape").
- Sonnet (in Claude Code) generated the currency code; reviewed it and **fixed
  three bugs in generate_synthetic.py**:
  1. G1 reproducibility — IDs used uuid4 (ignores seed); replaced with seeded
     `_rid()` helper so reruns are byte-identical.
  2. Currency-blind amounts — amounts were dollar-magnitude regardless of
     currency, making NGN/JPY cards spend sub-dollar USD-equiv; now drawn in
     USD-equivalent terms and converted to native units.
  3. Dead conditional in card_testing — simplified.
  Full acceptance battery re-run after fixes: G1–G6, X1, P2, F1–F4 all PASS.
  (One F1 "fail" was a test-side rounding artifact, not a code leak — confirmed
  trailing baseline uses only prior rows.)
- `.gitignore`: added `.claude/` (keep local, not committed — portfolio repo).
  `*.csv` still ignored, so `fx_rates.csv` is currently uncommitted (open
  question — it's tiny static reference data; may want `!fx_rates.csv`).

## Next steps (in order)

1. Commit the current flat baseline (clean: 5 .py files, PROJECT_BRIEF.md,
   README.md, .gitignore). Confirm `.claude/`, __pycache__, *.csv are ignored.
   If `.claude/` was ever tracked: `git rm -r --cached .claude/` first.
2. Build **evaluate.py** — the centrepiece. Threshold sweep; TWO configurable
   cost models (fixed FP:FN ratio + amount-weighted); cost-minimising threshold;
   per-scenario recall breakdown; hard-negative FP rate vs naive single-row
   baseline. NO accuracy as headline metric (brief E4).
3. **Restructure** as its own commit: move the 5 modules into a `fraud_eval/`
   package (with __init__.py), add `tests/` and gitignored `data/`. Imports
   change to `from fraud_eval.x import ...`, run via `python -m fraud_eval.x`.
   Decided to do this AFTER evaluate.py so files only move once.
4. Write the **README** (currently empty) — describe structure + usage; the
   README should reflect the final package layout, so do it after the
   restructure. Hard-negative twin table (brief §5) lifts well into it.

## Open decisions still pending (brief §11)

- Amount-weighted FN cost function: full amount lost / capped at issuer
  liability / flat per-miss. (Default proposal: amount-weighted with a cap.)
- Card aggregation default: `max` vs decaying_sum. (Both built + configurable;
  decaying_sum favoured for card-testing. Eval will show the tradeoff.)
- Trailing-baseline window: count-based vs time-based. (Chose **time-based**;
  velocity_1h/24h already computed.)

## Key context for whoever picks this up

- Repo name for resume: recommended **`fraud-eval-harness`** (skip
  "demo"/"dummy"); human title "Sequence-Aware Fraud Detection Evaluation
  Harness". Directory rename is free — nothing in code depends on it.
- Both target labs (Anthropic FDE / OpenAI Applied AI SA) weight evaluation
  methodology heavily, so evaluate.py is the most interview-relevant artifact.
- Filesystem connector (read/write to ~/Projects) and write_file are enabled.
  GitHub connector shows connected but its tools don't surface mid-session —
  may need a fresh chat to load. Public repos can be read via web fetch.
- Scott reviews each file before it lands; writes in his own voice; dislikes
  hyperbole and the word "shape". Plain, specific, verifiable.
