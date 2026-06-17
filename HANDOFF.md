# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-17 (session 2). Pick-up note for resuming, incl. a new chat._

## Where we are

**All six pipeline modules built, verified, and on disk:**
`fx.py` → `generate_synthetic.py` → `profile.py` → `features.py` → `score.py` → `evaluate.py`.
**Full pytest suite written and passing (27 tests, ~1s).**

The pipeline is functionally complete. Remaining work is packaging/presentation:
restructure into a package, then the README. Repo is at
`/Users/scott/Projects/fraud-detection/`. Git initialised; flat layout still
(restructure is the next deliberate commit).

## Done this session

- **evaluate.py** — the centrepiece. Card-level decision unit (row-level
  reported as secondary diagnostic). Threshold sweep; two configurable cost
  models (fixed FN:FP ratio, default 20:1; amount-weighted = full USD lost per
  miss + flat per-FP review cost). Reports: cost-minimising operating point
  under each model, per-scenario recall, hard-negative FP rate vs a naive
  single-row baseline. Outputs text report + sweep.csv + metrics.json.
  - Caught and handled a real issue: at aggressive ratios (e.g. 100:1) the
    cost-minimiser degenerates to "flag everything" (threshold 0). Fix: lowered
    default to 20:1 AND added degeneracy detection that names the flag-all case
    instead of presenting threshold 0 as a real operating point. Diagnostics are
    reported at TWO stated reference points (fixed 0.50, and target-recall 90%),
    never at the raw cost-min, so they stay meaningful.
  - Real findings on a 1500-card run: per-scenario recall at 0.50 was
    card_testing 71% / account_takeover 44% / impossible_travel 23% /
    stolen_spree 29% (the spread a blended number would hide). Hard-negative FP:
    sequence-aware 14% vs naive single-row 68% — the sequence approach earns its
    keep by ~5x fewer false positives. Verified E1–E4 + determinism.
- **tests/** — pytest suite, one test per acceptance criterion in brief §10,
  named for the criterion (test_G1_..., test_F1_..., etc.). conftest.py builds a
  small in-memory dataset via fixtures (no disk I/O, per NFR §9). Covers
  G1–G6, X1, P1–P5, F1–F4, S1–S3, E1–E4, plus 2 extra property tests
  (aggregation-methods-differ, degeneracy-detected). All 27 pass.
  - F1 (no-leakage) test independently reconstructs the trailing baseline from
    prior rows only and asserts the code matches — verifies the central claim
    from outside the implementation. S2 swaps in a stub scorer to prove the
    pluggability contract.

## Earlier (session 1) — currency dimension + bug fixes

- Added currency: transactions carry native amount + ISO-4217 currency;
  fx_rates.csv maps to USD; fx.py shared to_usd helper; profile + features
  normalise to USD; native amount/currency ride through for reporting.
- Fixed 3 generator bugs: G1 reproducibility (uuid4 → seeded _rid), currency-
  blind amounts (now USD-equivalent scaled), dead conditional in card_testing.
- PROJECT_BRIEF.md updated (fx_rates §4.4, currency invariants, §5 neutrality,
  G5/G6/X1/P5/F4). Language: Known limitations / Files and interfaces / "design"
  not "shape".
- .gitignore: .claude/ ignored; *.csv ignored (so fx_rates.csv currently
  uncommitted — may want !fx_rates.csv since it's static reference data).

## Next steps (in order)

1. **Restructure** as its own commit: move the 6 modules into a `fraud_eval/`
   package (with __init__.py); keep tests/ at root; add gitignored data/ for
   CSV outputs. Imports change `import fx` → `from fraud_eval import fx`; run via
   `python -m fraud_eval.generate_synthetic`. conftest.py's sys.path hack gets
   replaced by proper package imports. Do modules + tests + brief diagram in ONE
   pass — the tests will catch any missed import.
2. Add **requirements.txt**: runtime is stdlib-only; pytest is the one dev
   dependency. Note the runtime-vs-dev distinction.
3. Write the **README** (currently empty) — reflect the final package layout, so
   do it AFTER the restructure. The hard-negative twin table (brief §5) and the
   real eval findings above lift well into it. Repo name on resume:
   **fraud-eval-harness** (skip demo/dummy); title "Sequence-Aware Fraud
   Detection Evaluation Harness". Directory rename is free.
4. (Optional, later) the visualiser — reads sweep.csv / metrics.json to plot the
   threshold sweep and per-scenario recall. Belongs in its own viz/ dir with its
   own deps (matplotlib), separate from the stdlib-only pipeline.

## Open decisions (brief §11) — mostly resolved now

- Amount-weighted FN cost: **chose full amount lost** (capping at issuer
  liability documented as future refinement).
- Card aggregation: both built + configurable; **decaying_sum** is the default;
  eval shows the max-vs-decaying tradeoff.
- Trailing window: **time-based** (velocity_1h/24h).
- Still open / minor: evaluate.py hardcodes reference threshold 0.50 and target
  recall 0.90 — could expose as CLI args (Scott to decide on his run).

## Key context

- Both target labs (Anthropic FDE / OpenAI Applied AI SA) weight evaluation
  methodology heavily — evaluate.py + the test suite are the most
  interview-relevant artifacts.
- Filesystem connector with write_file is enabled (can read+write ~/Projects).
  GitHub connector shows connected but tools don't surface mid-session — try a
  fresh chat. Public repos readable via web fetch with a URL.
- Scott reviews each file before it lands; edits independently between rounds;
  plain, specific, non-hyperbolic language; never the word "shape" (use design/
  structure/layout); verifies claims from output, not just code.
