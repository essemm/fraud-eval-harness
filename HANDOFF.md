# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-19 (session 4). Pick-up note for resuming, incl. a new chat._

## Where we are

Pipeline complete, packaged, tested, documented. **ML scorer added (by Sonnet,
reviewed) and the rules-vs-ML A/B is done.** README written and tightened.
Next focus: the **visualiser** (spec drafted below).

Repo at `/Users/scott/Projects/fraud-detection/`. Package layout:
`fraud_eval/` holds fx, generate_synthetic, profile, features, scorer (the
protocol), score (RuleScorer), evaluate, and now **score_ml (MLScorer)**.
`tests/` has 27 passing tests. Run via `python -m fraud_eval.<module>` inside
the activated venv (`source .venv/bin/activate`).

## Done this session

- **Reviewed score_ml.py** (Sonnet-generated). Verdict: solid.
  - Strengths: train/eval split on DIFFERENT seeds (out-of-sample, no leakage —
    the standout decision); honours the swap contract (implements Scorer, imports
    aggregate_cards, drops into evaluate.py unchanged); class_weight='balanced'
    for imbalance; reason-string = top-contributing feature (keeps S1).
  - Two watch-items (not urgent): reason-string uses abs(coef*value) so it can
    surface a feature pushing AWAY from fraud as the "reason"; `self._coef`
    accessed from main() is a minor encapsulation smell (a coefficients() method
    would be cleaner).
- **Extended evaluate.py** (Option A): added a third diagnostics block,
  `diagnostics_at_operating_point`, reporting per-scenario recall + hard-negative
  FP at each scorer's OWN fixed-ratio cost-minimum (with degeneracy fallback to
  the reference threshold). Reason: comparing two scorers at a shared 0.50 is
  unfair — a calibrated ML probability and a hand-tuned rule score don't mean the
  same thing at 0.50. Change is additive; all 27 tests still pass. On disk +
  verified by diff.

## The A/B finding (important, and nuanced)

At a fixed 0.50 threshold, ML looked like it won on recall everywhere — but that
was an ARTIFACT of the shared threshold. At each scorer's OWN operating point:

  | scenario          | Rules @0.40 | ML @0.80 |
  | card_testing      | 83%         | 83%      |
  | account_takeover  | 80%         | 20%      |
  | impossible_travel | 22%         | 11%      |
  | stolen_spree      | 57%         | 43%      |
  | hard-neg FP       | 23%         | 3%       |

Real story: the two scorers sit at DIFFERENT points on the precision/recall
frontier. Rules-at-its-optimum is aggressive (higher recall, more false alarms);
ML-at-its-optimum is precise (fewer false alarms, lower recall). Neither strictly
wins — which you prefer depends on the cost model. That IS the harness's thesis.

CAVEAT: only 27 fraud cards in this run, so per-scenario percentages are noisy
(5-8 cards each). Trust the DIRECTION (rules-aggressive vs ML-precise) and the
hard-negative numbers (64 cards, stable), not the specific per-scenario figures,
until averaged over several seeds. Do NOT put specific per-scenario % in the
README as if fixed.

To make the ML feature improvements more legible, the synthetic test data (test
doubles) should double both the number of transaction rows and the number of cards.
More volume reduces per-scenario variance enough that the recall differences between
rules and ML become signal rather than noise, and the feature coefficients in
MLScorer stabilise across seeds.

## NEXT: the visualiser (spec)

Goal: turn evaluate.py's outputs (sweep.csv, metrics.json) into visuals that make
the findings legible at a glance. Reads existing artifacts — does NOT re-run the
pipeline or re-score. A pure consumer of the harness output, which keeps it
cleanly separate from the stdlib-only pipeline.

Placement: its own `viz/` directory at repo root (NOT inside fraud_eval/), with
its own dependency (matplotlib). Keeps the pipeline stdlib-only; the visualiser's
heavier deps stay quarantined. requirements-viz.txt or an extras section.

Core plots (priority order):
  1. Cost-vs-threshold curve — total cost across the sweep, one line per cost
     model, with the cost-minimum marked. Shows WHY a particular threshold is
     chosen and the degenerate-at-extremes behaviour.
  2. Precision-recall curve across the sweep — the threshold-independent view;
     overlay rules vs ML on the SAME axes for the real A/B (curve dominance, not
     point comparison). This is the fair scorer comparison.
  3. Per-scenario recall bar chart — at the operating point; grouped bars
     rules vs ML. The "catches sprees, misses card-testing" story, visual.
  4. Hard-negative FP comparison — sequence-aware vs naive bar, the "earns its
     complexity" point.

Open questions for the spec discussion:
  - Static PNG output (matplotlib savefig) vs interactive (HTML/plotly)? Static
    is simpler, repo-friendly, stdlib-adjacent; interactive is nicer to explore
    but heavier dep + not as clean in a repo. Lean static unless a reason emerges.
  - One figure with subplots, or separate files per plot? Separate is more
    reusable (drop one into a slide/README); subplots tell a single story.
  - Does it read sweep.csv (simple, per-run) or metrics.json (richer, has the
    operating points + diagnostics)? Probably metrics.json for the A/B plots,
    sweep.csv for the curves. May need BOTH a rules metrics.json and an ML one.
  - A/B plots need two metrics.json (rules + ML) — so the viz CLI likely takes
    --rules-metrics and --ml-metrics, or globs a directory.

## Outstanding to-do (from earlier, may already be done)

- `git rm OLD_score_DELETE_ME.py` if not yet done (stale duplicate of old
  root-level score.py).
- Confirm 27 tests pass in place: `python -m pytest tests/ -q`.
- Commit ML scorer + evaluate.py operating-point change.
- **Double the test-double data volume**: increase both transaction row count and
  card count in the synthetic generator (generate_synthetic.py / test fixtures) to
  reduce per-scenario noise and better demonstrate the ML feature improvements
  (see caveat above).

## Key context

- Both target labs (Anthropic FDE / OpenAI Applied AI SA) weight evaluation
  methodology heavily — evaluate.py, the test suite, and the rules-vs-ML A/B
  are the most interview-relevant artifacts. The "fair comparison at operating
  points, not a shared threshold" insight is itself a strong talking point.
- Filesystem connector with write_file enabled (read+write ~/Projects); cannot
  delete files (rename-to-DELETE_ME pattern instead). GitHub connector tools
  don't surface mid-session — fresh chat; public repos via web fetch with a URL.
- Scott reviews each file before it lands; edits independently between rounds;
  plain, specific, non-hyperbolic language; never the word "shape" (use design/
  structure/layout); verifies claims from output artifacts, not just code.
- Sonnet (Claude Code in VS Code) does code generation; this project (claude.ai)
  is for design, review, spec, and verification.
