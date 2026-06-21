# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-21 (session 5). Pick-up note for resuming, incl. a new chat._

## Where we are

Pipeline complete, packaged, tested, documented. ML scorer added (reviewed) and
the rules-vs-ML A/B is done. README written and tightened.

**This session (design only, no code yet):** worked out two specs to hand to
Claude Code, in order. (1) a **multi-seed aggregation harness** to replace the
noisy single-run per-scenario figures, and (2) the **visualiser**, now reading
the aggregated output. PROJECT_BRIEF.md updated to match (§3 architecture, new
§8.1/§8.2, rewritten §12 input contract, new acceptance criteria M1–M4 / V1–V5,
updated §11 resolved decisions). Both specs are reproduced in full below.

**Next focus: hand spec 1 (multi-seed harness) to Claude Code, then spec 2
(visualiser) once aggregate.json + per-seed sweep.csv exist.** The visualiser
cannot run until the harness outputs are on disk, so order matters.

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

## Decisions locked this session (the open questions are now closed)

- **More CARDS, not more days.** Per-scenario variance is driven by fraud-card
  count per scenario, not transaction volume. Days only thicken each card's
  baseline; they do nothing for the per-scenario denominator.
- **Sizing for ±7-point per-scenario CI:** ~200 fraud cards per scenario →
  ~800–900 total → **6 seeds × 3,000 cards × fraud-rate 0.10** (~135 fraud
  cards/run, ~34/scenario/run, pooling to ~200/scenario). Raising fraud-rate is
  legitimate — evaluate.py still reports cost/precision against whatever
  imbalance exists.
- **Multi-seed averaging, measure-then-average:** recall computed per seed, then
  reported as mean ± **sample** std (statistics.stdev, n−1 — NOT pstdev).
  Empirical scatter, not a formula CI. This is the defensible artifact.
- **Train/eval on different seeds** preserved: each eval seed pairs with a
  distinct train seed. Pairs: (1,101)…(6,106). 12 seeds total.
- **Framing:** "which scorer owns which attack" (rules-aggressive vs ML-precise),
  NOT "ML superior." If stabilised means still show a frontier trade-off, that's
  the finding — report it as such. Expect impossible_travel to have the widest
  std (low recall near the floor scatters); if still wide at 6 seeds, that's a
  signal to add seeds, not a bug.

---

## SPEC 1 (do first) — multi-seed A/B aggregation harness

Additive only. Do NOT modify generate_synthetic, profile, features, score,
score_ml, or evaluate — their interfaces are correct. Two new pieces:

**`scripts/run_seed.py`** (new `scripts/` dir at repo root — orchestration, not
pipeline). For one (eval_seed, train_seed) pair, runs the full pipeline twice
into `runs/seed_<eval_seed>/`:
  - ensure fx_rates.csv exists (derive from fraud_eval.fx.RATES if no generator:
    currency,rate_to_usd rows, USD=1.0).
  - EVAL branch: generate(eval_seed) → profile → features → featured_eval.csv;
    RuleScorer → scored rows/cards; evaluate → sweep_rules.csv + metrics_rules.json.
  - TRAIN branch: generate(train_seed) → profile → features → featured_train.csv.
  - ML: score_ml --train-featured featured_train --featured featured_eval → ML
    scored rows/cards; evaluate → sweep_ml.csv + metrics_ml.json.
  - Prefer library calls over -m shelling where functions already return dicts;
    evaluate.evaluate(...) returns the metrics dict — capture and json.dump it.
  - Keep --fn-fp-ratio 20 and --fp-review-cost 5 fixed across all seeds so
    operating points are comparable.

**`fraud_eval/aggregate_runs.py`** (pipeline-adjacent, stdlib-only, no plotting
dep). CLI: --rules-glob 'runs/seed_*/metrics_rules.json' --ml-glob '..._ml.json'.
For each scorer, pull from each seed's `diagnostics_at_operating_point`:
per_scenario_recall (4 scenarios) + hard_negative_analysis (sequence_fp_rate,
naive_fp_rate); also operating_point.fixed_ratio threshold/precision/recall.
Compute per scenario per scorer: mean, sample std (statistics.stdev), n_seeds
contributing (skip None scenarios; report the count). Guard: 1 contributing seed
→ std = null, NOT 0. Emit runs/aggregate.json (structure: per scorer → n_seeds,
operating_point{threshold/precision/recall mean+std}, per_scenario_recall{scenario
→ mean,std,n}, hard_negative{sequence_fp_rate, naive_fp_rate → mean,std}). Also
print a plain text mean±std table to stdout.

Acceptance: A1 metrics files have populated diagnostics_at_operating_point; A2
train≠eval every pair; A3 n=6 where present in all, lower where absent, std=null
only single-seed; A4 sample std (hand-check one scenario's 6 values); A5 existing
27 tests still pass untouched.

Reviewer watch-items: (1) evaluate() must run on ML scored cards with the SAME
ratio/review-cost as rules — comparability is the whole point, easy to get subtly
wrong. (2) impossible_travel std expected highest; large value = add seeds, not a
bug.

---

## SPEC 2 (do after aggregate.json exists) — visualiser (viz/)

Four committed PNGs, static (matplotlib) primary; Plotly --interactive is a
later follow-up, NOT part of acceptance. Lives in viz/ at repo root, own dep in
viz/requirements-viz.txt. Pure consumer — never re-runs pipeline or re-scores.

**Input split (deviation from old brief §12, now reconciled in the brief):**
  - Plots 1 & 2 (curves) read ONE representative seed's sweep_rules.csv +
    sweep_ml.csv (use seed 1). Stated on the figure. No error bars — a curve is
    a per-run object.
  - Plots 3 & 4 (bars) read runs/aggregate.json. Bar height = mean, ERROR BAR =
    sample std across seeds. These are the ONLY plots with error bars.

CLI: python -m viz.make_plots --rules-sweep runs/seed_1/sweep_rules.csv
--ml-sweep runs/seed_1/sweep_ml.csv --aggregate runs/aggregate.json --out-dir
viz/figures/. Separate PNG per plot: 01_cost_vs_threshold.png …
04_hard_negative_fp.png.

  1. Cost vs threshold — x=threshold, two cost lines, minima marked. Fixed-ratio
     (unitless) and amount-weighted (dollars) differ by orders of magnitude →
     STACKED PANELS sharing x-axis, NOT a twin y-axis (buries one curve). Repr.
     seed, labelled.
  2. Precision-recall, scorers overlaid on SAME axes; each operating point
     marked. Repr. seed, labelled.
  3. Per-scenario recall — grouped bars (4 scenarios × rules/ML), mean ± sd over
     6 seeds. Caption = "which scorer owns which attack," NOT "ML wins." Note n
     under groups if it varies.
  4. Hard-negative FP — sequence-aware vs naive, mean ± sd. Lower better, tight
     bars expected.

Style: colourblind-safe, IDENTICAL rules/ML colour across all 4 figs. Provenance
on every fig (1–2 "representative seed N"; 3–4 "mean ± 1 sd over 6 seeds").
null-std scenario → no error bar + caveat, never a zero bar. Word "accuracy"
appears nowhere.

Acceptance: V1 four standalone PNGs; V2 error bars from aggregate.json std,
null-std handled; V3 identical colour map; V4 matplotlib confined to viz/
(grep fraud_eval/ imports); V5 viz imports no pipeline scorer/generator; V6 no
"accuracy" in any title/axis/caption.

Reviewer watch-item: plot 1 dual-cost y-axis is the easy mistake — if Claude Code
reaches for twin y, check neither minimum is visually buried; stacked panels are
the honest render.

---

## Outstanding to-do (from earlier, may already be done)

- `git rm OLD_score_DELETE_ME.py` if not yet done (stale duplicate of old
  root-level score.py).
- Confirm 27 tests pass in place: `python -m pytest tests/ -q`.
- Commit ML scorer + evaluate.py operating-point change.
- **Data-volume question is now RESOLVED** — superseded by Spec 1's multi-seed
  approach (6 × 3,000 cards × fraud-rate 0.10, cards not days). No generator code
  change needed; volume comes from run invocation, variance from seed averaging.

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
