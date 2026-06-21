# Project checkpoint — fraud-detection (Sequence-Aware Fraud Detection Eval Harness)

_Last updated: 2026-06-21 (session 6). Pick-up note for resuming, incl. a new chat._

## Where we are

Pipeline complete. Multi-seed aggregation harness complete and run (6 seeds).
Visualiser complete (4 PNGs committed). README needs updating with final figures
and multi-seed A/B findings.

**SPEC 1 — DONE.** `scripts/run_seed.py` + `fraud_eval/aggregate_runs.py` built
and verified. 6-seed run complete; `runs/aggregate.json` on disk.

**SPEC 2 — DONE.** `viz/make_plots.py` built and run. Four PNGs in `viz/figures/`.

**Next focus:** Update README with the final multi-seed A/B findings and embed
the four committed PNGs.

Repo at `/Users/scott/Projects/fraud-detection/`. Package layout:
`fraud_eval/` holds fx, generate_synthetic, profile, features, scorer (the
protocol), score (RuleScorer), score_ml (MLScorer), evaluate, aggregate_runs.
`scripts/` holds run_seed.py (orchestration). `viz/` holds make_plots.py.
`tests/` has 33 passing tests. Run via `python -m fraud_eval.<module>` inside
the activated venv (`source .venv/bin/activate`).

## Multi-seed A/B findings (stabilised, 6 seeds × 3,000 cards × fraud-rate 0.10)

Seed pairs (eval, train): (1,101)…(6,106). Data in `runs/seed_*/`.

At each scorer's OWN cost-minimising operating point (not a shared threshold):

  | scenario              | Rules @0.05     | ML @~0.30       |
  |-----------------------|-----------------|-----------------|
  | card_testing          | 0.850 ± 0.042   | 0.951 ± 0.034   |
  | account_takeover      | 0.853 ± 0.038   | 0.907 ± 0.018   |
  | impossible_travel     | 0.774 ± 0.055   | 0.871 ± 0.063   |
  | stolen_spree          | 0.824 ± 0.066   | 0.909 ± 0.037   |
  | hard-neg seq FP rate  | 0.625 ± 0.026   | 0.818 ± 0.046   |
  | hard-neg naive FP rate| 0.743 ± 0.030   | 0.743 ± 0.030   |

Real story: ML achieves higher recall across all scenarios; rules have a
meaningfully lower hard-negative FP rate (0.625 vs 0.818). The rules scorer
runs at threshold=0.05 (aggressive). This is a precision/recall frontier
trade-off, not ML dominance.

Note: the rules scorer's operating point at 0.05 is not degenerate (recall
~0.83, not ~1.0), but it is at the bottom of the threshold sweep. The
decaying_sum aggregation accumulates many small rule-fires into scores above
0.05; even a single fired rule pushes most cards above this threshold.

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

## SPEC 1 — DONE (multi-seed A/B aggregation harness)

`scripts/run_seed.py` and `fraud_eval/aggregate_runs.py` built and verified.
All acceptance criteria met (A1–A5). 6-seed run output in `runs/`.
See "Multi-seed A/B findings" section above for results.

---

`viz/make_plots.py` built and run. Four PNGs in `viz/figures/`.
All acceptance criteria met (V1–V6). See viz/figures/ for output.

---

## Outstanding to-do

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
