"""
Run the full fraud-eval pipeline for one (eval_seed, train_seed) pair.

Generates synthetic data, scores with both the RuleScorer and the MLScorer
(trained on a separate seed, always out-of-sample), evaluates each under the
same cost model, and writes per-seed artifacts to runs/seed_<eval_seed>/:

    sweep_rules.csv     metrics_rules.json
    sweep_ml.csv        metrics_ml.json

The cost knobs (FN_FP_RATIO, FP_REVIEW_COST) are fixed constants, not CLI
flags, so every seed's operating points are on a comparable scale. The data-
volume constants (N_CARDS, N_DAYS, FRAUD_RATE) are also fixed here so a
6-seed run is simply six invocations of this script with different seed pairs.

Usage (from repo root, with venv active):
    python scripts/run_seed.py --eval-seed 1 --train-seed 101
    python scripts/run_seed.py --eval-seed 2 --train-seed 102
    ...

Seed pairs (eval, train): (1,101) (2,102) (3,103) (4,104) (5,105) (6,106).
"""

import argparse
import csv
import json
import os
import sys

# Importable from repo root; guard lets the script also be run from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_eval.fx import RATES
from fraud_eval.generate_synthetic import generate
from fraud_eval.profile import build_profiles
from fraud_eval.features import build_features
from fraud_eval.score import RuleScorer, aggregate_cards
from fraud_eval.score_ml import fit_from_rows
from fraud_eval.evaluate import evaluate, render_report

# Fixed across all seeds so every scorer's operating point is comparable.
FN_FP_RATIO = 20.0
FP_REVIEW_COST = 5.0

# Data-volume constants (HANDOFF §8.2: 6 seeds × 3,000 cards × fraud-rate 0.10).
N_CARDS = 3000
N_DAYS = 30
FRAUD_RATE = 0.10
HARD_NEG_RATE = 0.06


def _write_sweep_csv(path, sweep_rows):
    if not sweep_rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)


def _pipeline_for_seed(seed):
    """Generate → profile → features for one seed. Returns featured rows."""
    txn_rows = generate(N_CARDS, N_DAYS, FRAUD_RATE, HARD_NEG_RATE, seed)
    profile_rows = build_profiles(txn_rows, RATES)
    return build_features(txn_rows, profile_rows, RATES)


def _run_scorer(label, scored_rows, scored_cards, out_dir):
    """Evaluate one scorer and write sweep CSV + metrics JSON to out_dir."""
    metrics = evaluate(scored_rows, scored_cards, FN_FP_RATIO, FP_REVIEW_COST)

    sweep_path = os.path.join(out_dir, f"sweep_{label}.csv")
    metrics_path = os.path.join(out_dir, f"metrics_{label}.json")

    _write_sweep_csv(sweep_path, metrics["sweep"])

    # Sweep rows are already on disk as CSV; exclude them from the JSON to keep
    # it small and directly readable by aggregate_runs.py.
    metrics_for_json = {k: v for k, v in metrics.items() if k != "sweep"}
    with open(metrics_path, "w") as f:
        json.dump(metrics_for_json, f, indent=2)

    op = metrics["operating_point"]["fixed_ratio"]
    deg = " [DEGENERATE]" if op["degenerate"] else ""
    print(f"  [{label}] threshold={op['threshold']:.2f}  "
          f"precision={op['precision']:.3f}  recall={op['recall']:.3f}{deg}")

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Run both scorers for one (eval, train) seed pair.")
    ap.add_argument("--eval-seed", type=int, required=True,
                    help="seed for the evaluation dataset")
    ap.add_argument("--train-seed", type=int, required=True,
                    help="seed for the ML training dataset (must differ)")
    args = ap.parse_args()

    if args.eval_seed == args.train_seed:
        ap.error("--eval-seed and --train-seed must differ "
                 "(ML must be trained out-of-sample)")

    out_dir = os.path.join("runs", f"seed_{args.eval_seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== eval_seed={args.eval_seed}  train_seed={args.train_seed}"
          f"  -> {out_dir} ===")
    print(f"data: {N_CARDS} cards, {N_DAYS} days, "
          f"fraud_rate={FRAUD_RATE}, hard_neg_rate={HARD_NEG_RATE}")
    print(f"cost: fn_fp_ratio={FN_FP_RATIO}, fp_review_cost={FP_REVIEW_COST}")

    print(f"\ngenerating eval data  (seed {args.eval_seed})...")
    eval_featured = _pipeline_for_seed(args.eval_seed)

    print(f"generating train data (seed {args.train_seed})...")
    train_featured = _pipeline_for_seed(args.train_seed)

    n_fraud_eval = sum(int(r["is_fraud"]) for r in eval_featured)
    n_fraud_train = sum(int(r["is_fraud"]) for r in train_featured)
    print(f"eval:  {len(eval_featured):,} rows, {n_fraud_eval} fraud rows")
    print(f"train: {len(train_featured):,} rows, {n_fraud_train} fraud rows")

    print("\nscoring with RuleScorer...")
    rule_scorer = RuleScorer()
    scored_rows_rules = [rule_scorer.score_row(r) for r in eval_featured]
    scored_cards_rules = aggregate_cards(scored_rows_rules,
                                         method="decaying_sum", decay=0.9)
    _run_scorer("rules", scored_rows_rules, scored_cards_rules, out_dir)

    print("\ntraining MLScorer and scoring...")
    ml_scorer = fit_from_rows(train_featured)
    scored_rows_ml = [ml_scorer.score_row(r) for r in eval_featured]
    scored_cards_ml = aggregate_cards(scored_rows_ml,
                                       method="decaying_sum", decay=0.9)
    _run_scorer("ml", scored_rows_ml, scored_cards_ml, out_dir)

    print(f"\nwrote: {out_dir}/sweep_rules.csv  metrics_rules.json"
          f"  sweep_ml.csv  metrics_ml.json")
    print("=== done ===\n")


if __name__ == "__main__":
    main()
