"""
Pool per-seed metrics into mean ± sample-std per scenario.

Reads per-seed metrics_{rules,ml}.json files and aggregates the
diagnostics_at_operating_point block across seeds. Each scorer's operating
point is its own fixed-ratio cost minimum (not a shared threshold), so the
pooled figures represent each scorer at the threshold it would actually run at.

Acceptance criteria (HANDOFF M1–M4):
  M2 — scenarios absent in a seed are skipped, not zero-filled; n reports
        the actual contributing seed count.
  M3 — sample std (statistics.stdev, n−1); single-seed → std = null, not 0.
  M4 — reads diagnostics_at_operating_point, not the reference block.

Usage:
    python -m fraud_eval.aggregate_runs \
        --rules-glob 'runs/seed_*/metrics_rules.json' \
        --ml-glob   'runs/seed_*/metrics_ml.json' \
        --out runs/aggregate.json
"""

import argparse
import glob
import json
import os
import statistics

FRAUD_SCENARIOS = ["card_testing", "account_takeover",
                   "impossible_travel", "stolen_spree"]


def _mean_std(values):
    """Mean and sample std for a non-empty list. Std is None when n == 1."""
    n = len(values)
    if n == 0:
        return None, None
    m = statistics.mean(values)
    s = statistics.stdev(values) if n > 1 else None
    return round(m, 4), (round(s, 4) if s is not None else None)


def _load_metrics(glob_pattern):
    """Return sorted list of (path, dict) pairs matching the glob pattern."""
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"no files matched: {glob_pattern!r}\n"
            "run scripts/run_seed.py for each seed pair first")
    result = []
    for p in paths:
        with open(p) as f:
            result.append((p, json.load(f)))
    return result


def _aggregate_scorer(seed_metrics):
    """
    Aggregate diagnostics_at_operating_point across all seeds for one scorer.

    Returns:
      n_seeds                — number of seed files loaded
      operating_point        — threshold / precision / recall, each {mean, std}
      per_scenario_recall    — {scenario -> {mean, std, n}}  (M2: skip None)
      hard_negative          — {sequence_fp_rate, naive_fp_rate -> {mean, std, n}}
    """
    op_thresholds, op_precisions, op_recalls = [], [], []
    scenario_values = {sc: [] for sc in FRAUD_SCENARIOS}
    seq_fp_rates, naive_fp_rates = [], []

    for path, m in seed_metrics:
        op = m.get("operating_point", {}).get("fixed_ratio", {})
        if op.get("threshold") is not None:
            op_thresholds.append(float(op["threshold"]))
            op_precisions.append(float(op["precision"]))
            op_recalls.append(float(op["recall"]))

        diag = m.get("diagnostics_at_operating_point", {})
        psr = diag.get("per_scenario_recall", {})
        for sc in FRAUD_SCENARIOS:
            val = psr.get(sc)
            if val is not None:          # absent or None scenario: skip (M2)
                scenario_values[sc].append(float(val))

        hn = diag.get("hard_negative_analysis", {})
        if hn.get("sequence_fp_rate") is not None:
            seq_fp_rates.append(float(hn["sequence_fp_rate"]))
        if hn.get("naive_fp_rate") is not None:
            naive_fp_rates.append(float(hn["naive_fp_rate"]))

    per_scenario = {}
    for sc in FRAUD_SCENARIOS:
        vals = scenario_values[sc]
        mn, sd = _mean_std(vals)
        per_scenario[sc] = {"mean": mn, "std": sd, "n": len(vals)}

    thr_m, thr_s = _mean_std(op_thresholds)
    prec_m, prec_s = _mean_std(op_precisions)
    rec_m, rec_s = _mean_std(op_recalls)

    seq_m, seq_s = _mean_std(seq_fp_rates)
    naive_m, naive_s = _mean_std(naive_fp_rates)

    return {
        "n_seeds": len(seed_metrics),
        "operating_point": {
            "threshold": {"mean": thr_m, "std": thr_s},
            "precision": {"mean": prec_m, "std": prec_s},
            "recall": {"mean": rec_m, "std": rec_s},
        },
        "per_scenario_recall": per_scenario,
        "hard_negative": {
            "sequence_fp_rate": {
                "mean": seq_m, "std": seq_s, "n": len(seq_fp_rates)},
            "naive_fp_rate": {
                "mean": naive_m, "std": naive_s, "n": len(naive_fp_rates)},
        },
    }


def _fmt(mean, std):
    """Format mean ± std for the text table; handles None gracefully."""
    if mean is None:
        return "n/a"
    s = f"{mean:.3f}"
    if std is not None:
        s += f" ± {std:.3f}"
    return s


def render_table(agg):
    """Return a plain-text mean ± std summary for stdout."""
    lines = []
    for scorer in ("rules", "ml"):
        data = agg.get(scorer)
        if not data:
            continue
        op = data["operating_point"]
        lines.append(
            f"\n{scorer.upper()}  (n_seeds={data['n_seeds']}  "
            f"op threshold={_fmt(op['threshold']['mean'], op['threshold']['std'])}  "
            f"precision={_fmt(op['precision']['mean'], op['precision']['std'])}  "
            f"recall={_fmt(op['recall']['mean'], op['recall']['std'])})"
        )
        lines.append(f"  {'scenario':<26} {'recall mean ± std':>20}  n")
        lines.append("  " + "-" * 52)
        for sc, v in data["per_scenario_recall"].items():
            lines.append(f"  {sc:<26} {_fmt(v['mean'], v['std']):>20}  {v['n']}")
        lines.append("")
        hn = data["hard_negative"]
        lines.append(
            f"  {'hard_neg sequence_fp_rate':<26} "
            f"{_fmt(hn['sequence_fp_rate']['mean'], hn['sequence_fp_rate']['std']):>20}"
            f"  {hn['sequence_fp_rate']['n']}"
        )
        lines.append(
            f"  {'hard_neg naive_fp_rate':<26} "
            f"{_fmt(hn['naive_fp_rate']['mean'], hn['naive_fp_rate']['std']):>20}"
            f"  {hn['naive_fp_rate']['n']}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-seed metrics into mean ± sample-std.")
    ap.add_argument("--rules-glob",
                    default="runs/seed_*/metrics_rules.json",
                    help="glob for rules metrics files")
    ap.add_argument("--ml-glob",
                    default="runs/seed_*/metrics_ml.json",
                    help="glob for ML metrics files")
    ap.add_argument("--out", default="runs/aggregate.json",
                    help="output path for aggregate JSON")
    args = ap.parse_args()

    rules_metrics = _load_metrics(args.rules_glob)
    ml_metrics = _load_metrics(args.ml_glob)
    print(f"loaded {len(rules_metrics)} rules files, {len(ml_metrics)} ML files")

    agg = {
        "rules": _aggregate_scorer(rules_metrics),
        "ml": _aggregate_scorer(ml_metrics),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"wrote {args.out}")

    print(render_table(agg))


if __name__ == "__main__":
    main()
