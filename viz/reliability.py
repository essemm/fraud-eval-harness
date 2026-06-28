"""
Row-level ML probability reliability diagnostic (brief §8.3, acceptance
E6–E9 / V6–V8).

This answers one narrow question: when the ML scorer assigns a row a score
near p, is the observed fraud rate among those rows also near p? That is, can
the ML *row* score be read as a probability before it is aggregated to a
card-level operating score? It is NOT a threshold selector, and it says
nothing about the card-level decaying-sum score, which is an operating score,
not a calibrated probability.

Scope rules baked in here:

  - Reads HELD-OUT row-level scored rows only (scored_rows_ml.csv), each row
    carrying a `score` in [0, 1] and a binary `is_fraud` label. It never reads
    training rows and never reads card-level aggregated scores (E6).
  - Bins predicted scores into fixed-width intervals over [0, 1]; the final
    bin is closed so a score of exactly 1.0 lands in the top bin.
  - Reports, per populated bin: bounds, row count, mean predicted probability,
    observed fraud rate (E7). Empty bins are skipped, and the populated count
    is reported alongside the total bin count.
  - Reports the Brier score (mean squared error between row score and label)
    and the count-weighted expected calibration error (E8).

The word "accuracy" appears nowhere in this module's output — not in the JSON
keys, not in the text report, not in the file name it suggests (E9 / V5). The
brief forbids it for this diagnostic specifically.

This module is stdlib-only and is a pure consumer of on-disk artifacts: it
imports nothing from fraud_eval/ and no plotting library. The diagram itself
is drawn in viz/make_plots.py.

Usage:
    python -m viz.reliability \\
        --rows runs/seed_1/scored_rows_ml.csv \\
        --out runs/seed_1/reliability_ml.json \\
        --bins 10 [--report]

Or as a library:
    from viz.reliability import compute_reliability
    result = compute_reliability(scored_rows, n_bins=10, source="...")
"""

import argparse
import csv
import json


def _bin_index(score, n_bins):
    """Index of the fixed-width bin a score falls in, over [0, 1].

    The final bin is closed on the right so score == 1.0 lands in the top bin
    rather than overflowing to index n_bins."""
    idx = int(score * n_bins)
    return min(idx, n_bins - 1)


def compute_reliability(rows, n_bins=10, source=None):
    """Compute the reliability diagnostic from row-level scored rows.

    `rows` is an iterable of dicts, each with a `score` (predicted probability
    in [0, 1]) and an `is_fraud` (0/1) field. Only those two fields are read:
    this is a row-level probability check, never a card-level one (E6).

    Returns a dict with: source, n_rows, n_bins, n_populated_bins,
    brier_score, expected_calibration_error, and a `bins` list of populated
    bins, each {bin_lower, bin_upper, count, mean_predicted_probability,
    observed_fraud_rate}. Bins are reported in ascending order of bound;
    empty bins are omitted (E7).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    width = 1.0 / n_bins
    sums_pred = [0.0] * n_bins   # sum of predicted scores in each bin
    sums_label = [0.0] * n_bins  # sum of labels (fraud count) in each bin
    counts = [0] * n_bins

    n_rows = 0
    brier_total = 0.0
    for r in rows:
        score = float(r["score"])
        label = int(r["is_fraud"])
        n_rows += 1
        brier_total += (score - label) ** 2

        b = _bin_index(score, n_bins)
        sums_pred[b] += score
        sums_label[b] += label
        counts[b] += 1

    if n_rows == 0:
        raise ValueError("no rows to evaluate")

    bins = []
    ece = 0.0
    for i in range(n_bins):
        if counts[i] == 0:
            continue  # empty bins are skipped (E7)
        mean_pred = sums_pred[i] / counts[i]
        obs_rate = sums_label[i] / counts[i]
        # count-weighted gap between observed rate and mean predicted prob
        ece += (counts[i] / n_rows) * abs(obs_rate - mean_pred)
        bins.append({
            "bin_lower": round(i * width, 6),
            "bin_upper": round((i + 1) * width, 6),
            "count": counts[i],
            "mean_predicted_probability": round(mean_pred, 6),
            "observed_fraud_rate": round(obs_rate, 6),
        })

    return {
        "source": source,
        "n_rows": n_rows,
        "n_bins": n_bins,
        "n_populated_bins": len(bins),
        "brier_score": round(brier_total / n_rows, 6),
        "expected_calibration_error": round(ece, 6),
        "bins": bins,
    }


def render_report(result):
    """Plain-text summary of a reliability result. No 'accuracy' anywhere."""
    lines = []
    lines.append("ML row-level probability reliability")
    lines.append("=" * 40)
    src = result.get("source") or "(in-memory rows)"
    lines.append(f"source:                       {src}")
    lines.append(f"rows:                         {result['n_rows']}")
    lines.append(f"bins (total / populated):     "
                 f"{result['n_bins']} / {result['n_populated_bins']}")
    lines.append(f"Brier score:                  {result['brier_score']:.4f}")
    lines.append(f"expected calibration error:   "
                 f"{result['expected_calibration_error']:.4f}")
    lines.append("")
    lines.append(f"{'bin':>12}  {'count':>7}  {'mean_pred':>9}  {'obs_rate':>9}")
    for b in result["bins"]:
        rng = f"[{b['bin_lower']:.2f},{b['bin_upper']:.2f})"
        lines.append(f"{rng:>12}  {b['count']:>7}  "
                     f"{b['mean_predicted_probability']:>9.4f}  "
                     f"{b['observed_fraud_rate']:>9.4f}")
    return "\n".join(lines)


def _load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description="Compute the ML row-level probability reliability "
                    "diagnostic from held-out scored rows.")
    ap.add_argument("--rows", default="runs/seed_1/scored_rows_ml.csv",
                    help="held-out ML scored-rows CSV (needs score, is_fraud)")
    ap.add_argument("--out", default="runs/seed_1/reliability_ml.json",
                    help="output JSON path")
    ap.add_argument("--bins", type=int, default=10,
                    help="number of fixed-width probability bins over [0,1]")
    ap.add_argument("--report", action="store_true",
                    help="also print a plain-text report to stdout")
    args = ap.parse_args()

    rows = _load_rows(args.rows)
    result = compute_reliability(rows, n_bins=args.bins, source=args.rows)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {args.out}  "
          f"(rows={result['n_rows']}, populated bins="
          f"{result['n_populated_bins']}, Brier={result['brier_score']:.4f}, "
          f"ECE={result['expected_calibration_error']:.4f})")

    if args.report:
        print()
        print(render_report(result))


if __name__ == "__main__":
    main()
