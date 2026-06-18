"""
Cost-weighted evaluation harness.

This is the centrepiece. It does not ask "is the scorer accurate?" -- at a
sub-1% fraud rate accuracy is actively misleading (a do-nothing model scores
~99.7%). It asks the questions that matter operationally:

  - At what decision threshold should the scorer run?
  - What does being wrong cost, given that a miss and a false alarm are not
    equally expensive?
  - Which attack types does it catch, and which does it miss? (A single
    blended recall hides that a detector may catch sprees and miss card
    testing entirely.)
  - Does the sequence-aware approach actually beat a naive single-row
    threshold on the hard negatives -- i.e. does the complexity earn itself?

Unit of decision is the CARD (accounts get actioned, not rows). A card counts
as fraud if it contains any fraud row. Row-level metrics are also reported, as
a diagnostic: they reveal whether the scorer fires on the actually-fraudulent
transactions or merely somewhere on the right card.

Two cost models, both configurable knobs:
  - fixed_ratio: a false negative costs N times a false positive (default
    20:1). Captures the asymmetry with no dollar figures. (At very aggressive
    ratios against severe class imbalance this minimiser can degenerate to
    "flag everything"; the report detects and names that case rather than
    presenting it as a usable operating point.)
  - amount_weighted: each missed fraud costs the full USD amount of its
    fraudulent transactions; each false positive costs a flat per-review
    operating cost. Now the threshold trades real dollars against review
    labour. (Full-amount-lost is the v1 policy; capping at issuer liability
    is a documented future refinement.)

Outputs: a printed text report, a threshold-sweep CSV (for plotting), and a
JSON metrics dump (for programmatic comparison, e.g. rules-vs-ML later).

Usage:
    python -m fraud_eval.evaluate --rows scored_rows.csv --cards scored_cards.csv \
        --fn-fp-ratio 20 --fp-review-cost 5 \
        --report-out report.txt --sweep-out sweep.csv --json-out metrics.json
"""

import argparse
import csv
import json
from collections import defaultdict

FRAUD_SCENARIOS = ["card_testing", "account_takeover",
                   "impossible_travel", "stolen_spree"]


# --- confusion counting ---------------------------------------------------

def confusion_at(items, threshold):
    """items: list of (score, is_fraud). Returns TP, FP, TN, FN at threshold.
    'Flagged' means score >= threshold."""
    tp = fp = tn = fn = 0
    for score, is_fraud in items:
        flagged = score >= threshold
        if is_fraud and flagged:
            tp += 1
        elif is_fraud and not flagged:
            fn += 1
        elif not is_fraud and flagged:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


# --- cost models ----------------------------------------------------------

def cost_fixed_ratio(tp, fp, tn, fn, ratio):
    """FN costs `ratio` times an FP; FP costs 1 unit. Unitless."""
    return fn * ratio + fp * 1.0


def cost_amount_weighted(flagged_fraud_loss, missed_fraud_loss,
                         fp_count, fp_review_cost):
    """Missed fraud costs the dollars lost; each false positive costs a flat
    review cost. (TP/caught fraud is assumed prevented -> no loss.)"""
    return missed_fraud_loss + fp_count * fp_review_cost


# --- the sweep ------------------------------------------------------------

def sweep(card_items, card_fraud_amounts, thresholds, ratio, fp_review_cost):
    """card_items: list of (score, is_fraud) per card.
       card_fraud_amounts: dict card_idx -> total fraud USD on that card,
       aligned by position with card_items.
    Returns a list of per-threshold metric dicts."""
    rows = []
    for thr in thresholds:
        tp = fp = tn = fn = 0
        missed_loss = 0.0
        for (score, is_fraud), loss in zip(card_items, card_fraud_amounts):
            flagged = score >= thr
            if is_fraud and flagged:
                tp += 1
            elif is_fraud and not flagged:
                fn += 1
                missed_loss += loss      # this fraud got through
            elif not is_fraud and flagged:
                fp += 1
            else:
                tn += 1
        precision, recall = precision_recall(tp, fp, fn)
        rows.append({
            "threshold": round(thr, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "cost_fixed_ratio": round(cost_fixed_ratio(tp, fp, tn, fn, ratio), 2),
            "cost_amount_weighted": round(
                cost_amount_weighted(0, missed_loss, fp, fp_review_cost), 2),
            "missed_fraud_loss": round(missed_loss, 2),
        })
    return rows


def min_cost_row(sweep_rows, cost_key):
    """The threshold row minimising the given cost column (lowest threshold
    wins ties, so the most-sensitive equivalent operating point is chosen)."""
    return min(sweep_rows, key=lambda r: (r[cost_key], r["threshold"]))


def is_degenerate(row, sweep_rows):
    """A cost-minimising point is degenerate if it flags (nearly) everything:
    the threshold sits at the bottom of the range and recall is ~1 because the
    model is flagging the whole population, not discriminating. This signals
    the cost ratio overwhelms prevalence -- a property worth naming, not a
    usable operating point."""
    max_flagged = max(r["tp"] + r["fp"] for r in sweep_rows)
    flagged = row["tp"] + row["fp"]
    return flagged >= 0.95 * max_flagged and row["recall"] >= 0.99


def threshold_for_target_recall(sweep_rows, target):
    """Highest threshold (most precise) that still achieves at least `target`
    recall. This is an operationally meaningful point: 'tightest we can run
    while still catching `target` of fraud.'"""
    qualifying = [r for r in sweep_rows if r["recall"] >= target]
    if not qualifying:
        return None
    return max(qualifying, key=lambda r: r["threshold"])


# --- per-scenario recall --------------------------------------------------

def per_scenario_recall(cards, threshold):
    """For each fraud scenario, recall = fraction of cards truly of that
    scenario that are flagged at the threshold."""
    out = {}
    for sc in FRAUD_SCENARIOS:
        relevant = [c for c in cards if c["fraud_scenario"] == sc]
        if not relevant:
            out[sc] = None
            continue
        flagged = sum(1 for c in relevant
                      if float(c["card_score"]) >= threshold)
        out[sc] = round(flagged / len(relevant), 4)
    return out


# --- hard-negative analysis ----------------------------------------------

def hard_negative_fp(cards, rows, threshold, naive_amount_ratio=4.0):
    """FP rate on hard-negative cards under (a) the sequence-aware card score
    and (b) a naive single-row baseline that flags any card with a row whose
    amount exceeds `naive_amount_ratio` x its static median. Lower-is-better
    comparison demonstrates whether the sequence approach earns its keep."""
    hn_cards = [c for c in cards if c["any_fraud"] == 0
                and c.get("has_hard_neg")]
    n = len(hn_cards)
    if n == 0:
        return {"n_hard_neg_cards": 0}

    seq_fp = sum(1 for c in hn_cards
                 if float(c["card_score"]) >= threshold)

    # naive baseline: per-card, does any row trip the amount ratio alone?
    rows_by_card = defaultdict(list)
    for r in rows:
        rows_by_card[r["card_id"]].append(r)
    naive_fp = 0
    for c in hn_cards:
        tripped = any(
            float(r["amount_vs_static_median"]) >= naive_amount_ratio
            for r in rows_by_card[c["card_id"]])
        if tripped:
            naive_fp += 1

    return {
        "n_hard_neg_cards": n,
        "sequence_fp": seq_fp,
        "sequence_fp_rate": round(seq_fp / n, 4),
        "naive_fp": naive_fp,
        "naive_fp_rate": round(naive_fp / n, 4),
    }


# --- assembly -------------------------------------------------------------

def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_card_index(rows):
    """Reconstruct per-card facts evaluate needs that may not be in the card
    file: total fraud USD on the card, and whether it carries a hard-negative
    row. Keyed by card_id."""
    fraud_loss = defaultdict(float)
    has_hn = defaultdict(bool)
    for r in rows:
        if int(r["is_fraud"]) == 1:
            fraud_loss[r["card_id"]] += float(r["amount_usd"])
        if r["scenario"].startswith("hard_neg"):
            has_hn[r["card_id"]] = True
    return fraud_loss, has_hn


def evaluate(rows, cards, ratio, fp_review_cost, step=0.05):
    fraud_loss, has_hn = build_card_index(rows)
    for c in cards:
        c["any_fraud"] = int(c["any_fraud"])
        c["has_hard_neg"] = has_hn.get(c["card_id"], False)

    card_items = [(float(c["card_score"]), c["any_fraud"]) for c in cards]
    card_losses = [fraud_loss.get(c["card_id"], 0.0) for c in cards]

    n_steps = int(round(1.0 / step))
    thresholds = [i * step for i in range(n_steps + 1)]
    sweep_rows = sweep(card_items, card_losses, thresholds, ratio,
                       fp_review_cost)

    best_fixed = min_cost_row(sweep_rows, "cost_fixed_ratio")
    best_amount = min_cost_row(sweep_rows, "cost_amount_weighted")

    # Diagnostics (per-scenario recall, hard-negative FP, row-level) are
    # reported at two STATED reference points, never at the raw cost-minimiser
    # -- because at aggressive ratios the cost-minimiser can degenerate to
    # "flag everything", at which every diagnostic trivially reads 100%.
    #   - reference: a fixed threshold (0.50) -- a neutral midpoint.
    #   - target-recall: the tightest threshold still catching `target_recall`.
    target_recall = 0.90
    ref_row = next((r for r in sweep_rows
                    if abs(r["threshold"] - 0.50) < 1e-9), sweep_rows[0])
    tr_row = threshold_for_target_recall(sweep_rows, target_recall)

    def diagnostics_at(thr):
        return {
            "threshold": round(thr, 4),
            "per_scenario_recall": per_scenario_recall(cards, thr),
            "hard_negative_analysis": hard_negative_fp(cards, rows, thr),
        }

    diag_reference = diagnostics_at(ref_row["threshold"])
    diag_target = (diagnostics_at(tr_row["threshold"])
                   if tr_row is not None else None)

    # row-level diagnostic at the fixed reference threshold
    row_items = [(float(r["score"]), int(r["is_fraud"])) for r in rows]
    rtp, rfp, rtn, rfn = confusion_at(row_items, ref_row["threshold"])
    row_prec, row_rec = precision_recall(rtp, rfp, rfn)

    return {
        "params": {
            "fn_fp_ratio": ratio,
            "fp_review_cost": fp_review_cost,
            "threshold_step": step,
            "target_recall": target_recall,
            "reference_threshold": ref_row["threshold"],
            "n_cards": len(cards),
            "n_fraud_cards": sum(c["any_fraud"] for c in cards),
            "n_rows": len(rows),
        },
        "sweep": sweep_rows,
        "operating_point": {
            "fixed_ratio": {
                "threshold": best_fixed["threshold"],
                "cost": best_fixed["cost_fixed_ratio"],
                "precision": best_fixed["precision"],
                "recall": best_fixed["recall"],
                "degenerate": is_degenerate(best_fixed, sweep_rows),
            },
            "amount_weighted": {
                "threshold": best_amount["threshold"],
                "cost_usd": best_amount["cost_amount_weighted"],
                "precision": best_amount["precision"],
                "recall": best_amount["recall"],
                "missed_fraud_loss_usd": best_amount["missed_fraud_loss"],
                "degenerate": is_degenerate(best_amount, sweep_rows),
            },
        },
        "diagnostics_at_reference": diag_reference,
        "diagnostics_at_target_recall": diag_target,
        "row_level_diagnostic": {
            "threshold": ref_row["threshold"],
            "tp": rtp, "fp": rfp, "tn": rtn, "fn": rfn,
            "precision": round(row_prec, 4),
            "recall": round(row_rec, 4),
        },
    }


# --- reporting ------------------------------------------------------------

def render_report(m):
    p = m["params"]
    L = []
    L.append("=" * 70)
    L.append("FRAUD SCORER EVALUATION  (card-level decision unit)")
    L.append("=" * 70)
    L.append(f"cards: {p['n_cards']}  ({p['n_fraud_cards']} fraud)   "
             f"rows: {p['n_rows']}")
    L.append(f"cost knobs: FN:FP = {p['fn_fp_ratio']}:1   "
             f"FP review cost = ${p['fp_review_cost']}")
    L.append("")
    L.append("NOTE: accuracy is deliberately not reported. At this class "
             "imbalance")
    L.append("      a do-nothing model scores ~"
             f"{100*(1 - p['n_fraud_cards']/p['n_cards']):.1f}% accuracy while "
             "catching zero fraud.")
    L.append("")

    of = m["operating_point"]["fixed_ratio"]
    oa = m["operating_point"]["amount_weighted"]
    L.append("-" * 70)
    L.append("COST-MINIMISING OPERATING POINT")
    L.append("-" * 70)
    L.append(f"  fixed-ratio ({p['fn_fp_ratio']}:1):  threshold "
             f"{of['threshold']:.2f}   cost {of['cost']:.0f} units   "
             f"precision {of['precision']:.3f}  recall {of['recall']:.3f}")
    if of["degenerate"]:
        L.append("      [DEGENERATE: this minimum flags ~everything. The cost "
                 "ratio overwhelms")
        L.append("       prevalence, so pure cost-minimisation says 'flag all' "
                 "-- not a usable")
        L.append("       operating point. Lower the FN:FP ratio or raise the "
                 "FP cost.]")
    L.append(f"  amount-weighted:    threshold {oa['threshold']:.2f}   "
             f"cost ${oa['cost_usd']:,.0f}   "
             f"precision {oa['precision']:.3f}  recall {oa['recall']:.3f}")
    if oa["degenerate"]:
        L.append("      [DEGENERATE: flags ~everything; see note above.]")
    if of["threshold"] != oa["threshold"]:
        L.append(f"  -> the two cost models prefer DIFFERENT operating points "
                 f"({of['threshold']:.2f} vs {oa['threshold']:.2f}); the choice "
                 "of cost model is itself a business decision.")
    L.append("")

    # diagnostics are reported at stated reference points, not the cost-min
    def render_diag(title, diag):
        if diag is None:
            L.append(f"  {title}: target recall not achievable on this data")
            L.append("")
            return
        thr = diag["threshold"]
        L.append("-" * 70)
        L.append(f"{title}  (at threshold {thr:.2f})")
        L.append("-" * 70)
        L.append("  per-scenario recall:")
        for sc, r in diag["per_scenario_recall"].items():
            bar = "n/a" if r is None else f"{r:.0%}"
            L.append(f"    {sc:20s} {bar}")
        hn = diag["hard_negative_analysis"]
        if hn.get("n_hard_neg_cards", 0) == 0:
            L.append("  hard-negative FP: no hard-negative cards present")
        else:
            L.append(f"  hard-negative FP  (n={hn['n_hard_neg_cards']}):")
            L.append(f"    sequence-aware scorer: {hn['sequence_fp']} "
                     f"({hn['sequence_fp_rate']:.0%})")
            L.append(f"    naive single-row rule: {hn['naive_fp']} "
                     f"({hn['naive_fp_rate']:.0%})")
            delta = hn["naive_fp_rate"] - hn["sequence_fp_rate"]
            verdict = ("sequence approach reduces hard-negative FPs"
                       if delta > 0 else
                       "sequence approach does not beat the naive rule here")
            L.append(f"    -> {verdict}")
        L.append("")

    render_diag(f"DIAGNOSTICS @ reference threshold {p['reference_threshold']:.2f}",
                m["diagnostics_at_reference"])
    render_diag(f"DIAGNOSTICS @ target recall {p['target_recall']:.0%}",
                m["diagnostics_at_target_recall"])
    L.append("  (a single blended recall would hide divergence between "
             "scenarios)")
    L.append("")

    rl = m["row_level_diagnostic"]
    L.append("-" * 70)
    L.append(f"ROW-LEVEL DIAGNOSTIC  (at threshold {rl['threshold']:.2f}; "
             "secondary)")
    L.append("-" * 70)
    L.append(f"  TP {rl['tp']}  FP {rl['fp']}  TN {rl['tn']}  FN {rl['fn']}   "
             f"precision {rl['precision']:.3f}  recall {rl['recall']:.3f}")
    L.append("  (row-level reveals whether the scorer fires on the actual "
             "fraud rows,")
    L.append("   not merely somewhere on a correctly-flagged card)")
    L.append("=" * 70)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="scored_rows.csv")
    ap.add_argument("--cards", default="scored_cards.csv")
    ap.add_argument("--fn-fp-ratio", type=float, default=20.0,
                    help="fixed-ratio model: FN cost as a multiple of FP cost")
    ap.add_argument("--fp-review-cost", type=float, default=5.0,
                    help="amount-weighted model: flat $ cost per false positive")
    ap.add_argument("--step", type=float, default=0.05,
                    help="threshold sweep granularity")
    ap.add_argument("--report-out", default="report.txt")
    ap.add_argument("--sweep-out", default="sweep.csv")
    ap.add_argument("--json-out", default="metrics.json")
    args = ap.parse_args()

    rows = load_rows(args.rows)
    cards = load_rows(args.cards)

    m = evaluate(rows, cards, args.fn_fp_ratio, args.fp_review_cost, args.step)

    report = render_report(m)
    print(report)

    with open(args.report_out, "w") as f:
        f.write(report + "\n")

    sweep_fields = list(m["sweep"][0].keys())
    with open(args.sweep_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields)
        w.writeheader()
        w.writerows(m["sweep"])

    with open(args.json_out, "w") as f:
        json.dump(m, f, indent=2)

    print(f"\nwrote {args.report_out}, {args.sweep_out}, {args.json_out}")


if __name__ == "__main__":
    main()
