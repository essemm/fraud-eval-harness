"""
Score featured transactions and aggregate to an account-level decision.

Two layers, deliberately separated:

  1. A PER-ROW scorer that maps a featured row to a score in [0, 1] plus a
     human-readable REASON naming which rule fired. The reason string is a
     hard requirement (brief S1): it is what makes the baseline explainable
     and it is reused verbatim in the stakeholder / README narrative and,
     later, by the agentic investigation layer (brief §12).

  2. A CARD-LEVEL aggregator that rolls per-row scores up to one decision
     per account. The aggregation function is an explicit, named design
     choice (brief §7), configurable between:
       - 'max'         : the single most suspicious row decides. Simple and
                         transparent; best when one big signal = fraud.
       - 'decaying_sum': suspicion accumulates across rows with exponential
                         decay, so MANY SMALL signals (card-testing: a dozen
                         tiny scores) can outweigh one isolated medium score.

PLUGGABILITY (brief S2): scoring goes through the `Scorer` protocol. The v1
`RuleScorer` is one implementation. An ML model is a later swap-in behind the
same `score_row` signature; features.py and evaluate.py never learn which
scorer they are talking to.

v1 rules map directly to the fraud fingerprints in brief §5. Each rule
contributes evidence; the row score is the strongest single piece of evidence
(a max over rule outputs), and the reason names that rule. Keeping per-rule
contributions visible -- rather than blending into one opaque number -- is the
whole point of the transparent baseline.

Usage:
    python -m fraud_eval.score --in featured.csv --agg decaying_sum \
        --row-out scored_rows.csv --card-out scored_cards.csv

Or as a library:
    from fraud_eval.score import RuleScorer, aggregate_cards
    scorer = RuleScorer()
    scored = [scorer.score_row(r) for r in featured_rows]
    cards  = aggregate_cards(scored, method="max")
"""

import argparse
import csv
import math
from collections import defaultdict

from .scorer import Scorer


# --- Tunable rule thresholds ---------------------------------------------
# Collected here so the baseline's behaviour is inspectable and adjustable
# in one place rather than scattered through the logic.

VELOCITY_1H_BURST = 6        # >= this many txns in a trailing hour -> burst
AMOUNT_SPIKE_RATIO = 4.0     # amount this many x the trailing median -> spike
HIGH_RISK_CATEGORIES = {"electronics", "gift_card", "crypto", "travel"}
# Country change with very little elapsed time = no plausible travel.
IMPOSSIBLE_TRAVEL_MAX_SECS = 3 * 3600  # under 3h between countries


class RuleScorer:
    """Transparent rule baseline. Each rule yields (score, reason); the row
    takes the strongest-firing rule as its score and that rule's reason.

    Satisfies the `Scorer` protocol (.scorer) structurally -- it implements
    `score_row`, which is all the protocol requires; no explicit inheritance
    needed."""

    def score_row(self, row: dict) -> dict:
        candidates = []  # (score, reason)

        v1h = int(row["velocity_1h"])
        amt_vs_trail = float(row["amount_vs_trailing_median"])
        trail_low_conf = int(row["trailing_low_confidence"])
        secs = int(row["secs_since_prev"])
        new_device = int(row["is_new_device"])
        country_change = int(row["is_country_change"])
        category = row["merchant_category"]

        # Rule 1: velocity burst (card_testing fingerprint).
        # Scales with how far over the burst threshold we are.
        if v1h >= VELOCITY_1H_BURST:
            s = min(1.0, 0.6 + 0.05 * (v1h - VELOCITY_1H_BURST))
            candidates.append((s, f"velocity_burst: {v1h} txns in trailing 1h"))

        # Rule 2: amount spike vs the card's OWN trailing baseline.
        # Suppressed when history is too thin to trust the baseline.
        if not trail_low_conf and amt_vs_trail >= AMOUNT_SPIKE_RATIO:
            s = min(1.0, 0.4 + 0.1 * (amt_vs_trail - AMOUNT_SPIKE_RATIO))
            candidates.append(
                (s, f"amount_spike: {amt_vs_trail:.1f}x trailing median"))

        # Rule 3: country change without plausible travel time
        # (impossible_travel fingerprint vs its hours-apart hard-negative twin).
        if country_change and 0 <= secs < IMPOSSIBLE_TRAVEL_MAX_SECS:
            s = 0.7
            candidates.append(
                (s, f"impossible_travel: country change after {secs}s"))

        # Rule 4: new device + foreign-IP + high-risk category
        # (account_takeover fingerprint; the home-country hard-neg won't fire
        # the country_change leg).
        if new_device and country_change and category in HIGH_RISK_CATEGORIES:
            s = 0.8
            candidates.append(
                (s, f"takeover_pattern: new device + country change + {category}"))

        # Rule 5: high-value purchase in an unusual category vs baseline
        # (stolen_spree fingerprint; one-off big-ticket is caught by the run
        #  length at the card level, not here).
        if (not trail_low_conf and amt_vs_trail >= 3.0
                and category in HIGH_RISK_CATEGORIES):
            s = min(1.0, 0.5 + 0.1 * (amt_vs_trail - 3.0))
            candidates.append(
                (s, f"spree_purchase: {amt_vs_trail:.1f}x median in {category}"))

        if candidates:
            score, reason = max(candidates, key=lambda c: c[0])
        else:
            score, reason = 0.0, "no_rule_fired"

        out = dict(row)
        out["score"] = round(score, 3)
        out["reason"] = reason
        return out


# --- Card-level aggregation ----------------------------------------------

def aggregate_cards(scored_rows, method="max", decay=0.9):
    """Roll per-row scores up to one decision per card.

    method='max'         -> card score is the single highest row score.
    method='decaying_sum'-> rows sorted by time; suspicion accumulates with
                            exponential decay, then squashed to [0,1]. Many
                            small scores (card-testing) can outweigh one
                            medium score, which 'max' would miss.

    Returns one dict per card: card_id, card_score, the row count, the
    top-scoring row's reason (the account's headline explanation), whether
    the card contains any fraud (for evaluation), and the worst scenario seen.
    """
    by_card = defaultdict(list)
    for r in scored_rows:
        by_card[r["card_id"]].append(r)

    cards = []
    for card_id, rows in by_card.items():
        rows = sorted(rows, key=lambda r: r["timestamp"])
        scores = [float(r["score"]) for r in rows]

        if method == "max":
            card_score = max(scores) if scores else 0.0
        elif method == "decaying_sum":
            acc = 0.0
            for s in scores:                 # time order: older decays first
                acc = acc * decay + s
            # squash unbounded accumulator into [0,1] without a hard clip
            card_score = 1.0 - math.exp(-acc)
        else:
            raise ValueError(f"unknown aggregation method: {method}")

        top = max(rows, key=lambda r: float(r["score"]))
        # ground-truth bookkeeping for evaluate.py (not used in scoring)
        any_fraud = max(int(r["is_fraud"]) for r in rows)
        fraud_scenarios = {r["scenario"] for r in rows
                           if int(r["is_fraud"]) == 1}
        worst_scenario = (sorted(fraud_scenarios)[0]
                          if fraud_scenarios else "none")

        cards.append({
            "card_id": card_id,
            "card_score": round(card_score, 4),
            "n_rows": len(rows),
            "top_reason": top["reason"],
            "any_fraud": any_fraud,
            "fraud_scenario": worst_scenario,
        })

    cards.sort(key=lambda c: c["card_id"])
    return cards


ROW_OUT_FIELDS_EXTRA = ["score", "reason"]
CARD_OUT_FIELDS = ["card_id", "card_score", "n_rows", "top_reason",
                   "any_fraud", "fraud_scenario"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="featured.csv")
    ap.add_argument("--agg", choices=["max", "decaying_sum"],
                    default="decaying_sum",
                    help="card-level aggregation function")
    ap.add_argument("--decay", type=float, default=0.9,
                    help="decay factor for decaying_sum aggregation")
    ap.add_argument("--row-out", default="scored_rows.csv")
    ap.add_argument("--card-out", default="scored_cards.csv")
    args = ap.parse_args()

    with open(args.infile, newline="") as f:
        rows = list(csv.DictReader(f))

    scorer: Scorer = RuleScorer()
    scored = [scorer.score_row(r) for r in rows]
    cards = aggregate_cards(scored, method=args.agg, decay=args.decay)

    row_fields = list(scored[0].keys())
    with open(args.row_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_fields)
        w.writeheader()
        w.writerows(scored)

    with open(args.card_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CARD_OUT_FIELDS)
        w.writeheader()
        w.writerows(cards)

    n_fired = sum(1 for r in scored if r["reason"] != "no_rule_fired")
    print(f"scored {len(scored)} rows ({n_fired} fired a rule), "
          f"{len(cards)} cards [agg={args.agg}]")
    # distribution sanity: how card scores split by true fraud status
    fraud_cards = [c for c in cards if c["any_fraud"] == 1]
    clean_cards = [c for c in cards if c["any_fraud"] == 0]
    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    print(f"  mean card_score | fraud cards: "
          f"{avg([c['card_score'] for c in fraud_cards]):.3f}")
    print(f"  mean card_score | clean cards: "
          f"{avg([c['card_score'] for c in clean_cards]):.3f}")


if __name__ == "__main__":
    main()
