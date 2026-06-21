"""
Reconstruct sequence context for each transaction.

This is where the project's thesis becomes executable: fraud is a property
of a SEQUENCE and of DEVIATION FROM A CARDHOLDER'S OWN BASELINE, not of any
row read alone. The flat transaction stream is turned into featured rows
that carry that context.

Two kinds of baseline are attached, deliberately:

  - STATIC profile (from profile.py / card_profiles.csv): computed from ALL
    of a card's rows. It is the one permitted look-ahead -- a card's own
    future transactions inform its profile. This is flagged, not hidden,
    because in production a profile genuinely is rebuilt on a slow cadence
    from whatever history exists. Carried for comparison.

  - TRAILING point-in-time baseline: running median/mean over the card's
    PRIOR transactions only. No look-ahead. This is the production-correct
    signal, and it recovers what the all-rows static baseline masks (fraud
    rows nudge the static mean upward, dampening the very anomaly they
    create). Where the two baselines diverge is itself informative.

LEAKAGE RULE (brief F1): no feature for a transaction may use information
from that transaction's own future, except the explicitly-flagged static
join. Every trailing computation below consumes only rows already seen.

`is_country_change` is a unified geography-change flag: it fires when either
the merchant country or the IP country differs from the previous transaction
on the same card. That keeps both impossible-travel and account-takeover
signals behind one stable downstream field.

Usage:
    python -m fraud_eval.features --txns transactions.csv --profiles card_profiles.csv \
        --out featured.csv

Or as a library:
    from fraud_eval.features import build_features
    featured = build_features(txn_rows, profile_rows)
"""

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime

from .fx import load_rates, to_usd

# Sentinel for "no previous transaction" on a card's first row.
NO_PRIOR = -1

# Categories the card's baseline treats as everyday; divergence from these
# is a (weak) signal. Kept here so scoring can stay rules-agnostic about it.
TRAILING_MIN_HISTORY = 3  # below this, trailing stats are low-confidence


def _parse_ts(s):
    return datetime.fromisoformat(s)


def _running_median(values):
    """Median of a non-empty list; caller guarantees non-empty."""
    return statistics.median(values)


def build_features(txn_rows, profile_rows, rates):
    """Join transactions to profiles and attach sequence + baseline signals.

    Pure function: takes lists of dicts, returns a list of dicts. No file
    I/O, so this is directly unit-testable on in-memory fixtures (brief NFR
    'determinism for tests').

    Input rows are not assumed sorted; this function sorts within each card
    by timestamp before computing trailing signals, so correctness does not
    depend on the upstream sort order.

    `rates` is the {currency: rate_to_usd} dict from fx.load_rates. All
    amount-based signals use amount_usd, never native amount (brief F4).
    """
    profiles = {p["card_id"]: p for p in profile_rows}

    by_card = defaultdict(list)
    for r in txn_rows:
        by_card[r["card_id"]].append(r)

    featured = []
    for card_id, txns in by_card.items():
        txns = sorted(txns, key=lambda r: r["timestamp"])
        prof = profiles.get(card_id)

        seen_devices = set()
        seen_merchants = set()
        prior_amounts = []
        prev_ts = None
        prev_country = None
        prev_ip_country = None
        # timestamps of prior txns, for the trailing-velocity window
        prior_ts = []

        for r in txns:
            ts = _parse_ts(r["timestamp"])
            amount = float(r["amount"])
            amount_usd = to_usd(amount, r["currency"], rates)

            # --- static join (the permitted look-ahead) ------------------
            static_median = float(prof["amount_median"]) if prof else 0.0
            static_mean = float(prof["amount_mean"]) if prof else 0.0
            # Profile stats are in USD (brief §4.2), so compare against amount_usd.
            amount_vs_static = (amount_usd / static_median
                                if static_median > 0 else 0.0)

            # --- trailing point-in-time baseline (no look-ahead) ---------
            # prior_amounts accumulates USD values (see advance-state block below).
            if prior_amounts:
                trail_median = _running_median(prior_amounts)
                trail_mean = statistics.mean(prior_amounts)
            else:
                trail_median = 0.0
                trail_mean = 0.0
            amount_vs_trailing = (amount_usd / trail_median
                                  if trail_median > 0 else 0.0)
            trailing_n = len(prior_amounts)
            trailing_low_confidence = int(trailing_n < TRAILING_MIN_HISTORY)

            # --- sequence deltas (no look-ahead) -------------------------
            if prev_ts is None:
                secs_since_prev = NO_PRIOR
            else:
                secs_since_prev = int((ts - prev_ts).total_seconds())

            # rolling velocity: count of prior txns within trailing 1h / 24h
            v_1h = sum(1 for t in prior_ts
                       if 0 <= (ts - t).total_seconds() <= 3600)
            v_24h = sum(1 for t in prior_ts
                        if 0 <= (ts - t).total_seconds() <= 86400)

            is_new_device = int(r["device_id"] not in seen_devices)
            is_new_merchant = int(r["merchant_id"] not in seen_merchants)
            is_country_change = int(
                prev_country is not None
                and (
                    r["merchant_country"] != prev_country
                    or r["ip_country"] != prev_ip_country
                )
            )

            featured.append({
                # passthrough identity / ordering
                "txn_id": r["txn_id"],
                "card_id": card_id,
                "timestamp": r["timestamp"],
                "amount": round(amount, 2),
                "currency": r["currency"],
                "amount_usd": round(amount_usd, 4),
                "merchant_id": r["merchant_id"],
                "merchant_category": r["merchant_category"],
                "merchant_country": r["merchant_country"],
                "device_id": r["device_id"],
                "ip_country": r["ip_country"],
                "entry_mode": r["entry_mode"],
                # static baseline signals (flagged look-ahead)
                "amount_vs_static_median": round(amount_vs_static, 3),
                "static_median": round(static_median, 2),
                "static_mean": round(static_mean, 2),
                # trailing baseline signals (production-correct)
                "amount_vs_trailing_median": round(amount_vs_trailing, 3),
                "trailing_median": round(trail_median, 2),
                "trailing_n": trailing_n,
                "trailing_low_confidence": trailing_low_confidence,
                # sequence deltas
                "secs_since_prev": secs_since_prev,
                "velocity_1h": v_1h,
                "velocity_24h": v_24h,
                "is_new_device": is_new_device,
                "is_new_merchant": is_new_merchant,
                "is_country_change": is_country_change,
                # carried through untouched for evaluation
                "is_fraud": int(r["is_fraud"]),
                "scenario": r["scenario"],
            })

            # --- advance state AFTER emitting (preserves no-look-ahead) ---
            seen_devices.add(r["device_id"])
            seen_merchants.add(r["merchant_id"])
            prior_amounts.append(amount_usd)  # trailing baseline in USD
            prior_ts.append(ts)
            prev_ts = ts
            prev_country = r["merchant_country"]
            prev_ip_country = r["ip_country"]

    # stable global order for reproducible output
    featured.sort(key=lambda r: (r["card_id"], r["timestamp"]))
    return featured


FIELDS = [
    "txn_id", "card_id", "timestamp", "amount", "currency", "amount_usd",
    "merchant_id", "merchant_category", "merchant_country",
    "device_id", "ip_country", "entry_mode",
    "amount_vs_static_median", "static_median", "static_mean",
    "amount_vs_trailing_median", "trailing_median", "trailing_n",
    "trailing_low_confidence",
    "secs_since_prev", "velocity_1h", "velocity_24h",
    "is_new_device", "is_new_merchant", "is_country_change",
    "is_fraud", "scenario",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txns", default="transactions.csv")
    ap.add_argument("--profiles", default="card_profiles.csv")
    ap.add_argument("--fx-rates", default="fx_rates.csv")
    ap.add_argument("--out", default="featured.csv")
    args = ap.parse_args()

    with open(args.txns, newline="") as f:
        txn_rows = list(csv.DictReader(f))
    with open(args.profiles, newline="") as f:
        profile_rows = list(csv.DictReader(f))

    rates = load_rates(args.fx_rates)
    featured = build_features(txn_rows, profile_rows, rates)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(featured)

    print(f"wrote {len(featured)} featured rows to {args.out}")
    # a couple of quick distributional sanity prints
    n_new_dev = sum(r["is_new_device"] for r in featured)
    n_country = sum(r["is_country_change"] for r in featured)
    max_v1h = max(r["velocity_1h"] for r in featured)
    print(f"  new-device rows: {n_new_dev}")
    print(f"  country-change rows: {n_country}")
    print(f"  max velocity_1h: {max_v1h}")


if __name__ == "__main__":
    main()
