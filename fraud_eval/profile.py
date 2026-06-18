"""
Build a per-card profile dimension from the transaction stream.

This is the SLOW-CHANGING dimension that the fast transaction stream is
scored against -- the classic fraud-ops shape: cheap per-event lookups
against a baseline that is rebuilt on a slower cadence.

The baseline is computed from ALL rows, not just legitimate ones. In
production you do not have fraud labels at profile-build time, so an
all-rows baseline is the honest reflection of what a real system knows.
The cost is small but real: fraud rows nudge the amount statistics upward,
slightly masking the very anomaly they create. The trailing point-in-time
baseline computed inside features.py is what recovers that signal without
peeking at the future.

All amount statistics are computed on USD-normalised amounts (brief §4.2).
The fx_rates join is performed here so this module remains independently
runnable without a pre-normalised input.

Usage:
    python -m fraud_eval.profile --in transactions.csv --fx-rates fx_rates.csv \
        --out card_profiles.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict

from .fx import load_rates, to_usd


def build_profiles(rows, rates):
    """Aggregate flat transaction rows into one profile row per card.

    Amounts are normalised to USD before aggregation (brief P5).
    """
    by_card = defaultdict(list)
    for r in rows:
        by_card[r["card_id"]].append(r)

    profiles = []
    for card_id, txns in by_card.items():
        amounts_usd = [to_usd(float(t["amount"]), t["currency"], rates)
                       for t in txns]
        profiles.append({
            "card_id": card_id,
            "n_txns": len(txns),
            "amount_max": round(max(amounts_usd), 2),
            "amount_mean": round(statistics.mean(amounts_usd), 2),
            "amount_median": round(statistics.median(amounts_usd), 2),
            "distinct_countries": len({t["merchant_country"] for t in txns}),
            "distinct_devices": len({t["device_id"] for t in txns}),
        })

    profiles.sort(key=lambda p: p["card_id"])
    return profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="transactions.csv")
    ap.add_argument("--fx-rates", default="fx_rates.csv")
    ap.add_argument("--out", default="card_profiles.csv")
    args = ap.parse_args()

    with open(args.infile, newline="") as f:
        rows = list(csv.DictReader(f))

    rates = load_rates(args.fx_rates)
    profiles = build_profiles(rows, rates)

    fields = ["card_id", "n_txns", "amount_max", "amount_mean",
              "amount_median", "distinct_countries", "distinct_devices"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(profiles)

    print(f"wrote {len(profiles)} card profiles to {args.out}")
    print(f"  from {len(rows)} transactions")
    means = [p["amount_mean"] for p in profiles]
    print(f"  mean-amount range across cards: "
          f"{min(means):.2f} .. {max(means):.2f}")


if __name__ == "__main__":
    main()
