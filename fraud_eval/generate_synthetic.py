"""
Synthetic card-fraud transaction generator.

Design principle: fraud is a property of a SEQUENCE, not a single row.
Labels are assigned by the injection process (causal), never sampled
independently of the features. Includes "hard negatives" -- legitimate
behaviour that superficially resembles fraud -- so downstream evaluation
can measure precision under genuine ambiguity.

NOTE: This produces SYNTHETIC data only. No real cardholder data is used.

Usage:
    python -m fraud_eval.generate_synthetic --cards 1000 --days 30 --out transactions.csv
"""

import argparse
import csv
import random
from datetime import datetime, timedelta

from .fx import COUNTRY_CURRENCY, RATES


def _rid(rng, prefix, nbits=48):
    """Reproducible hex id drawn from the seeded rng (not uuid4, which
    ignores the seed and breaks byte-identical reruns -- brief G1)."""
    return f"{prefix}_{rng.getrandbits(nbits):0{nbits // 4}x}"

# --- Reference dimensions -------------------------------------------------

COUNTRIES = ["US", "GB", "DE", "FR", "JP", "AU", "BR", "NG", "RU", "CN"]
# Categories with a rough baseline legitimacy weight (higher = more everyday)
MCC = {
    "grocery": 1.0, "fuel": 0.9, "restaurant": 0.8, "transport": 0.7,
    "retail": 0.7, "utilities": 0.6, "travel": 0.4, "electronics": 0.4,
    "gambling": 0.15, "crypto": 0.1, "gift_card": 0.1, "wire": 0.1,
}
ENTRY_MODES = ["chip", "contactless", "online", "manual"]


# --- Card "personalities" -------------------------------------------------

class Card:
    def __init__(self, idx, rng):
        self.card_id = f"card_{idx:06d}"
        self.home_country = rng.choices(COUNTRIES, weights=[5, 3, 2, 2, 2, 2, 1, 1, 1, 1])[0]
        self.home_currency = COUNTRY_CURRENCY[self.home_country]
        self.fav_categories = rng.sample(
            [c for c, w in MCC.items() if w >= 0.6], k=rng.randint(2, 4)
        )
        # log-normal amount baseline in USD-EQUIVALENT terms: most spend small,
        # occasional big legit buys. exp(mu) ~ 12..37 USD median.
        self.amount_mu = rng.uniform(2.5, 3.6)
        self.amount_sigma = rng.uniform(0.5, 0.9)
        self.home_device = _rid(rng, "dev", 40)
        self.txns_per_day = rng.uniform(0.5, 4.0)

    def legit_amount(self, rng, currency):
        """A legitimate amount in NATIVE currency. The baseline is drawn in
        USD-equivalent terms, then expressed in the transaction's currency by
        dividing by that currency's USD rate, so a card's real spend power is
        consistent regardless of denomination (a JPY/NGN card is not modelled
        as spending sub-dollar amounts -- brief realism)."""
        usd = rng.lognormvariate(self.amount_mu, self.amount_sigma)
        native = usd / RATES[currency]
        return round(native, 2)


def base_txn(card, ts, rng, **overrides):
    """A normal-looking transaction for this card; overrides force anomalies."""
    cat = overrides.get("merchant_category") or rng.choice(card.fav_categories)
    country = overrides.get("merchant_country", card.home_country)
    currency = COUNTRY_CURRENCY[country]
    amount = overrides.get("amount")
    if amount is None:
        amount = card.legit_amount(rng, currency)
    row = {
        "txn_id": _rid(rng, "txn", 48),
        "card_id": card.card_id,
        "timestamp": ts.isoformat(),
        "amount": amount,
        "currency": currency,
        "merchant_id": overrides.get("merchant_id", f"mer_{rng.randint(0, 4000):04d}"),
        "merchant_category": cat,
        "merchant_country": country,
        "device_id": overrides.get("device_id", card.home_device),
        "ip_country": overrides.get("ip_country", card.home_country),
        "entry_mode": overrides.get("entry_mode", rng.choice(["chip", "contactless"])),
        "is_fraud": overrides.get("is_fraud", 0),
        "scenario": overrides.get("scenario", "legit"),
    }
    return row


# --- Fraud scenarios (each leaves a distinct fingerprint) -----------------

def scenario_card_testing(card, ts, rng):
    """Many tiny online amounts in minutes from a new device/merchant."""
    dev = _rid(rng, "dev", 40)
    mer = f"mer_{rng.randint(8000, 9999):04d}"
    out, t = [], ts
    for _ in range(rng.randint(6, 15)):
        out.append(base_txn(
            card, t, rng, amount=round(rng.uniform(0.5, 3.0), 2),
            merchant_category="retail",
            merchant_id=mer, device_id=dev, entry_mode="online",
            is_fraud=1, scenario="card_testing",
        ))
        t += timedelta(seconds=rng.randint(20, 90))
    return out


def scenario_account_takeover(card, ts, rng):
    """New device + new IP country, then escalating amounts."""
    dev = _rid(rng, "dev", 40)
    ipc = rng.choice([c for c in COUNTRIES if c != card.home_country])
    # base amount in the card's home currency, floored at ~50 USD-equiv
    floor_native = 50.0 / RATES[card.home_currency]
    out, t = [], ts
    amt = max(floor_native, card.legit_amount(rng, card.home_currency))
    for _ in range(rng.randint(3, 6)):
        amt *= rng.uniform(1.4, 2.2)
        out.append(base_txn(
            card, t, rng, amount=round(amt, 2),
            device_id=dev, ip_country=ipc, entry_mode="online",
            merchant_category=rng.choice(["electronics", "gift_card", "crypto"]),
            is_fraud=1, scenario="account_takeover",
        ))
        t += timedelta(minutes=rng.randint(2, 20))
    return out


def scenario_impossible_travel(card, ts, rng):
    """A legit local txn, then a far-country txn minutes later."""
    far = rng.choice([c for c in COUNTRIES if c != card.home_country])
    far_currency = COUNTRY_CURRENCY[far]
    local = base_txn(card, ts, rng)  # legit anchor (label 0)
    # the away amount is in the far country's currency; size it ~1.5-4x the
    # card's usual USD-equivalent spend, then express in far currency
    away_native = card.legit_amount(rng, far_currency) * rng.uniform(1.5, 4)
    away = base_txn(
        card, ts + timedelta(minutes=rng.randint(3, 25)), rng,
        merchant_country=far, ip_country=far, entry_mode="manual",
        amount=round(away_native, 2),
        is_fraud=1, scenario="impossible_travel",
    )
    return [local, away]


def scenario_stolen_spree(card, ts, rng):
    """Run of mid/large purchases across unusual categories before kill."""
    out, t = [], ts
    for _ in range(rng.randint(4, 8)):
        amt = card.legit_amount(rng, card.home_currency) * rng.uniform(3, 8)
        out.append(base_txn(
            card, t, rng,
            amount=round(amt, 2),
            merchant_category=rng.choice(["electronics", "retail", "gift_card", "travel"]),
            merchant_id=f"mer_{rng.randint(0, 4000):04d}",
            entry_mode=rng.choice(["chip", "contactless", "manual"]),
            is_fraud=1, scenario="stolen_spree",
        ))
        t += timedelta(minutes=rng.randint(5, 40))
    return out


FRAUD_SCENARIOS = [
    scenario_card_testing, scenario_account_takeover,
    scenario_impossible_travel, scenario_stolen_spree,
]


# --- Hard negatives (legit but fraud-looking; label stays 0) --------------

def hard_negative(card, ts, rng):
    kind = rng.choice(["travel", "big_ticket", "new_device"])
    if kind == "travel":
        far = rng.choice([c for c in COUNTRIES if c != card.home_country])
        # a short legit trip: several txns abroad over hours, not minutes
        out, t = [], ts
        for _ in range(rng.randint(2, 4)):
            out.append(base_txn(
                card, t, rng, merchant_country=far, ip_country=far,
                scenario="hard_neg_travel",
            ))
            t += timedelta(hours=rng.randint(2, 8))
        return out
    if kind == "big_ticket":
        return [base_txn(
            card, ts, rng,
            amount=round(card.legit_amount(rng, card.home_currency)
                         * rng.uniform(5, 12), 2),
            merchant_category="electronics", scenario="hard_neg_big_ticket",
        )]
    # new_device: legit upgrade -- new device but home country, normal amounts
    return [base_txn(
        card, ts, rng, device_id=_rid(rng, "dev", 40),
        entry_mode="online", scenario="hard_neg_new_device",
    )]


# --- Main generation loop -------------------------------------------------

def generate(n_cards, n_days, fraud_rate, hard_neg_rate, seed):
    rng = random.Random(seed)
    cards = [Card(i, rng) for i in range(n_cards)]
    start = datetime(2025, 1, 1)
    rows = []

    for card in cards:
        for day in range(n_days):
            day_start = start + timedelta(days=day)
            n = max(0, int(rng.gauss(card.txns_per_day, 1.0)))
            for _ in range(n):
                ts = day_start + timedelta(seconds=rng.randint(0, 86399))
                rows.append(base_txn(card, ts, rng))

        # inject fraud sequences for a subset of cards
        if rng.random() < fraud_rate:
            ts = start + timedelta(
                days=rng.randint(0, n_days - 1), seconds=rng.randint(0, 86399))
            rows.extend(rng.choice(FRAUD_SCENARIOS)(card, ts, rng))

        # inject hard negatives independently
        if rng.random() < hard_neg_rate:
            ts = start + timedelta(
                days=rng.randint(0, n_days - 1), seconds=rng.randint(0, 86399))
            rows.extend(hard_negative(card, ts, rng))

    rows.sort(key=lambda r: (r["card_id"], r["timestamp"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, default=1000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--fraud-rate", type=float, default=0.03,
                    help="fraction of cards that experience a fraud sequence")
    ap.add_argument("--hard-neg-rate", type=float, default=0.06,
                    help="fraction of cards with a fraud-looking legit sequence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="transactions.csv")
    args = ap.parse_args()

    rows = generate(args.cards, args.days, args.fraud_rate,
                    args.hard_neg_rate, args.seed)

    fields = ["txn_id", "card_id", "timestamp", "amount", "currency",
              "merchant_id", "merchant_category", "merchant_country",
              "device_id", "ip_country", "entry_mode", "is_fraud", "scenario"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_fraud = sum(r["is_fraud"] for r in rows)
    n_hard = sum(1 for r in rows if r["scenario"].startswith("hard_neg"))
    print(f"wrote {len(rows)} txns to {args.out}")
    print(f"  fraud rows: {n_fraud} ({100*n_fraud/len(rows):.2f}%)")
    print(f"  hard-negative rows: {n_hard} ({100*n_hard/len(rows):.2f}%)")
    print(f"  unique cards: {len({r['card_id'] for r in rows})}")


if __name__ == "__main__":
    main()
