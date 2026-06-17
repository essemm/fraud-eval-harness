"""
Shared currency-to-USD conversion helper.

Used by profile.py and features.py. Keeping the conversion rule in one
place means a rate change or a new currency touches exactly one file.

The rates are static (no date dimension) — a deliberate simplification
flagged in the brief (§4.4). The interface could gain a date parameter
and a point-in-time lookup without disturbing the rest of the pipeline.

Usage (generate fx_rates.csv once):
    python fx.py --out fx_rates.csv

Usage as a library:
    from fx import COUNTRY_CURRENCY, load_rates, to_usd
    rates = load_rates("fx_rates.csv")
    usd = to_usd(amount, currency, rates)
"""

import argparse
import csv

# Approximate mid-market rates (illustrative, not live).
RATES = {
    "USD": 1.000,
    "GBP": 1.270,
    "EUR": 1.080,
    "JPY": 0.0067,
    "AUD": 0.650,
    "BRL": 0.200,
    "NGN": 0.00065,
    "RUB": 0.011,
    "CNY": 0.138,
}

# One dominant currency per country (used by the generator to assign
# currency from merchant_country, keeping currency neutral w.r.t. label).
COUNTRY_CURRENCY = {
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "JP": "JPY",
    "AU": "AUD",
    "BR": "BRL",
    "NG": "NGN",
    "RU": "RUB",
    "CN": "CNY",
}


def to_usd(amount, currency, rates):
    """Convert a native-currency amount to USD. Pure function."""
    return amount * rates[currency]


def load_rates(path):
    """Read fx_rates.csv; return {currency: rate_to_usd}."""
    with open(path, newline="") as f:
        return {r["currency"]: float(r["rate_to_usd"]) for r in csv.DictReader(f)}


def write_rates(path, rates=None):
    """Write fx_rates.csv from the given rates dict (default: RATES)."""
    if rates is None:
        rates = RATES
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["currency", "rate_to_usd"])
        w.writeheader()
        for currency in sorted(rates):
            w.writerow({"currency": currency, "rate_to_usd": rates[currency]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fx_rates.csv")
    args = ap.parse_args()
    write_rates(args.out)
    print(f"wrote {len(RATES)} currency rates to {args.out}")
    for cur, rate in sorted(RATES.items()):
        print(f"  {cur}: {rate}")


if __name__ == "__main__":
    main()
