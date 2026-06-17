"""
Generator (G1-G6) and FX-rate (X1) acceptance criteria from brief §10.

Each test name carries the criterion id so a failure points straight at the
clause of the spec it violates.
"""

from collections import Counter

import fx
import generate_synthetic as gen

SEED = 12345


def test_G1_reproducible_same_seed():
    """G1: same seed -> identical output (every field, every row)."""
    a = gen.generate(n_cards=100, n_days=10, fraud_rate=0.05,
                     hard_neg_rate=0.08, seed=SEED)
    b = gen.generate(n_cards=100, n_days=10, fraud_rate=0.05,
                     hard_neg_rate=0.08, seed=SEED)
    assert a == b


def test_G1_different_seed_differs():
    """G1 (converse): a different seed should produce different data, else the
    seed isn't actually driving generation."""
    a = gen.generate(100, 10, 0.05, 0.08, seed=1)
    b = gen.generate(100, 10, 0.05, 0.08, seed=2)
    assert a != b


def test_G2_label_scenario_consistency(txns):
    """G2: is_fraud=1 iff scenario is a fraud scenario; hard_neg_* are label 0;
    legit is label 0."""
    fraud_scenarios = {"card_testing", "account_takeover",
                       "impossible_travel", "stolen_spree"}
    for r in txns:
        is_fraud = int(r["is_fraud"])
        sc = r["scenario"]
        if is_fraud == 1:
            assert sc in fraud_scenarios, f"fraud row with scenario {sc}"
        else:
            assert sc == "legit" or sc.startswith("hard_neg"), \
                f"non-fraud row with scenario {sc}"


def test_G3_all_scenarios_present(txns):
    """G3: all four fraud scenarios and all three hard-negative types appear."""
    seen = {r["scenario"] for r in txns}
    required = {"card_testing", "account_takeover", "impossible_travel",
                "stolen_spree", "hard_neg_travel", "hard_neg_big_ticket",
                "hard_neg_new_device"}
    missing = required - seen
    assert not missing, f"missing scenarios: {missing}"


def test_G4_sorted_by_card_then_time(txns):
    """G4: output sorted by (card_id, timestamp)."""
    keys = [(r["card_id"], r["timestamp"]) for r in txns]
    assert keys == sorted(keys)


def test_G5_lossless_fx_join(txns):
    """G5: every currency in transactions has a rate (no row fails to
    normalise)."""
    currencies = {r["currency"] for r in txns}
    assert currencies <= set(fx.RATES), \
        f"currencies without a rate: {currencies - set(fx.RATES)}"


def test_G6_currency_not_a_label_proxy(txns):
    """G6: no single currency is a near-perfect fraud tell. A currency that is
    a majority of fraud while rare among legit rows would leak the label."""
    fraud = Counter(r["currency"] for r in txns if r["is_fraud"] == "1"
                    or r["is_fraud"] == 1)
    legit = Counter(r["currency"] for r in txns if r["is_fraud"] in ("0", 0))
    tot_f = sum(fraud.values())
    tot_l = sum(legit.values())
    assert tot_f > 0 and tot_l > 0
    for c in set(fraud) | set(legit):
        pf = fraud[c] / tot_f
        pl = legit[c] / tot_l
        assert not (pf > 0.5 and pl < 0.1), \
            f"currency {c} leaks the label: {pf:.0%} of fraud, {pl:.0%} of legit"


def test_X1_fx_rates_wellformed():
    """X1: one rate per currency, USD present at 1.0, all rates positive."""
    rates = dict(fx.RATES)
    assert rates.get("USD") == 1.0
    assert all(v > 0 for v in rates.values())
    # one entry per currency is guaranteed by dict structure; assert non-empty
    assert len(rates) >= 2
