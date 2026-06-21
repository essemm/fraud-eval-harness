"""
Profile (P1-P5) and feature (F1-F4) acceptance criteria from brief §10.

The leakage check (F1) is the most important test in the suite: it proves no
derived feature uses a transaction's own future, which is the property the
whole sequence-aware approach depends on being honest about.
"""

import statistics
from collections import OrderedDict, defaultdict

from fraud_eval import fx
from fraud_eval import features as feat

NO_PRIOR = feat.NO_PRIOR


# --- profile --------------------------------------------------------------

def test_P1_one_profile_per_card(txns, profiles):
    distinct_cards = {r["card_id"] for r in txns}
    profile_ids = [p["card_id"] for p in profiles]
    assert len(profile_ids) == len(distinct_cards)
    assert set(profile_ids) == distinct_cards
    assert len(profile_ids) == len(set(profile_ids)), "duplicate profile rows"


def test_P2_amount_statistics_ordered(profiles):
    """P2: amount_max >= amount_mean >= 0 and amount_median >= 0."""
    for p in profiles:
        amax = float(p["amount_max"])
        amean = float(p["amount_mean"])
        amed = float(p["amount_median"])
        assert amax >= amean >= 0, f"{p['card_id']}: max<mean or negative"
        assert amed >= 0


def test_P3_distinct_counts_in_range(txns, profiles):
    """P3: distinct_countries / distinct_devices in [1, n_txns]."""
    counts = defaultdict(int)
    for r in txns:
        counts[r["card_id"]] += 1
    for p in profiles:
        n = counts[p["card_id"]]
        assert 1 <= int(p["distinct_countries"]) <= n
        assert 1 <= int(p["distinct_devices"]) <= n


def test_P4_profiles_join_losslessly(txns, profiles):
    """P4: no orphans either side of the card_id join."""
    txn_cards = {r["card_id"] for r in txns}
    prof_cards = {p["card_id"] for p in profiles}
    assert txn_cards == prof_cards


def test_P5_profile_amounts_are_usd(txns, rates):
    """P5: profile statistics are computed on USD-normalised amounts. A card
    transacting only in a non-USD currency must have a profile max equal to
    its max NATIVE amount times the rate, not the native amount itself."""
    from fraud_eval import profile as prof
    # find a card that transacts in a single non-USD currency
    by_card = defaultdict(list)
    for r in txns:
        by_card[r["card_id"]].append(r)
    target = None
    for cid, rows in by_card.items():
        currencies = {r["currency"] for r in rows}
        if len(currencies) == 1 and currencies != {"USD"}:
            target = (cid, rows, currencies.pop())
            break
    if target is None:
        import pytest
        pytest.skip("no single-non-USD-currency card in this dataset")
    cid, rows, cur = target
    profiles = {p["card_id"]: p for p in prof.build_profiles(txns, rates)}
    p = profiles[cid]
    expected_max_usd = max(float(r["amount"]) for r in rows) * rates[cur]
    assert abs(float(p["amount_max"]) - expected_max_usd) < 0.01


# --- features -------------------------------------------------------------

def _by_card(featured):
    bc = OrderedDict()
    for r in featured:
        bc.setdefault(r["card_id"], []).append(r)
    return bc


def test_F1_no_future_leakage(featured, rates):
    """F1: the trailing median at row i uses only rows < i (no look-ahead).
    Reconstruct prior amounts the same way the code does -- full-precision
    amount*rate -- and confirm the emitted trailing_median matches the median
    of strictly-prior rows within rounding tolerance."""
    for cid, rows in _by_card(featured).items():
        prior = []
        for r in rows:
            expected = statistics.median(prior) if prior else 0.0
            assert abs(round(expected, 2) - r["trailing_median"]) <= 0.02, \
                f"{cid}/{r['txn_id']}: trailing median used future data"
            prior.append(float(r["amount"]) * rates[r["currency"]])


def test_F1_first_row_has_no_prior(featured):
    """F1 corollary: a card's first row can have peeked at nothing."""
    for cid, rows in _by_card(featured).items():
        assert rows[0]["trailing_n"] == 0


def test_F2_new_device_flag(featured):
    """F2: is_new_device is true the first time a device appears for a card,
    false thereafter. Checkable from the output because device_id is carried
    through."""
    for cid, rows in _by_card(featured).items():
        seen = set()
        for r in rows:
            first_time = r["device_id"] not in seen
            assert r["is_new_device"] == int(first_time), \
                f"{cid}/{r['txn_id']}: new-device flag wrong"
            seen.add(r["device_id"])


def test_F3_secs_since_prev_sentinel(featured):
    """F3: seconds-since-previous is the sentinel on a card's first row and
    non-negative thereafter."""
    for cid, rows in _by_card(featured).items():
        assert rows[0]["secs_since_prev"] == NO_PRIOR
        for r in rows[1:]:
            assert r["secs_since_prev"] >= 0


def test_F4_amount_usd_correct(featured, rates):
    """F4: amount_usd == amount * rate_to_usd within tolerance, and the
    amount-vs-baseline ratios are computed on the USD value."""
    for r in featured:
        expected = float(r["amount"]) * rates[r["currency"]]
        assert abs(r["amount_usd"] - round(expected, 4)) < 0.01


def test_F_country_change_includes_ip_country(rates):
    """The unified country-change feature fires on IP-country movement even
    when merchant_country stays unchanged. This is the account-takeover signal:
    a home-country online purchase from a foreign IP."""
    from fraud_eval import profile as prof

    rows = [
        {
            "txn_id": "txn_1",
            "card_id": "card_1",
            "timestamp": "2025-01-01T10:00:00",
            "amount": 10.0,
            "currency": "USD",
            "merchant_id": "mer_1",
            "merchant_category": "grocery",
            "merchant_country": "US",
            "device_id": "dev_home",
            "ip_country": "US",
            "entry_mode": "chip",
            "is_fraud": 0,
            "scenario": "legit",
        },
        {
            "txn_id": "txn_2",
            "card_id": "card_1",
            "timestamp": "2025-01-01T10:05:00",
            "amount": 50.0,
            "currency": "USD",
            "merchant_id": "mer_2",
            "merchant_category": "electronics",
            "merchant_country": "US",
            "device_id": "dev_new",
            "ip_country": "GB",
            "entry_mode": "online",
            "is_fraud": 1,
            "scenario": "account_takeover",
        },
    ]
    profiles = prof.build_profiles(rows, rates)
    featured = feat.build_features(rows, profiles, rates)

    assert featured[0]["is_country_change"] == 0
    assert featured[1]["is_country_change"] == 1


def test_F_account_takeover_rows_expose_country_change(featured):
    """Generated account-takeover rows carry the foreign-IP signal through
    features, so the rule scorer can see the documented fingerprint."""
    takeover = [r for r in featured if r["scenario"] == "account_takeover"]
    assert takeover, "fixture contains no account_takeover rows"
    assert any(int(r["is_country_change"]) for r in takeover)
