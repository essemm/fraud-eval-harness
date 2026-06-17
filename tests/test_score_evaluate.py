"""
Scoring (S1-S3) and evaluation (E1-E4) acceptance criteria from brief §10.
"""

import score as scr
import evaluate as ev


# --- scoring --------------------------------------------------------------

def test_S1_score_in_range_with_reason(scored_rows):
    """S1: every scored row has a score in [0,1] and a non-empty reason."""
    for r in scored_rows:
        assert 0.0 <= float(r["score"]) <= 1.0
        assert isinstance(r["reason"], str) and r["reason"], "empty reason"


def test_S2_scorer_interface_stable(featured):
    """S2: a replacement scorer implementing the same score_row interface
    produces rows with the same downstream-relevant keys, so features and
    evaluate are untouched by a scorer swap."""
    class StubScorer:
        def score_row(self, row):
            out = dict(row)
            out["score"] = 0.5
            out["reason"] = "stub"
            return out
    rule_out = scr.RuleScorer().score_row(featured[0])
    stub_out = StubScorer().score_row(featured[0])
    # same key set -> evaluate.py cannot tell which scorer produced the row
    assert set(rule_out) == set(stub_out)


def test_S3_aggregation_deterministic(scored_rows):
    """S3: card-level aggregation is deterministic for fixed input + config."""
    a = scr.aggregate_cards(scored_rows, method="decaying_sum")
    b = scr.aggregate_cards(scored_rows, method="decaying_sum")
    assert a == b


def test_S3_aggregation_methods_differ(scored_rows):
    """S3 corollary: the two aggregation methods are genuinely different
    functions (a sanity check that 'configurable' means something)."""
    mx = scr.aggregate_cards(scored_rows, method="max")
    ds = scr.aggregate_cards(scored_rows, method="decaying_sum")
    mx_scores = {c["card_id"]: c["card_score"] for c in mx}
    ds_scores = {c["card_id"]: c["card_score"] for c in ds}
    assert mx_scores != ds_scores


# --- evaluation -----------------------------------------------------------

def _metrics(scored_rows, scored_cards):
    return ev.evaluate(scored_rows, scored_cards, ratio=20.0,
                       fp_review_cost=5.0, step=0.05)


def test_E1_sweep_monotonic(scored_rows, scored_cards):
    """E1: raising the threshold cannot increase the flagged count."""
    m = _metrics(scored_rows, scored_cards)
    flagged = [r["tp"] + r["fp"] for r in m["sweep"]]
    for i in range(len(flagged) - 1):
        assert flagged[i] >= flagged[i + 1], \
            f"flagged count rose from threshold step {i} to {i+1}"


def test_E2_per_scenario_recall_reported(scored_rows, scored_cards):
    """E2: per-scenario recall is reported for all four fraud scenarios."""
    m = _metrics(scored_rows, scored_cards)
    psr = m["diagnostics_at_reference"]["per_scenario_recall"]
    for sc in ["card_testing", "account_takeover", "impossible_travel",
               "stolen_spree"]:
        assert sc in psr


def test_E3_cost_min_is_true_min(scored_rows, scored_cards):
    """E3: the reported cost-minimising threshold actually minimises total
    cost over the swept range, under each cost model."""
    m = _metrics(scored_rows, scored_cards)
    sweep = m["sweep"]
    true_min_fixed = min(r["cost_fixed_ratio"] for r in sweep)
    true_min_amount = min(r["cost_amount_weighted"] for r in sweep)
    assert abs(m["operating_point"]["fixed_ratio"]["cost"]
               - true_min_fixed) < 0.01
    assert abs(m["operating_point"]["amount_weighted"]["cost_usd"]
               - true_min_amount) < 0.01


def test_E4_accuracy_not_a_headline(scored_rows, scored_cards):
    """E4: accuracy does not appear as a headline metric. It may appear only
    in the explanatory note about WHY it is omitted."""
    m = _metrics(scored_rows, scored_cards)
    report = ev.render_report(m)
    for line in report.splitlines():
        if "accuracy" in line.lower():
            assert ("deliberately not" in line or "do-nothing" in line), \
                f"accuracy used as a metric: {line!r}"


def test_E_degeneracy_detected_at_extreme_ratio(scored_rows, scored_cards):
    """Not a numbered criterion, but a property we built in: a very aggressive
    ratio against this class imbalance should produce a degenerate (flag-all)
    cost minimum, and the harness should mark it as such rather than presenting
    it as a real operating point."""
    m = ev.evaluate(scored_rows, scored_cards, ratio=1000.0,
                    fp_review_cost=1.0, step=0.05)
    assert m["operating_point"]["fixed_ratio"]["degenerate"] is True
