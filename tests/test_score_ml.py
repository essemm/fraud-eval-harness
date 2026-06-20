"""
ML scorer acceptance tests.

S1 and S2 criteria applied to MLScorer, plus checks specific to the ML path:
interface compatibility with RuleScorer, out-of-sample scoring, and the basic
sanity that fraud cards score higher than clean cards on held-out data.
"""

import pytest

from fraud_eval import fx
from fraud_eval import generate_synthetic as gen
from fraud_eval import profile as prof
from fraud_eval import features as feat
from fraud_eval import score as scr
from fraud_eval.score_ml import ML_FEATURES, _featurise, fit_from_rows

RATES = dict(fx.RATES)


def _featured(seed, n_cards=200, n_days=20):
    txns = gen.generate(n_cards, n_days, fraud_rate=0.05,
                        hard_neg_rate=0.08, seed=seed)
    profiles = prof.build_profiles(txns, RATES)
    return feat.build_features(txns, profiles, RATES)


@pytest.fixture(scope="module")
def train_rows():
    return _featured(seed=1)


@pytest.fixture(scope="module")
def eval_rows():
    return _featured(seed=2)


@pytest.fixture(scope="module")
def ml_scorer(train_rows):
    return fit_from_rows(train_rows)


@pytest.fixture(scope="module")
def ml_scored(ml_scorer, eval_rows):
    return [ml_scorer.score_row(r) for r in eval_rows]


# --- S1 -------------------------------------------------------------------

def test_ML_S1_score_in_range_with_reason(ml_scored):
    """S1 for MLScorer: score in [0,1], non-empty reason on every row."""
    for r in ml_scored:
        assert 0.0 <= float(r["score"]) <= 1.0
        assert isinstance(r["reason"], str) and r["reason"]


# --- S2 -------------------------------------------------------------------

def test_ML_S2_key_set_matches_rule_scorer(train_rows, eval_rows):
    """S2: MLScorer and RuleScorer produce rows with identical key sets,
    so evaluate.py cannot tell which scorer produced the output."""
    ml = fit_from_rows(train_rows)
    rule = scr.RuleScorer()
    assert set(ml.score_row(eval_rows[0])) == set(rule.score_row(eval_rows[0]))


# --- featurise ------------------------------------------------------------

def test_featurise_length(eval_rows):
    """_featurise returns a vector with one value per ML_FEATURES entry."""
    fv = _featurise(eval_rows[0])
    assert len(fv) == len(ML_FEATURES)


def test_featurise_sentinel_replaced(eval_rows):
    """secs_since_prev sentinel (-1) is replaced with a positive value."""
    first_rows = [r for r in eval_rows if int(r["secs_since_prev"]) < 0]
    assert first_rows, "no sentinel rows found — fixture too small"
    for r in first_rows:
        fv = _featurise(r)
        secs_idx = ML_FEATURES.index("secs_since_prev")
        assert fv[secs_idx] > 0


# --- train/eval split -----------------------------------------------------

def test_ML_train_eval_no_overlap(train_rows, eval_rows):
    """Training and eval sets come from different seeds and share no rows."""
    train_ids = {r["txn_id"] for r in train_rows}
    eval_ids = {r["txn_id"] for r in eval_rows}
    assert train_ids.isdisjoint(eval_ids)


# --- sanity ---------------------------------------------------------------

def test_ML_fraud_cards_score_higher(ml_scorer, eval_rows):
    """Mean card score for fraud cards exceeds mean for clean cards."""
    scored = [ml_scorer.score_row(r) for r in eval_rows]
    cards = scr.aggregate_cards(scored, method="decaying_sum")
    fraud = [c["card_score"] for c in cards if c["any_fraud"] == 1]
    clean = [c["card_score"] for c in cards if c["any_fraud"] == 0]
    assert fraud and clean, "need both fraud and clean cards"
    assert sum(fraud) / len(fraud) > sum(clean) / len(clean)
