"""
Shared fixtures for the acceptance-criteria test suite.

Per the brief's NFR (§9), every module is callable as a pure function on
in-memory data, so these fixtures build a small dataset in memory and pass it
through the pipeline -- no files on disk, fast enough to run on every save.

The dataset is deliberately small but large enough that all four fraud
scenarios and all three hard-negative types appear (a fixed seed guarantees
they do, and G3 asserts it).
"""

import pytest

from fraud_eval import fx
from fraud_eval import generate_synthetic as gen
from fraud_eval import profile as prof
from fraud_eval import features as feat
from fraud_eval import score as scr
from fraud_eval import evaluate as ev

SEED = 12345


@pytest.fixture(scope="session")
def rates():
    """The default FX rate table, in memory (a copy, so tests can't mutate
    the module global and affect each other)."""
    return dict(fx.RATES)


@pytest.fixture(scope="session")
def txns():
    """A fixed-seed synthetic transaction set, as a list of dict rows."""
    return gen.generate(n_cards=400, n_days=20, fraud_rate=0.05,
                        hard_neg_rate=0.08, seed=SEED)


@pytest.fixture(scope="session")
def profiles(txns, rates):
    return prof.build_profiles(txns, rates)


@pytest.fixture(scope="session")
def featured(txns, profiles, rates):
    return feat.build_features(txns, profiles, rates)


@pytest.fixture(scope="session")
def scored_rows(featured):
    scorer = scr.RuleScorer()
    return [scorer.score_row(r) for r in featured]


@pytest.fixture(scope="session")
def scored_cards(scored_rows):
    return scr.aggregate_cards(scored_rows, method="decaying_sum")
