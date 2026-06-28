"""
Investigation-layer tests (brief §13; acceptance A1–A7).

All fixtures are built in memory as lists of string-valued dicts, exactly as
csv.DictReader would yield them from the scored artifacts — so the tests
exercise the layer the way the CLI does, with no files and no live LLM (A6).
"""

import json
import os

import pytest

from investigation import ALLOWED_ACTIONS
from investigation.build_cases import build_cases, assert_no_withheld_labels
from investigation.investigate import (
    FakeModel, investigate_case, investigate_all, validate_note,
    extract_json_object, NoteValidationError,
)
from investigation.evaluate_notes import evaluate_notes, grade_case, RUBRIC_KEYS

WITHHELD = ("is_fraud", "scenario", "any_fraud", "fraud_scenario",
            "has_hard_neg")


def _row(txn_id, card_id, score, **over):
    """A scored-row dict with CSV-style string values."""
    base = {
        "txn_id": txn_id, "card_id": card_id, "timestamp": "2025-01-01T00:00:00",
        "amount_usd": "100.0", "merchant_category": "electronics",
        "merchant_country": "US", "ip_country": "US", "entry_mode": "online",
        "amount_vs_trailing_median": "1.0", "velocity_1h": "1",
        "velocity_24h": "1", "is_new_device": "0", "is_new_merchant": "0",
        "is_country_change": "0", "score": str(score),
        "reason": "no_rule_fired", "is_fraud": "0", "scenario": "legit",
    }
    base.update({k: str(v) for k, v in over.items()})
    return base


@pytest.fixture
def scored_rows():
    return [
        # Card A: clear fraud-like card, well above threshold.
        _row("t_a1", "card_A", 0.9, velocity_1h=8, is_new_device=1,
             reason="velocity_burst: 8 txns in trailing 1h", is_fraud=1,
             scenario="card_testing"),
        _row("t_a2", "card_A", 0.7, amount_vs_trailing_median=5.0,
             reason="amount_spike: 5.0x trailing median", is_fraud=1,
             scenario="card_testing"),
        _row("t_a3", "card_A", 0.1),
        # Card B: a hard-negative card (legit, but trips a single-row signal).
        _row("t_b1", "card_B", 0.6, is_country_change=1,
             reason="impossible_travel: geography change after 100s",
             scenario="hard_neg_travel"),
        # Card C: below threshold, must be excluded.
        _row("t_c1", "card_C", 0.05),
    ]


@pytest.fixture
def card_rows():
    return [
        {"card_id": "card_A", "card_score": "0.95", "n_rows": "3",
         "top_reason": "velocity_burst: 8 txns in trailing 1h",
         "any_fraud": "1", "fraud_scenario": "card_testing",
         "has_hard_neg": "False"},
        {"card_id": "card_B", "card_score": "0.55", "n_rows": "1",
         "top_reason": "impossible_travel: geography change after 100s",
         "any_fraud": "0", "fraud_scenario": "none",
         "has_hard_neg": "True"},
        {"card_id": "card_C", "card_score": "0.05", "n_rows": "1",
         "top_reason": "no_rule_fired", "any_fraud": "0",
         "fraud_scenario": "none", "has_hard_neg": "False"},
    ]


@pytest.fixture
def cases(card_rows, scored_rows):
    return build_cases(card_rows, scored_rows, scorer="ml", threshold=0.30,
                       limit=20, top_rows=5)


# --- A1: compact cases, no ground-truth in prompt -------------------------

def test_A1_no_withheld_labels_in_prompt_payload(cases):
    """prompt_payload carries none of the withheld ground-truth fields."""
    assert cases, "fixture produced no cases"
    for case in cases:
        blob = json.dumps(case["prompt_payload"])
        for field in WITHHELD:
            assert f'"{field}"' not in blob
        # the defensive guard agrees
        assert_no_withheld_labels(case)


def test_A1_threshold_selection_and_ordering(cases):
    """Only cards >= threshold are kept, highest score first; card_C excluded."""
    ids = [c["case_id"] for c in cases]
    assert ids == ["card_A", "card_B"]   # card_C (0.05) dropped, A before B


def test_A1_top_rows_are_prompt_safe(cases):
    """Top suspicious rows expose features but not labels."""
    rows = cases[0]["prompt_payload"]["top_suspicious_rows"]
    assert rows
    for r in rows:
        assert "is_fraud" not in r and "scenario" not in r
        assert "txn_id" in r and "score" in r


def test_A1_evidence_facts_present(cases):
    """evidence_facts are non-empty and reference the card's txns."""
    facts = cases[0]["prompt_payload"]["evidence_facts"]
    assert facts
    assert any("t_a1" in f for f in facts)


def test_A1_evaluation_block_carries_ground_truth(cases):
    """The evaluation block (never prompted) holds the labels for the rubric."""
    by_id = {c["case_id"]: c for c in cases}
    assert by_id["card_A"]["evaluation"]["any_fraud"] == 1
    assert by_id["card_B"]["evaluation"]["has_hard_neg"] is True


# --- A2: one validated note per case --------------------------------------

def test_A2_one_note_per_case(cases):
    notes = investigate_all(FakeModel(), cases)
    assert len(notes) == len(cases)
    for note, case in zip(notes, cases):
        assert note["card_id"] == case["case_id"]
        validate_note(note, case)  # re-validate: must not raise


# --- A3: invalid action rejected ------------------------------------------

def test_A3_invalid_action_rejected(cases):
    case = cases[0]
    note = json.loads(FakeModel().generate(case))
    note["recommended_action"] = "delete_account"   # not in the enum
    with pytest.raises(NoteValidationError):
        validate_note(note, case)


def test_A3_allowed_actions_accepted(cases):
    case = cases[0]
    base = json.loads(FakeModel().generate(case))
    for action in ALLOWED_ACTIONS:
        note = dict(base, recommended_action=action)
        validate_note(note, case)  # must not raise


# --- A4: forbidden conclusions / accusations ------------------------------

class _ForbiddenModel:
    """A model that emits a note claiming confirmed fraud."""
    def generate(self, case):
        note = json.loads(FakeModel().generate(case))
        note["risk_summary"] = "This is confirmed fraud on the account."
        return json.dumps(note)


class _AccusatoryModel:
    def generate(self, case):
        note = json.loads(FakeModel().generate(case))
        note["customer_safe_language"] = "You committed fraud and we caught you."
        return json.dumps(note)


def test_A4_forbidden_conclusion_rejected_at_write(cases):
    with pytest.raises(NoteValidationError):
        investigate_case(_ForbiddenModel(), cases[0])


def test_A4_accusatory_language_rejected_at_write(cases):
    with pytest.raises(NoteValidationError):
        investigate_case(_AccusatoryModel(), cases[0])


def test_A4_evaluator_flags_forbidden_note(cases):
    """If an unsafe note somehow reaches the evaluator, the rubric fails it."""
    case = cases[0]
    bad = json.loads(FakeModel().generate(case))
    bad["risk_summary"] = "Confirmed fraud — the customer is a fraudster."
    graded = grade_case(bad, case)
    assert graded["no_forbidden_conclusion"] is False


def test_A4_invalid_json_rejected(cases):
    class _Garbage:
        def generate(self, case):
            return "not json at all"
    with pytest.raises(NoteValidationError):
        investigate_case(_Garbage(), cases[0])


# --- tolerant JSON extraction from weak-model replies ---------------------

def test_extract_json_from_prose_and_fences():
    """The first balanced object is recovered from prose / code-fence wrapping."""
    obj = '{"a": 1, "b": {"c": "}"}}'   # note the brace inside a string
    assert json.loads(extract_json_object(obj)) == {"a": 1, "b": {"c": "}"}}

    fenced = "Here is the note:\n```json\n" + obj + "\n```\nHope that helps!"
    assert json.loads(extract_json_object(fenced)) == {"a": 1, "b": {"c": "}"}}


def test_extract_json_returns_none_without_object():
    assert extract_json_object("no braces here") is None
    assert extract_json_object('{ "unterminated": true ') is None


def test_chatty_model_reply_still_validates(cases):
    """A note wrapped in prose + a code fence is extracted and accepted."""
    case = cases[0]
    good = FakeModel().generate(case)

    class _ChattyModel:
        def generate(self, case):
            return f"Sure! Here is the JSON note:\n```json\n{good}\n```\n"

    note = investigate_case(_ChattyModel(), case)
    validate_note(note, case)  # must not raise


# --- A5: per-case rubric + aggregates -------------------------------------

def test_A5_rubric_booleans_and_aggregates(cases):
    notes = investigate_all(FakeModel(), cases)
    result = evaluate_notes(cases, notes)

    assert result["n_cases"] == len(cases)
    assert set(result["aggregate"]) == set(RUBRIC_KEYS)
    for case_result in result["cases"]:
        for key in RUBRIC_KEYS:
            assert isinstance(case_result[key], bool)
    # The fake model is built to satisfy every rubric on these cases.
    for key in RUBRIC_KEYS:
        assert result["aggregate"][key] == 1.0


def test_A5_hard_negative_caution_catches_overconfident_block(cases):
    """A hard-negative case escalated to block_or_suspend with no caveat fails
    the hard_negative_caution rubric."""
    hn_case = next(c for c in cases if c["evaluation"]["has_hard_neg"])
    note = json.loads(FakeModel().generate(hn_case))
    note["recommended_action"] = "block_or_suspend"
    note["caveats"] = []
    graded = grade_case(note, hn_case)
    assert graded["hard_negative_caution"] is False


def test_A5_missing_note_is_reported_not_fatal(cases):
    """A case with no matching note (e.g. dropped by --skip-invalid) is counted
    under `missing` and excluded from the aggregates, not graded as a failure."""
    notes = investigate_all(FakeModel(), cases)
    dropped = notes[1:]   # pretend the first case's note was skipped
    result = evaluate_notes(cases, dropped)

    assert result["n_cases"] == len(cases) - 1
    assert result["n_missing"] == 1
    assert cases[0]["case_id"] in result["missing"]
    # aggregates still computed over the graded subset
    assert result["aggregate"]["valid_action"] == 1.0


def test_A5_grounded_evidence_fails_on_invented_fact(cases):
    """Evidence not present in the case fails the grounding check."""
    case = cases[0]
    note = json.loads(FakeModel().generate(case))
    note["supporting_evidence"] = ["the cardholder admitted to the purchase"]
    graded = grade_case(note, case)
    assert graded["grounded_evidence"] is False


# --- A7: downstream only, no fraud_eval coupling --------------------------

def test_A7_no_fraud_eval_imports():
    """No investigation module imports fraud_eval (downstream isolation).

    Parsed with ast so only real import statements count — a mention of
    'fraud_eval' in a docstring or comment is not a coupling."""
    import ast
    import investigation
    pkg_dir = os.path.dirname(investigation.__file__)
    for name in os.listdir(pkg_dir):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(pkg_dir, name)) as f:
            tree = ast.parse(f.read(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                assert not mod.startswith("fraud_eval"), \
                    f"{name} imports {mod!r}"
