"""
Downstream investigation layer (brief §13, acceptance A1–A7).

This package is a pure *consumer* of the scored artifacts the core harness
writes (scored_rows_*.csv, scored_cards_*.csv). It prepares structured case
notes for a human reviewer; it never decides whether fraud occurred, changes
scores, sets thresholds, alters features, or touches any core evaluation
output. It is deliberately decoupled from fraud_eval/: nothing here imports it
(A7), so the investigation layer cannot influence detection.

The constants below are shared by the investigator (which enforces them before
writing a note) and the note evaluator (which grades notes against them), so
the two never drift apart.
"""

# The only recommended_action values an investigation note may carry (brief
# §13.2, acceptance A3).
ALLOWED_ACTIONS = (
    "no_action",
    "manual_review",
    "step_up_auth",
    "customer_contact",
    "block_or_suspend",
)

# Conclusions the note must never state: it summarises evidence for a human,
# it does not adjudicate fraud (brief §13.2, acceptance A4).
FORBIDDEN_PHRASES = (
    "confirmed fraud",
    "fraud confirmed",
    "definitely fraud",
    "definite fraud",
    "is fraudulent",
    "this is fraud",
    "guaranteed fraud",
)

# Customer-accusatory phrasing the note must never use (acceptance A4 / A5
# customer-safe-language rubric).
ACCUSATORY_PHRASES = (
    "you committed",
    "you stole",
    "you are a fraud",
    "fraudster",
    "criminal",
    "thief",
    "you are guilty",
    "caught you",
)

# Ground-truth fields that must never reach the LLM prompt (brief §13.1,
# acceptance A1). build_cases keeps these out of prompt_payload.
WITHHELD_LABEL_FIELDS = (
    "is_fraud",
    "scenario",
    "any_fraud",
    "fraud_scenario",
    "has_hard_neg",
)
