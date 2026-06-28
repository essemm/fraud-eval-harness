"""
Build compact investigation cases from scored artifacts (brief §13.1,
acceptance A1).

Reads scored cards plus scored rows (the on-disk outputs of the core harness),
selects the high-score cards a reviewer would actually look at, and emits one
JSONL case per card. Each case cleanly separates two halves:

  - prompt_payload : everything the LLM is allowed to see. It contains NO
                     ground-truth label (is_fraud, scenario, any_fraud,
                     fraud_scenario, has_hard_neg are all withheld; A1).
  - evaluation     : ground-truth kept aside for the rubric evaluator only.
                     It is never sent to the model.

The `evidence_facts` list is the grounding anchor: exact, copy-pasteable
strings derived from the case. The investigator is instructed to draw its
supporting evidence from this list, which makes grounding checkable by exact
string match (see evaluate_notes.py).

Usage:
    python -m investigation.build_cases \\
        --rows runs/seed_1/scored_rows_ml.csv \\
        --cards runs/seed_1/scored_cards_ml.csv \\
        --scorer ml --threshold 0.30 --limit 20 --top-rows 5 \\
        --out runs/seed_1/investigation_cases.jsonl
"""

import argparse
import csv
import json
from collections import defaultdict

from . import WITHHELD_LABEL_FIELDS

# Row fields that are safe to show the model: transaction identity and the
# sequence signals that explain the score. Deliberately excludes is_fraud and
# scenario (the withheld labels) and raw PII-ish identifiers beyond ids.
SAFE_ROW_FIELDS = [
    "txn_id", "timestamp", "amount_usd", "merchant_category",
    "merchant_country", "ip_country", "entry_mode",
    "amount_vs_trailing_median", "velocity_1h", "velocity_24h",
    "is_new_device", "is_new_merchant", "is_country_change",
    "score", "reason",
]


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_row(row):
    """Project a scored row down to the prompt-safe fields only."""
    return {k: row[k] for k in SAFE_ROW_FIELDS if k in row}


def _evidence_facts(card_id, top_rows):
    """Exact, groundable strings derived from the card's top rows.

    These are the only strings the investigator is told it may cite as
    supporting evidence, so grounding is checkable by exact match (A5)."""
    facts = []
    for r in top_rows:
        txn = r["txn_id"]
        facts.append(
            f"txn {txn} scored {r['score']} because {r['reason']}")
        v1h = int(_to_float(r.get("velocity_1h")))
        if v1h:
            facts.append(
                f"txn {txn} had {v1h} transactions in the trailing 1h")
        if int(_to_float(r.get("is_new_device"))):
            facts.append(f"txn {txn} used a new device")
        if int(_to_float(r.get("is_country_change"))):
            facts.append(f"txn {txn} changed country versus the prior txn")
        if int(_to_float(r.get("is_new_merchant"))):
            facts.append(f"txn {txn} used a new merchant")
        ratio = _to_float(r.get("amount_vs_trailing_median"))
        if ratio >= 3.0:
            facts.append(
                f"txn {txn} amount was {ratio:.1f}x the card's trailing median")
    return facts


def build_cases(card_rows, scored_rows, scorer, threshold,
                limit=20, top_rows=5):
    """Build a list of case dicts from in-memory card and row dicts.

    Selection: cards with card_score >= threshold, highest score first, capped
    at `limit`. For each selected card, the `top_rows` highest-scoring rows are
    attached (prompt-safe projection) and turned into evidence_facts.
    """
    rows_by_card = defaultdict(list)
    for r in scored_rows:
        rows_by_card[r["card_id"]].append(r)

    selected = [c for c in card_rows
                if _to_float(c["card_score"]) >= threshold]
    selected.sort(key=lambda c: _to_float(c["card_score"]), reverse=True)
    selected = selected[:limit]

    cases = []
    for c in selected:
        card_id = c["card_id"]
        card_rows_sorted = sorted(
            rows_by_card.get(card_id, []),
            key=lambda r: _to_float(r["score"]), reverse=True)
        top = card_rows_sorted[:top_rows]

        prompt_payload = {
            "card_id": card_id,
            "card_score": round(_to_float(c["card_score"]), 4),
            "scorer": scorer,
            "decision_threshold": threshold,
            "top_suspicious_rows": [_safe_row(r) for r in top],
            "evidence_facts": _evidence_facts(card_id, top),
        }

        # Evaluation-only block: ground truth for the rubric, never prompted.
        # has_hard_neg is read from the card if persisted, else derived from
        # row scenarios; either way it stays out of prompt_payload.
        has_hn = c.get("has_hard_neg")
        if has_hn is None:
            has_hn = any(str(r.get("scenario", "")).startswith("hard_neg")
                         for r in rows_by_card.get(card_id, []))
        else:
            has_hn = str(has_hn).lower() in ("1", "true", "yes")

        evaluation = {
            "any_fraud": int(_to_float(c.get("any_fraud"))),
            "fraud_scenario": c.get("fraud_scenario", "none"),
            "has_hard_neg": has_hn,
        }

        cases.append({
            "case_id": card_id,
            "prompt_payload": prompt_payload,
            "evaluation": evaluation,
        })

    return cases


def assert_no_withheld_labels(case):
    """Defensive check: no withheld ground-truth field leaked into the prompt
    payload (A1). Raises ValueError if one is found."""
    payload = case["prompt_payload"]
    blob = json.dumps(payload)
    for field in WITHHELD_LABEL_FIELDS:
        if f'"{field}"' in blob:
            raise ValueError(
                f"withheld label {field!r} leaked into prompt_payload "
                f"for case {case['case_id']}")


def _load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description="Build compact investigation cases from scored artifacts.")
    ap.add_argument("--rows", required=True, help="scored rows CSV")
    ap.add_argument("--cards", required=True, help="scored cards CSV")
    ap.add_argument("--scorer", default="ml", choices=["rules", "ml"])
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--top-rows", type=int, default=5)
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    scored_rows = _load_csv(args.rows)
    card_rows = _load_csv(args.cards)
    cases = build_cases(card_rows, scored_rows, args.scorer, args.threshold,
                        limit=args.limit, top_rows=args.top_rows)

    for case in cases:
        assert_no_withheld_labels(case)

    with open(args.out, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    print(f"wrote {len(cases)} cases to {args.out} "
          f"(scorer={args.scorer}, threshold={args.threshold}, "
          f"limit={args.limit})")


if __name__ == "__main__":
    main()
