"""
Investigation-note generator (brief §13.2–13.3, acceptance A2–A4, A6).

Consumes the case JSONL from build_cases.py and writes one structured note per
case. The note is produced by a model behind a small, explicit adapter:

  - FakeModel    : deterministic, no network, used by tests and the --fake CLI
                   path. It behaves as a constrained, cautious summariser:
                   it grounds its evidence in evidence_facts, always asks for
                   missing information, always carries a caveat, and never
                   escalates straight to block_or_suspend.
  - CommandModel : runs a local LLM as a subprocess, sending the prompt on
                   stdin and reading the JSON note from stdout. This keeps the
                   real-model path runnable offline (e.g. `ollama run ...`).

Every note is validated BEFORE it is written (A2–A4). On invalid JSON or any
failed check the run raises and writes nothing further, so an unsafe or
malformed note never lands on disk silently.

Usage:
    # deterministic, no model required (tests + smoke):
    python -m investigation.investigate \\
        --cases runs/seed_1/investigation_cases.jsonl \\
        --out   runs/seed_1/investigation_notes.jsonl --fake

    # real local model:
    python -m investigation.investigate \\
        --cases runs/seed_1/investigation_cases.jsonl \\
        --out   runs/seed_1/investigation_notes.jsonl \\
        --llm-command "ollama run llama3.2:3b-instruct"
"""

import argparse
import json
import re
import subprocess

# Terminal control / ANSI escape sequences. CLI model runners (e.g. `ollama
# run`) stream tokens with cursor-movement and erase-line codes that land in
# captured stdout; strip them before parsing so the JSON text is clean.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

from . import ALLOWED_ACTIONS, FORBIDDEN_PHRASES, ACCUSATORY_PHRASES

# Note fields and their expected shapes (A2). str fields are non-empty; list
# fields are lists of strings.
_STR_FIELDS = ("card_id", "recommended_action", "risk_summary",
               "customer_safe_language")
_LIST_FIELDS = ("supporting_evidence", "missing_information", "caveats")


def build_prompt(case):
    """Render the instruction + prompt-safe payload sent to the LLM.

    Only prompt_payload is included — never the evaluation block. The
    instructions pin the output contract and the safety constraints so a weak
    local model is steered toward a constrained summary, not a verdict."""
    payload = case["prompt_payload"]
    schema = {
        "card_id": payload["card_id"],
        "recommended_action": "one of " + ", ".join(ALLOWED_ACTIONS),
        "risk_summary": "short, evidence-grounded, no verdict",
        "supporting_evidence": ["copy items verbatim from evidence_facts"],
        "missing_information": ["what a human reviewer should still obtain"],
        "customer_safe_language": "neutral, non-accusatory wording",
        "caveats": ["reasons this may be a false positive"],
    }
    return (
        "You are assisting a fraud-review analyst. You are NOT deciding whether "
        "fraud occurred. Summarise the evidence and recommend a review step.\n"
        "Rules:\n"
        "- Do not claim fraud is confirmed; do not accuse the customer.\n"
        "- Draw supporting_evidence verbatim from the evidence_facts list.\n"
        "- Always include at least one missing_information item and one caveat.\n"
        "- recommended_action must be one of: " + ", ".join(ALLOWED_ACTIONS)
        + ".\n"
        "Return ONLY a JSON object with this shape:\n"
        + json.dumps(schema, indent=2) + "\n\n"
        "CASE:\n" + json.dumps(payload, indent=2) + "\n"
    )


class FakeModel:
    """Deterministic stand-in for a local LLM. Produces a safe, grounded note
    from prompt_payload alone (it never sees the evaluation block)."""

    def generate(self, case):
        payload = case["prompt_payload"]
        card_id = payload["card_id"]
        facts = payload.get("evidence_facts", [])
        n_rows = len(payload.get("top_suspicious_rows", []))

        note = {
            "card_id": card_id,
            # Cautious default: route to a human, never auto-escalate to block.
            "recommended_action": "manual_review",
            "risk_summary": (
                f"Card {card_id} scored {payload['card_score']} under the "
                f"{payload['scorer']} scorer, at or above the review threshold "
                f"of {payload['decision_threshold']}. {n_rows} transactions "
                f"contributed to the score and warrant a human look."),
            # Grounded: evidence copied verbatim from evidence_facts.
            "supporting_evidence": list(facts[:4]),
            "missing_information": [
                "Confirmation from the cardholder about the recent transactions",
                "Whether the device and locations involved are recognised",
            ],
            "customer_safe_language": (
                "We noticed some unusual activity on your card and would like "
                "to confirm a few recent transactions with you."),
            "caveats": [
                "This is an automated score, not a confirmation of fraud.",
                "Legitimate travel or a new device can produce a similar "
                "pattern.",
            ],
        }
        return json.dumps(note)


class CommandModel:
    """Runs a local LLM via a shell command. The prompt goes to stdin; the raw
    note JSON is read from stdout."""

    def __init__(self, command, timeout=120):
        self.command = command
        self.timeout = timeout

    def generate(self, case):
        prompt = build_prompt(case)
        proc = subprocess.run(
            self.command, shell=True, input=prompt,
            capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"llm command failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:500]}")
        return _ANSI_RE.sub("", proc.stdout)


class NoteValidationError(ValueError):
    """Raised when a model note is malformed or violates a safety constraint."""


def _contains_any(text, phrases):
    low = text.lower()
    return [p for p in phrases if p in low]


def validate_note(note, case):
    """Validate a parsed note against the schema and safety rules (A2–A4).

    Raises NoteValidationError on any violation; returns the note unchanged on
    success."""
    if not isinstance(note, dict):
        raise NoteValidationError("note is not a JSON object")

    missing = [f for f in (_STR_FIELDS + _LIST_FIELDS) if f not in note]
    if missing:
        raise NoteValidationError(f"missing required fields: {missing}")

    for f in _STR_FIELDS:
        if not isinstance(note[f], str) or not note[f].strip():
            raise NoteValidationError(f"field {f!r} must be a non-empty string")

    for f in _LIST_FIELDS:
        v = note[f]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise NoteValidationError(f"field {f!r} must be a list of strings")

    if note["recommended_action"] not in ALLOWED_ACTIONS:
        raise NoteValidationError(
            f"recommended_action {note['recommended_action']!r} not in "
            f"allowed set {ALLOWED_ACTIONS}")

    if note["card_id"] != case["prompt_payload"]["card_id"]:
        raise NoteValidationError(
            f"note card_id {note['card_id']!r} does not match case "
            f"{case['prompt_payload']['card_id']!r}")

    # Safety: scan all free text for forbidden conclusions / accusations and
    # for any reference to a withheld label value.
    blob = " ".join([note[f] for f in _STR_FIELDS]
                    + [x for f in _LIST_FIELDS for x in note[f]])
    hit = _contains_any(blob, FORBIDDEN_PHRASES)
    if hit:
        raise NoteValidationError(f"forbidden conclusion phrase(s): {hit}")
    hit = _contains_any(blob, ACCUSATORY_PHRASES)
    if hit:
        raise NoteValidationError(f"accusatory phrase(s): {hit}")

    return note


def extract_json_object(text):
    """Pull the first balanced top-level JSON object out of a model reply.

    Weak local models rarely return bare JSON: they wrap it in prose, in a
    ```json fence, or after a leading line. This finds the first '{' and scans
    to its matching '}', respecting string literals and escapes, so such
    replies still parse. Returns the JSON substring, or None if no object is
    present. It does NOT relax validation — the extracted object still goes
    through validate_note, so an off-contract note is still rejected.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None  # opened but never closed


def investigate_case(model, case):
    """Generate, parse, and validate one note. Raises on any failure."""
    raw = model.generate(case)
    candidate = extract_json_object(raw)
    if candidate is None:
        raise NoteValidationError(
            f"no JSON object found in model reply for case "
            f"{case['case_id']}: {raw.strip()[:200]!r}")
    try:
        # strict=False tolerates literal newlines/tabs inside string values,
        # which weak models emit routinely. It does not relax the contract:
        # validate_note still enforces fields, action enum, and safety.
        note = json.loads(candidate, strict=False)
    except json.JSONDecodeError as e:
        raise NoteValidationError(
            f"model returned invalid JSON for case "
            f"{case['case_id']}: {e}") from e
    return validate_note(note, case)


def investigate_all(model, cases):
    """Run every case through the model, validating each. If any case fails,
    the exception propagates and no notes are returned (the CLI then writes
    nothing), so a partial/unsafe batch never lands on disk."""
    return [investigate_case(model, case) for case in cases]


def _load_cases(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Generate structured investigation notes from cases.")
    ap.add_argument("--cases", required=True, help="cases JSONL from build_cases")
    ap.add_argument("--out", required=True, help="output notes JSONL")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fake", action="store_true",
                   help="use the deterministic fake model (no network)")
    g.add_argument("--llm-command",
                   help="shell command running a local LLM (prompt on stdin, "
                        "JSON note on stdout)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-case timeout for --llm-command, seconds")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="skip cases whose note fails to parse/validate "
                         "(logging each) and write the valid ones, instead of "
                         "aborting the whole batch. Useful with weak local "
                         "models; the default stays strict (no partial write).")
    args = ap.parse_args()

    cases = _load_cases(args.cases)
    model = FakeModel() if args.fake else CommandModel(args.llm_command,
                                                       timeout=args.timeout)

    if args.skip_invalid:
        # Lenient: keep going past a bad generation, report what was dropped.
        notes, skipped = [], []
        for case in cases:
            try:
                notes.append(investigate_case(model, case))
            except (NoteValidationError, RuntimeError) as e:
                skipped.append(case["case_id"])
                print(f"  skipped {case['case_id']}: {e}")
        if skipped:
            print(f"skipped {len(skipped)}/{len(cases)} cases: {skipped}")
    else:
        # Strict default: validate the whole batch before writing anything.
        notes = investigate_all(model, cases)

    with open(args.out, "w") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")

    print(f"wrote {len(notes)} validated notes to {args.out} "
          f"(model={'fake' if args.fake else 'command'})")


if __name__ == "__main__":
    main()
