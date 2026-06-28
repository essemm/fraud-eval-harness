"""
Readable renderer for investigation JSON artifacts.

The investigation layer intentionally writes JSON/JSONL so notes and rubric
results are easy to test and diff. This helper is for humans: it joins notes
with their optional cases and rubric output, then prints a compact plain-text
view suitable for a terminal or a quick portfolio review.

Usage:
    python -m investigation.render_notes \\
        --cases runs/seed_1/investigation_cases.jsonl \\
        --notes runs/seed_1/investigation_notes.jsonl \\
        --eval  runs/seed_1/investigation_eval.json \\
        --limit 5
"""

import argparse
import json
import textwrap

from .evaluate_notes import RUBRIC_KEYS
from .investigate import clean_note_text


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _wrap(text, indent="  ", width=88):
    wrapped = textwrap.fill(str(text), width=width,
                            initial_indent=indent,
                            subsequent_indent=indent)
    return wrapped


def _list_block(label, items, limit=None):
    lines = [f"{label}:"]
    shown = list(items or [])
    if limit is not None:
        shown = shown[:limit]
    if not shown:
        lines.append("  - (none)")
        return lines
    for item in shown:
        lines.append(_wrap(f"- {item}", indent="  "))
    return lines


def _grade_summary(grade):
    if not grade:
        return "rubric: n/a"
    parts = []
    for key in RUBRIC_KEYS:
        if key in grade:
            parts.append(f"{key}={'pass' if grade[key] else 'fail'}")
    return "rubric: " + ", ".join(parts)


def render_note(note, case=None, grade=None, evidence_limit=6):
    """Return a readable text block for one investigation note."""
    note = clean_note_text(note)
    lines = []
    card_id = note.get("card_id", "(unknown card)")
    action = note.get("recommended_action", "(missing action)")
    score = None
    threshold = None
    if case:
        payload = case.get("prompt_payload", {})
        score = payload.get("card_score")
        threshold = payload.get("decision_threshold")

    heading = f"{card_id} | action={action}"
    if score is not None:
        heading += f" | score={score}"
    if threshold is not None:
        heading += f" | threshold={threshold}"
    lines.append(heading)
    lines.append("-" * len(heading))
    lines.append(_grade_summary(grade))
    lines.append("")

    lines.append("risk_summary:")
    lines.append(_wrap(note.get("risk_summary", ""), indent="  "))
    lines.append("")

    lines.extend(_list_block("supporting_evidence",
                             note.get("supporting_evidence", [])))
    lines.append("")
    lines.extend(_list_block("missing_information",
                             note.get("missing_information", [])))
    lines.append("")

    lines.append("customer_safe_language:")
    lines.append(_wrap(note.get("customer_safe_language", ""), indent="  "))
    lines.append("")

    lines.extend(_list_block("caveats", note.get("caveats", [])))

    if case:
        facts = case.get("prompt_payload", {}).get("evidence_facts", [])
        lines.append("")
        lines.extend(_list_block("case evidence_facts",
                                 facts, limit=evidence_limit))
        if evidence_limit is not None and len(facts) > evidence_limit:
            lines.append(f"  - ... {len(facts) - evidence_limit} more")

    return "\n".join(lines)


def render_notes(cases, notes, eval_result=None, limit=None):
    """Render a sequence of notes with optional cases and rubric results."""
    cases_by_id = {c["case_id"]: c for c in (cases or [])}
    grades_by_id = {
        c["card_id"]: c
        for c in (eval_result or {}).get("cases", [])
    }

    selected = list(notes)
    if limit is not None:
        selected = selected[:limit]

    blocks = []
    for note in selected:
        cid = note.get("card_id")
        blocks.append(render_note(
            note,
            case=cases_by_id.get(cid),
            grade=grades_by_id.get(cid),
        ))

    header = [f"Investigation notes: {len(selected)} shown"]
    if limit is not None and len(notes) > limit:
        header[0] += f" of {len(notes)}"
    if eval_result:
        agg = eval_result.get("aggregate", {})
        header.append("Aggregate rubric:")
        for key in RUBRIC_KEYS:
            if key in agg:
                val = agg[key]
                shown = "n/a" if val is None else f"{val:.3f}"
                header.append(f"  {key}: {shown}")
    header.append("")

    return "\n".join(header + ["\n\n".join(blocks)])


def main():
    ap = argparse.ArgumentParser(
        description="Render investigation notes JSONL as readable text.")
    ap.add_argument("--notes", required=True, help="notes JSONL")
    ap.add_argument("--cases", help="optional cases JSONL")
    ap.add_argument("--eval", dest="eval_path", help="optional eval JSON")
    ap.add_argument("--limit", type=int, help="maximum notes to display")
    ap.add_argument("--out", help="optional text output path")
    args = ap.parse_args()

    notes = _load_jsonl(args.notes)
    cases = _load_jsonl(args.cases) if args.cases else []
    eval_result = _load_json(args.eval_path) if args.eval_path else None

    text = render_notes(cases, notes, eval_result=eval_result,
                        limit=args.limit)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
