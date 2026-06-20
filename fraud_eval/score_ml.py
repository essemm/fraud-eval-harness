"""
ML scorer: logistic regression trained on featured rows.

Implements the Scorer protocol (scorer.py) — score_row is the only method
evaluate.py calls, so MLScorer is a drop-in for RuleScorer with no changes
to features.py or evaluate.py.

Training is separate from inference: MLScorer.__init__ takes a fitted sklearn
model; fit_from_rows() is the factory that trains it. This keeps evaluate.py
scorer-agnostic — it never calls fit, only score_row.

Reason string: the top-contributing feature by abs(scaled_value * coefficient).
This satisfies brief S1 (non-empty reason) and keeps the ML decision as legible
as the rule baseline when inspecting flagged accounts.

Train/eval split: the CLI takes a separate --train-featured CSV so the model
is always evaluated out-of-sample. Generate the two CSVs with different seeds:

    python -m fraud_eval.generate_synthetic --seed 1 --out transactions_train.csv
    ... (profile, features) ... -> featured_train.csv

    python -m fraud_eval.generate_synthetic --seed 2 --out transactions_eval.csv
    ... (profile, features) ... -> featured_eval.csv

    python -m fraud_eval.score_ml \\
        --train-featured featured_train.csv --featured featured_eval.csv \\
        --row-out scored_rows_ml.csv --card-out scored_cards_ml.csv

Then pass scored_rows_ml.csv + scored_cards_ml.csv to evaluate.py as normal.

Or as a library:
    from fraud_eval.score_ml import fit_from_rows
    scorer = fit_from_rows(train_featured_rows)
    scored = [scorer.score_row(r) for r in eval_featured_rows]
"""

import csv

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .score import CARD_OUT_FIELDS, aggregate_cards

# Feature columns drawn from featured rows (brief §6 / features.py FIELDS).
# secs_since_prev uses -1 as a sentinel for "no prior transaction" (brief F3);
# _featurise replaces it with a large value so the model sees a long gap
# rather than a negative number with no real meaning.
ML_FEATURES = [
    "amount_vs_trailing_median",
    "amount_vs_static_median",
    "velocity_1h",
    "velocity_24h",
    "is_new_device",
    "is_country_change",
    "is_new_merchant",
    "trailing_low_confidence",
    "secs_since_prev",          # sentinel -1 -> 86400 (see _featurise)
]

_SECS_NO_PRIOR = 86_400         # substitute for the NO_PRIOR sentinel


def _featurise(row):
    """Extract the ML feature vector from a featured row dict.
    Returns a list of floats in ML_FEATURES order."""
    secs = int(row["secs_since_prev"])
    return [
        float(row["amount_vs_trailing_median"]),
        float(row["amount_vs_static_median"]),
        float(row["velocity_1h"]),
        float(row["velocity_24h"]),
        float(row["is_new_device"]),
        float(row["is_country_change"]),
        float(row["is_new_merchant"]),
        float(row["trailing_low_confidence"]),
        float(_SECS_NO_PRIOR if secs < 0 else secs),
    ]


class MLScorer:
    """Logistic-regression scorer. Satisfies the Scorer protocol structurally."""

    def __init__(self, model, scaler):
        self._model = model
        self._scaler = scaler
        self._coef = model.coef_[0]     # coefficients for the fraud class

    def score_row(self, row: dict) -> dict:
        fv = _featurise(row)
        fv_scaled = self._scaler.transform([fv])[0]
        prob = float(self._model.predict_proba([fv_scaled])[0][1])

        # reason: highest-magnitude contribution (scaled value * coefficient)
        contribs = [abs(v * c) for v, c in zip(fv_scaled, self._coef)]
        top = max(range(len(contribs)), key=lambda i: contribs[i])
        reason = f"ml:{ML_FEATURES[top]}={fv[top]:.3g}"

        out = dict(row)
        out["score"] = round(prob, 3)
        out["reason"] = reason
        return out


def fit_from_rows(featured_rows):
    """Train an MLScorer on a list of featured row dicts. Returns MLScorer.

    LogisticRegression with class_weight='balanced' handles the severe class
    imbalance without resampling. StandardScaler puts amount ratios, velocities,
    and binary flags on comparable scales before fitting."""
    X = [_featurise(r) for r in featured_rows]
    y = [int(r["is_fraud"]) for r in featured_rows]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=42)
    model.fit(X_scaled, y)
    return MLScorer(model, scaler)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-featured", required=True,
                    help="featured CSV to train on (different seed from eval)")
    ap.add_argument("--featured", required=True,
                    help="featured CSV to score")
    ap.add_argument("--agg", choices=["max", "decaying_sum"],
                    default="decaying_sum")
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--row-out", default="scored_rows_ml.csv")
    ap.add_argument("--card-out", default="scored_cards_ml.csv")
    args = ap.parse_args()

    with open(args.train_featured, newline="") as f:
        train_rows = list(csv.DictReader(f))
    with open(args.featured, newline="") as f:
        eval_rows = list(csv.DictReader(f))

    scorer = fit_from_rows(train_rows)
    scored = [scorer.score_row(r) for r in eval_rows]
    cards = aggregate_cards(scored, method=args.agg, decay=args.decay)

    row_fields = list(scored[0].keys())
    with open(args.row_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_fields)
        w.writeheader()
        w.writerows(scored)

    with open(args.card_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CARD_OUT_FIELDS)
        w.writeheader()
        w.writerows(cards)

    n_fraud_train = sum(int(r["is_fraud"]) for r in train_rows)
    print(f"trained on {len(train_rows)} rows "
          f"({n_fraud_train} fraud, {100*n_fraud_train/len(train_rows):.2f}%)")
    print(f"scored {len(scored)} eval rows, {len(cards)} cards [agg={args.agg}]")

    fraud_cards = [c for c in cards if c["any_fraud"] == 1]
    clean_cards = [c for c in cards if c["any_fraud"] == 0]
    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    print(f"  mean card_score | fraud: "
          f"{avg([c['card_score'] for c in fraud_cards]):.3f}")
    print(f"  mean card_score | clean: "
          f"{avg([c['card_score'] for c in clean_cards]):.3f}")

    print("\nfeature coefficients (fraud class, by magnitude):")
    pairs = sorted(zip(ML_FEATURES, scorer._coef),
                   key=lambda x: abs(x[1]), reverse=True)
    for name, coef in pairs:
        print(f"  {name:30s} {coef:+.4f}")


if __name__ == "__main__":
    main()
