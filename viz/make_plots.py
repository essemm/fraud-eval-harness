"""
Generate five diagnostic figures from the fraud-eval pipeline outputs.

Two backends share all data-loading and preparation; only the final draw call
differs:

  Static PNGs (default, matplotlib):
      python -m viz.make_plots --out-dir viz/figures/
      Output: 01_cost_vs_threshold.png … 05_ml_reliability_diagram.png

  Interactive HTML (Plotly):
      python -m viz.make_plots --interactive --out-dir viz/figures/
      Output: fraud_eval_interactive.html  (standalone; works offline)

Plots 1–2 (curves) come from one representative seed's sweep CSV files.
Plots 3–4 (bars) come from aggregate.json (mean ± sample-std over 6 seeds).
Plot 5 (ML row-level probability reliability) comes from the representative
seed's reliability_ml.json (or is computed from its held-out scored_rows_ml.csv
via viz/reliability.py). It is a row-level probability diagnostic, drawn before
card-level aggregation, and never describes the card-level score as a
probability.

This module is a pure consumer: it reads CSV and JSON artifacts already on
disk and never re-runs the pipeline, re-scores rows, or imports any scorer.
matplotlib and plotly are confined to viz/; fraud_eval/ is not imported here.
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# Colourblind-safe palette (Wong 2011). Identical mapping across all 4 figures.
COLOUR = {
    "rules": "#0072B2",   # blue
    "ml":    "#D55E00",   # vermillion
    "naive": "#999999",   # grey  (the naive single-row baseline; not a scorer)
}
SCORER_LABEL = {
    "rules": "Rules",
    "ml":    "ML (logistic regression)",
}

REP_SEED = 1   # representative seed for curve plots (stated in figure text)

SCENARIOS = ["card_testing", "account_takeover", "impossible_travel", "stolen_spree"]
SCENARIO_LABELS = ["Card testing", "Account\ntakeover",
                   "Impossible\ntravel", "Stolen spree"]
# Plotly tick labels don't need forced newlines — the axis is wider.
SCENARIO_LABELS_HTML = ["Card testing", "Account takeover",
                        "Impossible travel", "Stolen spree"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sweep(path):
    """Load sweep CSV; return list of dicts with numeric values cast to float."""
    numeric = {"threshold", "tp", "fp", "tn", "fn",
               "precision", "recall",
               "cost_fixed_ratio", "cost_amount_weighted", "missed_fraud_loss"}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in numeric:
            if k in r:
                r[k] = float(r[k])
    return rows


def _op_row(sweep_rows):
    """
    Fixed-ratio cost-minimising row, with degeneracy fallback to threshold=0.50.
    Mirrors the logic in evaluate.py so the marked operating points are consistent
    with the metrics files.
    """
    best = min(sweep_rows, key=lambda r: (r["cost_fixed_ratio"], r["threshold"]))
    max_flagged = max(r["tp"] + r["fp"] for r in sweep_rows)
    degenerate = (best["tp"] + best["fp"] >= 0.95 * max_flagged
                  and best["recall"] >= 0.99)
    if degenerate:
        ref = next((r for r in sweep_rows
                    if abs(r["threshold"] - 0.50) < 1e-9), None)
        return ref if ref is not None else sweep_rows[0]
    return best


# ── Plot 1: Cost vs threshold (stacked panels) ────────────────────────────────

def plot_cost_vs_threshold(rules_sweep, ml_sweep, out_dir):
    """
    Two stacked panels sharing the x-axis (threshold). Top panel: fixed-ratio
    cost (unitless). Bottom panel: amount-weighted cost (USD). Stacked panels
    are required because the two y-scales differ by orders of magnitude; a twin
    y-axis buries one curve visually.
    """
    fig, (ax_fr, ax_aw) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    fig.suptitle(
        f"Cost vs Decision Threshold  (representative seed {REP_SEED})",
        fontsize=12, y=0.98)

    for label, sweep in (("rules", rules_sweep), ("ml", ml_sweep)):
        thresholds = [r["threshold"] for r in sweep]
        cost_fr    = [r["cost_fixed_ratio"] for r in sweep]
        cost_aw    = [r["cost_amount_weighted"] for r in sweep]
        colour     = COLOUR[label]

        ax_fr.plot(thresholds, cost_fr, color=colour, lw=1.8,
                   label=SCORER_LABEL[label])
        ax_aw.plot(thresholds, cost_aw, color=colour, lw=1.8,
                   label=SCORER_LABEL[label])

        # mark cost minimum for each model on each panel
        best_fr = min(sweep, key=lambda r: (r["cost_fixed_ratio"], r["threshold"]))
        best_aw = min(sweep, key=lambda r: (r["cost_amount_weighted"], r["threshold"]))
        ax_fr.axvline(best_fr["threshold"], color=colour, lw=1.0, ls="--", alpha=0.6)
        ax_aw.axvline(best_aw["threshold"], color=colour, lw=1.0, ls="--", alpha=0.6)
        ax_fr.scatter([best_fr["threshold"]], [best_fr["cost_fixed_ratio"]],
                      color=colour, s=50, zorder=5)
        ax_aw.scatter([best_aw["threshold"]], [best_aw["cost_amount_weighted"]],
                      color=colour, s=50, zorder=5)

    ax_fr.set_ylabel("Cost (unitless)")
    ax_fr.set_title("Fixed-ratio cost  (FN costs 20× a FP)", fontsize=9)
    ax_fr.legend(fontsize=9)
    ax_fr.grid(True, lw=0.4, alpha=0.5)

    ax_aw.set_ylabel("Cost (USD)")
    ax_aw.set_xlabel("Decision threshold")
    ax_aw.set_title("Amount-weighted cost  (missed fraud = full USD lost)", fontsize=9)
    ax_aw.legend(fontsize=9)
    ax_aw.grid(True, lw=0.4, alpha=0.5)
    ax_aw.set_xlim(0, 1)

    fig.tight_layout()
    path = os.path.join(out_dir, "01_cost_vs_threshold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ── Plot 2: Precision–recall curve ────────────────────────────────────────────

def plot_precision_recall(rules_sweep, ml_sweep, out_dir):
    """
    Both scorers overlaid on the same axes. Each scorer's operating point (its
    own fixed-ratio cost minimum) is marked. Comparing scorers at a shared
    threshold is misleading; the operating-point markers are the fair comparison.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_title(
        f"Precision–Recall Curve  (representative seed {REP_SEED})\n"
        "Operating point = each scorer's own cost-minimising threshold",
        fontsize=11)

    for label, sweep in (("rules", rules_sweep), ("ml", ml_sweep)):
        # Sort by threshold descending so the curve goes from low recall
        # (strict threshold) to high recall (permissive), left-to-right.
        ordered = sorted(sweep, key=lambda r: -r["threshold"])
        recall    = [r["recall"]    for r in ordered]
        precision = [r["precision"] for r in ordered]
        op        = _op_row(sweep)

        ax.plot(recall, precision, color=COLOUR[label], lw=1.8,
                label=SCORER_LABEL[label])
        ax.scatter(op["recall"], op["precision"],
                   color=COLOUR[label], s=80, zorder=5,
                   label=f"{SCORER_LABEL[label]} op-point "
                         f"(thr={op['threshold']:.2f}, "
                         f"p={op['precision']:.2f}, r={op['recall']:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "02_precision_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ── Plot 3: Per-scenario recall (grouped bars with error bars) ────────────────

def plot_per_scenario_recall(agg, out_dir):
    """
    Four scenarios × two scorers. Bar height = mean across seeds; error bar =
    sample std. Scorers are compared at their own operating points, not a shared
    threshold. Null-std (single contributing seed) renders no error bar.
    """
    n_seeds = agg["rules"]["n_seeds"]
    x = np.arange(len(SCENARIOS))
    width = 0.35
    offsets = {"rules": -width / 2, "ml": width / 2}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title(
        "Per-scenario recall at each scorer's own operating point\n"
        f"Mean ± 1 sd over {n_seeds} seeds  —  which scorer owns which attack?",
        fontsize=11)

    null_std_any = False
    for label in ("rules", "ml"):
        data    = agg[label]["per_scenario_recall"]
        means   = [data[sc]["mean"] for sc in SCENARIOS]
        stds    = [data[sc]["std"]  for sc in SCENARIOS]
        pos     = x + offsets[label]

        ax.bar(pos, means, width=width, color=COLOUR[label], alpha=0.85,
               label=SCORER_LABEL[label])

        # error bars only where std is non-null (brief V2)
        for xj, m, s in zip(pos, means, stds):
            if m is None:
                continue
            if s is not None:
                ax.errorbar(xj, m, yerr=s, fmt="none", capsize=4,
                            color="black", lw=1.0, zorder=5)
            else:
                null_std_any = True

    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=10)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.35)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    if null_std_any:
        ax.text(0.01, 0.01, "† one or more scenarios have a single contributing seed "
                "(no std available; error bar omitted)",
                transform=ax.transAxes, fontsize=7.5, color="gray")

    fig.tight_layout()
    path = os.path.join(out_dir, "03_per_scenario_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ── Plot 4: Hard-negative false-positive rate ─────────────────────────────────

def plot_hard_negative_fp(agg, out_dir):
    """
    Three bars: Rules (sequence-aware), ML (sequence-aware), Naive (amount ratio
    only). Naive is the single-row baseline that ignores sequence context; lower
    sequence FP rate vs naive demonstrates the sequence approach earns its keep.
    The naive rate is identical for both scorers (computed from the feature, not
    the score); a single grey bar represents it.
    """
    n_seeds = agg["rules"]["n_seeds"]

    items = [
        ("rules", "sequence_fp_rate", "Rules\n(sequence-aware)", COLOUR["rules"]),
        ("ml",    "sequence_fp_rate", "ML\n(sequence-aware)",    COLOUR["ml"]),
        ("rules", "naive_fp_rate",    "Naive\n(amount ratio)",   COLOUR["naive"]),
    ]

    means, stds, labels, colours = [], [], [], []
    for scorer, key, lbl, col in items:
        entry = agg[scorer]["hard_negative"][key]
        means.append(entry["mean"])
        stds.append(entry["std"])
        labels.append(lbl)
        colours.append(col)

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        "Hard-negative false-positive rate  (lower is better)\n"
        f"Mean ± 1 sd over {n_seeds} seeds",
        fontsize=11)

    ax.bar(x, means, color=colours, alpha=0.85)

    null_std_any = False
    for xj, m, s in zip(x, means, stds):
        if m is None:
            continue
        if s is not None:
            ax.errorbar(xj, m, yerr=s, fmt="none", capsize=5,
                        color="black", lw=1.0, zorder=5)
        else:
            null_std_any = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("False-positive rate on hard-negative cards")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.35)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    if null_std_any:
        ax.text(0.01, 0.01, "† single contributing seed — no std available",
                transform=ax.transAxes, fontsize=7.5, color="gray")

    fig.tight_layout()
    path = os.path.join(out_dir, "04_hard_negative_fp.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ── Plot 5: ML probability reliability diagram ────────────────────────────────

def load_reliability(reliability_path, rows_path, n_bins):
    """Return a reliability result dict.

    Prefers the precomputed reliability_ml.json; if it is absent, computes the
    diagnostic from held-out scored_rows_ml.csv (so the figure can be drawn in
    one step). Either way the computation lives in viz/reliability.py — this is
    a pure consumer of row-level ML scores, never card-level aggregated ones.
    """
    if reliability_path and os.path.exists(reliability_path):
        with open(reliability_path) as f:
            return json.load(f)

    from viz.reliability import compute_reliability
    with open(rows_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return compute_reliability(rows, n_bins=n_bins, source=rows_path)


def build_reliability_figure(reliability):
    """
    Build (and return) the ML ROW-LEVEL reliability diagram: mean predicted
    probability (x) vs observed fraud rate (y) per score bin, against the
    diagonal perfect-calibration reference. Points are probability bins, sized
    by row count. Brier score and expected calibration error are reported in
    the caption. This is a calibration diagnostic for the row-level score
    before card-level aggregation — not a threshold-selection figure, and the
    card-level decaying-sum score is never described as a probability.

    Returns the matplotlib Figure (kept separate from saving so the figure's
    labels and reference line are inspectable in tests; V7).
    """
    bins = reliability["bins"]
    xs = [b["mean_predicted_probability"] for b in bins]
    ys = [b["observed_fraud_rate"] for b in bins]
    counts = [b["count"] for b in bins]

    # Marker area scaled by row count, clamped so a huge bin doesn't dominate.
    max_count = max(counts) if counts else 1
    sizes = [30 + 270 * (c / max_count) for c in counts]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_title(
        "ML Row-Level Probability Reliability\n"
        "Predicted probability vs observed fraud rate, by score bin",
        fontsize=11)

    # Diagonal perfect-calibration reference line (neutral reference).
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.2, ls="--",
            label="Perfect calibration")

    ax.scatter(xs, ys, s=sizes, color=COLOUR["ml"], alpha=0.8, zorder=5,
               edgecolor="white", linewidth=0.8,
               label="Probability bins (size ∝ row count)")

    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraud rate (bin)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, lw=0.4, alpha=0.5)

    caption = (
        f"Held-out ML row-level scores: {reliability['source'] or 'in-memory'}.  "
        f"{reliability['n_populated_bins']}/{reliability['n_bins']} bins populated, "
        f"n={reliability['n_rows']} rows.\n"
        f"Brier score = {reliability['brier_score']:.4f}   "
        f"Expected calibration error = "
        f"{reliability['expected_calibration_error']:.4f}.  "
        f"Row-level probability diagnostic, before card-level aggregation.")
    fig.text(0.5, -0.02, caption, ha="center", va="top",
             fontsize=7.5, color="#444")

    fig.tight_layout()
    return fig


def plot_ml_reliability(reliability, out_dir):
    """Draw the reliability diagram and write 05_ml_reliability_diagram.png."""
    fig = build_reliability_figure(reliability)
    path = os.path.join(out_dir, "05_ml_reliability_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ── Plotly (interactive) backend ─────────────────────────────────────────────
#
# Each function returns a plotly.graph_objects.Figure.
# write_interactive_html() assembles all four into one standalone HTML file.
# Plotly is imported lazily so the static PNG path works without it installed.

def _plotly_cost_vs_threshold(rules_sweep, ml_sweep):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Fixed-ratio cost  (FN costs 20× a FP)  — unitless",
            "Amount-weighted cost  (missed fraud = full USD lost)  — USD",
        ),
        vertical_spacing=0.12,
    )

    for label, sweep in (("rules", rules_sweep), ("ml", ml_sweep)):
        thresholds = [r["threshold"] for r in sweep]
        cost_fr    = [r["cost_fixed_ratio"]    for r in sweep]
        cost_aw    = [r["cost_amount_weighted"] for r in sweep]
        colour     = COLOUR[label]
        name       = SCORER_LABEL[label]

        # Custom hover text: show threshold + all key metrics on every point.
        hover_fr = [
            f"threshold={r['threshold']:.2f}<br>"
            f"cost={r['cost_fixed_ratio']:.1f}<br>"
            f"precision={r['precision']:.3f}<br>"
            f"recall={r['recall']:.3f}<br>"
            f"tp={int(r['tp'])}  fp={int(r['fp'])}  fn={int(r['fn'])}"
            for r in sweep
        ]
        hover_aw = [
            f"threshold={r['threshold']:.2f}<br>"
            f"cost=${r['cost_amount_weighted']:,.0f}<br>"
            f"missed loss=${r['missed_fraud_loss']:,.0f}<br>"
            f"precision={r['precision']:.3f}<br>"
            f"recall={r['recall']:.3f}"
            for r in sweep
        ]

        fig.add_trace(go.Scatter(
            x=thresholds, y=cost_fr, name=name,
            line=dict(color=colour, width=2),
            text=hover_fr, hoverinfo="text",
            legendgroup=label,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=thresholds, y=cost_aw, name=name,
            line=dict(color=colour, width=2),
            text=hover_aw, hoverinfo="text",
            legendgroup=label, showlegend=False,
        ), row=2, col=1)

        # Mark cost minima.
        best_fr = min(sweep, key=lambda r: (r["cost_fixed_ratio"], r["threshold"]))
        best_aw = min(sweep, key=lambda r: (r["cost_amount_weighted"], r["threshold"]))

        for row_idx, best, cost_key in (
            (1, best_fr, "cost_fixed_ratio"),
            (2, best_aw, "cost_amount_weighted"),
        ):
            fig.add_trace(go.Scatter(
                x=[best["threshold"]], y=[best[cost_key]],
                mode="markers",
                marker=dict(color=colour, size=10, symbol="circle",
                            line=dict(color="white", width=1.5)),
                name=f"{name} minimum",
                hovertext=(f"minimum<br>threshold={best['threshold']:.2f}<br>"
                           f"precision={best['precision']:.3f}<br>"
                           f"recall={best['recall']:.3f}"),
                hoverinfo="text",
                legendgroup=label, showlegend=False,
            ), row=row_idx, col=1)

    fig.update_xaxes(title_text="Decision threshold", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Cost (unitless)", row=1, col=1)
    fig.update_yaxes(title_text="Cost (USD)", row=2, col=1)
    fig.update_layout(
        title=f"Cost vs Decision Threshold  (representative seed {REP_SEED})",
        height=620, hovermode="x unified",
    )
    return fig


def _plotly_precision_recall(rules_sweep, ml_sweep):
    import plotly.graph_objects as go

    fig = go.Figure()

    for label, sweep in (("rules", rules_sweep), ("ml", ml_sweep)):
        ordered = sorted(sweep, key=lambda r: -r["threshold"])
        recall    = [r["recall"]    for r in ordered]
        precision = [r["precision"] for r in ordered]
        op        = _op_row(sweep)
        colour    = COLOUR[label]
        name      = SCORER_LABEL[label]

        hover = [
            f"threshold={r['threshold']:.2f}<br>"
            f"recall={r['recall']:.3f}<br>"
            f"precision={r['precision']:.3f}<br>"
            f"tp={int(r['tp'])}  fp={int(r['fp'])}  fn={int(r['fn'])}"
            for r in ordered
        ]

        fig.add_trace(go.Scatter(
            x=recall, y=precision, name=name,
            line=dict(color=colour, width=2),
            text=hover, hoverinfo="text",
        ))
        fig.add_trace(go.Scatter(
            x=[op["recall"]], y=[op["precision"]],
            mode="markers",
            marker=dict(color=colour, size=12, symbol="circle",
                        line=dict(color="white", width=1.5)),
            name=f"{name} op-point (thr={op['threshold']:.2f})",
            hovertext=(f"operating point<br>"
                       f"threshold={op['threshold']:.2f}<br>"
                       f"precision={op['precision']:.3f}<br>"
                       f"recall={op['recall']:.3f}"),
            hoverinfo="text",
        ))

    fig.update_layout(
        title=(f"Precision–Recall Curve  (representative seed {REP_SEED})<br>"
               "<sup>Operating point = each scorer's own cost-minimising threshold</sup>"),
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]),
        hovermode="closest",
    )
    return fig


def _plotly_per_scenario_recall(agg):
    import plotly.graph_objects as go

    n_seeds = agg["rules"]["n_seeds"]
    fig = go.Figure()
    null_std_note = False

    for label in ("rules", "ml"):
        data   = agg[label]["per_scenario_recall"]
        means  = [data[sc]["mean"] for sc in SCENARIOS]
        stds   = [data[sc]["std"]  for sc in SCENARIOS]
        ns     = [data[sc]["n"]    for sc in SCENARIOS]

        # Plotly error_y accepts None to suppress individual bars.
        err_vals = [s if s is not None else 0.0 for s in stds]
        err_vis  = [s is not None for s in stds]
        if not all(err_vis):
            null_std_note = True

        hover = [
            f"{SCENARIO_LABELS_HTML[i]}<br>"
            f"{SCORER_LABEL[label]}<br>"
            f"recall={m:.3f}"
            + (f" ± {stds[i]:.3f}" if stds[i] is not None else " (no std)")
            + f"<br>n={ns[i]} seeds"
            for i, m in enumerate(means)
        ]

        fig.add_trace(go.Bar(
            name=SCORER_LABEL[label],
            x=SCENARIO_LABELS_HTML,
            y=means,
            error_y=dict(type="data", array=err_vals, visible=True),
            marker_color=COLOUR[label],
            opacity=0.85,
            text=hover, hoverinfo="text",
        ))

    title = (f"Per-scenario recall at each scorer's own operating point<br>"
             f"<sup>Mean ± 1 sd over {n_seeds} seeds — which scorer owns which attack?")
    if null_std_note:
        title += "  † no error bar where n=1"
    title += "</sup>"

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(title="Recall", range=[0, 1.18]),
        hovermode="closest",
    )
    fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dash"),
                  opacity=0.35)
    return fig


def _plotly_hard_negative_fp(agg):
    import plotly.graph_objects as go

    n_seeds = agg["rules"]["n_seeds"]

    items = [
        ("rules", "sequence_fp_rate", "Rules (sequence-aware)",  COLOUR["rules"]),
        ("ml",    "sequence_fp_rate", "ML (sequence-aware)",     COLOUR["ml"]),
        ("rules", "naive_fp_rate",    "Naive (amount ratio)",    COLOUR["naive"]),
    ]

    x_labels, means, stds, colours = [], [], [], []
    for scorer, key, lbl, col in items:
        entry = agg[scorer]["hard_negative"][key]
        x_labels.append(lbl)
        means.append(entry["mean"])
        stds.append(entry["std"])
        colours.append(col)

    err_vals = [s if s is not None else 0.0 for s in stds]
    null_std_note = any(s is None for s in stds)

    hover = [
        f"{x_labels[i]}<br>"
        f"FP rate={means[i]:.3f}"
        + (f" ± {stds[i]:.3f}" if stds[i] is not None else " (no std)")
        + f"<br>n={agg['rules']['hard_negative']['sequence_fp_rate']['n']} seeds"
        for i in range(len(x_labels))
    ]

    fig = go.Figure(go.Bar(
        x=x_labels, y=means,
        error_y=dict(type="data", array=err_vals, visible=True),
        marker_color=colours,
        opacity=0.85,
        text=hover, hoverinfo="text",
    ))

    title = (f"Hard-negative false-positive rate  (lower is better)<br>"
             f"<sup>Mean ± 1 sd over {n_seeds} seeds")
    if null_std_note:
        title += "  † no error bar where n=1"
    title += "</sup>"

    fig.update_layout(
        title=title,
        yaxis=dict(title="False-positive rate on hard-negative cards",
                   range=[0, 1.15]),
        hovermode="closest",
    )
    fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dash"),
                  opacity=0.35)
    return fig


def _plotly_ml_reliability(reliability):
    import plotly.graph_objects as go

    bins = reliability["bins"]
    xs = [b["mean_predicted_probability"] for b in bins]
    ys = [b["observed_fraud_rate"] for b in bins]
    counts = [b["count"] for b in bins]
    max_count = max(counts) if counts else 1
    sizes = [8 + 30 * (c / max_count) for c in counts]

    hover = [
        f"bin [{b['bin_lower']:.2f}, {b['bin_upper']:.2f})<br>"
        f"mean predicted prob={b['mean_predicted_probability']:.3f}<br>"
        f"observed fraud rate={b['observed_fraud_rate']:.3f}<br>"
        f"rows={b['count']}"
        for b in bins
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#999999", width=1.5, dash="dash"),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(color=COLOUR["ml"], size=sizes,
                    line=dict(color="white", width=1)),
        text=hover, hoverinfo="text",
        name="Probability bins",
    ))
    fig.update_layout(
        title=("ML Row-Level Probability Reliability<br>"
               f"<sup>Brier={reliability['brier_score']:.4f}, "
               f"ECE={reliability['expected_calibration_error']:.4f}, "
               f"n={reliability['n_rows']} held-out rows — "
               "row-level diagnostic, before card-level aggregation</sup>"),
        xaxis=dict(title="Mean predicted probability (bin)", range=[0, 1]),
        yaxis=dict(title="Observed fraud rate (bin)", range=[0, 1],
                   scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return fig


def write_interactive_html(rules_sweep, ml_sweep, agg, reliability, out_dir):
    """Build all five Plotly figures and write a single standalone HTML file."""
    import plotly.io as pio

    figures = [
        (_plotly_cost_vs_threshold(rules_sweep, ml_sweep),
         "1. Cost vs Decision Threshold",
         (f"Representative seed {REP_SEED}. Stacked panels because the two cost "
          "scales differ by orders of magnitude. Hover for threshold / precision "
          "/ recall at any point; markers show cost-minimising operating points.")),

        (_plotly_precision_recall(rules_sweep, ml_sweep),
         "2. Precision–Recall Curve",
         (f"Representative seed {REP_SEED}. Hover over the curve to read "
          "threshold, precision, and recall. Markers show each scorer's own "
          "cost-minimising operating point.")),

        (_plotly_per_scenario_recall(agg),
         "3. Per-Scenario Recall",
         (f"Mean ± 1 sd over {agg['rules']['n_seeds']} seeds. Each scorer is "
          "evaluated at its own operating point, not a shared threshold.")),

        (_plotly_hard_negative_fp(agg),
         "4. Hard-Negative False-Positive Rate",
         (f"Mean ± 1 sd over {agg['rules']['n_seeds']} seeds. Lower is better. "
          "Rules beats the naive single-row baseline; ML does not at its "
          "operating point.")),

        (_plotly_ml_reliability(reliability),
         "5. ML Row-Level Probability Reliability",
         (f"Held-out ML row-level scores ({reliability['source'] or 'in-memory'}). "
          f"Brier={reliability['brier_score']:.4f}, "
          f"ECE={reliability['expected_calibration_error']:.4f}. "
          "Whether the row-level score reads as a probability, before "
          "card-level aggregation. Not a threshold-selection figure.")),
    ]

    sections = []
    for i, (fig, heading, caption) in enumerate(figures):
        # First figure bundles plotly.js inline (self-contained, works offline).
        # Remaining figures reference the global Plotly object already loaded.
        div = pio.to_html(fig, full_html=False,
                          include_plotlyjs=(i == 0),
                          config={"displayModeBar": True, "scrollZoom": True})
        sections.append(
            f'<section>\n'
            f'  <h2>{heading}</h2>\n'
            f'  {div}\n'
            f'  <p class="caption">{caption}</p>\n'
            f'</section>\n'
        )

    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <title>Fraud Detection Eval — Interactive Figures</title>\n"
        "  <style>\n"
        "    body { font-family: system-ui, sans-serif; max-width: 1100px;\n"
        "           margin: 40px auto; padding: 0 24px; color: #222; }\n"
        "    h1   { font-size: 1.4em; border-bottom: 1px solid #ddd;\n"
        "           padding-bottom: 8px; }\n"
        "    h2   { font-size: 1.1em; margin-top: 48px; color: #333; }\n"
        "    .caption { font-size: 0.88em; color: #666; max-width: 800px;\n"
        "               margin-top: 4px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Sequence-Aware Fraud Detection — Interactive Evaluation Figures</h1>\n"
        "  <p style='color:#666; font-size:0.9em'>"
        f"Hover to inspect values. Plots 1–2: representative seed {REP_SEED}. "
        f"Plots 3–4: mean ± 1 sd over {agg['rules']['n_seeds']} seeds. "
        f"Plot 5: held-out ML row-level scores (seed {REP_SEED})."
        "</p>\n"
        + "".join(sections)
        + "</body>\n</html>\n"
    )

    path = os.path.join(out_dir, "fraud_eval_interactive.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate four diagnostic figures from fraud-eval outputs.")
    ap.add_argument("--rules-sweep",
                    default=f"runs/seed_{REP_SEED}/sweep_rules.csv",
                    help="sweep CSV for the rules scorer (representative seed)")
    ap.add_argument("--ml-sweep",
                    default=f"runs/seed_{REP_SEED}/sweep_ml.csv",
                    help="sweep CSV for the ML scorer (representative seed)")
    ap.add_argument("--aggregate", default="runs/aggregate.json",
                    help="aggregate.json from aggregate_runs.py")
    ap.add_argument("--reliability",
                    default=f"runs/seed_{REP_SEED}/reliability_ml.json",
                    help="reliability_ml.json from viz.reliability; if absent, "
                         "computed from --reliability-rows")
    ap.add_argument("--reliability-rows",
                    default=f"runs/seed_{REP_SEED}/scored_rows_ml.csv",
                    help="held-out ML scored rows, used if --reliability JSON "
                         "is absent")
    ap.add_argument("--reliability-bins", type=int, default=10,
                    help="bins to use when computing reliability from rows")
    ap.add_argument("--out-dir",  default="viz/figures",
                    help="directory to write output into")
    ap.add_argument("--interactive", action="store_true",
                    help="write a standalone interactive HTML file (Plotly) "
                         "instead of static PNGs (matplotlib)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rules_sweep = load_sweep(args.rules_sweep)
    ml_sweep    = load_sweep(args.ml_sweep)
    with open(args.aggregate) as f:
        agg = json.load(f)
    reliability = load_reliability(args.reliability, args.reliability_rows,
                                   args.reliability_bins)

    print(f"sweep rows: rules={len(rules_sweep)}, ml={len(ml_sweep)}")
    print(f"aggregate: rules n_seeds={agg['rules']['n_seeds']}, "
          f"ml n_seeds={agg['ml']['n_seeds']}")
    print(f"reliability: n_rows={reliability['n_rows']}, "
          f"populated bins={reliability['n_populated_bins']}, "
          f"Brier={reliability['brier_score']:.4f}, "
          f"ECE={reliability['expected_calibration_error']:.4f}")
    print()

    if args.interactive:
        try:
            write_interactive_html(rules_sweep, ml_sweep, agg, reliability,
                                   args.out_dir)
        except ImportError:
            ap.error("plotly is required for --interactive. "
                     "Run: pip install plotly")
    else:
        plot_cost_vs_threshold(rules_sweep, ml_sweep, args.out_dir)
        plot_precision_recall(rules_sweep, ml_sweep, args.out_dir)
        plot_per_scenario_recall(agg, args.out_dir)
        plot_hard_negative_fp(agg, args.out_dir)
        plot_ml_reliability(reliability, args.out_dir)
        print(f"\nall five figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
