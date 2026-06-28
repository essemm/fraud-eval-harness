"""
ML row-level probability reliability tests (brief §8.3; acceptance
E6–E9 and V5–V8).

The diagnostic itself (viz/reliability.py) is stdlib-only, so its tests run
everywhere. The figure tests need matplotlib/numpy, which are viz-only
dependencies not installed in the core CI job (requirements.txt); those tests
import the plotting module lazily behind pytest.importorskip so they skip
cleanly rather than erroring when the plotting stack is absent.
"""

import json
import os

import pytest

from viz.reliability import compute_reliability, render_report


# A tiny, hand-checkable dataset. Two populated bins:
#   bin [0.2,0.3): two rows, scores 0.2, labels {0,1} -> mean_pred 0.2, obs 0.5
#   bin [0.9,1.0): two rows, scores 0.9, labels {1,1} -> mean_pred 0.9, obs 1.0
# Brier = ((0.2-0)^2 + (0.2-1)^2 + (0.9-1)^2 + (0.9-1)^2) / 4
#       = (0.04 + 0.64 + 0.01 + 0.01) / 4 = 0.175
# ECE   = (2/4)*|0.5-0.2| + (2/4)*|1.0-0.9| = 0.5*0.3 + 0.5*0.1 = 0.20
TINY_ROWS = [
    {"score": "0.2", "is_fraud": "0"},
    {"score": "0.2", "is_fraud": "1"},
    {"score": "0.9", "is_fraud": "1"},
    {"score": "0.9", "is_fraud": "1"},
]


# --- E8: Brier and ECE on a deterministic input ---------------------------

def test_E8_brier_and_ece_known_values():
    """Brier score and ECE match a hand-computed reference exactly."""
    result = compute_reliability(TINY_ROWS, n_bins=10)
    assert result["brier_score"] == pytest.approx(0.175)
    assert result["expected_calibration_error"] == pytest.approx(0.20)


# --- E7: bin records and empty-bin handling -------------------------------

def test_E7_bins_report_required_fields():
    """Every populated bin reports count, mean predicted prob, observed rate."""
    result = compute_reliability(TINY_ROWS, n_bins=10)
    for b in result["bins"]:
        assert set(b) >= {"bin_lower", "bin_upper", "count",
                          "mean_predicted_probability", "observed_fraud_rate"}
        assert b["count"] > 0


def test_E7_empty_bins_skipped():
    """Empty bins are omitted; only the two populated bins appear."""
    result = compute_reliability(TINY_ROWS, n_bins=10)
    assert result["n_bins"] == 10
    assert result["n_populated_bins"] == 2
    assert len(result["bins"]) == 2
    lowers = {round(b["bin_lower"], 2) for b in result["bins"]}
    assert lowers == {0.2, 0.9}


def test_E7_bin_values_correct():
    """The two populated bins carry the expected means and observed rates."""
    result = compute_reliability(TINY_ROWS, n_bins=10)
    by_lower = {round(b["bin_lower"], 2): b for b in result["bins"]}
    assert by_lower[0.2]["mean_predicted_probability"] == pytest.approx(0.2)
    assert by_lower[0.2]["observed_fraud_rate"] == pytest.approx(0.5)
    assert by_lower[0.9]["mean_predicted_probability"] == pytest.approx(0.9)
    assert by_lower[0.9]["observed_fraud_rate"] == pytest.approx(1.0)


def test_final_bin_includes_one():
    """A score of exactly 1.0 lands in the closed top bin, not an overflow."""
    result = compute_reliability(
        [{"score": "1.0", "is_fraud": "1"}], n_bins=10)
    assert result["n_populated_bins"] == 1
    assert round(result["bins"][0]["bin_lower"], 2) == 0.9
    assert round(result["bins"][0]["bin_upper"], 2) == 1.0


# --- E6: row-level scores only, never card-level --------------------------

def test_E6_uses_row_score_not_card_fields():
    """compute_reliability reads only `score` and `is_fraud`. A misleading
    card-level field on the same rows must not change the result."""
    base = compute_reliability(TINY_ROWS, n_bins=10)

    # Same rows, but each carries a card_score that contradicts the row score.
    poisoned = [dict(r, card_score="0.999", card_id="c1") for r in TINY_ROWS]
    poisoned_result = compute_reliability(poisoned, n_bins=10)

    assert poisoned_result["brier_score"] == base["brier_score"]
    assert poisoned_result["bins"] == base["bins"]


def test_E6_works_without_card_fields_present():
    """Rows that contain no card-level fields at all are sufficient input —
    the diagnostic never depends on aggregated columns."""
    minimal = [{"score": r["score"], "is_fraud": r["is_fraud"]}
               for r in TINY_ROWS]
    result = compute_reliability(minimal, n_bins=10)
    assert result["n_rows"] == 4


# --- E9 / V5: the word "accuracy" appears nowhere -------------------------

def test_E9_no_accuracy_in_metadata():
    """Neither the JSON result nor the text report uses the word 'accuracy'."""
    result = compute_reliability(TINY_ROWS, n_bins=10, source="held_out.csv")
    assert "accuracy" not in json.dumps(result).lower()
    assert "accuracy" not in render_report(result).lower()


# --- Figure tests (need the viz plotting stack) ---------------------------

def _reliability_fixture():
    return compute_reliability(TINY_ROWS, n_bins=10, source="held_out.csv")


def test_V6_plot_writes_named_png(tmp_path):
    """V6: the plotter writes 05_ml_reliability_diagram.png."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from viz.make_plots import plot_ml_reliability

    plot_ml_reliability(_reliability_fixture(), str(tmp_path))
    assert (tmp_path / "05_ml_reliability_diagram.png").exists()


def test_V7_diagonal_reference_and_bin_labels():
    """V7: the diagram has a diagonal perfect-calibration line and labels its
    points as probability bins."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    import matplotlib
    matplotlib.use("Agg")
    from viz.make_plots import build_reliability_figure

    fig = build_reliability_figure(_reliability_fixture())
    ax = fig.axes[0]

    # A line running corner to corner is the perfect-calibration reference.
    has_diagonal = any(
        list(ln.get_xdata()) == [0, 1] and list(ln.get_ydata()) == [0, 1]
        for ln in ax.lines)
    assert has_diagonal, "no (0,0)->(1,1) diagonal reference line found"

    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts()).lower()
    assert "perfect calibration" in labels
    assert "probability bins" in labels

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_V5_no_accuracy_anywhere_in_figure():
    """V5: 'accuracy' appears in no title, axis label, legend, or caption."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    import matplotlib
    matplotlib.use("Agg")
    from viz.make_plots import build_reliability_figure

    fig = build_reliability_figure(_reliability_fixture())
    ax = fig.axes[0]

    texts = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    texts += [t.get_text() for t in ax.get_legend().get_texts()]
    texts += [t.get_text() for t in fig.texts]  # the caption lives on the fig
    blob = " ".join(texts).lower()
    assert "accuracy" not in blob

    import matplotlib.pyplot as plt
    plt.close(fig)


# --- V8: plotting stays in viz/, never in fraud_eval/ ----------------------

def test_V8_fraud_eval_does_not_import_plotting():
    """V8: no core fraud_eval module imports matplotlib, plotly, or viz."""
    import fraud_eval
    pkg_dir = os.path.dirname(fraud_eval.__file__)
    forbidden = ("matplotlib", "plotly", "import viz", "from viz")
    for name in os.listdir(pkg_dir):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(pkg_dir, name)) as f:
            src = f.read()
        for token in forbidden:
            assert token not in src, f"{name} references {token!r}"
