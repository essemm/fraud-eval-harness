"""
Sequence-aware fraud detection evaluation harness.

A pipeline of small, independently-testable modules connected by CSV
interfaces:

    fx -> generate_synthetic -> profile -> features -> score -> evaluate

The scorer is pluggable behind the `Scorer` protocol (scorer.py); the rule
baseline (`score.RuleScorer`) is one implementation, and the evaluation
harness (evaluate.py) judges any scorer under configurable cost models.

See PROJECT_BRIEF.md for the full specification.
"""

__version__ = "0.1.0"
