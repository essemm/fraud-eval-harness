"""
The scorer interface.

This is the seam the whole project is organised around (brief S2, the "swap
contract"): any scorer -- the transparent `RuleScorer` baseline today, an ML
model later -- implements `score_row` and nothing downstream changes.
`features.py` and `evaluate.py` never learn which scorer produced a row.

The protocol lives in its own module, rather than next to the rule baseline,
so that every implementation imports the *contract* from one neutral place as
an equal. When the ML scorer is added (`score_ml.py`), it imports `Scorer`
from here exactly as `score.py` does -- neither implementation owns the
interface.
"""

from typing import Protocol


class Scorer(Protocol):
    """Maps a featured row to a scored row: the input row plus a `score` in
    [0, 1] and a non-empty `reason` string (brief S1). Implementations must
    not change the row's other keys, so downstream modules are scorer-agnostic.
    """

    def score_row(self, row: dict) -> dict:
        ...
