"""Turn snapshots into one fill record per section.

Snapshots are sparse (the Wayback Machine caught a term's page 4-35 times), so we
rarely see the moment a section fills. We see the last capture where it still had
seats and the first capture where it had none. The fill time lies between them:
an interval-censored observation. Sections that never hit zero are right-censored
at the last capture. Sections already full at the first capture are left-censored.

This is the reason the model can pool many past terms even though no term has
dense data: survival models handle censored intervals directly (e.g. a Turnbull
estimator, or an interval-censored Weibull / accelerated failure time model).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime

from .terms import REGISTRATION_OPENS


@dataclass(frozen=True)
class FillRecord:
    term: str
    class_number: int
    course: str
    section: str
    lower: datetime | None  # last capture with seats (None = full at first capture)
    upper: datetime | None  # first capture with no seats (None = never seen full)
    n_obs: int

    @property
    def kind(self) -> str:
        if self.upper is None:
            return "right"
        if self.lower is None:
            return "left"
        return "interval"


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def fill_records(con: sqlite3.Connection, term: str) -> list[FillRecord]:
    """One record per section in the term, using every non-empty snapshot."""
    cur = con.execute(
        """
        SELECT o.class_number, o.course, o.section, s.captured_at, o.open_seats
        FROM observation o JOIN snapshot s ON s.id = o.snapshot_id
        WHERE s.term = ? AND s.n_sections > 0 AND o.open_seats IS NOT NULL
        ORDER BY o.class_number, s.captured_at
        """,
        (term,),
    )
    by_class: dict[int, list[tuple[str, str, datetime, int]]] = {}
    for class_number, course, section, captured_at, seats in cur:
        by_class.setdefault(class_number, []).append((course, section, _ts(captured_at), seats))

    out = []
    for class_number, obs in by_class.items():
        lower: datetime | None = None
        upper: datetime | None = None
        for _, _, t, seats in obs:
            if seats <= 0:
                upper = t
                break  # first fill only; seats that reopen after drops are a later problem
            lower = t
        course, section = obs[-1][0], obs[-1][1]
        out.append(FillRecord(term, class_number, course, section, lower, upper, len(obs)))
    return out


def days_since_open(term: str, t: datetime | None) -> float | None:
    """Days from the term's registration start to t, or None if either is unknown."""
    opens: date | None = REGISTRATION_OPENS.get(term)
    if t is None or opens is None:
        return None
    return (t - datetime(opens.year, opens.month, opens.day)).total_seconds() / 86400
