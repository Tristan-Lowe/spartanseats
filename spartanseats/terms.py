"""Registration start dates (first day of the continuing-student registration period).

Time zero for every section in a term.

Confirmed dates come from SJSU registrar calendars (checked 2026-11-18):
  fall-2024   Tue Apr 23 2024   registrar/docs/Fall 2024.pdf
  spring-2025 Tue Oct 29 2024   registrar/docs/Spring 2025.pdf
  fall-2026   Tue Apr 21 2026   registrar/calendar/fall-2026.php
  spring-2027 Tue Oct 27 2026   registrar/calendar/spring-2027.php

All four fit one rule: spring opens the last Tuesday of October, fall opens the
Tuesday nearest April 22. Other terms use that rule and are marked estimated
until an archived calendar confirms them.
"""

from __future__ import annotations

from datetime import date, timedelta

CONFIRMED: dict[str, date] = {
    "fall-2024": date(2024, 4, 23),
    "spring-2025": date(2024, 10, 29),
    "fall-2026": date(2026, 4, 21),
    "spring-2027": date(2026, 10, 27),
}

TUESDAY = 1  # date.weekday()


def _last_tuesday_of_october(year: int) -> date:
    d = date(year, 10, 31)
    return d - timedelta(days=(d.weekday() - TUESDAY) % 7)


def _tuesday_nearest_april_22(year: int) -> date:
    d = date(year, 4, 22)
    back = (d.weekday() - TUESDAY) % 7
    return d - timedelta(days=back) if back <= 3 else d + timedelta(days=7 - back)


def estimated_open(term: str) -> date:
    season, year_s = term.split("-")
    year = int(year_s)
    if season == "spring":
        return _last_tuesday_of_october(year - 1)
    if season == "fall":
        return _tuesday_nearest_april_22(year)
    raise ValueError(f"no registration rule for {term}")


def registration_opens(term: str) -> tuple[date, bool]:
    """(date, confirmed) for the term's registration start."""
    if term in CONFIRMED:
        return CONFIRMED[term], True
    return estimated_open(term), False


def term_start(term: str) -> date:
    """Approximate first day of instruction; captures after this are add/drop noise, not registration."""
    season, year_s = term.split("-")
    year = int(year_s)
    return date(year, 1, 20) if season == "spring" else date(year, 8, 20)


# Keep collecting through add/drop, when seats still move.
COLLECT_AFTER_START = timedelta(days=30)


def _next_terms(today: date) -> list[str]:
    y = today.year
    return [f"spring-{y}", f"fall-{y}", f"spring-{y + 1}", f"fall-{y + 1}"]


def collectable_terms(today: date) -> list[str]:
    """Terms whose registration window (opens - 14 days through start + 30 days) contains today."""
    out = []
    for term in _next_terms(today):
        opens, _ = registration_opens(term)
        if opens - timedelta(days=14) <= today <= term_start(term) + COLLECT_AFTER_START:
            out.append(term)
    return out
