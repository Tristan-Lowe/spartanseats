"""Registration start dates (first day of the continuing-student registration period).

Time zero for every section in a term. Source: SJSU registrar calendar pages
(www.sjsu.edu/registrar/calendar/<term>.php). Terms set to None still need a date
from an archived calendar page; they can't be aligned until then.
"""

from __future__ import annotations

from datetime import date

REGISTRATION_OPENS: dict[str, date | None] = {
    "fall-2020": None,
    "spring-2021": None,
    "fall-2021": None,
    "spring-2022": None,
    "fall-2022": None,
    "spring-2023": None,
    "fall-2023": None,
    "spring-2024": None,
    "fall-2024": None,
    "spring-2025": None,
    "fall-2025": None,
    "spring-2026": None,
    "fall-2026": date(2026, 4, 21),  # registrar calendar, checked 2026-11-18
    "spring-2027": date(2026, 10, 27),  # registrar calendar, checked 2026-11-18
}
