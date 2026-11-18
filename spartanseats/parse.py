"""Parse an SJSU class schedule page (www.sjsu.edu/classes/schedules/<term>.php).

The page is one big HTML table, id="classSchedule", with these columns:
Section, Class Number, Mode of Instruction, Course Title, Satisfies, Units,
Type, Days, Times, Instructor, Location, Dates, Open Seats, Notes.

There is no capacity column, so a section counts as "full" when open seats <= 0.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Columns every layout must have. Older pages (2020) lack "Satisfies" and add "Fees",
# so optional columns are filled with "" when absent.
REQUIRED_HEADERS = [
    "Section",
    "Class Number",
    "Mode of Instruction",
    "Course Title",
    "Units",
    "Type",
    "Days",
    "Times",
    "Instructor",
    "Location",
    "Dates",
    "Open Seats",
]
OPTIONAL_HEADERS = ["Satisfies", "Notes", "Fees"]

# Current (2021+) layout, kept for reference and tests.
EXPECTED_HEADERS = [
    "Section",
    "Class Number",
    "Mode of Instruction",
    "Course Title",
    "Satisfies",
    "Units",
    "Type",
    "Days",
    "Times",
    "Instructor",
    "Location",
    "Dates",
    "Open Seats",
    "Notes",
]

SECTION_RE = re.compile(r"^(?P<course>.+?)\s*\(Section\s+(?P<section>[^)]+)\)$")


@dataclass(frozen=True)
class SectionRow:
    class_number: int
    course: str  # e.g. "CS 146"
    section: str  # e.g. "01"
    mode: str
    title: str
    satisfies: str
    units: str
    type: str
    days: str
    times: str
    instructor: str
    location: str
    dates: str
    open_seats: int | None
    notes: str


class ScheduleFormatError(ValueError):
    """The page has a schedule table, but its columns are not the ones we expect."""


def decode_html(raw: bytes) -> str:
    """Wayback 'id_' responses are sometimes still gzip-compressed; handle both."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _to_int(text: str) -> int | None:
    text = _clean(text)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def parse_schedule(html: str) -> list[SectionRow]:
    """Return every section row on the page.

    Returns [] when the page has no schedule table (SJSU posts a placeholder
    page before a term's schedule is released). Raises ScheduleFormatError if
    the table exists but the columns changed, so a silent layout change can't
    poison the dataset.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="classSchedule")
    if table is None:
        return []

    headers = [_clean(th.get_text()) for th in table.find_all("th")]
    missing = [h for h in REQUIRED_HEADERS if h not in headers]
    unknown = [h for h in headers if h not in REQUIRED_HEADERS + OPTIONAL_HEADERS]
    if missing or unknown:
        raise ScheduleFormatError(f"unexpected columns: {headers}")
    col = {h: i for i, h in enumerate(headers)}

    rows: list[SectionRow] = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cells = [_clean(td.get_text()) for td in tr.find_all("td")]
        if len(cells) != len(headers):
            continue

        def get(name: str, cells: list[str] = cells) -> str:
            return cells[col[name]] if name in col else ""

        m = SECTION_RE.match(get("Section"))
        class_number = _to_int(get("Class Number"))
        if not m or class_number is None:
            continue
        rows.append(
            SectionRow(
                class_number=class_number,
                course=m.group("course"),
                section=m.group("section"),
                mode=get("Mode of Instruction"),
                title=get("Course Title"),
                satisfies=get("Satisfies"),
                units=get("Units"),
                type=get("Type"),
                days=get("Days"),
                times=get("Times"),
                instructor=get("Instructor"),
                location=get("Location"),
                dates=get("Dates"),
                open_seats=_to_int(get("Open Seats")),
                notes=get("Notes"),
            )
        )
    return rows
