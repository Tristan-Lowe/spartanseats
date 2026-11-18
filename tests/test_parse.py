import gzip
from pathlib import Path

import pytest

from spartanseats import db
from spartanseats.parse import ScheduleFormatError, decode_html, parse_schedule

SNIPPET = (Path(__file__).parent / "fixtures" / "schedule_snippet.html").read_text(encoding="utf-8")


def test_parses_real_rows():
    rows = parse_schedule(SNIPPET)
    assert len(rows) == 3
    first = rows[0]
    assert first.course == "AAS 1"
    assert first.section == "01"
    assert first.class_number == 27070
    assert first.open_seats == 39
    assert first.instructor == "Joanne Rondilla"
    assert first.notes == ""


def test_placeholder_page_has_no_rows():
    assert parse_schedule("<html><body><p>Schedule coming soon</p></body></html>") == []


def test_changed_columns_raise():
    broken = SNIPPET.replace(">Open Seats</th>", ">Seats Left</th>")
    assert broken != SNIPPET
    with pytest.raises(ScheduleFormatError):
        parse_schedule(broken)


def test_decode_gzip():
    assert decode_html(gzip.compress("héllo".encode())) == "héllo"
    assert decode_html(b"plain") == "plain"


def test_save_is_idempotent(tmp_path):
    con = db.connect(tmp_path / "t.db")
    rows = parse_schedule(SNIPPET)
    kw = {"term": "spring-2025", "captured_at": "2024-11-08T07:15:34Z", "source": "wayback", "url": "u", "digest": "d"}
    assert db.save_snapshot(con, rows=rows, **kw) is not None
    assert db.save_snapshot(con, rows=rows, **kw) is None
    assert con.execute("SELECT COUNT(*) FROM observation").fetchone()[0] == 3


def test_2020_layout_without_satisfies_and_with_fees():
    old = SNIPPET.replace(">Satisfies</th>", ">Fees</th>")
    rows = parse_schedule(old)
    assert len(rows) == 3
    assert rows[0].open_seats == 39
    # The "Satisfies" cell is now read as Fees, which we don't keep.
    assert rows[0].satisfies == ""


def test_missing_required_column_raises():
    with pytest.raises(ScheduleFormatError):
        parse_schedule(SNIPPET.replace(">Open Seats</th>", ">Seats</th>"))
