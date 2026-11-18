from datetime import datetime

from spartanseats import db
from spartanseats.intervals import days_since_open, fill_records
from spartanseats.parse import SectionRow


def _row(class_number: int, seats: int) -> SectionRow:
    return SectionRow(
        class_number, "CS 146", "01", "In Person", "Data Structures", "", "3.0", "LEC",
        "MW", "", "", "", "", seats, "",
    )


def _snap(con, when: str, rows):
    db.save_snapshot(con, term="fall-2026", captured_at=when, source="wayback", url="u", digest=when, rows=rows)


def test_interval_right_and_left_censoring(tmp_path):
    con = db.connect(tmp_path / "t.db")
    # 1 fills between captures 2 and 3; 2 never fills; 3 is already full at the first capture.
    _snap(con, "2026-04-20T00:00:00Z", [_row(1, 10), _row(2, 30), _row(3, 0)])
    _snap(con, "2026-04-25T00:00:00Z", [_row(1, 4), _row(2, 25), _row(3, 0)])
    _snap(con, "2026-05-01T00:00:00Z", [_row(1, 0), _row(2, 20), _row(3, 1)])
    recs = {r.class_number: r for r in fill_records(con, "fall-2026")}

    assert recs[1].kind == "interval"
    assert recs[1].lower == datetime(2026, 4, 25)
    assert recs[1].upper == datetime(2026, 5, 1)
    assert recs[2].kind == "right"
    assert recs[2].lower == datetime(2026, 5, 1)
    assert recs[3].kind == "left"


def test_placeholder_snapshots_are_ignored(tmp_path):
    con = db.connect(tmp_path / "t.db")
    _snap(con, "2026-04-01T00:00:00Z", [])
    _snap(con, "2026-04-25T00:00:00Z", [_row(1, 0)])
    (rec,) = fill_records(con, "fall-2026")
    assert rec.kind == "left"
    assert rec.n_obs == 1


def test_days_since_open():
    assert days_since_open("fall-2026", datetime(2026, 4, 22, 12)) == 1.5
    assert days_since_open("fall-2020", datetime(2020, 4, 22)) is None
