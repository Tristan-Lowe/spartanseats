from datetime import date
from pathlib import Path

from spartanseats import db, fetch
from spartanseats.terms import collectable_terms

SNIPPET = (Path(__file__).parent / "fixtures" / "schedule_snippet.html").read_bytes()


class _Resp:
    content = SNIPPET


def test_csv_round_trip(tmp_path, monkeypatch):
    real_connect = db.connect
    monkeypatch.setattr(fetch, "_get", lambda url, **kw: _Resp())
    monkeypatch.setattr(fetch.db, "connect", lambda *a, **k: real_connect(tmp_path / "t.db"))

    path = fetch.collect_csv("spring-2027", out_dir=tmp_path / "snapshots")
    assert path.parent.parent.name == "spring-2027"
    assert (tmp_path / "snapshots" / "spring-2027" / "listing.csv.gz").exists()
    assert fetch.import_csv(tmp_path / "snapshots") == 1
    assert fetch.import_csv(tmp_path / "snapshots") == 0  # idempotent

    con = real_connect(tmp_path / "t.db")
    (seats,) = con.execute("SELECT open_seats FROM observation WHERE class_number = 27070").fetchone()
    assert seats == 39
    (source, n) = con.execute("SELECT source, n_sections FROM snapshot").fetchone()
    assert (source, n) == ("actions", 3)


def test_collectable_terms():
    assert collectable_terms(date(2026, 11, 18)) == ["spring-2027"]
    assert collectable_terms(date(2027, 3, 1)) == []
    assert collectable_terms(date(2027, 4, 25)) == ["fall-2027"]
