"""SQLite storage. One row per page snapshot, one row per section per snapshot.

SQLite for now (one file, no server). The proposal names PostgreSQL; the schema
is plain SQL so moving later is a dump-and-load.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .parse import SectionRow

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "spartanseats.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot (
    id           INTEGER PRIMARY KEY,
    term         TEXT NOT NULL,          -- e.g. 'spring-2027'
    captured_at  TEXT NOT NULL,          -- UTC ISO-8601
    source       TEXT NOT NULL,          -- 'live' or 'wayback'
    url          TEXT NOT NULL,
    digest       TEXT NOT NULL,          -- sha1 of the decoded HTML
    n_sections   INTEGER NOT NULL,
    UNIQUE (term, captured_at, source)
);

CREATE TABLE IF NOT EXISTS observation (
    snapshot_id  INTEGER NOT NULL REFERENCES snapshot(id),
    class_number INTEGER NOT NULL,
    course       TEXT NOT NULL,
    section      TEXT NOT NULL,
    mode         TEXT,
    title        TEXT,
    satisfies    TEXT,
    units        TEXT,
    type         TEXT,
    days         TEXT,
    times        TEXT,
    instructor   TEXT,
    location     TEXT,
    dates        TEXT,
    open_seats   INTEGER,
    notes        TEXT,
    PRIMARY KEY (snapshot_id, class_number)
);

CREATE INDEX IF NOT EXISTS obs_class ON observation (class_number);
CREATE INDEX IF NOT EXISTS snap_term ON snapshot (term, captured_at);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def have_snapshot(con: sqlite3.Connection, term: str, captured_at: str, source: str) -> bool:
    cur = con.execute(
        "SELECT 1 FROM snapshot WHERE term=? AND captured_at=? AND source=?",
        (term, captured_at, source),
    )
    return cur.fetchone() is not None


def save_snapshot(
    con: sqlite3.Connection,
    *,
    term: str,
    captured_at: str,
    source: str,
    url: str,
    digest: str,
    rows: list[SectionRow],
) -> int | None:
    """Insert a snapshot and its rows in one transaction. Returns the id, or None if it already existed."""
    with con:
        cur = con.execute(
            "INSERT OR IGNORE INTO snapshot (term, captured_at, source, url, digest, n_sections)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (term, captured_at, source, url, digest, len(rows)),
        )
        if cur.rowcount == 0:
            return None
        snap_id = cur.lastrowid
        # A class number can repeat on the page for multi-meeting sections; keep the first.
        seen: set[int] = set()
        params = []
        for r in rows:
            if r.class_number in seen:
                continue
            seen.add(r.class_number)
            params.append(
                (
                    snap_id, r.class_number, r.course, r.section, r.mode, r.title,
                    r.satisfies, r.units, r.type, r.days, r.times, r.instructor,
                    r.location, r.dates, r.open_seats, r.notes,
                )
            )
        con.executemany(
            "INSERT INTO observation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", params
        )
        return snap_id
