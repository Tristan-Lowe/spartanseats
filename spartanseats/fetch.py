"""Collect schedule snapshots: live from sjsu.edu, and historical from the Wayback Machine.

Why the Wayback backfill exists: one semester of our own snapshots is one
registration season. The Internet Archive has captured SJSU's public schedule
pages since 2020 (roughly 4-14 distinct captures per term), which gives many
past registration seasons. Captures are sparse, so a section's fill time is only
known to fall between two captures (interval-censored), and the model has to
treat it that way.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from . import db
from .parse import ScheduleFormatError, decode_html, parse_schedule

USER_AGENT = "SpartanSeats/0.1 (student research project; public schedule pages only)"
LIVE_URL = "https://www.sjsu.edu/classes/schedules/{term}.php"
ARCHIVE_URL = "https://www.sjsu.edu/classes/schedules/archive/{term}.php"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_RAW = "https://web.archive.org/web/{ts}id_/{url}"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
POLITE_DELAY_S = 5.0  # between requests to any one host

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def _get(url: str, **kw) -> requests.Response:
    # The Internet Archive refuses connections for a while when hit too fast, so back off hard.
    for attempt in range(6):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == 5:
                raise
            time.sleep(min(POLITE_DELAY_S * 3 ** (attempt + 1), 300))
    raise AssertionError("unreachable")


def _store(term: str, captured_at: str, source: str, url: str, raw: bytes) -> tuple[int | None, int]:
    html = decode_html(raw)
    digest = hashlib.sha1(html.encode("utf-8")).hexdigest()
    # Keep the raw page first, so a page we can't parse yet is still on disk to re-parse later.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = captured_at.replace(":", "").replace("-", "")
    (RAW_DIR / f"{term}_{source}_{stamp}.html").write_text(html, encoding="utf-8")
    rows = parse_schedule(html)
    con = db.connect()
    try:
        snap_id = db.save_snapshot(
            con, term=term, captured_at=captured_at, source=source, url=url, digest=digest, rows=rows
        )
    finally:
        con.close()
    return snap_id, len(rows)


def collect_live(term: str) -> None:
    url = LIVE_URL.format(term=term)
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = _get(url).content
    snap_id, n = _store(term, captured_at, "live", url, raw)
    print(f"live {term} {captured_at}: {n} sections (snapshot {snap_id})")


SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
LISTING_FIELDS = ["class_number", "course", "section", "mode", "title", "satisfies", "units", "type",
                  "days", "times", "instructor", "location", "dates", "notes"]


def _gz_csv(header: list[str], rows: list[list]) -> bytes:
    import csv
    import gzip
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    # mtime=0 makes the bytes deterministic, so an unchanged listing is not a git change.
    return gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)


def _read_gz_csv(path: Path) -> list[dict]:
    import csv
    import gzip
    import io

    return list(csv.DictReader(io.StringIO(gzip.decompress(path.read_bytes()).decode("utf-8"))))


def collect_csv(term: str, out_dir: Path = SNAPSHOT_DIR) -> Path:
    """Snapshot to small gzipped CSVs (for GitHub Actions; SQLite and raw HTML don't belong in git).

    snapshots/<term>/seats/<UTC timestamp>.csv.gz   class_number, open_seats   (~20 KB, every run)
    snapshots/<term>/listing.csv.gz                  everything else            (rewritten; git only
                                                                                stores it when it changes)
    """
    url = LIVE_URL.format(term=term)
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = parse_schedule(decode_html(_get(url).content))
    if not rows:
        raise SystemExit(f"{term}: page has no schedule table, nothing written")
    term_dir = out_dir / term
    (term_dir / "seats").mkdir(parents=True, exist_ok=True)

    listing = {r.class_number: r for r in rows}  # first row per class number, as in db.save_snapshot
    (term_dir / "listing.csv.gz").write_bytes(
        _gz_csv(LISTING_FIELDS, [[getattr(r, f) for f in LISTING_FIELDS] for r in sorted(
            listing.values(), key=lambda r: r.class_number)])
    )
    path = term_dir / "seats" / f"{captured_at.replace(':', '')}.csv.gz"
    path.write_bytes(_gz_csv(["class_number", "open_seats"],
                             [[r.class_number, "" if r.open_seats is None else r.open_seats]
                              for r in sorted(listing.values(), key=lambda r: r.class_number)]))
    print(f"csv {term} {captured_at}: {len(listing)} sections -> {path.name}")
    return path


def import_csv(out_dir: Path = SNAPSHOT_DIR) -> int:
    """Load every snapshots/<term>/seats/*.csv.gz not yet in SQLite. Returns how many were added.

    Listing fields come from the term's current listing.csv.gz; sections missing from it
    (dropped from the schedule since) keep only their seat counts.
    """
    from .parse import SectionRow

    con = db.connect()
    added = 0
    for term_dir in sorted(p for p in out_dir.glob("*") if p.is_dir()):
        term = term_dir.name
        listing_path = term_dir / "listing.csv.gz"
        listing = {int(r["class_number"]): r for r in _read_gz_csv(listing_path)} if listing_path.exists() else {}
        for path in sorted((term_dir / "seats").glob("*.csv.gz")):
            stamp = path.name.removesuffix(".csv.gz")  # 2026-11-18T193420Z
            captured_at = f"{stamp[:13]}:{stamp[13:15]}:{stamp[15:]}"
            if db.have_snapshot(con, term, captured_at, "actions"):
                continue
            rows = []
            for rec in _read_gz_csv(path):
                cn = int(rec["class_number"])
                info = listing.get(cn, {})
                fields = {f: info.get(f, "") for f in LISTING_FIELDS if f != "class_number"}
                if not fields["course"]:
                    fields["course"], fields["section"] = "?", "?"
                seats = rec["open_seats"]
                rows.append(SectionRow(class_number=cn, open_seats=int(seats) if seats else None, **fields))
            db.save_snapshot(con, term=term, captured_at=captured_at, source="actions",
                             url=LIVE_URL.format(term=term),
                             digest=hashlib.sha1(path.read_bytes()).hexdigest(), rows=rows)
            added += 1
    con.close()
    return added


def wayback_captures(url: str) -> list[str]:
    """Timestamps of distinct (by content digest) successful captures of url."""
    r = _get(
        CDX_URL,
        params={
            "url": url,
            "fl": "timestamp",
            "filter": "statuscode:200",
            "collapse": "digest",
        },
    )
    return [line.strip() for line in r.text.splitlines() if line.strip()]


def backfill(term: str, all_captures: bool = False) -> None:
    """Pull distinct Wayback captures of a term's schedule page (live and archive URLs).

    By default only captures inside the term's registration window are fetched; later
    captures of the archived page just repeat the frozen end-of-term state.
    """
    from .intervals import registration_window

    lo, hi = (s.replace("-", "").replace(":", "").replace("T", "").rstrip("Z") for s in registration_window(term))
    con = db.connect()
    stamps: list[tuple[str, str]] = []
    for template in (LIVE_URL, ARCHIVE_URL):
        page = template.format(term=term)
        stamps += [(ts, page) for ts in wayback_captures(page.removeprefix("https://www."))]
        time.sleep(POLITE_DELAY_S)
    total = len(stamps)
    if not all_captures:
        stamps = [(ts, page) for ts, page in stamps if lo <= ts <= hi]
    print(f"{term}: {total} distinct captures, {len(stamps)} in the registration window")
    for ts, page in sorted(stamps):
        captured_at = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        if db.have_snapshot(con, term, captured_at, "wayback"):
            continue
        try:
            raw = _get(WAYBACK_RAW.format(ts=ts, url=page)).content
        except requests.RequestException as e:
            print(f"  {ts} failed: {e}", file=sys.stderr)
            continue
        try:
            _, n = _store(term, captured_at, "wayback", page, raw)
        except ScheduleFormatError as e:
            print(f"  {captured_at}: SKIPPED, {e}", file=sys.stderr)
            continue
        print(f"  {captured_at}: {n} sections")
        time.sleep(POLITE_DELAY_S)
    con.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="spartanseats.fetch")
    sub = p.add_subparsers(dest="cmd", required=True)
    live = sub.add_parser("live", help="snapshot the current schedule page for each term")
    live.add_argument("terms", nargs="*", help="e.g. spring-2027 (default: every term in its registration window)")
    live.add_argument("--csv", action="store_true", help="write snapshots/<term>/*.csv.gz instead of SQLite")
    sub.add_parser("import", help="load snapshots/*.csv.gz (from GitHub Actions) into SQLite")
    back = sub.add_parser("backfill", help="load past captures from the Wayback Machine")
    back.add_argument("terms", nargs="+", help="e.g. fall-2024 spring-2025")
    back.add_argument("--all", action="store_true", help="also fetch captures outside the registration window")
    args = p.parse_args(argv)
    if args.cmd == "import":
        print(f"imported {import_csv()} snapshots")
        return
    terms = args.terms
    if args.cmd == "live" and not terms:
        from .terms import collectable_terms

        terms = collectable_terms(datetime.now(UTC).date())
        print(f"terms in their registration window: {terms or 'none'}")
    for term in terms:
        if args.cmd == "live":
            collect_csv(term) if args.csv else collect_live(term)
        else:
            backfill(term, all_captures=args.all)


if __name__ == "__main__":
    main()
