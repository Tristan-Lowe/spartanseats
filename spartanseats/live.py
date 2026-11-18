"""Live predictions from the snapshot CSVs, with no database.

Two steps:
  export   (run locally, where the SQLite history lives) trains the landmark model on every
           past term and writes model/landmark.joblib, model/history.json, model/meta.json.
  predict  (runs anywhere, including GitHub Actions) reads snapshots/<term>/ plus model/ and
           writes site/data/<term>.json for the website.

For each open section the model gives P(full within g days) for a grid of g. Those are
forced to be non-decreasing in g, then read off as: chance of filling within 7 days, chance
of filling before classes start, and an estimated fill window (the days where the chance
passes 25%, 50%, and 75%). The horizon stops at the first day of class, because training
captures stop there.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import landmark
from .features import KEEP_TYPES, apply_history, history_tables, listing_features
from .fetch import LISTING_FIELDS, SNAPSHOT_DIR, _read_gz_csv
from .terms import registration_opens, term_start

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
SITE_DATA = ROOT / "site" / "data"
GAPS = np.array([0.5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120], dtype=float)


# ---------- export (local) ----------

def export(train_terms: list[str]) -> dict:
    from . import db
    from .features import build

    con = db.connect()
    sections = build(con, train_terms)
    rng = np.random.default_rng(0)
    pairs = pd.concat(
        [landmark.pairs_for_term(con, sections[sections["term"] == t], t, rng) for t in train_terms],
        ignore_index=True,
    )
    clf = landmark.fit(pairs)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "landmark.joblib", compress=3)
    (MODEL_DIR / "history.json").write_text(json.dumps(history_tables(sections)), encoding="utf-8")
    import sklearn

    meta = {
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_terms": train_terms,
        "train_pairs": len(pairs),
        "train_sections": len(sections),
        "sklearn": sklearn.__version__,
        "features": landmark.FEATURES,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ---------- predict (anywhere) ----------

def _seats_files(term_dir: Path) -> list[tuple[datetime, Path]]:
    out = []
    for p in sorted((term_dir / "seats").glob("*.csv.gz")):
        out.append((datetime.strptime(p.name.removesuffix(".csv.gz"), "%Y-%m-%dT%H%M%SZ"), p))
    return out


def _seats(path: Path) -> pd.Series:
    rows = _read_gz_csv(path)
    return pd.Series(
        {int(r["class_number"]): (float(r["open_seats"]) if r["open_seats"] else np.nan) for r in rows}
    )


def current_state(term: str, snapshots: Path = SNAPSHOT_DIR) -> tuple[pd.DataFrame, datetime]:
    """Listing + seats now + seats about a day earlier, one row per section."""
    term_dir = snapshots / term
    files = _seats_files(term_dir)
    if not files:
        raise SystemExit(f"no snapshots for {term}")
    now_t, now_p = files[-1]
    # Drop-rate baseline: the capture closest to 24 h before now (at least 2 h earlier).
    earlier = [(t, p) for t, p in files[:-1] if (now_t - t) >= timedelta(hours=2)]
    prev = min(earlier, key=lambda tp: abs((now_t - tp[0]) - timedelta(days=1)), default=None)

    listing = pd.DataFrame(_read_gz_csv(term_dir / "listing.csv.gz"), columns=LISTING_FIELDS)
    listing["class_number"] = listing["class_number"].astype(int)
    df = listing.copy()
    df["seats_now"] = df["class_number"].map(_seats(now_p))
    if prev is not None:
        prev_seats = df["class_number"].map(_seats(prev[1]))
        days = (now_t - prev[0]).total_seconds() / 86400
        df["seats_drop_rate"] = (prev_seats - df["seats_now"]) / days
    else:
        df["seats_drop_rate"] = np.nan
    return df, now_t


def predict(term: str, snapshots: Path = SNAPSHOT_DIR, model_dir: Path = MODEL_DIR) -> dict:
    clf = joblib.load(model_dir / "landmark.joblib")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    df, now_t = current_state(term, snapshots)

    opens, confirmed = registration_opens(term)
    t0 = datetime(opens.year, opens.month, opens.day)
    start = term_start(term)
    day_now = (now_t - t0).total_seconds() / 86400
    days_to_start = (datetime(start.year, start.month, start.day) - now_t).total_seconds() / 86400

    df = listing_features(df)
    df = apply_history(df, hist)
    df = landmark._codes(df)
    full_now = df["seats_now"] <= 0
    df["share_course_full_a"] = full_now.groupby(df["course"]).transform("mean")
    predictable = df["type"].isin(KEEP_TYPES) & (df["seats_now"] > 0)

    # One feature row per (open section, gap).
    # The model only learned up to the first day of class, so the horizon stops there.
    horizon = max(days_to_start, 0.5)
    gaps = np.append(GAPS[GAPS < horizon], horizon)
    base = df[predictable]
    p = np.zeros((len(base), len(gaps)))
    if len(base):
        for j, g in enumerate(gaps):
            x = base.assign(day_a=day_now, gap=g, seats_a=base["seats_now"])
            p[:, j] = landmark.predict(clf, x)
        p = np.maximum.accumulate(p, axis=1)  # P(full within g) can't fall as g grows

    def when(row: np.ndarray, q: float) -> float | None:
        hit = np.nonzero(row >= q)[0]
        return float(gaps[hit[0]]) if len(hit) else None

    def to_date(gap: float | None) -> str | None:
        return None if gap is None else (now_t + timedelta(days=gap)).date().isoformat()

    preds = {}
    for i, cn in enumerate(base["class_number"]):
        row = p[i]
        preds[int(cn)] = {
            "p7": round(float(np.interp(min(7, horizon), gaps, row)), 3),
            "p_start": round(float(row[-1]), 3),
            "early": to_date(when(row, 0.25)),
            "median": to_date(when(row, 0.5)),
            "late": to_date(when(row, 0.75)),
        }

    sections = []
    for _, r in df.sort_values(["course", "section"]).iterrows():
        cn = int(r["class_number"])
        seats = None if pd.isna(r["seats_now"]) else int(r["seats_now"])
        status = "full" if seats is not None and seats <= 0 else ("open" if cn in preds else "na")
        sections.append({
            "cn": cn, "course": r["course"], "sec": r["section"], "title": r["title"],
            "type": r["type"], "mode": r["mode"], "days": r["days"], "times": r["times"],
            "instructor": r["instructor"], "ge": r["satisfies"], "seats": seats,
            "status": status, **preds.get(cn, {}),
        })
    return {
        "term": term,
        "updated": now_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration_opened": opens.isoformat(),
        "registration_date_confirmed": confirmed,
        "day_of_registration": round(day_now, 1),
        "days_to_term_start": round(days_to_start, 1),
        "term_start": start.isoformat(),
        "model": {k: meta[k] for k in ("trained_at", "train_terms", "train_pairs")},
        "counts": {s: sum(1 for x in sections if x["status"] == s) for s in ("open", "full", "na")},
        "sections": sections,
    }


def write_site_data(terms: list[str], out_dir: Path = SITE_DATA) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for term in terms:
        data = predict(term)
        path = out_dir / f"{term}.json"
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        written.append(path)
        print(f"{term}: {data['counts']} -> {path}")
    (out_dir / "terms.json").write_text(json.dumps(terms), encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spartanseats.live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="train on past terms and write model/")
    e.add_argument("terms", nargs="+", help="past terms, oldest first")
    pr = sub.add_parser("predict", help="write site/data/<term>.json from snapshots/ and model/")
    pr.add_argument("terms", nargs="*", help="default: terms with snapshots in their registration window")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export(a.terms), indent=1))
        return
    terms = a.terms
    if not terms:
        from .terms import collectable_terms

        open_now = collectable_terms(datetime.now(UTC).date())
        terms = [t for t in open_now if (SNAPSHOT_DIR / t / "listing.csv.gz").exists()]
    write_site_data(terms)


if __name__ == "__main__":
    main()
