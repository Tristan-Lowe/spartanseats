"""End-to-end check on synthetic terms where the true fill pattern is known."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from spartanseats import db
from spartanseats.features import build
from spartanseats.model import evaluate
from spartanseats.parse import SectionRow
from spartanseats.terms import registration_opens

TERMS = ["fall-2024", "spring-2025", "fall-2025"]
# Popular courses fill in ~5 days, the rest in ~60 (often never, before classes start).
COURSES = {"CS 46A": 5, "CS 146": 5, "MATH 30": 5, "ART 101": 60, "HIST 170": 60, "PHIL 250": 60}


def _make_db(path):
    rng = np.random.default_rng(0)
    con = db.connect(path)
    for term in TERMS:
        opens, _ = registration_opens(term)
        t0 = datetime(opens.year, opens.month, opens.day)
        sections = []
        cn = 1000 * (TERMS.index(term) + 1)
        for course, scale in COURSES.items():
            for s in range(8):
                cn += 1
                sections.append((cn, course, f"{s + 1:02d}", rng.weibull(1.5) * scale))
        for day in [1, 3, 7, 14, 30, 60, 90]:
            rows = [
                SectionRow(cn, course, sec, "In Person", "t", "", "3.0", "LEC", "MW",
                           "10:30AM-11:45AM", "", "", "", 0 if fill <= day else 10, "")
                for cn, course, sec, fill in sections
            ]
            db.save_snapshot(con, term=term, captured_at=(t0 + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             source="wayback", url="u", digest=f"{term}{day}", rows=rows)
    return con


def test_history_feature_uses_only_earlier_terms(tmp_path):
    con = _make_db(tmp_path / "t.db")
    df = build(con, TERMS)
    first = df[df["term"] == TERMS[0]]
    assert not first["hist_course_seen"].any()
    assert df[df["term"] == TERMS[-1]]["hist_course_seen"].all()


def test_model_beats_overall_baseline(tmp_path):
    con = _make_db(tmp_path / "t.db")
    df = build(con, TERMS)
    res = evaluate(con, df, TERMS[-1], TERMS[:-1])
    assert res["model"]["auc"] > 0.8
    assert res["model"]["brier"] < res["baseline_overall"]["brier"]


def test_landmark_pairs_and_fit(tmp_path):
    from spartanseats import landmark

    con = _make_db(tmp_path / "t.db")
    df = build(con, TERMS)
    pairs = landmark.pairs_for_term(con, df[df["term"] == TERMS[0]], TERMS[0])
    # Only sections still open at A appear, and every pair has A before B.
    assert (pairs["seats_a"] > 0).all()
    assert (pairs["gap"] > 0).all()
    train = pd.concat([landmark.pairs_for_term(con, df[df["term"] == t], t) for t in TERMS[:2]])
    clf = landmark.fit(train)
    test = landmark.pairs_for_term(con, df[df["term"] == TERMS[2]], TERMS[2])
    p = landmark.predict(clf, test)
    assert ((p >= 0) & (p <= 1)).all()
    # Longer horizons should get higher predicted fill chances on average.
    assert p[test["gap"].to_numpy() >= 30].mean() > p[test["gap"].to_numpy() <= 3].mean()
