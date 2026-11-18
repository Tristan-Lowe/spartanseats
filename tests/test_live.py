"""Live predictor: snapshot CSVs + exported model -> site JSON, with no database."""

import json

import joblib
import numpy as np
import pandas as pd
from test_model import TERMS, _make_db

from spartanseats import landmark, live
from spartanseats.features import build, history_tables
from spartanseats.fetch import LISTING_FIELDS, _gz_csv


def _model_dir(tmp_path):
    con = _make_db(tmp_path / "t.db")
    sections = build(con, TERMS)
    pairs = pd.concat([landmark.pairs_for_term(con, sections[sections["term"] == t], t) for t in TERMS])
    d = tmp_path / "model"
    d.mkdir()
    joblib.dump(landmark.fit(pairs), d / "landmark.joblib")
    (d / "history.json").write_text(json.dumps(history_tables(sections)))
    (d / "meta.json").write_text(json.dumps({"trained_at": "x", "train_terms": TERMS, "train_pairs": len(pairs)}))
    return d


def _snapshots(tmp_path):
    term_dir = tmp_path / "snapshots" / "spring-2027"
    (term_dir / "seats").mkdir(parents=True)
    listing = [
        [1, "CS 46A", "01", "In Person", "Intro", "", "3.0", "LEC", "MW", "10:30AM-11:45AM", "", "", "", ""],
        [2, "PHIL 250", "01", "In Person", "Ethics", "", "3.0", "LEC", "MW", "10:30AM-11:45AM", "", "", "", ""],
        [3, "CS 146", "01", "In Person", "DS", "", "3.0", "LEC", "MW", "10:30AM-11:45AM", "", "", "", ""],
        [4, "CS 180", "01", "In Person", "Indep", "", "3.0", "SUP", "TBA", "TBA", "", "", "", ""],
    ]
    (term_dir / "listing.csv.gz").write_bytes(_gz_csv(LISTING_FIELDS, listing))
    for stamp, seats in [("2026-11-17T200000Z", [10, 40, 0, 5]), ("2026-11-18T200000Z", [2, 40, 0, 5])]:
        rows = [[cn, s] for cn, s in zip([1, 2, 3, 4], seats, strict=True)]
        (term_dir / "seats" / f"{stamp}.csv.gz").write_bytes(_gz_csv(["class_number", "open_seats"], rows))
    return tmp_path / "snapshots"


def test_predict_from_csvs(tmp_path):
    out = live.predict("spring-2027", _snapshots(tmp_path), _model_dir(tmp_path))
    by_cn = {s["cn"]: s for s in out["sections"]}

    assert out["counts"] == {"open": 2, "full": 1, "na": 1}
    assert by_cn[3]["status"] == "full"
    assert by_cn[4]["status"] == "na"  # supervision sections get no prediction
    for cn in (1, 2):
        s = by_cn[cn]
        assert 0 <= s["p7"] <= s["p_start"] <= 1  # a longer horizon can only be likelier
        for key in ("early", "median", "late"):
            assert s[key] is None or s[key] <= out["term_start"]
    # 2 seats left and dropping fast in a popular course vs 40 untouched seats.
    assert by_cn[1]["p7"] > by_cn[2]["p7"]
    assert np.isclose(out["day_of_registration"], 22.8, atol=0.1)
