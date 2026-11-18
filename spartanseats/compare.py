"""Compare models on a held-out term, on the same question for all of them:

    a section is open at capture A; is it full by capture B?

Models:
  landmark  gradient-boosted trees on the state at A (seats left, drop rate, ...)
  aft       interval-censored Weibull AFT on listing + history: 1 - S(day_B) / S(day_A)
  gap-only  training share of open-at-A sections full at B, by gap bucket (no section info)

Scores: Brier (lower is better), AUC, and the same two on short gaps (<= 3 days),
which is the horizon a student actually cares about.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import db, landmark
from .features import build
from .model import design
from .model import fit as fit_aft

GAP_BINS = [0, 1, 3, 7, 14, 30, 60, 1000]


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0}
    auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
    return {"n": len(y), "positives": int(y.sum()), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(float(auc), 4)}


def compare(test_term: str, train_terms: list[str]) -> dict:
    con = db.connect()
    sections = build(con, [*train_terms, test_term])
    rng = np.random.default_rng(0)
    train_pairs = pd.concat(
        [landmark.pairs_for_term(con, sections[sections["term"] == t], t, rng) for t in train_terms],
        ignore_index=True,
    )
    test_sections = sections[sections["term"] == test_term]
    test_pairs = landmark.pairs_for_term(con, test_sections, test_term)
    if train_pairs.empty or test_pairs.empty:
        return {"error": "not enough captures", "train_pairs": len(train_pairs), "test_pairs": len(test_pairs)}
    y = test_pairs["full_b"].to_numpy().astype(float)

    clf = landmark.fit(train_pairs)
    p_land = landmark.predict(clf, test_pairs)

    aft, cols = fit_aft(sections[sections["term"].isin(train_terms)])
    x = design(test_sections.set_index("class_number").loc[test_pairs["class_number"]].reset_index(), cols)
    p_aft = np.empty(len(test_pairs))
    for (a, b), idx in test_pairs.groupby(["day_a", test_pairs["day_a"] + test_pairs["gap"]]).indices.items():
        s = aft.predict_survival_function(x.iloc[idx], times=[max(a, 0.01), max(b, 0.02)]).to_numpy()
        p_aft[idx] = 1 - s[1] / np.maximum(s[0], 1e-9)

    bins_train = pd.cut(train_pairs["gap"], GAP_BINS)
    rate = train_pairs.groupby(bins_train, observed=False)["full_b"].mean()
    p_gap = pd.cut(test_pairs["gap"], GAP_BINS).map(rate).astype(float).fillna(y.mean()).to_numpy()

    short = test_pairs["gap"].to_numpy() <= 3
    res = {
        "test_term": test_term,
        "train_terms": train_terms,
        "train_pairs": len(train_pairs),
        "test_capture_days": sorted(set(test_pairs["day_a"]).union(test_pairs["day_a"] + test_pairs["gap"])),
    }
    for name, p in (("landmark", p_land), ("aft", p_aft), ("gap_only", p_gap)):
        res[name] = {"all": _score(y, p), "gap<=3d": _score(y[short], p[short])}
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spartanseats.compare")
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(compare(a.test, a.train), indent=1, default=float))


if __name__ == "__main__":
    main()
