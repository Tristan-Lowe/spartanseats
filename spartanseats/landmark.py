"""Landmark model: given a section's state at capture A, will it be full by capture B?

Why: the AFT model in model.py only knows the listing, so every section of a course
gets the same prediction. At prediction time we also know how many seats are left
and how fast they have been going, which is the strongest signal there is. Using it
is not leakage as long as training uses the seat count *at the earlier capture*.

Every pair of in-window captures (A, B), A before B, where the section is open at A
becomes one training row. Sparse Archive captures still give many pairs, with gaps
from hours to months, so the model learns P(full within g days | state now) for any g.

Gradient-boosted trees with monotone constraints: more days ahead can only raise
the chance of filling, more open seats can only lower it.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import KEEP_TYPES

STATIC = ["level_code", "type_code", "mode_code", "units_f", "tba", "evening", "early", "ge",
          "n_sections", "hist_course_full", "hist_subject_full", "hist_course_seen"]
DYNAMIC = ["day_a", "gap", "seats_a", "seats_drop_rate", "share_course_full_a"]
FEATURES = STATIC + DYNAMIC
# +1 = prediction can only rise with the feature, -1 = only fall, 0 = free.
MONOTONE = {"gap": 1, "seats_a": -1}
MAX_PAIRS_PER_TERM = 400_000


def _codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["level_code"] = df["level"].map({"lower": 0, "upper": 1, "grad": 2}).fillna(1)
    df["type_code"] = df["type"].map({t: i for i, t in enumerate(sorted(KEEP_TYPES))}).fillna(-1)
    df["mode_code"] = df["mode"].map({"In Person": 0, "Hybrid": 1, "Fully Online": 2}).fillna(-1)
    return df


def pairs_for_term(con: sqlite3.Connection, static: pd.DataFrame, term: str,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
    """All (A, B) capture pairs for open-at-A sections, with features and label full_at_B."""
    from .model import status_at_captures  # needs lifelines; kept out of the live/site path

    st = status_at_captures(con, term)
    st = st[st["class_number"].isin(static["class_number"])]
    # Collapse near-duplicate captures (same day) to one per section per day.
    st = st.assign(dayr=st["day"].round(1)).drop_duplicates(["class_number", "dayr"]).sort_values("day")
    days = np.sort(st["dayr"].unique())
    if len(days) < 2:
        return pd.DataFrame()
    wide = st.pivot_table(index="class_number", columns="dayr", values="open_seats", aggfunc="first")
    s = _codes(static.set_index("class_number")).reindex(wide.index)
    course_of = s["course"]

    frames = []
    for i, a in enumerate(days[:-1]):
        seats_a = wide[a]
        prev = wide[days[i - 1]] if i > 0 else None
        full_a = seats_a <= 0
        share_full = full_a.groupby(course_of).transform("mean")
        for b in days[i + 1:]:
            seats_b = wide[b]
            ok = seats_a.notna() & seats_b.notna() & (seats_a > 0)
            if not ok.any():
                continue
            f = s.loc[ok, STATIC].copy()
            f["day_a"] = a
            f["gap"] = b - a
            f["seats_a"] = seats_a[ok]
            if prev is not None:
                f["seats_drop_rate"] = ((prev[ok] - seats_a[ok]) / max(a - days[i - 1], 0.1)).to_numpy()
            else:
                f["seats_drop_rate"] = np.nan
            f["share_course_full_a"] = share_full[ok]
            f["full_b"] = (seats_b[ok] <= 0).to_numpy()
            f["term"] = term
            frames.append(f.reset_index())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if rng is not None and len(out) > MAX_PAIRS_PER_TERM:
        out = out.sample(MAX_PAIRS_PER_TERM, random_state=int(rng.integers(1 << 31)))
    return out


def fit(pairs: pd.DataFrame) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        monotonic_cst=[MONOTONE.get(c, 0) for c in FEATURES], random_state=0,
    )
    clf.fit(pairs[FEATURES].astype(float), pairs["full_b"].astype(int))
    return clf


def predict(clf: HistGradientBoostingClassifier, pairs: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(pairs[FEATURES].astype(float))[:, 1]
