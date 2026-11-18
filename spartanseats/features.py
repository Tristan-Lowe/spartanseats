"""One row per section-term: fill interval in days since registration opened, plus features.

Features are only things known before registration opens (the schedule listing and
how the course filled in earlier terms). Seat counts are deliberately left out:
a section already full at the first capture would show 0 seats, which leaks the answer.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from .intervals import days_since_open, fill_records

# Supervision, clinical and practicum sections are individual arrangements, not seats students race for.
KEEP_TYPES = {"LEC", "SEM", "LAB", "ACT", "DIS"}
EPS = 0.01  # survival bounds must be > 0


def _start_hour(times: str) -> float:
    m = re.match(r"(\d{1,2}):(\d{2})(AM|PM)", times or "")
    if not m:
        return np.nan
    h = int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0)
    return h + int(m.group(2)) / 60


def _course_number(course: str) -> float:
    m = re.search(r"(\d+)", course.split(" ", 1)[-1])
    return float(m.group(1)) if m else np.nan


def _listing(con: sqlite3.Connection, term: str) -> pd.DataFrame:
    """Latest listing fields per section in the term (title, days, times, ...)."""
    return pd.read_sql_query(
        """
        SELECT o.class_number, o.course, o.type, o.mode, o.units, o.days, o.times, o.satisfies
        FROM observation o JOIN snapshot s ON s.id = o.snapshot_id
        WHERE s.term = ? AND s.id = (
            SELECT MAX(s2.id) FROM snapshot s2 JOIN observation o2 ON o2.snapshot_id = s2.id
            WHERE s2.term = ? AND o2.class_number = o.class_number)
        """,
        con,
        params=(term, term),
    ).drop_duplicates("class_number")


def term_frame(con: sqlite3.Connection, term: str) -> pd.DataFrame:
    recs = fill_records(con, term)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "term": term,
            "class_number": [r.class_number for r in recs],
            "kind": [r.kind for r in recs],
            "lower": [days_since_open(term, r.lower) for r in recs],
            "upper": [days_since_open(term, r.upper) for r in recs],
        }
    )
    df = df.merge(_listing(con, term), on="class_number", how="left")
    df = df[df["type"].isin(KEEP_TYPES)].copy()

    # Survival bounds in days. Left-censored: filled somewhere in (0, first capture].
    df["lower"] = df["lower"].fillna(0).clip(lower=EPS)
    df["upper"] = df["upper"].fillna(np.inf)
    df["upper"] = np.maximum(df["upper"], df["lower"] + EPS)
    df["ever_full"] = np.isfinite(df["upper"])

    df["subject"] = df["course"].str.split(" ").str[0]
    df["number"] = df["course"].map(_course_number)
    df["level"] = pd.cut(df["number"], [-1, 99, 199, 1000], labels=["lower", "upper", "grad"]).astype(str)
    df["units_f"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["start_hour"] = df["times"].map(_start_hour)
    df["tba"] = df["start_hour"].isna()
    df["evening"] = df["start_hour"] >= 18
    df["early"] = df["start_hour"] < 9
    df["ge"] = df["satisfies"].fillna("").str.startswith("GE")
    df["n_sections"] = df.groupby("course")["class_number"].transform("count")
    return df


def add_history(df: pd.DataFrame, terms_in_order: list[str], prior: float = 5.0) -> pd.DataFrame:
    """Course and subject fill rates from strictly earlier terms (smoothed toward the overall rate)."""
    out = []
    for i, term in enumerate(terms_in_order):
        cur = df[df["term"] == term].copy()
        past = df[df["term"].isin(terms_in_order[:i])]
        base = past["ever_full"].mean() if len(past) else 0.3
        for key, col in (("course", "hist_course_full"), ("subject", "hist_subject_full")):
            g = past.groupby(key)["ever_full"].agg(["sum", "count"])
            rate = (g["sum"] + prior * base) / (g["count"] + prior)
            cur[col] = cur[key].map(rate).fillna(base)
        cur["hist_course_seen"] = cur["course"].isin(set(past["course"]))
        out.append(cur)
    return pd.concat(out, ignore_index=True) if out else df


def build(con: sqlite3.Connection, terms_in_order: list[str]) -> pd.DataFrame:
    frames = [f for t in terms_in_order if not (f := term_frame(con, t)).empty]
    if not frames:
        return pd.DataFrame()
    return add_history(pd.concat(frames, ignore_index=True), terms_in_order)
