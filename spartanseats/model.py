"""Interval-censored Weibull AFT model of days-to-fill, with baselines and evaluation.

Evaluation question per held-out term: at each capture day d, which sections are
already full? The model gives P(full by d) for every section; we compare it with
what the capture showed. Scored with the Brier score (lower is better) and AUC.

Baselines:
  - overall: one curve for everyone (no covariates): same P(full by d) for all sections.
  - history: the course's smoothed fill rate in earlier terms, ignoring d.
"""

from __future__ import annotations

import argparse
import sqlite3
import warnings

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.metrics import roc_auc_score

from . import db
from .features import build
from .intervals import _ts, days_since_open, registration_window

CATEGORICAL = ["level", "type", "mode"]
NUMERIC = [
    "units_f", "tba", "evening", "early", "ge", "n_sections",
    "hist_course_full", "hist_subject_full", "hist_course_seen",
]


def design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    x = pd.get_dummies(df[CATEGORICAL], drop_first=False).astype(float)
    x = pd.concat([x, df[NUMERIC].astype(float)], axis=1)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def fit(train: pd.DataFrame) -> tuple[WeibullAFTFitter, list[str]]:
    x = design(train)
    x = x.loc[:, x.std() > 0]  # constant columns break the fit
    data = pd.concat([x, train[["lower", "upper"]]], axis=1)
    model = WeibullAFTFitter(penalizer=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_interval_censoring(data, lower_bound_col="lower", upper_bound_col="upper")
    return model, list(x.columns)


def p_full_by(model: WeibullAFTFitter, columns: list[str], df: pd.DataFrame, day: float) -> np.ndarray:
    surv = model.predict_survival_function(design(df, columns), times=[day])
    return 1.0 - surv.to_numpy().ravel()


def status_at_captures(con: sqlite3.Connection, term: str) -> pd.DataFrame:
    """(class_number, day, full) for every in-window capture of the term."""
    start, end = registration_window(term)
    rows = con.execute(
        """
        SELECT o.class_number, s.captured_at, o.open_seats
        FROM observation o JOIN snapshot s ON s.id = o.snapshot_id
        WHERE s.term = ? AND s.n_sections > 0 AND o.open_seats IS NOT NULL
          AND s.captured_at BETWEEN ? AND ?
        """,
        (term, start, end),
    ).fetchall()
    out = pd.DataFrame(rows, columns=["class_number", "captured_at", "open_seats"])
    out["day"] = [days_since_open(term, _ts(t)) for t in out["captured_at"]]
    out["full"] = out["open_seats"] <= 0
    return out


def evaluate(con: sqlite3.Connection, all_df: pd.DataFrame, test_term: str, train_terms: list[str]) -> dict:
    train = all_df[all_df["term"].isin(train_terms)]
    test = all_df[all_df["term"] == test_term].set_index("class_number")
    model, cols = fit(train)

    status = status_at_captures(con, test_term)
    status = status[status["class_number"].isin(test.index)]
    # One capture per distinct day keeps near-duplicate captures from dominating the score.
    status = status.assign(dayr=status["day"].round()).drop_duplicates(["class_number", "dayr"])

    preds, base_overall, base_hist, ys = [], [], [], []
    for _, grp in status.groupby("dayr"):
        rows = test.loc[grp["class_number"]].reset_index()
        d = max(float(grp["day"].mean()), 0.01)
        preds.append(p_full_by(model, cols, rows, d))
        base_overall.append(_overall_curve(train, d, len(rows)))
        base_hist.append(rows["hist_course_full"].to_numpy())
        ys.append(grp["full"].to_numpy())
    y = np.concatenate(ys).astype(float)
    p = np.concatenate(preds)
    b1 = np.concatenate(base_overall)
    b2 = np.concatenate(base_hist)

    def score(q: np.ndarray) -> dict:
        auc = roc_auc_score(y, q) if 0 < y.mean() < 1 else float("nan")
        return {"brier": float(np.mean((q - y) ** 2)), "auc": float(auc)}

    return {
        "test_term": test_term,
        "train_terms": train_terms,
        "n_train_sections": len(train),
        "n_test_points": len(y),
        "capture_days": sorted(status["dayr"].unique().tolist()),
        "share_full": float(y.mean()),
        "model": score(p),
        "baseline_overall": score(b1),
        "baseline_history": score(b2),
    }


def _overall_curve(train: pd.DataFrame, day: float, n: int) -> np.ndarray:
    """Covariate-free Weibull fit: one P(full by day) for everyone."""
    from lifelines import WeibullFitter

    wf = WeibullFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf.fit_interval_censoring(train["lower"], train["upper"])
    return np.full(n, 1.0 - float(wf.survival_function_at_times([day]).iloc[0]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="spartanseats.model")
    p.add_argument("--test", required=True, help="held-out term, e.g. spring-2027")
    p.add_argument("--train", nargs="+", required=True, help="earlier terms, oldest first")
    args = p.parse_args(argv)
    con = db.connect()
    df = build(con, [*args.train, args.test])
    res = evaluate(con, df, args.test, args.train)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
