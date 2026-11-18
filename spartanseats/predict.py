"""Predicted fill date for every open section in the current term.

Trains on all past terms, then writes a CSV: course, section, open seats now,
predicted median fill date, a 50% range (25th-75th percentile), and the chance
the section fills in the next 7 days. Sections already full are listed as full.
Fill times are conditioned on the section still being open today.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .features import build
from .model import design, fit
from .terms import registration_opens


def predict_term(term: str, train_terms: list[str], out: Path) -> pd.DataFrame:
    con = db.connect()
    df = build(con, [*train_terms, term])
    train = df[df["term"].isin(train_terms)]
    cur = df[df["term"] == term].reset_index(drop=True)
    model, cols = fit(train)

    snap = con.execute(
        "SELECT id, captured_at FROM snapshot WHERE term=? AND n_sections>0 ORDER BY captured_at DESC LIMIT 1",
        (term,),
    ).fetchone()
    seats = dict(con.execute("SELECT class_number, open_seats FROM observation WHERE snapshot_id=?", (snap[0],)))
    now = datetime.strptime(snap[1], "%Y-%m-%dT%H:%M:%SZ")
    opens, _ = registration_opens(term)
    t0 = datetime(opens.year, opens.month, opens.day)
    today = (now - t0).total_seconds() / 86400

    x = design(cur, cols)
    grid = np.arange(0.5, 200.5, 0.5)
    surv = model.predict_survival_function(x, times=grid).to_numpy()  # (len(grid), n)
    s_now = model.predict_survival_function(x, times=[today]).to_numpy().ravel()

    rows = []
    for i, r in cur.iterrows():
        open_now = seats.get(r["class_number"])
        base = {"course": r["course"], "class_number": r["class_number"], "type": r["type"], "open_seats": open_now}
        if open_now is not None and open_now <= 0:
            rows.append({**base, "status": "full"})
            continue
        # Condition on surviving to today: S(t | T > today) = S(t) / S(today).
        cond = surv[:, i] / max(s_now[i], 1e-9)
        cond[grid <= today] = 1.0

        def quantile(q: float, cond: np.ndarray = cond) -> str:
            hit = np.nonzero(cond <= 1 - q)[0]
            return (t0 + timedelta(days=float(grid[hit[0]]))).date().isoformat() if len(hit) else "after 200 days"

        week = 1 - np.interp(today + 7, grid, cond)
        rows.append({
            **base,
            "status": "open",
            "fill_median": quantile(0.5),
            "fill_25": quantile(0.25),
            "fill_75": quantile(0.75),
            "p_full_7d": round(float(week), 3),
        })
    result = pd.DataFrame(rows).sort_values(["status", "p_full_7d"], ascending=[False, False])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="spartanseats.predict")
    p.add_argument("term")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    out = a.out or Path("data") / f"predictions_{a.term}.csv"
    res = predict_term(a.term, a.train, out)
    print(res["status"].value_counts().to_string())
    print(res[res["status"] == "open"].head(15).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
