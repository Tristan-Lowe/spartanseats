# SpartanSeats

Predicts when each SJSU class section will fill, using only the public class schedule
(`www.sjsu.edu/classes/schedules/<term>.php`). Never logs in to MySJSU.

## Data: why not just one semester

Feedback on the proposal: one semester of our own snapshots is too little training data.
That one semester is a single registration season, so the model would see each course
fill exactly once. Three fixes, in order of how much data they add:

1. **Wayback backfill (fall 2020 → fall 2026, 13 terms).** The Internet Archive has
   saved SJSU's schedule pages many times per term (4–35 distinct captures). Each
   capture is the full table, about 5,000–6,700 sections with open seats.
2. **Model the section, not the semester.** The unit of analysis is a section-term.
   13 terms × ~5,000 sections is ~65,000 records, and the same course shows up in
   every term, so the model learns how *CS 146 at 10:30 MW* fills across years.
3. **Censoring-aware survival model.** Archive captures are sparse, so a fill time is
   usually only known to fall between two captures. That is interval censoring, which
   survival analysis (Cox 1972; Turnbull; interval-censored AFT) handles directly
   rather than throwing those sections away.

Live snapshots (several a day, starting spring 2027) give dense data going forward
and are the test set: train on past terms, predict spring 2027, measure error in days.

## Layout

| Path | What it does |
|---|---|
| `spartanseats/parse.py` | Schedule HTML → section rows. Matches columns by name (the 2020 layout differs) and refuses unknown layouts. |
| `spartanseats/db.py` | SQLite: `snapshot` (one per page capture) and `observation` (one per section per capture). |
| `spartanseats/fetch.py` | `live` snapshot of sjsu.edu; `backfill` from the Wayback CDX API. Raw HTML kept in `data/raw/`. |
| `spartanseats/intervals.py` | Per-section fill interval (interval / right / left censored). |
| `spartanseats/terms.py` | Registration start date per term (time zero). |

## Run

```powershell
python -m venv .venv; .venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m spartanseats.fetch live spring-2027
.venv\Scripts\python -m spartanseats.fetch backfill fall-2024 spring-2025
.venv\Scripts\python -m pytest
```

`collect.ps1` runs one live snapshot and appends to `data/collect.log`; schedule it with
Windows Task Scheduler every 4 hours during registration.

## Etiquette

Public pages only, a descriptive User-Agent, 5 s between requests, and long backoff
when the Internet Archive throttles. A live snapshot is one request.
