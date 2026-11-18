# Results (2026-11-18)

Question scored: a section is open at capture A; is it full by capture B?
Each held-out term is trained only on earlier terms. AUC: higher is better. Brier: lower is better.

## Archive coverage (captures inside the registration window)

| Term | Captures | Days since registration opened |
|---|---|---|
| fall-2020 | 0 | (all captures are after the term started) |
| spring-2021 | 10 | -1, 50-57, 82 |
| fall-2021 | 3 | 64, 65, 117 |
| spring-2022 | 3 | -7, 35, 43 |
| fall-2022 | 2 | 24, 30 |
| spring-2023 | 0 | |
| fall-2023 | 3 | 24, 36, 40 |
| spring-2024 | 2 | 30, 37 |
| fall-2024 | 6 | 8, 28, 32, 51, 88, 89 |
| spring-2025 | 5 | 4, 10, 46, 72, 81 |
| fall-2025 | 7 | 6, 12, 21, 42, 44, 83, 91 |
| spring-2026 | 5 | -3, 14, 17, 39, 76 |
| fall-2026 | 3 | 6, 19, 29 |

Rows with about 1,700-2,000 sections are partial pages the Archive saved; missing sections are skipped.

## Held-out terms

| Test term | Train pairs | Landmark AUC / Brier | AFT AUC / Brier | Gap-only AUC / Brier |
|---|---|---|---|---|
| spring-2022 | 158,242 | 0.852 / 0.142 | 0.832 / 0.148 | 0.35 / 0.208 |
| fall-2025 | 227,481 | **0.888 / 0.105** | 0.814 / 0.123 | 0.666 / 0.156 |
| spring-2026 | 321,111 | **0.865 / 0.126** | 0.787 / 0.148 | 0.610 / 0.182 |
| fall-2026 | 365,169 | **0.887 / 0.085** | 0.827 / 0.104 | 0.579 / 0.125 |

Short gaps (B within 3 days of A), where the Archive happens to have them:

| Test term | Pairs (filled) | Landmark AUC / Brier | AFT AUC / Brier |
|---|---|---|---|
| fall-2025 (days 42 -> 44) | 4,114 (38) | 0.980 / 0.007 | 0.788 / 0.009 |
| spring-2026 (days 14 -> 17) | 4,280 (486) | 0.910 / 0.097 | 0.853 / 0.106 |

## Calibration (landmark model): predicted chance vs. share that actually filled

| Predicted | spring-2026 actual | fall-2026 actual |
|---|---|---|
| 0-5% (avg 1.5%) | 3.3% | 1.4% |
| 5-10% (7%) | 12.2% | 8.7% |
| 10-20% (14.5%) | 22.6% | 14.0% |
| 20-30% (24.6%) | 32.5% | 23.9% |
| 30-50% (39%) | 47.2% | 39.0% |
| 50-70% (60%) | 59.5% | 53.8% |
| 70-90% (79%) | 72.5% | 81.6% |
| 90-100% (93%) | 84.9% | 100% (35 pairs) |

Fall 2026 is close to exact. Spring 2026 runs up to ~10 points off (that term filled faster than
earlier ones). The site tells students to read a percentage as roughly +/-10 points.

Not yet measured: error in days for predicted fill dates, and short horizons in the first week
of registration. Both need the 4-hourly live snapshots.
