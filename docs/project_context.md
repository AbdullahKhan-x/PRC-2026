# PRC Data Challenge 2026 — Complete Context Handoff

**Team:** `logical-ladder` · **Best official score:** 303.4643 (v35) · **Leader:** ~213 (charming-eel) · **Final deadline:** **2026-10-14T10:00Z** (noon CET)

This document is the condensed memory of roughly three weeks of work and about 40 analysis scripts. It is written for an agent (Claude Code) that will continue the work, and for the team re-reading it to look for what was missed. Every number in it was measured on the data unless explicitly labelled as a hypothesis, estimate or guess. Where something was believed and later disproved, both are recorded, because the disproofs are as valuable as the findings.

Read sections 0, 1.7, 5, 12 and 13 first if time is short.

---

## 0. Rules of engagement — read before running anything

These are the things that have cost real submissions. Each one happened at least once.

1. **The scorer aligns by POSITION, not by ID.** The submission must have exactly the template's `MVT_ID_mvt` order. A file with the right IDs in the wrong order scores ~546 (it compares every prediction against a different flight). Always reindex onto the template and assert element-wise equality before writing, and re-verify the exact file you upload:
   ```fish
   python -c 'import pandas as pd; t=pd.read_parquet("data/raw/submitting.parquet"); s=pd.read_parquet("FILE.parquet"); print((s.MVT_ID_mvt.values==t.MVT_ID_mvt.values).all(), len(s), (s.TAXITIME_SEC_mvt>10800).sum())'
   ```
   Must print `True 344841 <count>` (row count will change in the final phase — see 1.7).
2. **Never reuse a version number.** A teammate uploads to the same bucket with overlapping filenames (`logical-ladder_v13/14/15`). Result JSONs were silently overwritten (v10 read 329.169 one day and 548.1375 the next). Use a prefix that cannot collide.
3. **Generate submissions on the laptop, not Kaggle.** Kaggle's different thread count changes LightGBM's histogram summation order → different trees → base drift (measured: 3.6 s mean / 109 s max on non-pool rows, best_iteration 1326 vs local). Fine for experiments at multi-point resolution; not fine for anything compared to v35 byte-for-byte.
4. **Every new version must prove it is built on the incumbent.** Scripts 33–35 refit v35's base and detector, compare against `logical-ladder_v35.parquet`, and **refuse to write** unless `max diff < 1e-6`. Never weaken this gate. A previous edit replaced it with "load v35 and compare it with itself", which passes on any file (demonstrated: it printed `REPRODUCED` against a v35 corrupted on 10,000 rows by +500 s).
5. **Never reorder the training frame.** LightGBM bagging and the 5% early-stopping slice depend on row positions. Features that need sorting (e.g. runway sequence) are computed on a sorted copy and written back by position.
6. **Change one thing per submission.** v10 changed five things at once and lost 25 points with no way to attribute it.
7. **The local metric is trustworthy for bulk changes, untrustworthy for detector thresholds.** Bulk (base-model) changes measured across 344k rows transferred every time. Detector/pool changes rest on ~49 extreme events and failed to transfer repeatedly. See §7.
8. **Shell is `fish`.** Backticks inside pasted text trigger `command substitutions not allowed`; use `python -c '...'` with single quotes; `VAR=value cmd` → use `env VAR=value cmd`. Pipe long runs through `| tee name.log`.
9. **Machine:** Arch Linux laptop, ~7.4 GB RAM, i5-8365U. A full base fit takes tens of minutes; scripts with several fits run for one to several hours. `num_threads=-1` takes every core.
10. **5 uploads/day, reset 00:00 UTC.** On the current leaderboard only the best score counts, so a bad experiment costs a slot but never the incumbent. **The final phase is a single submission** — no second chances there.

### Paths

```
/home/abdullah/Dev-Environment/Personal/PRC/prc_1/            DATA_FOLDER
  DATA_ANALYTICS/                                             all scripts
    24_final.py              v4 pipeline (scored 304.3859; byte-reproducible)
    24_v35.py                v35 pipeline (== "24_final-downloaded to-check.py", verified by diff)
    30_noise_floor.py ... 35_v38_v39_mechanism.py             harnesses and experiments
  data/raw/training_2025-MM-01_....parquet                    12 raw monthly files
  data/raw/ranking.parquet, data/raw/submitting.parquet       ranking set and template
  dep_q_train.parquet / dep_q_rank.parquet                    queue features (from 12_queue_features.py)
  dep_qw_train.parquet / dep_qw_rank.parquet                  + weather (from 15_weather.py)  <- models read these
  metar_cache/                                                raw METAR downloads
  logical-ladder_v35.parquet                                  current best; back it up
  ../logical-ladder_v4_304.parquet                            v4 backup (byte-identical to a fresh 24_final.py run)
```

Upload and score:
```fish
mcli cp logical-ladder_vNN.parquet prc/prc-2026-logical-ladder/
mcli cat prc/prc-2026-logical-ladder/logical-ladder_vNN.parquet_result.json
```

---

## 1. The competition

### 1.1 Organisers and purpose

The **Performance Review Commission (PRC)** of EUROCONTROL, with the **OpenSky Network (OSN)**, runs an annual data challenge whose output feeds open research (results published in the *Journal of Open Aviation Science*; previous editions: 2024 take-off weight estimation, 2025 fuel-burn estimation). The 2026 edition asks for **taxi-out time** of departures at 10 major European airports. The stated rationale: taxi-out is hard to measure and hard to predict, and a good estimator lets analysts quantify excess fuel burn and CO2 from congestion. Organiser contact on the challenge mailing list: Enrico Spinelli (EUROCONTROL). Community channel: OSN Discord `#prc-data-competition`.

### 1.2 Prize and eligibility

- Combined prize for the top 3 teams: **5,000 EUR**.
- A winning solution only qualifies if its **source is public on GitHub under GNU GPLv3** (the challenge account forks it) and **every external dataset used is openly accessible, openly licensed and documented**. Teams from sanctioned countries are excluded.
- External data other teams documented: Iowa State University Environmental Mesonet (METAR), OurAirports (runways), OpenStreetMap (stand/runway geometry), OPDI (PRC + OSN ADS-B flight events). We use IEM METAR only.

### 1.3 Timeline

- Competition ran from 1 Sep 2026; original close **11 Oct 2026 23:59:59 CET**.
- **Extended to 2026-10-14T10:00Z** with the introduction of the final phase (1.7).

### 1.4 Task and scoring

- Predict `TAXITIME_SEC_mvt` for every **departure** in the ranking set.
- Metric: **RMSE in seconds over all scored departures, uncapped.** No clipping, no exclusion of absurd values on the scorer's side. This one fact reshaped the whole project (§5).
- Current leaderboard months: **January and July 2026**: 152,709 Jan + 192,111 Jul + 21 boundary rows = **344,841** departures.

### 1.5 Submission mechanics

- Template: `submitting.parquet` with columns `MVT_ID_mvt, TAXITIME_SEC_mvt`.
- File name `<team>_v<n>.parquet`, uploaded to the team's MinIO bucket via `mcli`; a `<file>_result.json` appears with `score` and `used_pairs`.
- Rejection reasons enforced by the ranking script: row count, unknown or missing `MVT_ID_mvt`, duplicates, **row order**, nulls.
- Max 5 submissions per day, resetting at 00:00 UTC (a rejected upload still consumes one).

### 1.6 The rules question we have not resolved — `AOBT_3_flt` and `LOBT_flt`

Other teams quote the challenge brief as warning that using `AOBT_3_flt` and `LOBT_flt` **"is probably exploiting the ranking process,"** and one team states **"the brief forbids learning from the ranking process"** (meaning: no tuning to the leaderboard via near-duplicate submissions). `kind-mango` excludes both columns and reports that excluding them "sets the floor near 300 s"; `Phoenix-Ops` keeps an AOBT-free primary mode because an AOBT-based model "must not be represented as a predeparture forecast."

**Our model depends heavily on `AOBT_3_flt`**: `implied_taxi_aobt = MVT − AOBT` and `pushback_delay = AOBT − EOBT` are among the top features, and the detector uses AOBT missingness. The final phase adds **code and documentation review**. This is a team decision that must be made explicitly and documented honestly in the GPLv3 write-up. Note the logic of the warning: in a real pre-departure system you would know neither `AOBT` nor `MVT` (the takeoff time) at prediction time. The ranking file nevertheless provides both.

### 1.7 The final phase (announced by email, the most important recent change)

> "We introduced a final phase with a single final submission of your predictions. The final phase will ask to predict 4 months of taxi-out, the two initial months (Jan and July 2026) plus two additional ones (Feb and Jun 2026). We decided to introduce a final phase because we have seen signs of distorting the goal of the challenge which is to (try to) build a model useful beyond the specifics of the competition, i.e. not to just ranking the highest on two specific months. … We will be able to rank per Jan/Jul, Feb/Jun and the four months combined Jan/Feb/Jun/Jul which together with the review of the code and documentation would provide (we hope) enough evidence to select the best teams. … extended the deadline to noon 14th Oct CET, or 2026-10-14T10:00Z."

Consequences:
- **One shot.** Every safeguard in §0 matters more than ever.
- **Validation must cover Feb and Jun.** Everything so far was validated on Jan/Jul 2025 holdouts and tuned against the Jan/Jul leaderboard. The pool floor, the detector and v38 are unverified on Feb/Jun.
- A new ranking file covering four months will presumably be released. The feature pipeline (12 → 15) must regenerate on it. The METAR cache window (2024-12-28 → 2026-08-05) covers Feb–Jun 2026 if the download completed; verify.
- The organisers explicitly suspect top scores of fitting Jan/Jul. Some of the 213–235 scores may not generalise. Mechanism-based features should hold up better than leaderboard-tuned thresholds.
- Documentation quality is now scored. This file (and the earlier `PRC_2026_technical_record.md`, if kept) is the raw material.

### 1.8 Leaderboard landscape

| when | entry | score | notes |
|---|---|---|---|
| 9 Sep | best | ~246 | top ten inside ~275 |
| 24 Sep | jolly-lobster | 225.84 | publishes a validation design matching ours |
| later | jolly-lobster | 224.8810 | 64 submissions |
| later | vigorous-whistle | 229.6196 | 28 subs |
| later | quick-boat | 231.7292 | 69 subs |
| later | strong-crane | 233.7136 | 53 subs |
| later | zesty-puzzle | 234.3683 | 68 subs |
| later | upstanding-firefly | 234.9886 | 17 subs |
| latest | **charming-eel** | **~213** | current leader |
| — | kind-mango | 296.57 | **no AOBT/LOBT**; per-airport delegation, LIRF regime head, 5-member ensembles, 117 features, 3-seed LightGBM with `linear_tree`; prediction = `p_fb·sched_dev + p_24h·(86400+extra) + p_norm·mixture` |
| — | elegant-alligator | 268.56 | reported local 257.13 → official (+11.4) |
| — | **logical-ladder** | **303.4643** | v35 |

---

## 2. The data

### 2.1 Files and sizes

| | rows | notes |
|---|---|---|
| training (12 monthly parquets, 2025) | 4,167,797 | DEP + ARR, all answers present |
| training departures | 2,085,047 | 2,084,659 after dropping y≤0 and repairing rollovers |
| training arrivals | 2,082,750 | |
| ranking (Jan + Jul 2026) | 689,534 | 344,841 DEP + 344,693 ARR |
| all combined | 4,857,331 | ~0.47 GB in memory with categoricals |

Ten airports: **EDDF** Frankfurt, **EDDM** Munich, **EGLL** London Heathrow, **EHAM** Amsterdam Schiphol, **LEBL** Barcelona, **LEMD** Madrid, **LFPG** Paris CDG, **LIRF** Rome Fiumicino, **LSZH** Zurich, **LTFM** Istanbul. (The challenge page text says "11 airport movements"; the data contains 10.)

### 2.2 Columns — two source systems

Suffix `_mvt` = **movement table** (the airport's own record of the movement). Suffix `_flt` = **flight table** (the EUROCONTROL Network Manager's flight record, joined on).

| column | meaning |
|---|---|
| `MVT_ID_mvt` | unique movement id; submission key; no duplicates |
| `FLIGHT_ID_mvt` | flight id; missing 0.5–1.3%/month; pairs an ARR and a DEP when a flight connects two of the ten airports (73,667 such pairs in Q1 alone) |
| `PHASE_mvt` | `DEP` or `ARR` |
| `ADEP_mvt`, `ADES_mvt` | origin, destination airport |
| `MVT_TIME_UTC_mvt` | DEP: take-off time; ARR: landing time. Second resolution. Present 100% in ranking |
| `BLOCK_TIME_UTC_mvt` | DEP: off-block; ARR: on-block. **Blanked for DEP in ranking**, present 100% for ARR in ranking |
| `SCHED_TIME_UTC_mvt` | scheduled time; 100% present |
| `TAXITIME_SEC_mvt` | the target; blanked for DEP in ranking |
| `RUNWAY_mvt`, `STAND_mvt` | runway and stand; ~70 airport–runway combinations; only 0.08% of ranking departures use a stand unseen in training |
| `AIRCRAFT_TYPE_mvt` | type |
| `AOBT_3_flt` | NM actual off-block time; ~99% present on ranking departures (see 1.6) |
| `EOBT_1_flt`, `IOBT_flt`, `LOBT_flt` | NM estimated / initial / last off-block times (plans and updates) |
| `ARVT_1_flt`, `ARVT_3_flt` | NM arrival times |
| `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` | airline, segment (Mainline/Lowcost/Regional/Business/Military…), flight type, wake category |
| `ADES_flt`, `ADES_FILED_flt`, `CALLSIGN_flt`, `FLIGHT_RULE_*`, `AIRCRAFT_TYPE_flt` | flight-plan details |

**All `_flt` columns go missing together.** When `AOBT_3_flt` is missing, `EOBT_1_flt` is missing on 100% of those rows: the movement simply did not match a Network Manager flight record ("unmatched flight"). A `flight_matched` flag captures this.

---

## 3. The target, mathematically

### 3.1 The identity everything rests on

```
TAXITIME_SEC_mvt  =  MVT_TIME_UTC_mvt  −  BLOCK_TIME_UTC_mvt        (100% of training DEP rows, max diff 0 s)
```

`MVT_TIME` is given in the ranking file. **Predicting taxi time is therefore exactly the problem of estimating the airport's recorded off-block timestamp `BLOCK_TIME`.** Not "how long does taxiing take" in the physical sense — what number did the airport's system write into the off-block field. That distinction explains most of the strange behaviour in this dataset.

### 3.2 Distribution

Training departures, raw: min −12, max 131,167, mean 991.2, median 912, std 546.4.

| threshold | rows | share |
|---|---|---|
| y < 0 | 369 | 0.018% — range −12 to −1 s, 298 of them at LSZH (clock skew) |
| y = 0 | 19 | |
| y > 1,800 | 74,167 | 3.56% |
| y > 3,600 | 4,126 | 0.20% |
| y > 10,800 | 202 | 0.0097% |
| y > 86,400 | 15 | |

After cleaning to `0 < y ≤ 10,800` (2,084,457 rows): mean 989, median 912, **std 430**. Percentiles: 1% 293 · 5% 480 · 25% 719 · 75% 1,188 · 95% 1,684 · 99% 2,337 · 99.9% 4,385. Single hump around 700–800 s with a long right tail. Arrivals (taxi-in) average ~535 s — never mix the two.

Mean departure taxi by airport: EGLL 1,364 · LIRF 1,165 · LTFM 1,053 · LFPG 1,019 · LEMD 1,015 · LEBL 957 · EDDF 863 · EDDM 812 · EHAM 784 · LSZH 741. LIRF is the least predictable (std 658 even after cleaning).

### 3.3 Moments — why RMSE here is fragile

| | skewness | excess kurtosis |
|---|---|---|
| all departures | **48.5** | **7,023.6** |
| y ≤ 10,800 | 3.2 | 34.0 |

Removing ~200 rows out of two million collapses kurtosis from 7,023 to 34. The higher moments of this target are made of a handful of flights. Under RMSE, **one row can be worth tens of points**: a single 82,500 s error is 6.8e9 of squared error, about a third of January 2025's entire SSE (§5.5).

Statistics can find the tail but cannot separate the contaminated rows inside it (tested, §4.13). The supervised detector is better at the part that matters.

### 3.4 The two-clock decomposition

For rows with `AOBT_3_flt` present:

```
taxi  =  MVT − BLOCK  =  (MVT − AOBT) − (BLOCK − AOBT)  =  implied_taxi − delta
```

`delta = BLOCK − AOBT` is the disagreement between the airport's and the Network Manager's record of the same pushback. On clean rows **std(delta) = 371 s vs std(taxi) = 401 s.** So predicting taxi on these rows *is* predicting delta, and delta carries almost all of the target's variance. Our model's RMSE on ordinary rows (~244) means it explains roughly 57% of delta's variance.

---

## 4. How the data was generated — measured mechanisms

### 4.1 `BLOCK_TIME` is often a copy of another timestamp

| `BLOCK_TIME` equals (±5 s) | rows | share | then the target is |
|---|---|---|---|
| `SCHED_TIME` | 88,741 | 4.26% | `MVT − SCHED` = `sched_dev`, exactly (median err 3.0 s, 100% within 5 s) |
| `AOBT_3_flt` | 148,274 | 7.11% | `MVT − AOBT` = `implied_taxi`, exactly |
| neither | 1,855,689 | 89.02% | must be predicted |

Overlap: SCHED-only 80,696 · AOBT-only 140,229 · both 8,045. About 11% of all targets are exactly reconstructable *if you knew which regime a row is in*.

### 4.2 Contamination (`BLOCK = SCHED`) — the mechanism behind the extremes

When the off-block message is lost, the schedule is written instead. The recorded "taxi" becomes taxi **plus the whole departure delay**. If the flight left hours late, the target is tens of thousands of seconds.

Rate by airport: **LIRF 16.55%** · LEBL 5.13% · LTFM 4.72% · EDDM 4.59% · EGLL 4.45% · LEMD 4.09% · LFPG 3.77% · EHAM 0.68% · LSZH 0.65% · EDDF 0.64%. At LIRF contaminated rows average 1,572.7 s vs 1,119.6 s for the rest.

**It is an outage, not per-flight noise.** LIRF departures per hour: mean contamination 19.1%, sd 17.3%; 8.0% of hours at 0% and **2.6% of hours at 100%** — impossible if each flight failed independently.

**Arrivals see the same outage.** Airport-level contamination rate DEP vs ARR: correlation 0.9855. ARR contamination is computable in the ranking file (2.13% in 2026 vs 2.20% in 2025). Inside the candidate pool, the extreme rate climbs from 0.23% (ARR feed clean) to 33% (ARR feed >50% contaminated). But used as a detector feature it **failed** (§9): it deflated the detector's confidence on true extremes.

**Contamination is a property of the stand.** For stands with ≥200 departures, a rate learned on Jan–Jun predicts Jul–Dec:

| stand rate in H1 | actual rate in H2 |
|---|---|
| 0–1% | 0.60% |
| 1–5% | 3.45% |
| 5–20% | 6.77% |
| 20%+ | 26.72% |

A 45× spread, stable across half-years. (Hypothesis, not verified: stands without docking sensors fall back to the schedule. EUROCONTROL documentation does not confirm this.)

### 4.3 The two clocks differ by airport, in resolution and offset

`AOBT_3_flt` is **minute-rounded** everywhere (99–100% of values on `:00`; EHAM 83.8%). `BLOCK_TIME` has **1 s resolution at EDDF, EHAM, LSZH** and **5 s resolution at the other seven** (8.3% on `:00` = 1/12).

delta = BLOCK − AOBT on clean rows, by airport:

| airport | median | share with \|delta\| > 5 min |
|---|---|---|
| LTFM | **+296 s** | 55.6% |
| EHAM | +107 s | 19.1% |
| EDDF | +87 s | 14.6% |
| EGLL | +55 s | 27.3% |
| LIRF | −66 s | 31.8% |
| EDDM, LEBL, LEMD, LFPG | ~0 s | 13–26% |

Only ~21% of clean rows have the two clocks within ±60 s.

### 4.4 What drives delta: `AOBT`'s own error, not congestion

- corr(delta, `AOBT − EOBT`) = **−0.42 to −0.64** at nine of ten airports (EHAM −0.15). The negative sign is the signature of measurement noise in `AOBT`: when it is recorded late, the true off-block sits earlier.
- Queue/congestion features correlate with delta at |r| < 0.17.
- Predicting delta from congestion/time features: constant 372 → one global model 278.0 → **ten per-airport experts 277.6**. Per-airport experts add nothing; airport as a feature suffices.

Independent EDA, correlation of `(MVT − X)` with taxi: **EOBT_1 0.551** · AOBT_3 0.533 · IOBT 0.509 · LOBT 0.507 · SCHED 0.310. A *plan* (EOBT) correlates better than the *actual* (AOBT). Median absolute error of `MVT − AOBT` as a taxi guess: 120–180 s at most airports, 356 s at LTFM; ~125 s at 06:00 and 19:00–20:00, 235–300 s at night; Lowcost 124 s, Regional 147, Mainline 177, Business 236, Military 238.

### 4.5 Back-filled `AOBT` — real pattern, no cost

`implied_taxi` piles onto a single 10-second bin at several airports — a value computed as takeoff minus a fixed allowance, not measured (who computes it is unverified):

| airport | spike | share 2025 | share 2026 ranking |
|---|---|---|---|
| LTFM | 1,200 s | 38.4% | 35.3% |
| EDDM | 720 s | 22.0% | 23.2% |
| LIRF | 840 s | 13.7% | 12.8% |
| LFPG | 780 s | 10.6% | 9.1% |
| EGLL | 1,320 s | 6.2% | 6.0% |
| LEBL | 1,220 s | 5.5% | 5.1% |
| EHAM | 620 s | 3.4% | 2.7% |

Stable year over year. **But the model predicts these rows exactly as well as observed ones** (RMSE 243.7 vs 243.6) and an explicit flag ranked #78 of 81 by gain. The tree already handles them. Do not chase this.

### 4.6 Unmatched flights (`AOBT` missing)

1.55% of departures (5,321 in Jan+Jul 2025). `EOBT` is also missing on 100% of them — the whole Network Manager record is absent. Composition in Jan+Jul 2025: 47 recoverable extremes, 4 stale-BLOCK extremes, 140 contaminated normal-range rows, 5,130 clean normal rows. AOBT is missing on **91.7% of true extremes** vs 12% of false alarms in the pool — the detector's strongest signal.

Their stand records are also unreliable: a "next arrival at my stand" physical bound holds on only 35.9% of these rows vs 99.7% for matched flights (§4.14).

### 4.7 The extreme rows (y > 10,800)

202 raw; 14–15 are date-rollover errors (below), leaving **188**. By airport: **LIRF 175**, LFPG 6, EGLL 4, EHAM 2, LSZH 1, others 0. By month: Jul 45 · Aug 25 · Sep 20 · May 16 · Jun 16 · Oct 14 · Feb 12 · Dec 11 · Jan 9 · Apr 9 · Mar 7 · Nov 4. Note Feb (12) and Jun (16) — the new final-phase months — have their own extremes.

Two classes:
- **Recoverable** — `BLOCK ≈ SCHED`, so `sched_dev` reproduces the target. 157 at a 5 s tolerance, 171 at 60 s (worst error 54 s), 180 at 1,800 s.
- **Stale-BLOCK** — 22 rows where `BLOCK` sits a median 22,562 s *before* the schedule; median y 87,169; 13 of 22 within an hour of 86,400 (24 h). Probably return-to-stand events or carried-over records. Three reconstruction hypotheses all failed: `BLOCK` = arrival on-block at the same stand (1/22 within 60 s), = a prior departure's block at that stand (0/22), = the same callsign earlier (0/22). **Undetectable from ranking-visible data.**

### 4.8 Other data defects

- **Date rollover:** 14–15 rows where `MVT_TIME` carries the wrong date (y ≈ 86,400 × k + a normal taxi). Repaired by subtracting whole days from `mvt_i` *before* any derived column is computed (an earlier version repaired only the target and left `sched_dev` with an 86,400 s offset, which produced bogus oracle numbers).
- **Negative taxi:** 369 rows, clock skew, dropped.
- Arrivals: 220 rows where landing is after on-block.

### 4.9 Inside the candidate pool (`sched_dev > 7,200`)

39,804 rows, 193 extremes, base rate 0.48%. Four regimes:

| regime | rows | `sched_dev == y` |
|---|---|---|
| contaminated, y > 10,800 | 157 | 100.0% |
| contaminated, y ≤ 10,800 | 238 | 100.0% |
| not contaminated, y > 10,800 (stale) | 36 | 38.9% |
| not contaminated, y ≤ 10,800 (clean, genuinely late) | 39,373 | 0.1% |

The `is_extreme` label mislabels the 238 (as negatives) and the 36 (as positives). Relabelling to `contam` was tested and lost (§9) — the 238 are rare in the scored months (12 in Jan+Jul) and the relabel doubled false-alarm mass.

### 4.10 LIRF in detail

The dominant error source. July's tail widens while the median barely moves: p99 2,757 → 5,038 s; share of taxis > 30 min 4.32% → 14.37%. Morning bank 08–10 runs 27–33% tail vs 5–7% at 15–19. Remote stands 901/902/903 run 31–36% tail vs ~3% at contact stands. Runway 16L 38.4% tail vs runway 25 6.9%. Stand+runway geometry explains only 7% of LIRF's variance (within-pair residual std 611 vs 658), against 48% at EDDF.

### 4.11 Drift between 2025 and 2026

Arrival taxi-in medians, same month a year apart (computed from the ranking file's arrival rows):
- January: EDDM **+13.1%**, EGLL **+11.0%**, LIRF −8.8%, LSZH +8.5%, LEBL +6.3%, EHAM +2.5%, LEMD −3.7%, LTFM −1.2%, LFPG +0.6%, EDDF +1.1%.
- July: EHAM **−18.7%**, EDDF **−11.2%**, LEBL +4.8%, LEMD −4.6%, others within ±2%.

Some airports genuinely changed. A model that memorises 2025 levels will be biased there. Arrival on-block times are fully present in the ranking file, so a daily per-airport arrival-speed level is computable for 2026 (untested lead, §11).

### 4.12 Physical sequencing on the precise clock

`MVT` is recorded to the second for every flight. The runway departure sequence is therefore known exactly. Time since the previous takeoff and until the next one on the same runway (`gap_prev_tko`, `gap_next_tko`) have ~zero linear correlation with taxi (threshold effects) yet rank #14–17 of ~81 by gain. All our older queue features were anchored on `AOBT`, the noisy clock.

### 4.13 Statistics vs the detector (inside the pool, Jan+Jul 2025: 9,093 rows, 57 extremes)

| method | AUC | precision in top 57 |
|---|---|---|
| `sched_dev` alone | 0.893 | 8.8% |
| per-airport z-scores (moments) | 0.945 | 17.5% |
| isolation forest, no labels | 0.968 | 38.6% |
| **supervised detector (v35)** | **0.982** | **70.2%** |

### 4.14 Stand occupancy (tested while writing this document)

Two aircraft cannot occupy a stand at once, so if the next arrival on-blocked at my stand before my takeoff, my taxi ≥ (my MVT − that on-block). Using `AOBT` (else `SCHED`) to identify "next":
- Active on 9.4% of departures; holds on 97.9%; active on ~21% of taxis > 30 min.
- **With AOBT present:** holds 99.7%, corr 0.512 — but by construction never exceeds `implied_taxi`, so it adds nothing.
- **With AOBT missing:** holds only 35.9%, corr 0.019 — unusable. Unmatched flights' stand records contradict the arrival stream.
- **Verdict: closed.** The second bullet is itself a data-quality fact about unmatched flights.

---

## 5. Where the error actually lives

### 5.1 The evaluation discovery that reshaped everything

For most of the project, local evaluation filtered to `y ≤ 10,800`. The scorer does not. Measured on Jan/Jul 2025 with the model of the time:

```
                                       Jan      Jul   combined
capped eval (what we had been quoting)  228.11   289.87    264.30
UNCAPPED eval (what the scorer sees)    396.30   433.25    417.29
extreme rows' share of total SSE        66.9%    55.3%
```

60 rows out of 344,336 more than doubled the RMSE. That led to the extreme detector (v4: 521 → 304).

### 5.2 Regime map at the current architecture (v38 eval, Jan+Jul 2025, uncapped)

Arms: A = v35-style, B = + sequence features (v38), C = + reliability features in the detector (v39). Eval ran in "fast" mode (lr 0.05) on a base with `min_data_in_leaf 20` — not v35's exact base (see §6.3), but the same base for all three arms.

| AOBT regime | rows | share of SSE | RMSE A | RMSE B | RMSE C |
|---|---|---|---|---|---|
| **AOBT missing** | 5,320 | **48.3%** | 1,877.6 | 1,878.6 | 1,865.6 |
| AOBT observed | 304,565 | 46.5% | 243.6 | 242.0 | 241.0 |
| AOBT back-filled | 34,451 | 5.3% | 243.7 | 242.1 | 242.1 |

| delay band (`sched_dev`) | rows | share | RMSE A |
|---|---|---|---|
| 10–30 min | 174,110 | 29.9% | 258.5 |
| 30–60 min | 95,085 | 26.8% | 331.1 |
| > 2 hours | 9,088 | 23.5% | 1,003.0 |
| 1–2 hours | 33,301 | 18.2% | 461.0 |
| < 10 min | 32,752 | 1.5% | 134.0 |

| airport | share of SSE | RMSE A |
|---|---|---|
| LFPG | 34.2% | 579.6 |
| LIRF | 29.4% | 656.3 |
| EGLL | 8.6% | 287.6 |
| LTFM | 8.5% | 266.6 |
| EHAM | 4.1% | 196.3 |
| LEBL | 4.0% | 230.7 (C: **256.2**) |
| EDDF / LEMD / LSZH / EDDM | 2.3–3.5% each | 180–206 |

LFPG's 34% is inflated by one row (§5.5).

### 5.3 What is inside the AOBT-missing 48%

87% of that regime's error (by a squared-distance proxy) sits in 51 rows: 47 recoverable extremes and 4 stale-BLOCK extremes. The detector catches most recoverable ones; the stale ones are undetectable.

### 5.4 Direction of error (July 2025, rebuilt v35-style model, RMSE 338.1)

| regime | rows | mean error (pred − truth) | share of SSE | systematic part |
|---|---|---|---|---|
| contaminated extreme | 42 | **−594** (too low) | 1.1% | 6% |
| stale-BLOCK extreme | 3 | −9,047 | 1.1% | 99% |
| contaminated, normal y, in pool | 38 | −2,950 | 2.1% | 72% |
| contaminated, normal y, outside pool | 7,017 | −178 | 9.2% | 11% |
| **clean, late (in pool)** | 5,776 | +78 | **32.9%** | 0% |
| clean, ordinary | 177,802 | +11 | 53.5% | 0% |

Contaminated rows are systematically under-predicted, but together they are ~13% of July error. On true extremes the detector works: **40 of 42 caught, mean error −77 s**; the 2 missed ones are off by ~−10,900 s each.

### 5.5 The 32 false alarms — about a fifth of July's error

Inside "clean, late": SSE with the detector 7.17e9 vs **base alone 2.96e9**. The detector's false alarms add **+4.21e9 = 19.3% of all July error**, and almost all of it comes from **32 rows** with p > 0.5, predicted at their `sched_dev` and wrong by **+11,344 s on average**. Without them July RMSE would fall from 338 to ~304. These are the "32 false positives" visible since the first detector run (July: 41 TP vs 32 FP at p > 0.1, precision 0.562, recall 0.911).

They cannot be separated from the 40 caught true extremes on anything observed:

| | 40 true extremes | 32 false alarms |
|---|---|---|
| airport | all LIRF | all LIRF |
| AOBT missing | 100% | 100% |
| stand contamination rate (median) | 0.185 | 0.145 |
| operator contamination rate | 0.043 | 0.043 |
| detector p (median) | 1.000 | 0.999 |
| `sched_dev` (median) | 16,948 s | 12,274 s |

At Rome, when AOBT is missing and the flight is hours late, it is roughly 56/44 whether the off-block was substituted. That is the unobservable event, measured precisely. Physically both groups are identical (the aircraft sat at the gate, then taxied normally); only the *record* differs.

**The one lever:** the detector is overconfident (p ≈ 1 on both). If the groups are indistinguishable, the squared-error-optimal p is near the true ratio, which would trade some error on true extremes for a much larger cut on false alarms — very roughly 4–9% of July SSE by back-of-envelope arithmetic. A previous calibration attempt lost, but it was fitted on the detector's own training predictions (overfit). **Out-of-fold calibration has never been tried.**

### 5.6 The Paris monster and the "−32 offset"

One January 2025 row: LFPG, `y = 84,240`, `sched_dev = 1,740` (BLOCK 2025-01-18 19:30, SCHED 01-19 18:25, MVT 01-19 18:54) — a stale-BLOCK row outside the pool. The model predicts a normal ~1,700 s and misses by ~82,500 s → **~6.8e9 SSE, about a third of January's total.** Remove it:

```
January    345.95  ->  274.6
combined   335.93  ->  305.3       official v35 = 303.46
```

For weeks we used a "local − 32 ≈ official" rule. **It was almost entirely this one row.** January 2025 contains a monster; the 2026 set evidently does not. Local evaluation should exclude stale-BLOCK rows (`y > 10,800` and `|sched_dev − y| > 60`) so local tracks official without a fudge.

### 5.7 What the leaders' scores imply (arithmetic, not speculation)

At 213, a team's **entire** error budget over 344,841 rows is 213² × 344,841 ≈ **1.56e10**. Our error on **ordinary observed-AOBT flights alone** is about 304,565 × 243.6² ≈ **1.81e10**. Even predicting every bad row perfectly, we would still sit above 213; we would need ordinary-flight RMSE ≤ ~227. **The leaders predict ordinary flights roughly 30 points better than we do.** The gap is not only the 15–35 extreme rows — an earlier claim to that effect was wrong (it rested on the −32 offset, which is one row). The organisers' email also suggests some top scores reflect tuning to Jan/Jul specifically; we cannot tell which.

---

## 6. Our pipeline

### 6.1 Feature engineering

**`12_queue_features.py`** (writes `dep_q_train/rank.parquet`): epoch columns `mvt_i, blk_i, sch_i, aobt_i, eobt_i`; surface queue at pushback (aircraft pushed back but not airborne, airport and runway level), pushes/takeoffs in prev 15/60 and next 15/30 min, arrivals landing prev 15/60 next 30 min, arrivals on the surface (via ARR on-block), Little's-law wait estimates, `rwys_active_this_hour`, `implied_taxi_aobt = mvt − aobt`, `has_aobt`, `pushback_delay = aobt − eobt`, `sched_dev = mvt − sch`, time features (hour, dow, month, doy, minute_of_day, cyclical). **Reference time for queues is AOBT (or SCHED when missing) — the noisy clock.** It sorts by `['apt','date','RUNWAY_mvt']`, which is why submissions must be reindexed onto the template.

**`15_weather.py`** (writes `dep_qw_*`): METAR from IEM ASOS, cached quarterly in `metar_cache/` (2024-12-28 → 2026-08-05), backward as-of join on pushback time with 2 h tolerance (no future weather). 28 columns: temp, dewpoint, spread, wind, gust, direction, visibility (m), ceiling (BKN/OVC only, NaN when unknown), present-weather flags from `wxcodes`/group-matched raw METAR (TS, SN, FZ, RA, FG, BR, GR, SHRA), LVP, de-icing risk, freezing, 3–6 h rolling persistence, METAR age, cross/headwind vs runway heading. `p01i` is US-only — never build precipitation from it. Coverage 99.96–100% per station.

### 6.2 Model architecture (v4 → v35)

```
base regressor   LightGBM on y <= 10,800 (training cap), predicts ordinary taxi
detector         LightGBM binary, trained ONLY on the candidate pool (sched_dev > 7,200),
                 label is_extreme = (y > 10,800), 6 inputs
blend            inside the pool:  pred = p · sched_dev + (1 − p) · base
                 outside the pool: pred = base
clip             [60, 172,800]  — never clip at 10,800 (that made extremes unwinnable)
```

Base features (75 present): categoricals `RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, AIRCRAFT_OPERATOR_flt, MARKET_SEGMENT_flt, WK_TBL_CAT_flt, FLIGHT_TYPE_flt, apt, ADES_flt`; time (8); `implied_taxi_aobt, has_aobt, pushback_delay`; target encodings fit on capped training rows only (`stand_unimp, rwy_stand_unimp, apt_rwy_unimp` = 10th-percentile taxi per stand / runway+stand / airport+runway with ≥30 rows; `rwy_stand_delay` = mean − p10; `stand_te, operator_te` = smoothed means, k=20); queue (24); weather (28); `sched_dev_clip` (clipped to [−3600, 10800]), `sched_dev_abs`, `sched_dev_pos`.

Detector features: `sched_dev_log, implied_taxi_aobt, has_aobt, imp_minus_sched, imp_is_odd, apt` (`imp_is_odd` = AOBT missing or implied taxi > 7200 or < 0).

Iteration count: early stopping on a 5% random slice (`default_rng(42)`), then refit on all capped rows with that count.

### 6.3 v35 exactly (verified from the file that produced it)

`24_v35.py` is byte-identical (`diff`) to `24_final-downloaded to-check.py`, which wrote v35.

```
BASE PARAMS   objective regression, learning_rate 0.04, num_leaves 255, min_data_in_leaf 60,
              feature_fraction 0.60, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0,
              seed / bagging_seed / feature_fraction_seed 42, num_threads -1
DET_PARAMS    binary, learning_rate 0.02, num_leaves 31, min_data_in_leaf 20,
              feature_fraction 0.9, bagging_fraction 0.9, bagging_freq 1, seed 42, is_unbalance True
DET_ROUNDS 400 · POOL_MIN 7200 · TRAIN_CAP 10800 · EARLY_STOP 100 · MAX_ROUNDS 3000 · HOLDOUT_FRAC 0.05
best_iteration 1783 · predictions above 10,800: 77
```

**So v35 = v4 + base `feature_fraction 0.60` + detector `learning_rate 0.02` (was 0.03).** The bulk sweep recommended three base changes (ff 0.60, min_data_in_leaf 20, lr 0.02); only ff reached the base. The lr change landed in the detector by accident, and min_data_in_leaf 20 was already the detector's value. Consequences: v35's +0.92 is not purely a bulk gain; **base `min_data_in_leaf 20` (+1.03 locally) and base `lr 0.02` (+0.56) have never reached the leaderboard**; and script versions that forced lr 0.02 / min_leaf 20 onto the base produced a different model (best_iteration 3000) and failed the reproduction gate. Fixed by using `V4_SRC`'s parameters verbatim.

### 6.4 Script inventory

| script | purpose | status |
|---|---|---|
| 12_queue_features.py | queue/time/AOBT features → dep_q_* | in use |
| 13_model.py | 3 forward folds, base LightGBM | superseded |
| 14_submit.py | **BROKEN**: set-wise ID check missed row order | do not use |
| 15_weather.py | METAR → dep_qw_* | in use |
| 16 / 17 / 18 | weather experiment; Jan/Jul seasonal CV; sched_dev + rollover audit | historical |
| 19 / 20 | global two-population mixture and its diagnosis | dead |
| 21_fix.py | template-order-gated submitter | pattern reused everywhere |
| 22 (21.py) | established the uncapped metric | historical |
| 23_extreme_detector.py | detector research | historical |
| **24_final.py** | **v4** | reference |
| **24_v35.py** | **v35, current incumbent** | reference |
| 25_v5_seeds.py | recoverable label + 3 seeds (v5) | scored 306.23 |
| 26_v6.py | three-way AOBT mixture + local eval | dead |
| 27_detector_sweep.py | 48-config sweep — the overfitting lesson | do not reuse |
| 28_v7_submit.py | swept config (v10) | scored 329.17 |
| 29_variants.py | re-implementation of v4 + knobs | **abandoned**: not v4-identical |
| 30_noise_floor.py | imports 24_final.py; seed noise floor + knob sweep | trusted harness |
| 31_bulk_sweep.py | 13 base configs vs in-run v4 | done (see §7.3) |
| 32_arr_outage.py | arrival-outage detector features | dead (−9.23) |
| 33_v36_contam.py | contamination label + reproduction gate | dead (−4.40) |
| 34_v37_routed.py | clean specialist routed by p_contam | dead (Jan negative) |
| **35_v38_v39_mechanism.py** | sequence features (v38), reliability detector (v39), regime map, gate | **ready to submit** |
| teammate 37/39/40/41 | separate pipeline (Open-Meteo weather, 62 features, airport overlays, log1p) | scored 307–358 |

---

## 7. Validation methodology — what works, what lies

### 7.1 The harness

Hold out a full month of 2025, train on the other eleven, evaluate **uncapped**, with encoders fit on training months only. Jan and Jul so far; **Feb and Jun must now be added** (final phase). Combined score weights months by their ranking-set row counts (Jan 152,709, Jul 192,111). `30_noise_floor.py` imports `24_final.py` directly — no reimplementation — and reproduced v4's local number to 0.02 (336.65 vs 336.59).

Schemes tried and rejected: single December holdout (wrong month); leave-one-month-out (optimistic: 254.95 vs a real 272.6 at the time); forward folds (wrong months; three of four validation pairs were never scored).

### 7.2 Noise floor

Same configuration, five detector seeds: combined 336.52–336.93, **spread 0.41**. Any local difference below ~0.4 is not a measurement. Note different scripts set seeds differently (some set `bagging_seed`/`feature_fraction_seed` explicitly, v4 sets only `seed`), so absolute local numbers differ between scripts by ~1–2 points; **only compare arms within one run.**

### 7.3 What transfers to the leaderboard and what does not

| change | local | official | transferred? |
|---|---|---|---|
| extreme detector (v3 → v4) | +80.7 | 521 → 304 | yes |
| bulk params (v4 → v35) | ff 0.60 alone +1.36 | 304.39 → 303.46 | yes, partly (also a detector lr change) |
| POOL_MIN 9000 | **+2.07 "REAL"** | **305.59 (worse)** | **no** |
| swept detector config | +12 | 329.17 (much worse) | **no** |

Rule: **bulk changes measured across 344k rows transfer; detector thresholds fitted to ~49 extreme events do not.** The 2025 extremes and the 2026 extremes sit in different places in the `sched_dev` distribution, and ~20 specific flights decide any pool comparison (between v4 and pool_9000, 20 rows carried 97% of all prediction change).

### 7.4 The bulk sweep (script 31), corrected

The script compared against a hard-coded reference (336.65) taken from a different script; the in-run v4 reference was 333.82, so every printed delta was inflated by 2.83 and every printed "REAL" verdict was wrong. Corrected against the in-run baseline:

| config | vs in-run v4 | verdict |
|---|---|---|
| feature_fraction 0.60 | **+1.36** | real |
| min_data_in_leaf 20 | **+1.03** | real |
| learning_rate 0.02 | +0.56 | marginal |
| callsign target encoding + 511 leaves | +0.29 | noise |
| lambda_l2 10 | +0.16 | noise |
| callsign target encoding | −0.05 | noise |
| num_leaves 511 / 127 | −0.12 / −0.18 | noise |
| feature_fraction 0.90 | −0.74 | worse |
| min_data_in_leaf 200 | −1.03 | worse |
| learning_rate 0.08 | −1.06 | worse |
| linear_tree | −1.52 | worse |

Direction: more regularisation via sampling helps; bigger/greedier models don't.

### 7.5 Detector-side knob sweep (script 30, phase 2; for the record — none transferred)

pool 9000 +2.07 (official: worse) · pool 10800 +0.78 · recoverable label +0.17 (noise) · hard blend t=0.20 −0.26 / t=0.50 −0.49 · detector rounds 150 −0.34 / 800 −0.19 · pool 5400 −298 (catastrophic: a few clean rows flagged at huge `sched_dev`).

---

## 8. Complete submission history

| ver | what it was | official |
|---|---|---|
| v1 | teammate's early CatBoost/XGBoost model (local ~272.6) — row order scrambled | 546.4895 |
| v2 | teammate's revision — same defect | 545.5379 |
| v3 | queue model, row order fixed | 521.4497 |
| **v4** | `24_final.py`: base + extreme detector (pool 7200, `is_extreme`, 6 features), soft blend | **304.3859** |
| v5 | recoverable label + 3 seeds (two changes) | 306.2316 |
| v8, v9 | identical score — duplicate/misaligned uploads | 548.437 |
| v10 | swept config: pool 14400, wide detector, isotonic, 3 seeds, recoverable (5 changes) | 329.169 (later overwritten in the bucket → 548.1375) |
| v11 | "baseline" from `29_variants.py` — corrupt upload | 533.3469 |
| v12 | calib (isotonic, in-sample), from 29_variants | 309.3505 |
| v13 | callsign encoding, from 29_variants | 306.9124 |
| v14 | pool 5400 train+apply, from 29_variants | 334.5881 |
| v15 | recoverable label — corrupt upload | 655.486 |
| v21 | pool 9000 — **row order broken** | 349.5348 (invalid) |
| v24 | pool 9000, from `24_final.py`, verified | 305.5878 |
| v31 | detector trained on pool 5400, applied at 7200 (Kaggle) | 335.7295 |
| v32 | teammate's detector package on our base: no `is_unbalance`, 15 leaves, 40 min_data, 200 rounds, train pool 5400 (Kaggle) | 328.296 |
| **v35** | **v4 + base ff 0.60 + detector lr 0.02** | **303.4643 ← best** |
| v36, v37 | contam label; routed specialist — killed locally, never submitted | — |
| v38, v39 | sequence features; + reliability detector — generated by script 35, pending | — |

`POOL_MIN` is closed: 5400 → 334.59, **7200 → 304.39**, 9000 → 305.59, 14400 → 329.17.

The teammate's separate pipeline scored 358 → 346 → 340 → 308 → 307 (never beat v4).

---

## 9. Tested and dead — do not retry without a new reason

| idea | result | why it failed |
|---|---|---|
| rolling realised-taxi regime | +0.54 | LIRF's median is flat all year; no regime to detect |
| upper-tail target encodings (p90 per stand, etc.) | +0.33 | rank #2 by gain, collinear with `rwy_stand_unimp` |
| AOBT − LOBT (ATFM signal) | ~0 | 0.876 correlated with `pushback_delay` |
| out-of-fold target encoding | −0.11 | p10 over 30+ rows barely moves |
| global soft mixture (p_contam everywhere) | −3.75 | injected bias into 183k clean rows to help 7k |
| three-way AOBT mixture | −12.70 | `implied_taxi` already the #2 feature; noisier duplicate |
| removing the training cap | −25.18 | extremes are recording errors; base shades everything up |
| stale-BLOCK reconstruction (3 hypotheses) | 0–1 of 22 | not copied from any neighbour |
| per-airport experts (for delta) | 277.6 vs 278.0 | airport-as-feature already enough |
| contamination label for the detector | −4.40 | 238 relabelled rows are only 12 in Jan+Jul; false alarms doubled |
| clean specialist routed by p_contam | −0.12, **Jan negative at every threshold** | +42 s "late-flight bias" was an artefact of a weakened subsample model |
| arrival-outage features in the detector | −9.23, seed spread 0.15 → 2.55 | deflated p on true extremes (0.88 → 0.74) |
| isotonic calibration (in-sample) | official +4.96 worse | fit on the detector's own training predictions |
| teammate's detector package | official 328.30 | wide training pool harmful on our base |
| wide training pool with `is_unbalance` | official 335.73 | doubling negatives doubles the positive weight → p inflates |
| log1p target (teammate) | worse | loses precision on ordinary rows |
| linear_tree | −1.52 | |
| moment / unsupervised outlier detection | P@57 39% vs detector 70% | finds the tail, not the contamination |
| stand-occupancy bound | redundant / unreliable | §4.14 |
| back-filled-AOBT flag | gain rank #78 | rows already predicted as well as observed ones |

---

## 10. Tested, positive, not yet on the leaderboard

- **v38** — v35 base + `gap_prev_tko, gap_next_tko, tko_before_mvt_{300,600,1200}` + `aobt_backfilled`, detector unchanged. **+1.03 combined, Jan +1.21, Jul +0.87** (both months, above noise). Measured on a base with min_leaf 20; real gain on v35's exact base may differ. Ready in script 35.
- **v39** — v38 + `stand_contam_rate, operator_contam_rate, aobt_backfilled` in the detector. +2.85 combined but **Jan −0.80**, Jul +5.95, and LEBL damaged (230.7 → 256.2). The premise (stand rate separates false alarms) was later refuted (§5.5). Optional at best.
- **Base `min_data_in_leaf 20`** (+1.03) and **base `lr 0.02`** (+0.56) — never actually applied to the base (§6.3).

---

## 11. Untested leads, ranked by evidence × size × how well they should generalise

1. **Multi-anchor features and a residual target (the main thesis, §12).** Add `MVT − EOBT_1`, `MVT − IOBT`, `MVT − LOBT` explicitly (the strongest single clue, 0.551, is not a column we have — it exists only as `implied_taxi + pushback_delay`, a sum trees cannot form in one split), plus pairwise differences between the anchors and an "anchor agreement" spread. Add clock-offset encodings: training means of `BLOCK − AOBT` per (airport, stand), per (airport, operator), per (airport, hour), and `implied_taxi` corrected by them. Then try training on `y − implied_taxi` for rows with an observed AOBT (residual target), adding it back at prediction. Bulk change, measured on 344k rows — the category that transfers. Size unknown.
2. **Seed averaging for the final.** Average 3–5 seeds of the base and of the detector. Never cleanly tested on v35 (v5 bundled it with a label change). With one final submission, lower variance is worth having for its own sake.
3. **Base `min_data_in_leaf 20`** (+1.03 locally, never applied) and **base `lr 0.02`** (+0.56).
4. **Out-of-fold calibrated detector probability**, aimed at the 32 false alarms (19% of July error). Fit the mapping from out-of-fold p to observed extreme rate across K folds of the 2025 pool; never on the detector's own training predictions. Detector change → judge on four months and expect noise.
5. **2026 drift from the ranking file's own arrivals.** Daily per-airport arrival taxi-in level (raw and relative to the airport's 2025 norm). Before modelling it, cheaply check per-airport stability 2025 → 2026 of: arrival contamination rate (BLOCK = SCHED on ARR rows), arrival clock relationship, arrival taxi-in. If LIRF's 2026 arrival contamination is lower, its departure extremes are probably rarer too — that alone would explain part of why 2026 scores better than 2025.
6. **A CatBoost model blended with LightGBM.** Never tried. Handles the high-cardinality categoricals (≈1,700 stands, 1,560 destinations, operators) differently; blends of different learners typically give a few percent. Expensive on this laptop.

---

## 12. Synthesis — re-reading everything from the top

### 12.1 What the problem actually is

Not "how long does taxiing take." It is: **estimate the number an airport system wrote into its off-block field**, given the takeoff time to the second and several other systems' records of roughly the same moment. It is a record-reconciliation problem with three kinds of evidence:

- **Other records of the off-block moment**, each with its own resolution, bias and failure modes: `AOBT_3` (minute-rounded; sometimes synthetic, computed as takeoff minus a fixed allowance; airport-specific offset up to +296 s at LTFM), `EOBT_1`, `IOBT`, `LOBT` (plans and updates), `SCHED`.
- **Physical sequencing on the precise clock**: the exact runway order of takeoffs. (Stand occupancy was checked and adds nothing; §4.14.)
- **Knowledge of when the recorder fails**: schedule substitution during feed outages, persistent by stand; stale records; unmatched flights.

### 12.2 Where the remaining gap must be

The arithmetic in §5.7 is the hardest fact in this document: a 213 team spends less total error than we spend on ordinary flights alone. So most of the gap to the top is on **ordinary flights with an observed AOBT** — 88% of rows, RMSE ~244 — not only in the extremes.

On those rows, `taxi = implied_taxi − delta`, and delta (the disagreement between two clocks) carries almost all of the variance. What drives delta is `AOBT`'s own error (corr −0.42 to −0.64 with `AOBT − EOBT`), not traffic (|r| < 0.17). Our congestion features were always going to be weak on this part. The only new features that helped recently were *clock and sequence* features, not traffic features.

### 12.3 The thesis

We give the tree each off-block record as an unrelated column. What it actually has to learn is something like:

```
taxi  ≈  MVT  −  [ reliability-weighted blend of AOBT, EOBT, IOBT, LOBT, SCHED ]  −  offset(airport, stand, operator)
```

That is an additive structure over continuous anchors with context-dependent weights and offsets. Trees can approximate it, but only piecewise, one split at a time. Three facts point here:

1. A *plan* (`MVT − EOBT`, 0.551) correlates with taxi better than the *actual* (`MVT − AOBT`, 0.533). No single record is best, so a fusion of them should beat any one — and **we have never given the model `MVT − EOBT` directly.**
2. `AOBT`'s reliability varies sharply and predictably: median absolute error 120–180 s at most airports but 356 s at LTFM; ~125 s in the morning vs 235–300 s at night; 124 s for low-cost carriers vs 236 s for business aviation. Those are learnable reliability weights.
3. Delta has stable, per-airport (and plausibly per-stand, per-operator — handlers report differently) offsets.

**This is a hypothesis.** It is the best-supported explanation of the 30-point gap on ordinary flights we have, it is a bulk change (the kind that transfers), and it is built from mechanisms rather than from the two leaderboard months — which is what the final phase rewards. Lead §11.1 is how to test it.

### 12.4 The extremes are an information limit, with one lever left

The detector finds true extremes almost perfectly (40/42, mean error −77 s). What it cannot do is separate them from clean late flights at Rome that look identical on every observed column (§5.5). The remaining lever is honest probability, not a new feature: out-of-fold calibration. And remember the stale-BLOCK rows are unpredictable — the Paris monster is one; any equivalent in Feb/Jun 2026 will simply cost what it costs.

### 12.5 Generalisation — what the final phase changes

Things fitted to the Jan/Jul leaderboard (`POOL_MIN`, detector knobs) are now liabilities until re-verified on Feb/Jun. Things derived from the data-generating process — the contamination mechanism, the sched_dev identity, clock features, sequence features, honest calibration — should hold. If some leaders tuned to Jan/Jul (the organisers say they saw signs of it), the four-month ranking may reshuffle; a model validated on four months is the best preparation.

### 12.6 The tension the team must resolve

The thesis leans harder on `AOBT_3_flt`, the column the brief warns about, and the final phase includes code review. Options: keep AOBT (document it honestly, and argue that the ranking file provides it); or build an AOBT-free model (others report a ~300 floor) — the same fusion idea still applies with `EOBT/IOBT/SCHED` anchors, minus LOBT. **Decide this before investing the last days.**

### 12.7 Questions still open

- Why does `EOBT_1` beat `AOBT_3`? Is `EOBT_1` updated close to pushback (airline delay messages)? What does the distribution of `AOBT − EOBT_1` look like near zero?
- Is there a per-operator reporting convention in delta (handlers reporting "doors closed" vs "pushback start")?
- Did any airport's recording system change between 2025 and 2026? (Check with arrivals: contamination, clock relationship.)
- Do Feb/Jun 2025 extremes look like Jan/Jul (Rome, AOBT missing, in the pool)?
- What do the published repositories of teams at 213–235 use as features?

---

## 13. Plan for the final phase (deadline 2026-10-14T10:00Z)

1. **Submit v38** with the fixed script 35 (gate must print `best_iteration 1783`, `max diff 0.000000`, `above 10800: 77`). Verify order. Upload. Record the score.
2. **Decide the AOBT policy** (§12.6).
3. **Build the four-month harness**: hold out Jan, Feb, Jun, Jul 2025 separately (train on the other eleven months each time), uncapped, reporting each month, the Jan/Jul and Feb/Jun pairs, and the combined score — plus a version that excludes stale-BLOCK rows. Re-measure v35, v38, v39 on it. This also re-validates `POOL_MIN 7200` on Feb/Jun.
4. **Cheap 2026 stability checks** from arrival rows (§11.5).
5. **Bulk experiments, one at a time, on the four-month harness**: anchors and residual target (§11.1); min_data_in_leaf 20; seed averaging; drift features. Keep what is positive in all four months.
6. **Detector**: out-of-fold calibration, judged on four months.
7. **When the four-month ranking file arrives**: regenerate `dep_q_rank` (12) and `dep_qw_rank` (15) from it, confirm METAR coverage for Feb/Jun 2026, update the template row count in every check, run the final pipeline with the reproduction gate, verify, and submit **once**.
8. **Documentation and GPLv3 repository** — the final ranking includes code and documentation review.

---

## 14. Runbook

```fish
# evaluate (Jan+Jul, fast mode, ~1-1.5 h) — read A/B/C comparisons, not levels
env V4_SRC=DATA_ANALYTICS/24_v35.py python -u DATA_ANALYTICS/35_v38_v39_mechanism.py eval fast 2>&1 | tee v38_eval.log

# build v38 + v39 behind the reproduction gate (~1 h)
env V4_SRC=DATA_ANALYTICS/24_v35.py python -u DATA_ANALYTICS/35_v38_v39_mechanism.py submit 2>&1 | tee v38_submit.log

# verify, then upload
python -c 'import pandas as pd; t=pd.read_parquet("data/raw/submitting.parquet"); s=pd.read_parquet("logical-ladder_v38.parquet"); print((s.MVT_ID_mvt.values==t.MVT_ID_mvt.values).all(), len(s), (s.TAXITIME_SEC_mvt>10800).sum())'
mcli cp logical-ladder_v38.parquet prc/prc-2026-logical-ladder/
mcli cat prc/prc-2026-logical-ladder/logical-ladder_v38.parquet_result.json
```

Before writing any new experiment script:
- import the incumbent's file (`V4_SRC`) rather than re-implementing it (`29_variants.py` re-implemented v4 and silently produced a different model);
- compare arms inside one run, never against a constant from another script;
- for a new submission, refit the incumbent and pass the reproduction gate first;
- if a script exits with "does NOT reproduce", read its diagnosis (base vs detector) — don't work around it.

---

## 15. Glossary

| term | meaning |
|---|---|
| `implied_taxi` | `MVT − AOBT_3_flt`; the AOBT-based taxi estimate |
| delta | `BLOCK − AOBT`, disagreement between the airport's and NM's clocks |
| `sched_dev` | `MVT − SCHED`; equals the target exactly on contaminated rows |
| `pushback_delay` | `AOBT − EOBT_1` (teammate's version: `AOBT − SCHED`) |
| contaminated | abs(BLOCK − SCHED) ≤ 5 s: schedule written instead of the off-block |
| extreme | `y > 10,800` |
| recoverable / stale-BLOCK | extremes whose target `sched_dev` reproduces / does not |
| candidate pool | rows with `sched_dev > POOL_MIN` (7,200); only these can be flagged |
| detector `p` | probability a pool row is extreme; blend `p·sched_dev + (1−p)·base` |
| unmatched flight | movement with no Network Manager record; all `_flt` columns missing |
| back-filled AOBT | `implied_taxi` equal to an airport's fixed allowance (LTFM 1200 s…) |
| reproduction gate | refit the incumbent, compare to its saved file, refuse unless identical |
| noise floor | 0.41 RMSE: spread of local scores across detector seeds |
| bulk vs detector change | base-model change (transfers) vs pool/detector change (does not, reliably) |
