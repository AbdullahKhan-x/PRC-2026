# PRC Data Challenge 2026: taxi-out time, team `logical-ladder`

GPLv3 solution to the EUROCONTROL Performance Review Commission (PRC) / OpenSky Network Data Challenge 2026:
predict `TAXITIME_SEC_mvt` for departures at 10 European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF,
LSZH, LTFM). The metric is **uncapped RMSE**.

| version | what changed | official RMSE (Jan + Jul 2026) |
|---|---|---|
| v35 | LightGBM base + extreme-value ("schedule copy") detector | 303.46 |
| v105 | + post-rules from measured recording mechanisms + context features | 284.68 |
| v107 | + flight-plan gaps, live arrival-copy rates, clock-offset encodings (ctx2id) | 277.22 |
| v110 | + feature-subset member + per-airport models | 274.18 |
| v112 | + residual-target member + record-provenance member | 273.33 |
| v114 | + **ADS-B** off-block witness member (adsb.lol ground traces) | 269.37 |
| **v116** | + EUROCONTROL daily delay series + OPDI flight lists in the ADS-B member | **268.60** |

Every change was first validated on held-out months of 2025 (Jan, Feb, Jun, Jul) and only then submitted.
Full history, including what failed: [`docs/submissions.md`](docs/submissions.md) and
[`docs/project_context.md`](docs/project_context.md) (measured mechanisms and the reasoning behind each component).

## 1. What is actually being predicted

`TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` on every 2025 departure. The take-off time is given, so
the task is to reconstruct the off-block timestamp **as the airport's system recorded it**. The 2025 data shows:

* **Schedule copy.** On 4.3 % of departures (16.5 % at Rome) `BLOCK = SCHED`: when the off-block message is lost
  the schedule is written instead, so a flight that left 5 h late has a recorded "taxi" of 5 h.
* **Two clocks.** The Network Manager's `AOBT_3_flt` is minute-rounded, sometimes back-computed, and differs from
  the airport's value by an airport-specific offset (−66 s at LIRF to +296 s at LTFM); this disagreement carries
  most of the error on ordinary flights.
* **LOBT window.** On all 2,062,577 departures of 2025 with a `LOBT_flt`, `|BLOCK − LOBT| ≤ 3606 s`.
* **Rome date stamp.** Some Rome flights delayed past local midnight are stamped with the scheduled date (+86,400 s).

## 2. Model (v116)

```
features   queue/surface counts, IEM METAR weather, all gaps between MVT/AOBT/EOBT/IOBT/LOBT/SCHED,
           neighbouring departures' (MVT − AOBT) and arrivals' taxi-in right now, exact runway take-off sequence,
           per-stand/operator/runway clock-offset and schedule-copy-rate encodings (fitted on training rows only),
           ADS-B movement start / first ground contact (adsb.lol), EUROCONTROL daily delay series, OPDI rotations
members    LightGBM regressors on 2025 departures with y ≤ 3 h:
           ctx2id, ctx2rep (no NM operator/destination), ctx2res (target y − (MVT − AOBT)), ctx3 (+ record-provenance
           features), kae (+ ADS-B + external data), 10 per-airport models
blend      g = 0.5 · [0.75 · mean(ctx2id, ctx2rep, ctx2res) + 0.25 · ctx3] + 0.5 · kae;  base = 0.7 g + 0.3 per-airport
           (flights > 2 h late keep g)
detector   LightGBM classifier for "schedule copy" among flights > 2 h late; pred = p · (MVT − SCHED) + (1 − p) · base
rules      A  LOBT window (p = 0 when a copy is impossible; projection onto [MVT − LOBT ∓ 3606])
           C  Rome, no NM record, 3–5.5 h late: measured 2025 copy rate replaces the detector
           D  no NM record outside Rome: p = 0
```

## 3. Setup

Python ≥ 3.11 (developed on 3.14, Linux; works on Windows). About 8 GB RAM; ~45 GB free disk if ADS-B is kept locally.

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate        Windows (PowerShell):  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Challenge data is credentialed and **not** in this repo: put the `credentials.json` you received from the organisers in
the repo root (it is git-ignored), then `python download.py` (writes `data/raw/`).

## 4. Running

```bash
python run_all.py --list          # steps and their status
python run_all.py                 # everything that is missing, in order
python run_all.py --only members  # a single step
```

| step | script(s) | output |
|---|---|---|
| download | `download.py` (needs `credentials.json`), `download_external.py` | `data/raw/`, `data/ext/` |
| metar | `DATA_ANALYTICS/15_weather.py` (`download_all`) | `metar_cache/` |
| base | `DATA_ANALYTICS/10_build_base_features.py train / rank` | `dep_qw_train.parquet`, `dep_qw_rank.parquet` |
| v35 | `24_v35.py`, `39_repro_v35_parts.py` | detector + `candidates/v35_parts.parquet` |
| context | `42`, `42b`, `42f`, `42d` | `ctxall_*`, `ctx3_*`, `ext_*` |
| adsb | `62_adsb_features.py` (needs ADS-B point files, see below) | `adsb_*`, `ctxae_*` |
| members | `44_train_full.py` (one run per member) | `candidates/parts_*.parquet`, models |
| blend | `53b_make_blend4.py`, `40_postrules.py` | `candidates/logical-ladder_v116.parquet` |
| final | `55_final_v116.py final_ranking.parquet final_submitting.parquet OUT` | 4-month final submission |

**ADS-B point files.** The adsb.lol archives are 3–4 GB per day, so they are filtered on Kaggle, where downloads are fast:
`python kaggle/make_adsb_kernels.py A B C D` (needs `pip install kaggle` and a Kaggle API token), then download each
notebook's output into `data/adsb/<period>/` and unzip it. `DATA_ANALYTICS/60_adsb_extract.py` does the same filtering
locally for a single day (`python 60_adsb_extract.py OUT.parquet day.tar.aa day.tar.ab`).

**Validation harness.** `41_oof_harness.py MONTHS` holds out whole 2025 months (train on the other eleven) and scores
the uncapped target; `43_eval_postrules.py TAG` applies the rules with every rate re-estimated without the scored
month. Compare variants only within one environment: LightGBM results depend slightly on the thread count.

**Reproducibility.** `55_final_v116.py` rebuilds every feature from a raw ranking file; run on `data/raw/ranking.parquet`
it reproduces our v116 submission exactly (0 differing rows) when given our trained model files.

## 5. External data (all openly licensed, attribution)

| source | use | licence |
|---|---|---|
| Iowa Environmental Mesonet ASOS/METAR | weather at the 10 airports | free use with attribution |
| adsb.lol `globe_history` | ground traces → ADS-B off-block witness | ODbL 1.0 |
| EUROCONTROL daily airport series (ATFM slot adherence, pre-departure delays) | daily airport state | free use with attribution |
| OPDI v0.0.2 flight lists (PRC + OpenSky Network) | airframe rotation, first contact | open data with attribution |

No OpenSky Trino / state-vector data and no 2026 ground truth are used. The leaderboard was used only to score
changes that had already been validated on 2025.

## 6. On `AOBT_3_flt` and `LOBT_flt`

Both are provided by the organisers in the ranking file and are used as retrospective inputs (the take-off time is
also given). This is a **reconstruction** model of the recorded taxi-out, not a pre-departure forecast.

## License

GNU GPL v3, see `LICENSE`.
