# PRC Data Challenge 2026: taxi-out time, team `logical-ladder`

GPLv3 solution to the EUROCONTROL Performance Review Commission (PRC) / OpenSky Network Data Challenge 2026:
predict `TAXITIME_SEC_mvt` for departures at 10 European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF,
LSZH, LTFM). The metric is **uncapped RMSE**.


**Result:** ranked **45th of 250** teams (best RMSE **247.02**) the top team from Pakistan (1 of 2 Pakistani teams). Official final ranking pending.


| version | what changed | official RMSE (Jan + Jul 2026) |
|---|---|---|
| v35 | LightGBM base + extreme-value ("schedule copy") detector | 303.46 |
| v105 | + post-rules from measured recording mechanisms + context features | 284.68 |
| v107 | + flight-plan gaps, live arrival-copy rates, clock-offset encodings (ctx2id) | 277.22 |
| v110 | + feature-subset member + per-airport models | 274.18 |
| v112 | + residual-target member + record-provenance member | 273.33 |
| v114 | + **ADS-B** off-block witness member (adsb.lol ground traces) | 269.37 |
| v116 | + EUROCONTROL daily delay series + OPDI flight lists in the ADS-B member | 268.60 |
| v126 | every member with ADS-B + external data, ADS-B for all 12 months of 2025, 3 seeds per member, per-airport layer retrained with ADS-B | 262.39 |
| v127 | v126 + record-provenance member with ADS-B | 261.77 |
| v133 | v126 + rule G (Rome schedule copies: copy rate by flight-code format and airline × destination) | 259.17 |
| v137 | geometry (ADS-B ground-track) LightGBM members + ctx3ae12 member + 0.5 × CatBoost (3 seeds) | 253.24 |
| v141 | + rules T/P (twin-arrival LOBT window, LFPG duplicate records) and DD (EUROCONTROL daily delay totals) | 251.54 |
| v143 | geometry CatBoost (2 seeds) replaces plain CatBoost | 249.84 |
| v151 | + FL (Rome flight-number copy habit), DDG (adaptive daily-delay guard), NX residual corrector on all airports, K (LIRF monthly count), S′ (Rome date-wrap expectation) | 248.17 |
| v155 | 5 CatBoost seeds, CAP (LEMD/LEBL ≤ 3,604 s), guarded monthly-mean rule M | 247.31 |
| v157 | + rule ADSB (blend toward the direct ADS-B taxi observation, grade-A rows) | 247.24 |
| **v159** | + linear-tree leaves in the NX corrector (twin of the **final submission**, see §2b) | **247.02** |

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

## 2. Model (v133)

```
features   queue/surface counts, IEM METAR weather, all gaps between MVT/AOBT/EOBT/IOBT/LOBT/SCHED,
           neighbouring departures' (MVT − AOBT) and arrivals' taxi-in right now, exact runway take-off sequence,
           per-stand/operator/runway clock-offset and schedule-copy-rate encodings (fitted on training rows only),
           ADS-B movement start / first ground contact (adsb.lol, all twelve months of 2025),
           EUROCONTROL daily delay series, OPDI rotations
members    LightGBM regressors on 2025 departures with y ≤ 3 h, 3 seeds each (42, 2, 3), all with ADS-B + external data:
           kae12 (all features), krepae12 (without NM operator/destination), kresae12 (target y − (MVT − AOBT)),
           kae12apt (10 per-airport models);  ctx2id (no ADS-B) is kept only as the clean anchor of rule G
blend      g = mean(kae12, krepae12, kresae12);  base = 0.7 g + 0.3 kae12apt  (flights > 2 h late keep g)
detector   LightGBM classifier for "schedule copy" among flights > 2 h late; pred = p · (MVT − SCHED) + (1 − p) · base
rules      A  LOBT window (p = 0 when a copy is impossible; projection onto [MVT − LOBT ∓ 3606])
           D  no NM record outside Rome: p = 0
           G  Rome, no NM record, 3–5.5 h late: P(copy) = mean of the 2025 copy rate for the flight-code format
              (long / odd prefixes such as ITYTY680 come from another feed and are copies more often) and the
              airline × destination rate;  pred = P · (MVT − SCHED) + (1 − P) · ctx2id base   (replaces rule C)
           S  (v134, for the final file) Rome, no NM record, scheduled and actual local dates differ, 65–100 ks late:
              BLOCK is stamped with the scheduled date (+86,400 s) -> 0.65 weight on 86,400 · days + 1,000 s
```
`recipes/v133.json` / `recipes/v134.json` describe the blend and rules for `55b_final_recipe.py`.

## 2b. Final submission (14 Oct 2026)

**Final submission = v163**, uploaded successfully as `logical-ladder_final.parquet`. It uses recipe `recipes/v158.json`, followed by the ADS-B post-rule and Rome date-wrap hedge. The NX corrector stacks an averaged sequence-model bag of two bi-GRU models, C and D. The final artifact was validated for row count, template order, finite predictions, minimum values and duplicates.

The sequence models use neighbouring departure and arrival movements, including timing, delay, runway and flight attributes, to predict taxi-out directly. Their predictions provide complementary information to the existing model through the NX corrector. The final C+D bag was selected over the C+D+F alternative because F's additional held-out gain was only 0.057 RMSE, below the 0.10 threshold for replacing the simpler bag.

The Jan/Jul 2026 score of **247.02 belongs to v159**, the earlier twin of the v161 submission—not v163. The final v163 leaderboard score has not been recorded here.

Full experiment history, including unsuccessful alternatives: `docs/submissions.md`.

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
| adsb | `62_adsb_features.py` (needs ADS-B point files for all 2025 days, see below) | `adsb_*`, `ctxae12_*` |
| members | `44_train_full.py` (one run per member, `SEEDS=42,2,3`; `APT_ONLY=1` for kae12apt) | `candidates/parts_*.parquet`, models |
| blend | `53c_make_blend_ads3.py` (`APT_PURE=kae12apt`), `40_postrules.py ADG` | `candidates/logical-ladder_v133.parquet` |
| final | `55b_final_recipe.py recipes/v134.json final_ranking.parquet final_submitting.parquet OUT` | 4-month final submission |

**Sequence model (v162 candidate).** `python DATA_ANALYTICS/72_seq_rows.py` (held-out p + label-free inputs →
`candidates/seq/seq_rows.parquet`), run `DATA_ANALYTICS/70_seq_direct_kernel.py` on a GPU with the raw training months and
`seq_rows.parquet` as inputs (writes `dq_*.parquet` = held-out q and `dseq_*.pt` = full fit), score the stacking with
`73_seq_stack_eval.py dq_*.parquet`, write 2026 q with `OUT_Q=q_final.parquet python DATA_ANALYTICS/71_seq_apply.py dseq_*.pt w0.json BASE OUT`
(`w0.json` = `{"global": 0}`), then rebuild with `NXQ_TRAIN=dq_*.parquet NXQ_FINAL=q_final.parquet python DATA_ANALYTICS/55b_final_recipe.py recipes/v158.json ...`
followed by `56a_adsb_post.py` and `56b_wraphedge_post.py`.

**ADS-B point files.** The adsb.lol archives are 3–4 GB per day, so they are filtered on Kaggle, where downloads are fast:
`python kaggle/make_adsb_kernels.py A B C D E G H I` (A/B = 2026, C–I = all of 2025; needs `pip install kaggle` and a Kaggle API token), then download each
notebook's output into `data/adsb/<period>/` and unzip it. `DATA_ANALYTICS/60_adsb_extract.py` does the same filtering
locally for a single day (`python 60_adsb_extract.py OUT.parquet day.tar.aa day.tar.ab`).

**Validation harness.** `41_oof_harness.py MONTHS` holds out whole 2025 months (train on the other eleven) and scores
the uncapped target; `43_eval_postrules.py TAG` applies the rules with every rate re-estimated without the scored
month. Compare variants only within one environment: LightGBM results depend slightly on the thread count.

**Reproducibility.** `55b_final_recipe.py RECIPE.json RANKING.parquet TEMPLATE.parquet OUT.parquet` rebuilds every
feature from a raw ranking file (the organisers jitter the take-off times between exports, so nothing is reused); run on
`data/raw/ranking.parquet` it reproduces our leaderboard files exactly (0 differing rows; checked for v126, v131 and v134)
when given our trained model files. `55_final_v116.py` is the same for the earlier v116 recipe.

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
