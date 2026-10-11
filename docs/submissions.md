# Submission history (official RMSE, January + July 2026)

All submissions are listed, including failed ones. "Order bug" means the file had the right IDs
in the wrong row order; the scorer aligns by position. Every submission from v4 onward is
written behind an element-wise template-order gate.

| version | what it was | official RMSE |
|---|---|---|
| v1, v2 | early CatBoost / XGBoost models (order bug) | 546.49, 545.54 |
| v3 | queue-feature LightGBM, order fixed, predictions capped at 3 h | 521.45 |
| **v4** | + extreme detector: schedule-copy rows among flights > 2 h late, soft blend | **304.39** |
| v5 | recoverable label + 3 seeds | 306.23 |
| v8, v9, v11, v15, v18, v20, v41 | duplicate / misaligned / corrupt uploads | 533-655 |
| v10 | detector configuration swept on 2025 (did not transfer; the bucket file was later overwritten, so its result now reads 548.14) | 329.17 |
| v12 | in-sample isotonic calibration of the detector | 309.35 |
| v13 | callsign target encoding | 306.91 |
| v14 | candidate pool from 1.5 h late | 334.59 |
| v24 | candidate pool from 2.5 h late | 305.59 |
| v31, v32 | wider detector training pools | 335.73, 328.30 |
| v19, v22, v23, v25-v27, v29, v30, v34, v40, v42, v43 | teammate pipeline variants | 307.9-358.7 |
| **v35** | v4 + base `feature_fraction` 0.60 + detector learning rate 0.02 | **303.46** |
| **v101** | v35 + rule A (LOBT window) | **299.13** |
| v102 | v101 + Rome 24 h mixture (rejected rule) | 299.59 |
| **v103** | v101 + rule C (Rome copy rate, 3-5.5 h late) | **291.21** |
| **v104** | v103 + rule D (no NM record outside Rome: p = 0) | **291.12** |
| **v105** | v104 rules on a base retrained with the context features | **284.68** |
| v106 | (rejected upload: daily limit) | - |
| v107 | v105 rules on the ctx2id base (flight-plan gaps, live arrival-copy rates, clock-offset encodings) | 277.22 |
| v110 | 0.5 ctx2id + 0.5 ctx2rep (no NM operator/destination), then 0.3 per-airport models | 274.18 |
| v112 | + residual-target member (ctx2res) and record-provenance member (ctx3) | 273.33 |
| v114 | + ADS-B member (adsb.lol ground traces, 4 months of 2025) | 269.37 |
| **v116** | ADS-B member with EUROCONTROL daily series + OPDI flight lists | **268.60** |
| **v126** | every member with ADS-B + external data, ADS-B for all 12 months of 2025, 3 seeds, per-airport layer with ADS-B | **262.39** |
| v127 | v126 + record-provenance member with ADS-B (ctx3ae12) | 261.77 |
| **v133** | v126 with rule G instead of C (Rome copy rate by flight-code format and airline × destination, ctx2id anchor; 47 rows) | **259.17** |

Lessons that shaped the method:

* **Changes measured on all 344k rows transferred to the leaderboard.** Changes driven by the
  ~50 extreme rows of the 2025 holdout did not.
* **Every rule here comes from a 2025 mechanism** (the LOBT bound, the schedule copy, the Rome
  date stamp) and was checked on held-out 2025 months before submission. The one rule that
  lost officially (v102) is reported and was removed.
