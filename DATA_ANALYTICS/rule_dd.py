"""
Rule DD (daily delay constraint). EUROCONTROL publishes per airport and UTC off-block date the total pre-departure delay
DLY_ALL_PRE_2 (minutes) = sum over that day's departures of max(0, BLOCK - SCHED)   (2025: corr 0.99989, 95 % of days within 1 %).
With d = MVT - SCHED known, the gap between the published total and the total implied by our predictions equals the summed
error of that day's delayed flights. Spread the gap over the day's delayed non-pool rows (d - pred > 0, d <= 2 h),
weighted by pred, each row's shift capped at CAP s.
Held-out 2025 (s6cat50 + ADGS): 289.36 -> 288.45 (-0.91; Jan -0.43 / Feb -2.21 / Jun -0.49 / Jul -0.97).
Source: ansperformance.eu all_pre_departure_delays_{YEAR}.csv (data/ext/), already used as daily features (42d).
"""
import os, glob
import numpy as np, pandas as pd

CAP = 60.0


def load_daily(D):
    x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'data/ext/all_pre_departure_delays_*.csv')))])
    x = x[x.DLY_ALL_PRE_2.notna()]
    return {(a, t): v * 60.0 for a, t, v in zip(x.APT_ICAO.values, x.FLT_DATE.values, x.DLY_ALL_PRE_2.values)}


def load_counts(D):
    x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'data/ext/all_pre_departure_delays_*.csv')))])
    x = x[x.DLY_ALL_PRE_2.notna()]
    return {(a, t): v for a, t, v in zip(x.APT_ICAO.values, x.FLT_DATE.values, x.FLT_DEP_1.values)}


def apply(pred, mvt, sd, apt, daily, cap=CAP, elig_max=7200, counts=None, cmin=0.95, adapt=None):
    """counts (load_counts): partial-day guard, skip airport-days where our departures < cmin x published FLT_DEP_1
    (month-file edge days). adapt: per-day cap = max(cap, adapt x |gap| / n eligible) (disruption / snow days).
    'DDG' = elig_max 1e9 + guard 0.95 + adapt 0.5: held-out vs DDA −0.98 (−0.01/−5.61/+0.01/−0.03)."""
    """pred, mvt (epoch s), sd (MVT - SCHED), apt (ADEP): arrays over ALL departures of the file. Returns (pred, shift)."""
    day = pd.to_datetime(mvt - pred, unit='s', utc=True).strftime('%Y-%m-%d').values
    k = pd.Series([f'{a}|{t}' for a, t in zip(apt, day)])
    dl = np.maximum(0.0, sd - pred)
    elig = (dl > 0) & (sd <= elig_max)   # elig_max 1e9 ('DDA'): held-out −0.11/−0.69/−0.06/−0.31 vs 7200
    obs = np.array([daily.get((a, t), np.nan) for a, t in zip(apt, day)])
    imp = pd.Series(dl).groupby(k.values).transform('sum').values
    w = np.where(elig, pred, 0.0)
    W = pd.Series(w).groupby(k.values).transform('sum').values
    gap = np.where(np.isnan(obs), 0.0, obs - imp)
    if counts is not None:
        nours = pd.Series(np.ones(len(pred))).groupby(k.values).transform('sum').values
        npub = np.array([counts.get((a, t), np.nan) for a, t in zip(apt, day)])
        gap = np.where(nours >= cmin * np.nan_to_num(npub, nan=0.0), gap, 0.0)
    if adapt is not None:
        n = pd.Series(elig.astype(float)).groupby(k.values).transform('sum').values
        cap = np.maximum(cap, adapt * np.abs(gap) / np.maximum(n, 1))
    s = np.clip(gap * w / np.where(W > 0, W, 1.0), -cap, cap)
    s = np.where(elig, s, 0.0)
    return np.clip(pred - s, 60, None), s
