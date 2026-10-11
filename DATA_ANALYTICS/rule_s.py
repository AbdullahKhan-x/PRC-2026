"""
Rule S (= B', Rome date stamps): Rome, no NM record, scheduled and actual LOCAL dates differ, 65,000 < MVT - SCHED <= 100,000 s.
  BLOCK is sometimes logged with the scheduled date + actual time of day -> taxi = 86,400 * cross + normal taxi.
  2025: 6/6 rows in this band were stamps (LOMO -1.15e9, 4/4 months); held-out stack (Jan/Feb/Jun/Jul): -1.5 RMSE.
  pred = 0.65 * (86,400 * cross + 1,000) + 0.35 * pred      (hedged: rule B on the 55-65k band lost on the leaderboard)
"""
import numpy as np, pandas as pd

W, T_CLEAN, LO, HI = 0.65, 1000.0, 65000, 100000


def cross_days(mvt_utc, sched_utc, tz='Europe/Rome'):
    loc = lambda s: pd.to_datetime(pd.Series(s), utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    return (loc(mvt_utc) - loc(sched_utc)).dt.days.values


def apply(pred, sd, rome, cross):
    m = rome & (cross >= 1) & (sd > LO) & (sd <= HI)
    return np.where(m, W * (86400 * cross + T_CLEAN) + (1 - W) * pred, pred), m
