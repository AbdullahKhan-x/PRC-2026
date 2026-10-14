"""
Rule S' (agent_rome_datewrap, 12 Oct 2026; replaces rule S on its rows, applied LAST and frozen).
Rome (LIRF), no NM record, local MVT date > local SCHED date (cross >= 1), 50,000 < MVT - SCHED <= 100,000 s:
such a row is never a clean taxi in 2025 (0/31 above 40 ks) - it is a schedule copy (y = sd) or a date wrap
(BLOCK = scheduled local date + actual clock time: y = 86,400*cross + ~1,105 s).
  pred = p * (86,400*cross + 1,105) + (1 - p) * sd        (exact E[y])
  p = 5/12 for sd <= 65 ks (2025: 5 wraps / 12 rows), 0.86 above (2025: 6/6, LOMO-Laplace)  -- 2025 labels ONLY
Held-out on the v150 stack (Jan/Feb/Jun/Jul 2025): see WORKLOG.
"""
import numpy as np
import rule_s

LO, MID, HI, T_WRAP = 50000, 65000, 100000, 1105.0
P_LO, P_HI = 5 / 12, 0.86


def apply(pred, sd, rome, cross):
    sd = np.asarray(sd, float); cross = np.asarray(cross)
    m = np.asarray(rome, bool) & (cross >= 1) & (sd > LO) & (sd <= HI)
    p = np.where(sd <= MID, P_LO, P_HI)
    return np.where(m, p * (86400.0 * cross + T_WRAP) + (1 - p) * sd, pred), m
