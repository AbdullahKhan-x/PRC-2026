"""
Rule F = rule C refined (Rome, no NM record, 3-5.5 h late vs schedule):
  P(BLOCK == SCHED) depends on the FLIGHT_mvt code format. Long / odd prefixes (ITYTY680, WZZZ2062, BAWW561D: a different
  feed) are copies 83 % of the time in 2025, standard ICAO+digits codes 55 %.  pred = P * sd + (1 - P) * t_clean (median
  clean taxi of the zone, as rule C). NOT the model base: on the ADS-B stack the base already hedges (3,000-6,800 s on these
  rows) and a base anchor double-counts the copy risk (held-out stack +0.74, Jul +4.8). With t_clean: stack -0.32 (Jul -1.3).
"""
import os, glob
import numpy as np, pandas as pd

LO, HI = 10800, 20000


def fmt_long(fl):
    fl = pd.Series(fl).astype(str)
    return ((fl.str.extract(r'^([A-Z]*)')[0].str.len().fillna(0) >= 4) | ~fl.str.match(r'^[A-Z]{3}\d')).values


def fit(D):
    c = ['PHASE_mvt', 'ADEP_mvt', 'AOBT_3_flt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
    t = pd.concat([pd.read_parquet(f, columns=c) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))])
    t = t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LIRF') & t.AOBT_3_flt.isna()]
    e = lambda k: (pd.to_datetime(t[k], utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
    sd = e('MVT_TIME_UTC_mvt') - e('SCHED_TIME_UTC_mvt'); y = e('MVT_TIME_UTC_mvt') - e('BLOCK_TIME_UTC_mvt')
    z = (sd > LO) & (sd <= HI); cp = np.abs(y - sd) <= 60; f = fmt_long(t.FLIGHT_mvt.values)
    rate = {g: float(cp[z & (f == g)].mean()) for g in (True, False)}
    rate['t_clean'] = float(np.median(y[z & ~cp & (y <= 10800)]))
    return rate


def apply(pred, base, sd, rome, flight, rate):
    fr = fmt_long(flight)
    pf = np.where(fr, rate[True], rate[False])
    m = rome & (sd > LO) & (sd <= HI)
    return np.where(m, pf * sd + (1 - pf) * rate['t_clean'], pred), m, fr
