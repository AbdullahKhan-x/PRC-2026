"""
Rule FL (Rome flight-number copy history) -- seg_rome_copy researcher, 12 Oct 2026.

Zone: LIRF departures WITHOUT an NM record, 1 h < MVT - SCHED <= 3 h (outside rule G's 3-5.5 h zone and rule S's band).
At LIRF the airport sometimes logs BLOCK = SCHED ("copy"). Whether a flight is copied is a habit of the FLIGHT NUMBER
(same handler / same slot / same feed), beyond operator, format and airline x destination (rules F/E/G).
  lfr(flight) = log((copies + K) / (expected copies + K)) over 2025 LIRF late departures where a copy is possible
                (no-NM, or NM with |SCHED - LOBT| <= 3606 s; 30 min - 5.5 h; expected = mean copy rate of the
                 NM-status x lateness-band cell), K = 1.
  q   = copy weight implied by the current prediction: (pred - anchor) / (sd - anchor), clipped [0.01, 0.99]
        (anchor = ctx2id base = the clean-taxi anchor already used by rule G)
  q'  = sigmoid(logit(q) + GAMMA * lfr)       GAMMA = 1.5 (LOMO-fitted logistic coefficient 1.46-1.52 on no-NM rows)
  pred <- pred + (q' - q) * (sd - anchor)
Label-free at prediction time: needs FLIGHT_mvt, MVT, SCHED, AOBT_3 presence and the ctx2id anchor; the table is fitted on
2025 labels only. Apply after rules A D G S and BEFORE T/P and DD (T keeps priority on twin-window rows).
Held-out 2025 (s6cat50 ADGS + TP + DD, table refitted without the scored month): see handover/agent_seg_rome_copy.md.
"""
import os, glob
import numpy as np, pandas as pd

LO, HI = 3600, 10800          # zone of application (MVT - SCHED, s)
GAMMA, K = 1.5, 1.0
FIT_LO, FIT_HI, WIN = 1800, 19800, 3606
BANDS = [1800, 3600, 7200, 10800, 19800]


def load_2025(D):
    c = ['PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'AOBT_3_flt', 'LOBT_flt']
    out = []
    for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet'))):
        t = pd.read_parquet(f, columns=c)
        out.append(t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LIRF')])
    t = pd.concat(out, ignore_index=True)
    e = lambda k: (pd.to_datetime(t[k], utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
    r = pd.DataFrame({'fl': t.FLIGHT_mvt.astype(str).values, 'nonm': t.AOBT_3_flt.isna().values,
                      'sd': e('MVT_TIME_UTC_mvt') - e('SCHED_TIME_UTC_mvt'), 'sl': e('SCHED_TIME_UTC_mvt') - e('LOBT_flt'),
                      'y': t.TAXITIME_SEC_mvt.astype(float).values,
                      'mon': pd.to_datetime(t.SCHED_TIME_UTC_mvt, utc=True).dt.month.values})
    ok = (r.sd > FIT_LO) & (r.sd <= FIT_HI) & (r.y <= 43200) & (r.nonm | (np.abs(r.sl) <= WIN))
    return r[ok].reset_index(drop=True)


def fit(r, k=K):
    r = r.assign(cp=(np.abs(r.y - r.sd) <= 60), band=pd.cut(r.sd, BANDS, labels=False))
    r['pc'] = r.groupby(['nonm', 'band']).cp.transform('mean')
    g = r.groupby('fl').agg(c=('cp', 'sum'), e=('pc', 'sum'))
    return np.log((g.c + k) / (g.e + k)).to_dict()


def apply(pred, sd, anchor, rome, flight, tab, gamma=GAMMA):
    """rome = LIRF departure without NM record. Returns (new pred, mask, lfr)."""
    pred = np.asarray(pred, float); sd = np.asarray(sd, float); anchor = np.asarray(anchor, float)
    m = np.asarray(rome, bool) & (sd > LO) & (sd <= HI) & (np.abs(sd - anchor) > 300)
    lfr = pd.Series(flight).astype(str).map(tab).fillna(0.0).values
    q = np.clip((pred - anchor) / np.where(m, sd - anchor, 1.0), 0.01, 0.99)
    q1 = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + gamma * lfr)))
    return np.where(m, pred + (q1 - q) * (sd - anchor), pred), m, lfr
