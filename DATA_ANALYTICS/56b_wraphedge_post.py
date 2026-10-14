"""FINAL_v161 = FINAL_v159 + Rome after-midnight date-wrap hedge (USER-CHOSEN GAMBLE, 14 Oct 06:05 UTC).
Rows: LIRF, no NM record, local MVT date > local SCHED date (cross >= 1), sd <= 50,000 s (S' covers 50-100 ks), local take-off 00:20-06:00.
2025: among such rows that are not schedule copies, 1 of 23 was a date wrap (ITY332) -> p_w = 1/23.
E[y] = pred + (1 - q) * p_w * (W - m),  W = 86,400*cross + 1,105 (S' wrap value), m = 900 s (clean Rome taxi), q = clip((pred - m)/(sd - m), 0, 1)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'DATA_ANALYTICS'); import rule_s
IN, RAW, OUT, TMPL = sys.argv[1:5]
P_W, M, WX = 1 / 23, 900.0, 1105.0
f = pd.read_parquet(IN)
r = pd.read_parquet(RAW, columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
r = r[r.PHASE_mvt == 'DEP'].set_index('MVT_ID_mvt').loc[f.MVT_ID_mvt.values]
sd = (pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(r.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().values
cross = rule_s.cross_days(r.MVT_TIME_UTC_mvt.values, r.SCHED_TIME_UTC_mvt.values)
ml = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert('Europe/Rome'); tmin = (ml.dt.hour * 60 + ml.dt.minute).values
m = (r.ADEP_mvt.astype(str).values == 'LIRF') & r.AOBT_3_flt.isna().values & (cross >= 1) & (sd <= 50000) & (tmin >= 20) & (tmin < 360)
p = f.TAXITIME_SEC_mvt.values.astype(float).copy()
q = np.clip((p - M) / np.maximum(sd - M, 1.0), 0, 1)
add = (1 - q) * P_W * (86400.0 * cross + WX - M)
p[m] = p[m] + add[m]
out = f.copy(); out['TAXITIME_SEC_mvt'] = p
t = pd.read_parquet(TMPL)
assert (out.MVT_ID_mvt.values == t.MVT_ID_mvt.values).all() and np.isfinite(p).all() and (p >= 60).all() and len(out) == len(t)
out.to_parquet(OUT, index=False)
dd = p - f.TAXITIME_SEC_mvt.values
print(OUT, 'rows hedged', int(m.sum()), ' changed', int((np.abs(dd) > 1e-6).sum()), ' added: min %.0f median %.0f max %.0f s' % (dd[m].min(), np.median(dd[m]), dd[m].max()), ' q median %.2f' % np.median(q[m]))
