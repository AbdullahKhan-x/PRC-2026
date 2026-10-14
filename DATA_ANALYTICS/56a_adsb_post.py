"""post-apply rule ADSB to a built file (equivalent to 55b order ... KCNT, ADSB, SP, CAP: S' rows excluded, CAP + 60 s floor re-applied)
   python candidates/make_adsb.py IN.parquet RAW.parquet WORKDIR OUT.parquet TEMPLATE.parquet"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'DATA_ANALYTICS'); import rule_adsb, rule_s
IN, RAW, W, OUT, TMPL = sys.argv[1:6]
f = pd.read_parquet(IN); tab = json.load(open('recipes/constants/rule_adsb.json'))
r = pd.read_parquet(RAW, columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'])
r = r[r.PHASE_mvt == 'DEP'].set_index('MVT_ID_mvt').loc[f.MVT_ID_mvt.values]
a = pd.read_parquet(W + '/adsb.parquet', columns=['MVT_ID_mvt', 'adsb_matched', 'adsb_move_gap', 'adsb_gs0']).set_index('MVT_ID_mvt').reindex(f.MVT_ID_mvt.values)
ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
sd = ep(r.MVT_TIME_UTC_mvt) - ep(r.SCHED_TIME_UTC_mvt); apt = r.ADEP_mvt.astype(str).values
rome = (apt == 'LIRF') & r.AOBT_3_flt.isna().values
cross = rule_s.cross_days(r.MVT_TIME_UTC_mvt.values, r.SCHED_TIME_UTC_mvt.values)
spm = rome & (cross >= 1) & (sd > 50000) & (sd <= 100000)
p, g = rule_adsb.apply(f.TAXITIME_SEC_mvt.values, sd, apt, a.adsb_matched.values, a.adsb_move_gap.values, a.adsb_gs0.values, tab, exclude=spm)
p = np.maximum(np.where(np.isin(apt, ['LEMD', 'LEBL']) & (p > 3604), 3604.0, p), 60.0)
out = f.copy(); out['TAXITIME_SEC_mvt'] = p
t = pd.read_parquet(TMPL); assert (out.MVT_ID_mvt.values == t.MVT_ID_mvt.values).all() and np.isfinite(p).all()
out.to_parquet(OUT, index=False)
dd = p - f.TAXITIME_SEC_mvt.values
print(OUT, 'grade rows', int(g.sum()), 'changed', int((np.abs(dd) > 1e-6).sum()), 'rms %.1f mean %+.2f' % (np.sqrt((dd ** 2).mean()), dd.mean()))
