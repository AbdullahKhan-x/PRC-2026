"""
Post-processing rules on top of v35 (no refit). Every rule is derived from the 2025
training data only and changes a small, named set of rows; all other rows stay
byte-identical to v35.

  A  LOBT window   In 100% of the 2,062,577 departures of 2025 with a LOBT,
                   |BLOCK - LOBT| <= 3606 s. So (1) a row cannot be a schedule copy
                   (BLOCK == SCHED) when SCHED is outside that window -> detector p = 0,
                   and (2) every prediction is projected onto
                   [MVT - LOBT - 3606, MVT - LOBT + 3606].
  B  Rome 24h      LIRF, no NM record, scheduled-local-date < take-off-local-date and
                   55,000 < MVT-SCHED <= 100,000: in 2025, 10 of 15 had BLOCK = real
                   off-block stamped with the scheduled date, i.e. y = 86400 + taxi,
                   5 of 15 were schedule copies, 0 clean. Expectation mixture.
  C  Rome 3-5.5h   LIRF, no NM record, 10,800 < MVT-SCHED <= 20,000 (not B):
                   2025 copy rate 107/173 = 0.62 vs detector p ~ 1. Expectation mixture
                   with the honest rate.
  D  no-NM outside Rome: 2025 had 1 schedule copy in 4,317 pool rows -> p = 0.

Usage:  python 40_postrules.py RULES VERSION      e.g.  python 40_postrules.py A 101
"""
import sys, os, numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(D, 'candidates', 'v35_parts.parquet')
RULES, VERSION = sys.argv[1].upper(), int(sys.argv[2])
WIN = 3606.0
W_SHIFT, T_CLEAN = 11/17, 950.0          # Beta(1,1)-smoothed 10/15; median clean LIRF taxi
P_ROME_3H = 107/173

tmpl = pd.read_parquet(os.path.join(D, 'data/raw/submitting.parquet'))
v35 = pd.read_parquet(os.path.join(D, 'logical-ladder_v35.parquet'))
assert (v35.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all()
parts = pd.read_parquet(PARTS)
assert (parts.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all()
if os.path.basename(PARTS) == 'v35_parts.parquet':      # the v35 rebuild must be exact
    assert np.allclose(parts.pred.values, v35.TAXITIME_SEC_mvt.values, atol=1e-6, rtol=0)
else:
    print(f'parts from {os.path.basename(PARTS)} (new base); mean |pred - v35| before rules: '
          f'{np.abs(parts.pred.values - v35.TAXITIME_SEC_mvt.values).mean():.1f}s')

cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'AOBT_3_flt', 'LOBT_flt', 'FLIGHT_mvt', 'ADES_mvt']
rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=cols)
rk = rk[rk.PHASE_mvt == 'DEP'].set_index('MVT_ID_mvt').loc[tmpl.MVT_ID_mvt.values].reset_index()
ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
mvt, sch, lobt = ep(rk.MVT_TIME_UTC_mvt), ep(rk.SCHED_TIME_UTC_mvt), ep(rk.LOBT_flt)
apt = rk.ADEP_mvt.astype(str).values
nonm = rk.AOBT_3_flt.isna().values
rome = (apt == 'LIRF') & nonm
loc = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
cross = (loc(rk.MVT_TIME_UTC_mvt) - loc(rk.SCHED_TIME_UTC_mvt)).dt.days.values

base, p, sd = parts.base.values.copy(), parts.p.values.copy(), parts.sd.values
pool = sd > 7200
pred = parts.pred.values.copy()
changed = {}

def blend(pp):
    return np.where(pool, pp * sd + (1 - pp) * base, base)

if 'D' in RULES:
    m = pool & nonm & (apt != 'LIRF')
    p = np.where(m, 0.0, p); changed['D'] = m
if 'A' in RULES:
    has = ~np.isnan(lobt)
    m = has & pool & (np.abs(sch - lobt) > WIN)
    p = np.where(m, 0.0, p); changed['A_p0'] = m & (parts.p.values > 0)
pred = np.clip(blend(p), 60, 172800)
if 'C' in RULES:
    m = rome & (sd > 10800) & (sd <= 20000)
    pred = np.where(m, P_ROME_3H * sd + (1 - P_ROME_3H) * T_CLEAN, pred); changed['C'] = m
if 'E' in RULES:          # Rome, no NM record, 1-5.5 h late: band x airline x destination copy rate
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import rome_rates
    tab = rome_rates.fit(rome_rates.load_2025(D))
    pe, pp = rome_rates.predict(tab, sd, rk.FLIGHT_mvt.values, rk.ADES_mvt.values)
    m = rome & (sd > 3600) & (sd <= 20000)
    pred = np.where(m, pe, pred); changed['E'] = m
    print(f'   rule E: {int(m.sum())} rows, mean P(copy) {pp[m].mean():.3f}, t_clean {tab["t_clean"]:.0f}')
if 'B' in RULES:
    m = rome & (cross >= 1) & (sd > 55000) & (sd <= 100000)
    pred = np.where(m, W_SHIFT * (86400 * cross + T_CLEAN) + (1 - W_SHIFT) * pred, pred); changed['B'] = m
if 'A' in RULES:
    has = ~np.isnan(lobt)
    lo, hi = np.maximum(mvt - lobt - WIN, 0), mvt - lobt + WIN
    clipped = np.where(has, np.clip(pred, lo, np.maximum(hi, lo)), pred)
    changed['A_clip'] = has & (np.abs(clipped - pred) > 1e-9)
    pred = clipped

old = v35.TAXITIME_SEC_mvt.values
moved = np.abs(pred - old) > 1e-9
print(f'rules {RULES} -> v{VERSION}:  rows changed vs v35: {int(moved.sum())}   sum of squared changes {((pred-old)**2).sum():.3e}')
for k, m in changed.items():
    print(f'   {k:7s} rows {int(m.sum()):5d}   squared change on them {((pred-old)[m]**2).sum():.3e}')
sub = pd.DataFrame({'MVT_ID_mvt': tmpl.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': pred.astype(np.float64)})
assert len(sub) == len(tmpl) and (sub.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all()
assert np.isfinite(sub.TAXITIME_SEC_mvt).all() and (sub.TAXITIME_SEC_mvt >= 0).all()
out = os.path.join(D, 'candidates', f'logical-ladder_v{VERSION}.parquet')
sub.to_parquet(out, index=False)
chk = pd.read_parquet(out)
print(f'   wrote {out}   order ok: {(chk.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all()}  rows {len(chk)}  >10800: {(chk.TAXITIME_SEC_mvt > 10800).sum()}')
