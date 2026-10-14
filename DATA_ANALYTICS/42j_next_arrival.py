"""
Stand-turnover witnesses of the hidden off-block (label-free; arrivals' in-block times are visible in every file):
a stand holds one aircraft, so the NEXT arrival at my stand cannot block in before I have pushed back.
In 2025, for ~10 % of departures (EGLL 26 %) the next aircraft blocks in at my stand BEFORE my take-off; its in-block is
then a median 654 s (IQR 423-904) after my real off-block.
  na_g1      MVT - in-block of the latest arrival at my stand before my take-off (<= 6 h)
  na_g2      MVT - in-block of the arrival before that
  na_l_aobt  that latest in-block - NM AOBT_3   (> 0: it blocked in after my NM off-block -> likely the NEXT aircraft)
  na_l_sch   that latest in-block - my SCHED
  na_l_lobt  that latest in-block - my LOBT
  na_next    in-block of the first arrival at my stand after my take-off - MVT
  na_l_tin   taxi-in of that latest arrival (long = it waited for my stand -> my off-block close to its in-block)
  python 42j_next_arrival.py      -> na_train.parquet (2025 training months), na_rank.parquet (ranking.parquet)
"""
import os, glob
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
        'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', 'TAXITIME_SEC_mvt']
NA = ['na_g1', 'na_g2', 'na_l_aobt', 'na_l_sch', 'na_l_lobt', 'na_next', 'na_l_tin']
TOL = 6 * 3600


def _ep(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values


def build(raw):
    t = raw[COLS].copy()
    t['apt'] = np.where(t.PHASE_mvt == 'DEP', t.ADEP_mvt, t.ADES_mvt).astype(str)
    t['stand'] = t.STAND_mvt.astype(object).where(t.STAND_mvt.notna(), None)
    t['mvt'], t['blk'], t['sch'] = _ep(t.MVT_TIME_UTC_mvt), _ep(t.BLOCK_TIME_UTC_mvt), _ep(t.SCHED_TIME_UTC_mvt)
    t['aobt'], t['lobt'] = _ep(t.AOBT_3_flt), _ep(t.LOBT_flt)
    a = t[(t.PHASE_mvt == 'ARR') & t.stand.notna() & ~np.isnan(t.blk.values)][['apt', 'stand', 'blk', 'TAXITIME_SEC_mvt']]
    a = a.rename(columns={'blk': 'ai', 'TAXITIME_SEC_mvt': 'tin'}).assign(k=lambda z: z.ai).sort_values('k')
    a['tin'] = a.tin.astype(float)
    dep = t[t.PHASE_mvt == 'DEP'][['MVT_ID_mvt', 'apt', 'stand', 'mvt', 'sch', 'aobt', 'lobt']].copy()
    out = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.values})
    d = dep[dep.stand.notna()].copy()
    if len(d) and len(a):
        m = pd.merge_asof(d.assign(k=d.mvt).sort_values('k'), a.rename(columns={'ai': 'ai1'}), on='k', by=['apt', 'stand'],
                          direction='backward', tolerance=TOL)
        m = m.drop(columns='k').assign(k=lambda z: (z.ai1 - 1e-3).fillna(-1e12)).sort_values('k')
        m = pd.merge_asof(m, a.drop(columns='tin').rename(columns={'ai': 'ai2'}), on='k', by=['apt', 'stand'], direction='backward', tolerance=TOL)
        m = m.drop(columns='k').assign(k=lambda z: z.mvt + 1e-3).sort_values('k')
        m = pd.merge_asof(m, a.drop(columns='tin').rename(columns={'ai': 'ain'}), on='k', by=['apt', 'stand'], direction='forward', tolerance=TOL)
        f = pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt.values, 'na_g1': (m.mvt - m.ai1).values, 'na_g2': (m.mvt - m.ai2).values,
                          'na_l_aobt': (m.ai1 - m.aobt).values, 'na_l_sch': (m.ai1 - m.sch).values,
                          'na_l_lobt': (m.ai1 - m.lobt).values, 'na_next': (m.ain - m.mvt).values, 'na_l_tin': m.tin.values})
        out = out.merge(f, on='MVT_ID_mvt', how='left')
    for c in NA:
        if c not in out.columns:
            out[c] = np.nan
    return out[['MVT_ID_mvt'] + NA]


if __name__ == '__main__':
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))],
                   ignore_index=True)
    build(tr).to_parquet(os.path.join(D, 'na_train.parquet'), index=False)       # month boundaries: arrivals of all months together
    build(pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)).to_parquet(os.path.join(D, 'na_rank.parquet'), index=False)
    for s in ('train', 'rank'):
        x = pd.read_parquet(os.path.join(D, f'na_{s}.parquet'))
        print(f'na_{s}: {len(x):,} rows; non-null share', {c: round(float(x[c].notna().mean()), 3) for c in NA}, flush=True)
