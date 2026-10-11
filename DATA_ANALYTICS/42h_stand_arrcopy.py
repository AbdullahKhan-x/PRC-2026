"""
Sixth context batch: stand-level outage signal from ARRIVALS (label-free: arrival on-blocks are visible in the ranking file).

When the stand's off/on-block recording fails, the schedule is written instead (copy). Failures persist per stand, and
arrivals at the stand show it: in 2025, departures 30 min-3 h late were copies 8.9 % of the time when > 50 % of the
same-stand arrivals in the previous 12 h had BLOCK == SCHED, vs 1.2 % when none had (LIRF 24.4 % vs 6.4 %).

  sac_share_3h / sac_n_3h     share / count of copied arrival on-blocks at my stand, landing in [MVT - 3 h, MVT]
  sac_share_12h / sac_n_12h   same over [MVT - 12 h, MVT]
writes sac_train.parquet / sac_rank.parquet
"""
import os, glob, gc
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']


def ep(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def build(raw):
    r = raw[COLS].copy()
    r['apt'] = np.where(r.PHASE_mvt == 'DEP', r.ADEP_mvt.astype(str), r.ADES_mvt.astype(str))
    r['key'] = r.apt + '|' + r.STAND_mvt.astype(str)
    r['t'] = ep(r.MVT_TIME_UTC_mvt)
    dep = r[r.PHASE_mvt == 'DEP'].reset_index(drop=True)
    arr = r[r.PHASE_mvt == 'ARR'].copy()
    arr['cp'] = (np.abs(ep(arr.BLOCK_TIME_UTC_mvt) - ep(arr.SCHED_TIME_UTC_mvt)) <= 60).astype(float)
    arr = arr.sort_values(['key', 't'])
    ag = {k: (g.t.to_numpy(), np.concatenate([[0.0], np.cumsum(g.cp.to_numpy())])) for k, g in arr.groupby('key')}
    out = {c: np.full(len(dep), np.nan, dtype=np.float32) for c in ['sac_share_3h', 'sac_n_3h', 'sac_share_12h', 'sac_n_12h']}
    for k, idx in dep.groupby('key').indices.items():
        if k not in ag:
            continue
        t, cs = ag[k]; x = dep.t.to_numpy()[idx]
        for h, nm in [(10800, '3h'), (43200, '12h')]:
            a = np.searchsorted(t, x - h, 'left'); b = np.searchsorted(t, x, 'right'); n = b - a
            out[f'sac_n_{nm}'][idx] = n
            out[f'sac_share_{nm}'][idx] = np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)
    res = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.to_numpy(), **out})
    return res


if __name__ == '__main__':
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data', 'raw', 'training_2025-*.parquet')))],
                   ignore_index=True)
    a = build(tr); del tr; gc.collect()
    a.to_parquet(os.path.join(D, 'sac_train.parquet'), index=False)
    print('train', a.shape, a.drop(columns='MVT_ID_mvt').describe().T[['mean', '50%']].round(3).to_dict('index'), flush=True)
    for name, path in [('rank', 'ranking.parquet'), ('final', 'final_ranking.parquet')]:
        p = os.path.join(D, 'data', 'raw', path)
        if os.path.exists(p):
            b = build(pd.read_parquet(p, columns=COLS))
            b.to_parquet(os.path.join(D, f'sac_{name}.parquet'), index=False)
            print(name, b.shape, b.drop(columns='MVT_ID_mvt').describe().T[['mean', '50%']].round(3).to_dict('index'), flush=True)
