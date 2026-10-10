"""
Fourth context batch (label-free; literature hypotheses H2, H3 + flight-number digits). Computable from the
ranking file alone, so 2026 rows get them from 2026 data (new stands: EDDF T3, EDDM renumbering).

  H3  per (airport, stand, UTC month), from the same month's own rows:
        lf_st_imp_med   median MVT - AOBT_3 of departures (60..3600 s)
        lf_st_eobt_med  median MVT - EOBT_1 of departures (60..3600 s)
        lf_st_n         departures at the stand that month
        lf_st_tin_med   median arrival taxi-in (on-block - landing, 0..3600 s) at the stand that month
  H2  relative position vs the airport right now (+- 30 min of my take-off, other departures):
        lf_rel_imp      (MVT - AOBT_3) - window mean of (MVT - AOBT_3)
        lf_rel_eobt     (MVT - EOBT_1) - window mean of (MVT - EOBT_1)
        lf_rel_imp_st   (MVT - AOBT_3) - lf_st_imp_med
  flight number: lf_fno (numeric part of FLIGHT_mvt), lf_fno_k (thousands), lf_fno_len (digits)

writes lf_train.parquet / lf_rank.parquet
"""
import os, glob, gc
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt',
        'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt']
W = 1800.0


def ep(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def win_mean(t, v, lo, hi):
    """mean of v over rows with t in [t_i + lo, t_i + hi], excluding row i itself; t sorted."""
    ok = np.isfinite(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))]); cn = np.concatenate([[0], np.cumsum(ok)])
    a = np.searchsorted(t, t + lo, 'left'); b = np.searchsorted(t, t + hi, 'right')
    s = cs[b] - cs[a] - np.where(ok, v, 0.0); n = cn[b] - cn[a] - ok
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def build(raw):
    d = raw[raw.PHASE_mvt == 'DEP'].copy()
    a = raw[raw.PHASE_mvt == 'ARR'].copy()
    d['mvt'], d['aobt'], d['eobt'] = ep(d.MVT_TIME_UTC_mvt), ep(d.AOBT_3_flt), ep(d.EOBT_1_flt)
    d['imp'] = d.mvt - d.aobt; d['ime'] = d.mvt - d.eobt
    d['apt'] = d.ADEP_mvt.astype(str); d['st'] = d.STAND_mvt.astype(str)
    d['mon'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    a['tin'] = ep(a.BLOCK_TIME_UTC_mvt) - ep(a.MVT_TIME_UTC_mvt)
    a['apt'] = a.ADES_mvt.astype(str); a['st'] = a.STAND_mvt.astype(str)
    a['mon'] = pd.to_datetime(a.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    good = lambda x: x.where((x > 60) & (x < 3600))
    k = ['apt', 'st', 'mon']
    g = d.assign(imp_g=good(d.imp), ime_g=good(d.ime)).groupby(k)
    st = pd.DataFrame({'lf_st_imp_med': g.imp_g.median(), 'lf_st_eobt_med': g.ime_g.median(), 'lf_st_n': g.size()})
    tin = a.assign(t=a.tin.where((a.tin > 0) & (a.tin < 3600))).groupby(k).t.median().rename('lf_st_tin_med')
    d = d.join(st, on=k).join(tin, on=k)
    d['lf_rel_imp'] = np.nan; d['lf_rel_eobt'] = np.nan
    for ap, idx in d.groupby('apt').indices.items():
        sub = d.iloc[idx].sort_values('mvt')
        t = sub.mvt.to_numpy()
        d.loc[sub.index, 'lf_rel_imp'] = sub.imp.to_numpy() - win_mean(t, good(sub.imp).to_numpy(), -W, W)
        d.loc[sub.index, 'lf_rel_eobt'] = sub.ime.to_numpy() - win_mean(t, good(sub.ime).to_numpy(), -W, W)
    d['lf_rel_imp_st'] = d.imp - d.lf_st_imp_med
    num = d.FLIGHT_mvt.astype(str).str.extract(r'(\d+)', expand=False)
    d['lf_fno'] = pd.to_numeric(num, errors='coerce')
    d['lf_fno_k'] = np.floor(d.lf_fno / 1000)
    d['lf_fno_len'] = num.str.len()
    cols = ['lf_st_imp_med', 'lf_st_eobt_med', 'lf_st_n', 'lf_st_tin_med', 'lf_rel_imp', 'lf_rel_eobt', 'lf_rel_imp_st',
            'lf_fno', 'lf_fno_k', 'lf_fno_len']
    out = d[['MVT_ID_mvt'] + cols].reset_index(drop=True)
    for c in cols:
        out[c] = out[c].astype(np.float32)
    return out


if __name__ == '__main__':
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))],
                   ignore_index=True)
    q = build(tr); del tr; gc.collect()
    q.to_parquet(os.path.join(D, 'lf_train.parquet'), index=False)
    print('train', q.shape, q.drop(columns='MVT_ID_mvt').describe().T[['mean', '50%']].round(1).to_dict('index'), flush=True)
    rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)
    qr = build(rk)
    qr.to_parquet(os.path.join(D, 'lf_rank.parquet'), index=False)
    print('rank', qr.shape, qr.drop(columns='MVT_ID_mvt').describe().T[['mean', '50%']].round(1).to_dict('index'))
