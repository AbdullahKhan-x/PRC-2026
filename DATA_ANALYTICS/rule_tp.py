"""
Rules T (twin-arrival LOBT window) and P (LFPG duplicate pair) -- agent monster audit, 12 Oct 2026.
See handover/agent_monster_audit.md for evidence (LOMO 2025: v137eq stack 289.36 -> 288.34, 4/4 months; ctx2id 12/12 months).

Label-free features from ONE raw file (all phases; works on training, ranking and final_ranking files):
  T: departure without NM record whose own flight's ARRIVAL row at another of the 10 airports carries an NM record
     (matched on flight code = arrival FLIGHT_mvt or arrival CALLSIGN_flt, same ADEP/ADES, arrival MVT within 10 h after).
     The airport->NM match requires |BLOCK - LOBT| <= 3600 s, so the departure's y lies OUTSIDE
     [MVT - LOBT_twin - 3600, MVT - LOBT_twin + 3600]  (2025: 1,415/1,415 rows).
     Twin is discarded when its NM flight id is already linked to a departure row.
  P: two departure rows of the same FLIGHT_mvt at the same ADEP with MVT <= 300 s apart, one NM-linked and one not
     (LFPG). The NM-linked row carries the planned/first block: y ~= MVT - LOBT - 356 s.

apply():
  T-up  : MVT - LOBT_twin in (-3600, 3600] and pred < hi  -> pred = hi + D_UP   (D_UP = median(y - hi), ~870 s)
  T-down: MVT - LOBT_twin > 3600 and lo < pred < hi       -> pred = lo
  P     : NM row of a mixed pair                          -> pred = MVT - LOBT + D_P (D_P ~ -356 s)
Apply AFTER the existing rules (A D G S); rule A already keeps NM rows inside their own LOBT window, P stays inside it.
"""
import numpy as np
import pandas as pd

APT = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
WIN = 3600.0
D_UP, D_P = 870.0, -356.0      # 2025 medians (LOMO values ranged 729-903 and -418..-270)


def _sec(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True)
        return (s - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    return s.astype('float64')


def features(raw: pd.DataFrame) -> pd.DataFrame:
    """raw = full file (DEP + ARR rows). Returns one row per departure: MVT_ID_mvt, tp_ml_twin, tp_pair, tp_ml_own."""
    t = {c: _sec(raw[c]) for c in ['MVT_TIME_UTC_mvt', 'LOBT_flt']}
    nm = raw['AOBT_3_flt'].notna().values
    s = lambda c: raw[c].astype(str).values
    R = pd.DataFrame({'MVT_ID_mvt': raw.MVT_ID_mvt.values, 'phase': s('PHASE_mvt'), 'fl': s('FLIGHT_mvt'),
                      'cs': s('CALLSIGN_flt'), 'adep': s('ADEP_mvt'), 'ades': s('ADES_mvt'),
                      't': t['MVT_TIME_UTC_mvt'].values, 'lobt': t['LOBT_flt'].values, 'nm': nm,
                      'fid': raw.FLIGHT_ID_mvt.values})
    dep = R[R.phase == 'DEP'].copy()
    arr = R[(R.phase == 'ARR') & R.t.notna()].copy()
    arr['fid_used'] = arr.fid.isin(set(dep.fid.dropna()))
    arr = arr.sort_values('t')
    # ---- T: twin arrival
    x = dep[dep.ades.isin(APT) & (dep.ades != dep.adep) & ~dep.fl.isin(['nan', 'None'])][['MVT_ID_mvt', 'fl', 'adep', 'ades', 't']].sort_values('t')
    keep = ['t', 'adep', 'ades', 'lobt', 'nm', 'fid_used']
    m1 = pd.merge_asof(x, arr[keep + ['fl']].rename(columns={'t': 'ta'}).assign(t=lambda a: a.ta),
                       on='t', by=['fl', 'adep', 'ades'], direction='forward', tolerance=10 * 3600.)
    m2 = pd.merge_asof(x, arr[keep + ['cs']].rename(columns={'cs': 'fl', 't': 'ta'}).assign(t=lambda a: a.ta),
                       on='t', by=['fl', 'adep', 'ades'], direction='forward', tolerance=10 * 3600.)
    m1 = m1.set_index('MVT_ID_mvt'); m2 = m2.set_index('MVT_ID_mvt')
    use2 = m1.ta.isna()
    tw = m1[['ta', 'lobt', 'nm', 'fid_used']].copy(); tw.loc[use2] = m2.loc[use2, ['ta', 'lobt', 'nm', 'fid_used']]
    ok = tw.ta.notna() & tw.nm.astype(bool) & ~tw.fid_used.astype(bool)
    d = dep.set_index('MVT_ID_mvt')
    twin_lobt = tw.lobt.where(ok).reindex(d.index)
    out = pd.DataFrame(index=d.index)
    out['tp_ml_twin'] = np.where(~d.nm, d.t - twin_lobt, np.nan)
    # ---- P: near-simultaneous duplicate pair (one NM, one not)
    k = d.reset_index()[['MVT_ID_mvt', 'fl', 'adep', 't', 'nm']].sort_values(['fl', 'adep', 't'])
    g = k.groupby(['fl', 'adep'])
    close_n = (g.t.shift(-1) - k.t).abs() <= 300
    close_p = (k.t - g.t.shift(1)).abs() <= 300
    sib_nonm = (close_n & ~g.nm.shift(-1, fill_value=True).astype(bool)) | (close_p & ~g.nm.shift(1, fill_value=True).astype(bool))
    k['pair'] = k.nm & sib_nonm
    out['tp_pair'] = k.set_index('MVT_ID_mvt').pair.reindex(out.index).fillna(False).astype(bool)
    out['tp_ml_own'] = (d.t - d.lobt)
    return out.reset_index()


def apply(pred, f: pd.DataFrame, d_up=D_UP, d_p=D_P):
    """pred aligned with f rows. Returns new pred and dict of masks."""
    p = np.asarray(pred, dtype='float64').copy()
    ml = f.tp_ml_twin.values; lo, hi = ml - WIN, ml + WIN
    has = ~np.isnan(ml)
    up = has & (ml > -WIN) & (ml <= WIN) & (p < hi)
    p[up] = hi[up] + d_up
    dn = has & (ml > WIN) & (p > lo) & (p < hi)
    p[dn] = lo[dn]
    pp = f.tp_pair.values & ~np.isnan(f.tp_ml_own.values)
    p[pp] = f.tp_ml_own.values[pp] + d_p
    return p, {'T_up': up, 'T_down': dn, 'P': pp}


if __name__ == '__main__':      # self-check on 2025 vs the v137eq held-out OOF (constants fixed, i.e. not LOMO)
    import glob, sys
    root = (os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'
    cols = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt',
            'MVT_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt']
    files = sys.argv[1:] or sorted(glob.glob(root + 'data/raw/training_2025-*.parquet'))
    raw = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    f = features(raw); del raw
    o = pd.read_parquet(root + 'handover/oof_v137eq_heldout2025.parquet', columns=['MVT_ID_mvt', 'mon', 'y', 'pred'])
    x = o.merge(f, on='MVT_ID_mvt', how='left')
    x['tp_pair'] = x.tp_pair.fillna(False).astype(bool)
    newp, masks = apply(x.pred.values, x)
    for m in sorted(x.mon.unique()):
        s = (x.mon == m).values
        print(m, f'{np.sqrt(((x.pred.values[s] - x.y.values[s]) ** 2).mean()):.2f} -> {np.sqrt(((newp[s] - x.y.values[s]) ** 2).mean()):.2f}',
              {k: int(v[s].sum()) for k, v in masks.items()})
    print('all', f'{np.sqrt(((x.pred - x.y) ** 2).mean()):.3f} -> {np.sqrt(((newp - x.y) ** 2).mean()):.3f}')
