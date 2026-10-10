"""
Second batch of context features (label-free: no departure BLOCK / target is read).

  ctx2_arrcopy_pm60   share of arrivals landing within +-60 min whose on-block equals the
                      schedule (|BLOCK - SCHED| <= 5 s): a live view of a recorder outage,
                      which writes the schedule for departures too
  ctx2_arrcopy_day    the same over the airport's whole local day
  ctx2_dly_prev30     mean (AOBT - SCHED), clipped to [-1800, 7200], of departures that pushed
                      in the 30 min before my reference time (AOBT, else MVT - 15 min)
  ctx2_retention      departures already past SCHED at my reference time that have not yet
                      pushed (their AOBT is later): flights stuck at gates
  ctx2_tat            time from the last arrival on-block at my stand to my reference time
  ctx2_arr_dly        that arrival's own lateness (on-block - its schedule)

  python 42b_context_features2.py   -> ctx2_train.parquet, ctx2_rank.parquet (key MVT_ID_mvt)
"""
import os, glob, gc
import numpy as np
import pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt',
        'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt']
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London',
      'EHAM': 'Europe/Amsterdam', 'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid',
      'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich',
      'LTFM': 'Europe/Istanbul'}


def ep(s):
    t = pd.to_datetime(s, utc=True, errors='coerce')
    return (t - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy(dtype='float64')


def wmean(ref_t, ref_v, at, lo, hi):
    cs = np.concatenate([[0.0], np.cumsum(ref_v)])
    a = np.searchsorted(ref_t, at + lo, 'right'); b = np.searchsorted(ref_t, at + hi, 'right')
    n = (b - a).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan), n


def build(df):
    dep = df[df.PHASE_mvt == 'DEP'].copy(); arr = df[df.PHASE_mvt == 'ARR'].copy(); del df
    dep['apt'] = dep.ADEP_mvt.astype(str); arr['apt'] = arr.ADES_mvt.astype(str)
    dep['mvt'], dep['sch'], dep['aobt'] = ep(dep.MVT_TIME_UTC_mvt), ep(dep.SCHED_TIME_UTC_mvt), ep(dep.AOBT_3_flt)
    arr['land'], arr['onb'], arr['sch'] = ep(arr.MVT_TIME_UTC_mvt), ep(arr.BLOCK_TIME_UTC_mvt), ep(arr.SCHED_TIME_UTC_mvt)
    dep['ref'] = np.where(np.isnan(dep.aobt), dep.mvt - 900, dep.aobt)
    out = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.values}, index=dep.index)
    for c in ['ctx2_arrcopy_pm60', 'ctx2_arrcopy_day', 'ctx2_dly_prev30', 'ctx2_retention', 'ctx2_tat', 'ctx2_arr_dly']:
        out[c] = np.nan
    for a, g in dep.groupby('apt', sort=False):
        ga = arr[(arr.apt == a) & arr.land.notna() & arr.onb.notna() & arr.sch.notna()]
        o = np.argsort(ga.land.values, kind='stable')
        lt = ga.land.values[o]
        cp = (np.abs(ga.onb.values - ga.sch.values) <= 5).astype(float)[o]
        m, _ = wmean(lt, cp, g.mvt.values, -3600, 3600)
        out.loc[g.index, 'ctx2_arrcopy_pm60'] = m
        # whole local day
        tz = TZ.get(a, 'UTC')
        dday = pd.to_datetime(g.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert(tz).dt.date.astype(str).values
        aday = pd.to_datetime(ga.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert(tz).dt.date.astype(str).values
        rate = pd.Series((np.abs(ga.onb.values - ga.sch.values) <= 5).astype(float)).groupby(aday).mean()
        out.loc[g.index, 'ctx2_arrcopy_day'] = pd.Series(dday).map(rate).values
        # departure delay level and gate retention around my reference time
        ok = ~np.isnan(g.aobt.values)
        pt = g.aobt.values[ok]; dl = np.clip(g.aobt.values[ok] - g.sch.values[ok], -1800, 7200)
        o2 = np.argsort(pt, kind='stable'); pt, dl = pt[o2], dl[o2]
        m, _ = wmean(pt, dl, g.ref.values, -1800, 0)
        out.loc[g.index, 'ctx2_dly_prev30'] = m
        # retention: SCHED < ref  and  AOBT > ref  (counted over flights with an AOBT)
        s_sorted = np.sort(g.sch.values[ok]); a_sorted = np.sort(g.aobt.values[ok])
        due = np.searchsorted(s_sorted, g.ref.values, 'left')            # SCHED < ref
        pushed = np.searchsorted(a_sorted, g.ref.values, 'right')        # AOBT <= ref
        # flights due but not pushed ~ due - (due and pushed); approximate with due - pushed, floored at 0
        out.loc[g.index, 'ctx2_retention'] = np.maximum(due - pushed, 0)
        # turnaround at my stand
        st = g.STAND_mvt.astype(str).values
        gas = ga.STAND_mvt.astype(str).values
        tat = np.full(len(g), np.nan); adl = np.full(len(g), np.nan)
        arr_by_stand = {}
        for s_, on_, sc_ in zip(gas, ga.onb.values, ga.sch.values):
            arr_by_stand.setdefault(s_, []).append((on_, on_ - sc_))
        for s_ in arr_by_stand:
            v = np.array(sorted(arr_by_stand[s_]))
            arr_by_stand[s_] = v
        refv = g.ref.values
        for i in range(len(g)):
            v = arr_by_stand.get(st[i])
            if v is None:
                continue
            j = np.searchsorted(v[:, 0], refv[i], 'right') - 1
            if j >= 0:
                tat[i] = refv[i] - v[j, 0]; adl[i] = v[j, 1]
        out.loc[g.index, 'ctx2_tat'] = np.clip(tat, 0, 86400)
        out.loc[g.index, 'ctx2_arr_dly'] = np.clip(adl, -7200, 14400)
    for c in out.columns:
        if c != 'MVT_ID_mvt':
            out[c] = out[c].astype(np.float32)
    return out.reset_index(drop=True)


def read_compact(f):
    x = pd.read_parquet(f, columns=COLS)
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt']:
        x[c] = x[c].astype('category')
    return x


if __name__ == '__main__':
    tr = pd.concat([read_compact(f) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))], ignore_index=True)
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt']:
        tr[c] = tr[c].astype('category')
    c2 = build(tr); del tr; gc.collect()
    print('train', c2.shape, c2.drop(columns='MVT_ID_mvt').describe().T[['mean', '50%']].round(2).to_dict('index'))
    c2.to_parquet(os.path.join(D, 'ctx2_train.parquet'), index=False)
    rk = read_compact(os.path.join(D, 'data/raw/ranking.parquet'))
    build(rk).to_parquet(os.path.join(D, 'ctx2_rank.parquet'), index=False)
    print('rank done')
