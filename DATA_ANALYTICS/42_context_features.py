"""
Context features for departures, built from columns present in BOTH the 2025 training
files and the 2026 ranking file (no departure BLOCK / target is read anywhere).

  ctx_imp_*   what the airport's taxi times look like right now: mean of other
              departures' MVT - AOBT_3 (clipped to [0, 3600]) for take-offs in a window
              before / around my take-off, airport level and same-runway level
  ctx_tin_*   arrivals' taxi-in (on-block - landing, clipped) in a window before my take-off
  seq_*       exact runway sequence on the take-off clock (MVT is to the second)
  lobt_*      gaps between the flight-plan times and the NM / airport times;
              |BLOCK - LOBT| <= 3606 s held on 100% of 2025 departures
  loc_*       local (DST-aware) hour of take-off and of the schedule
  op2/ades2   operator / destination with the airport's own FLIGHT_mvt / ADES_mvt as
              fallback when there is no Network Manager record

  python 42_context_features.py      -> ctx_train.parquet, ctx_rank.parquet  (key MVT_ID_mvt)
"""
import os, glob, gc
import numpy as np
import pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London',
      'EHAM': 'Europe/Amsterdam', 'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid',
      'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich',
      'LTFM': 'Europe/Istanbul'}
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'RUNWAY_mvt',
        'STAND_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'AIRCRAFT_OPERATOR_flt', 'ADES_flt']


def ep(s):
    t = pd.to_datetime(s, utc=True, errors='coerce')
    v = (t - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    return v.to_numpy(dtype='float64')


def window_mean(ref_t, ref_v, at, lo, hi, exclude_self=None):
    """mean and count of ref_v for ref_t in (at+lo, at+hi]; ref_t sorted."""
    cs = np.concatenate([[0.0], np.cumsum(ref_v)])
    a = np.searchsorted(ref_t, at + lo, 'right')
    b = np.searchsorted(ref_t, at + hi, 'right')
    n = (b - a).astype(np.float64)
    s = cs[b] - cs[a]
    if exclude_self is not None:          # self value is inside the window by construction
        n = n - 1
        s = s - exclude_self
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m.astype(np.float32), n.astype(np.float32)


def build(df):
    dep = df[df.PHASE_mvt == 'DEP'].copy()
    arr = df[df.PHASE_mvt == 'ARR'].copy()
    del df
    gc.collect()
    dep['mvt'] = ep(dep.MVT_TIME_UTC_mvt); dep['sch'] = ep(dep.SCHED_TIME_UTC_mvt)
    dep['aobt'] = ep(dep.AOBT_3_flt); dep['eobt'] = ep(dep.EOBT_1_flt)
    dep['iobt'] = ep(dep.IOBT_flt); dep['lobt'] = ep(dep.LOBT_flt)
    arr['land'] = ep(arr.MVT_TIME_UTC_mvt); arr['onblk'] = ep(arr.BLOCK_TIME_UTC_mvt)
    dep['apt'] = dep.ADEP_mvt.astype(str); arr['apt'] = arr.ADES_mvt.astype(str)
    out = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.values}, index=dep.index)

    # ---- flight-plan / clock gaps (seconds) ----
    out['lobt_mvt_m_lobt'] = (dep.mvt - dep.lobt).astype(np.float32)
    out['lobt_aobt_m_lobt'] = (dep.aobt - dep.lobt).astype(np.float32)
    out['lobt_sch_m_lobt'] = (dep.sch - dep.lobt).astype(np.float32)
    out['lobt_eobt_m_lobt'] = (dep.eobt - dep.lobt).astype(np.float32)
    out['lobt_iobt_m_lobt'] = (dep.iobt - dep.lobt).astype(np.float32)
    out['mvt_m_eobt'] = (dep.mvt - dep.eobt).astype(np.float32)
    out['mvt_m_iobt'] = (dep.mvt - dep.iobt).astype(np.float32)

    # ---- identity fallbacks for flights without an NM record ----
    fl = dep.FLIGHT_mvt.astype(str).str[:3]
    out['op2'] = dep.AIRCRAFT_OPERATOR_flt.astype(object).where(dep.AIRCRAFT_OPERATOR_flt.notna(), fl)
    out['ades2'] = dep.ADES_flt.astype(object).where(dep.ADES_flt.notna(), dep.ADES_mvt.astype(object))

    # ---- local time ----
    lh_m = np.full(len(dep), np.nan, dtype=np.float32); lh_s = lh_m.copy()
    for a, tz in TZ.items():
        m = (dep.apt == a).values
        if not m.any():
            continue
        tm = pd.to_datetime(dep.loc[m, 'MVT_TIME_UTC_mvt'], utc=True).dt.tz_convert(tz)
        ts = pd.to_datetime(dep.loc[m, 'SCHED_TIME_UTC_mvt'], utc=True).dt.tz_convert(tz)
        lh_m[m] = (tm.dt.hour + tm.dt.minute / 60).values
        lh_s[m] = (ts.dt.hour + ts.dt.minute / 60).values
    out['loc_hour_mvt'] = lh_m
    out['loc_hour_sch'] = lh_s

    # ---- neighbour departures' implied taxi, airport level and runway level ----
    imp = (dep.mvt - dep.aobt).clip(0, 3600)
    for c in ['ctx_imp_prev30', 'ctx_imp_prev30_n', 'ctx_imp_pm15', 'ctx_imp_pm15_n',
              'ctx_imp_prev60', 'ctx_imp_rwy_prev30', 'ctx_imp_rwy_prev30_n',
              'ctx_tin_prev60', 'ctx_tin_prev60_n', 'ctx_tin_pm30',
              'seq_gap_prev', 'seq_gap_next', 'seq_n_prev300', 'seq_n_prev600', 'seq_n_prev1200',
              'seq_n_next600']:
        out[c] = np.float32(np.nan)
    for a, g in dep.groupby('apt', sort=False):
        ok = g.aobt.notna().values & g.mvt.notna().values
        gt = g.mvt.values[ok]; gv = imp.loc[g.index].values[ok]
        o = np.argsort(gt, kind='stable'); gt, gv = gt[o], gv[o]
        at = g.mvt.values
        selfv = np.where(ok, imp.loc[g.index].values, 0.0)
        m, n = window_mean(gt, gv, at, -1800, 0, exclude_self=np.where(ok, selfv, np.nan))
        # rows without AOBT are not in the reference set: recompute without self-exclusion
        m2, n2 = window_mean(gt, gv, at, -1800, 0)
        out.loc[g.index, 'ctx_imp_prev30'] = np.where(ok, m, m2)
        out.loc[g.index, 'ctx_imp_prev30_n'] = np.where(ok, n, n2)
        m, n = window_mean(gt, gv, at, -900, 900, exclude_self=np.where(ok, selfv, np.nan))
        m2, n2 = window_mean(gt, gv, at, -900, 900)
        out.loc[g.index, 'ctx_imp_pm15'] = np.where(ok, m, m2)
        out.loc[g.index, 'ctx_imp_pm15_n'] = np.where(ok, n, n2)
        m, _ = window_mean(gt, gv, at, -3600, 0, exclude_self=np.where(ok, selfv, np.nan))
        m2, _ = window_mean(gt, gv, at, -3600, 0)
        out.loc[g.index, 'ctx_imp_prev60'] = np.where(ok, m, m2)
        # arrivals' taxi-in
        ga = arr[arr.apt == a]
        okk = ga.land.notna().values & ga.onblk.notna().values
        lt = ga.land.values[okk]; tv = np.clip(ga.onblk.values[okk] - lt, 0, 3600)
        o = np.argsort(lt, kind='stable'); lt, tv = lt[o], tv[o]
        m, n = window_mean(lt, tv, at, -3600, 0)
        out.loc[g.index, 'ctx_tin_prev60'] = m; out.loc[g.index, 'ctx_tin_prev60_n'] = n
        m, _ = window_mean(lt, tv, at, -1800, 1800)
        out.loc[g.index, 'ctx_tin_pm30'] = m
    for (a, r), g in dep.groupby(['apt', 'RUNWAY_mvt'], sort=False, observed=True):
        ok = g.aobt.notna().values & g.mvt.notna().values
        gt = g.mvt.values[ok]; gv = imp.loc[g.index].values[ok]
        o = np.argsort(gt, kind='stable'); gt, gv = gt[o], gv[o]
        at = g.mvt.values
        selfv = np.where(ok, imp.loc[g.index].values, np.nan)
        m, n = window_mean(gt, gv, at, -1800, 0, exclude_self=selfv)
        m2, n2 = window_mean(gt, gv, at, -1800, 0)
        out.loc[g.index, 'ctx_imp_rwy_prev30'] = np.where(ok, m, m2)
        out.loc[g.index, 'ctx_imp_rwy_prev30_n'] = np.where(ok, n, n2)
        # exact take-off sequence on this runway (all departures, AOBT or not)
        t_all = np.sort(g.mvt.values[~np.isnan(g.mvt.values)])
        i = np.searchsorted(t_all, at, 'left')
        prev = np.where(i > 0, at - t_all[np.maximum(i - 1, 0)], np.nan)
        j = np.searchsorted(t_all, at, 'right')
        nxt = np.where(j < len(t_all), t_all[np.minimum(j, len(t_all) - 1)] - at, np.nan)
        out.loc[g.index, 'seq_gap_prev'] = np.clip(prev, 0, 7200)
        out.loc[g.index, 'seq_gap_next'] = np.clip(nxt, 0, 7200)
        for w, c in [(300, 'seq_n_prev300'), (600, 'seq_n_prev600'), (1200, 'seq_n_prev1200')]:
            out.loc[g.index, c] = (np.searchsorted(t_all, at, 'left') - np.searchsorted(t_all, at - w, 'left'))
        out.loc[g.index, 'seq_n_next600'] = (np.searchsorted(t_all, at + 600, 'right') - np.searchsorted(t_all, at, 'right'))
    for c in out.columns:
        if c not in ('MVT_ID_mvt', 'op2', 'ades2'):
            out[c] = out[c].astype(np.float32)
    out['op2'] = out['op2'].astype(str)
    out['ades2'] = out['ades2'].astype(str)
    return out.reset_index(drop=True)


def read_compact(f):
    x = pd.read_parquet(f, columns=COLS)
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'RUNWAY_mvt', 'STAND_mvt',
              'AIRCRAFT_OPERATOR_flt', 'ADES_flt']:
        x[c] = x[c].astype('category')
    return x


if __name__ == '__main__':
    fs = sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))
    tr = pd.concat([read_compact(f) for f in fs], ignore_index=True)
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'RUNWAY_mvt', 'STAND_mvt',
              'AIRCRAFT_OPERATOR_flt', 'ADES_flt']:
        tr[c] = tr[c].astype('category')
    ctx = build(tr)
    del tr; gc.collect()
    ctx.to_parquet(os.path.join(D, 'ctx_train.parquet'), index=False)
    print('train', ctx.shape, ctx.isna().mean().round(3).to_dict())
    rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)
    ctx = build(rk)
    ctx.to_parquet(os.path.join(D, 'ctx_rank.parquet'), index=False)
    print('rank', ctx.shape)
