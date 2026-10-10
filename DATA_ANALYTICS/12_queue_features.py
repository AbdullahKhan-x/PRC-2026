"""
STEP 1 - Surface queue features.

Builds the features your model is currently blind to: how much traffic is
physically on the airport surface at the moment this aircraft pushes back.

Uses ONLY columns present in the ranking file, so everything here is
computable at prediction time. No target involved anywhere.

Run once. Writes two parquet caches. Later scripts read those, never this.
"""
import pandas as pd
import numpy as np
import gc
import os

DATA_FOLDER = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL',
            'LEMD', 'LFPG', 'LIRF', 'LTFM', 'LSZH']

USECOLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt',
           'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
           'TAXITIME_SEC_mvt', 'AOBT_3_flt', 'RUNWAY_mvt', 'STAND_mvt',
           'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt',
           'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt',
           'ADES_flt', 'CALLSIGN_flt', 'EOBT_1_flt']

CATCOLS = ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
           'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt',
           'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt', 'ADES_flt']

SEC = 'datetime64[s]'


def _to_epoch(series):
    """datetime string column -> int64 epoch seconds, NaT becomes NaN float."""
    t = pd.to_datetime(series, errors='coerce', format='mixed', utc=True)
    v = t.values.astype(SEC).astype('float64')
    v[t.isna().values] = np.nan
    return v


def load_movements(path):
    """Chunked read. Returns (dep_dataframe, arr_context_dataframe)."""
    dep_parts, arr_parts = [], []
    for ch in pd.read_csv(path, usecols=USECOLS, chunksize=400_000,
                          low_memory=False):
        for c in CATCOLS:
            if c in ch.columns:
                ch[c] = ch[c].astype('category')

        ch['mvt_i'] = _to_epoch(ch['MVT_TIME_UTC_mvt'])
        ch['blk_i'] = _to_epoch(ch['BLOCK_TIME_UTC_mvt'])
        ch['sch_i'] = _to_epoch(ch['SCHED_TIME_UTC_mvt'])
        ch['aobt_i'] = _to_epoch(ch['AOBT_3_flt'])
        ch['eobt_i'] = _to_epoch(ch['EOBT_1_flt'])
        ch = ch.drop(columns=['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
                              'SCHED_TIME_UTC_mvt', 'AOBT_3_flt',
                              'EOBT_1_flt'])

        d = ch[ch.PHASE_mvt == 'DEP'].copy()
        d['apt'] = d['ADEP_mvt'].astype(str)
        dep_parts.append(d.drop(columns=['PHASE_mvt']))

        a = ch[ch.PHASE_mvt == 'ARR']
        arr_parts.append(pd.DataFrame({
            'apt': a['ADES_mvt'].astype(str).values,
            'land_i': a['mvt_i'].values,     # ARR movement time = touchdown
            'onblk_i': a['blk_i'].values,    # present in BOTH train and ranking
        }))
        del ch, d, a
        gc.collect()

    dep = pd.concat(dep_parts, ignore_index=True)
    arr = pd.concat(arr_parts, ignore_index=True)
    del dep_parts, arr_parts
    gc.collect()
    return dep, arr


def _counts(ref_sorted, at, lo=None, hi=None):
    """How many values of ref fall in (at-lo, at] or [at, at+hi)."""
    if lo is not None:
        return (np.searchsorted(ref_sorted, at, 'right')
                - np.searchsorted(ref_sorted, at - lo, 'left'))
    return (np.searchsorted(ref_sorted, at + hi, 'right')
            - np.searchsorted(ref_sorted, at, 'left'))


def _surface_block(g, prefix, out):
    """
    Queue features for one group (one airport, or one airport+runway).

    CRITICAL: pushback times and takeoff times must come from the SAME
    subset of rows (those with a usable AOBT), otherwise the concurrent
    count goes negative. That bug is easy to write and silent.
    """
    aobt = g['aobt_i'].values
    mvt = g['mvt_i'].values
    ok = ~np.isnan(aobt) & ~np.isnan(mvt)

    push = np.sort(aobt[ok]).astype('int64')      # pushbacks
    tko = np.sort(mvt[ok]).astype('int64')        # their matching takeoffs
    tko_all = np.sort(mvt[~np.isnan(mvt)]).astype('int64')

    # reference instant: my pushback, or my scheduled time if AOBT is missing
    at = np.where(ok, np.nan_to_num(aobt), g['sch_i'].values).astype('int64')

    # aircraft that pushed back before me and have not yet taken off
    q = (np.searchsorted(push, at, 'left')
         - np.searchsorted(tko, at, 'right'))
    out[prefix + 'queue'] = np.maximum(q, 0).astype(np.int16)

    # demand around me
    out[prefix + 'push_prev15'] = _counts(push, at, lo=900).astype(np.int16)
    out[prefix + 'push_prev60'] = _counts(push, at, lo=3600).astype(np.int16)
    out[prefix + 'push_next15'] = _counts(push, at, hi=900).astype(np.int16)
    out[prefix + 'push_next30'] = _counts(push, at, hi=1800).astype(np.int16)

    # achieved throughput = how fast the runway is actually clearing traffic
    out[prefix + 'tko_prev15'] = _counts(tko_all, at, lo=900).astype(np.int16)
    out[prefix + 'tko_prev60'] = _counts(tko_all, at, lo=3600).astype(np.int16)

    # Little's law: queue length divided by service rate = expected wait.
    # This is the single most physically meaningful thing we can hand a tree.
    rate_per_min = out[prefix + 'tko_prev60'].astype(np.float32) / 60.0
    out[prefix + 'wait_est_min'] = (
        out[prefix + 'queue'] / np.maximum(rate_per_min, 0.05)
    ).clip(0, 240).astype(np.float32)
    return out


def build_features(dep, arr):
    dep = dep.reset_index(drop=True)

    # ---------- airport level ----------
    parts = []
    for apt, g in dep.groupby('apt', sort=False):
        out = pd.DataFrame(index=g.index)
        _surface_block(g, 'apt_', out)
        parts.append(out)
    dep = dep.join(pd.concat(parts))
    del parts
    gc.collect()

    # ---------- runway level ----------
    parts = []
    for _, g in dep.groupby(['apt', 'RUNWAY_mvt'], sort=False, observed=True):
        out = pd.DataFrame(index=g.index)
        _surface_block(g, 'rwy_', out)
        parts.append(out)
    dep = dep.join(pd.concat(parts))
    del parts
    gc.collect()

    # ---------- arrival pressure on the same taxiways ----------
    # ARR on-block time is present in the ranking file, so we know exactly
    # which arrivals are still rolling in when this aircraft pushes back.
    parts = []
    for apt, g in dep.groupby('apt', sort=False):
        sub = arr[arr['apt'].values == apt]
        land = sub['land_i'].values
        onblk = sub['onblk_i'].values
        ok = ~np.isnan(land) & ~np.isnan(onblk)
        ls = np.sort(land[ok]).astype('int64')
        bs = np.sort(onblk[ok]).astype('int64')
        land_all = np.sort(land[~np.isnan(land)]).astype('int64')

        at = np.where(~np.isnan(g['aobt_i'].values),
                      np.nan_to_num(g['aobt_i'].values),
                      g['sch_i'].values).astype('int64')

        aq = (np.searchsorted(ls, at, 'left')
              - np.searchsorted(bs, at, 'right'))
        parts.append(pd.DataFrame({
            'arr_on_surface': np.maximum(aq, 0).astype(np.int16),
            'arr_land_prev15': _counts(land_all, at, lo=900).astype(np.int16),
            'arr_land_prev60': _counts(land_all, at, lo=3600).astype(np.int16),
            'arr_land_next30': _counts(land_all, at, hi=1800).astype(np.int16),
        }, index=g.index))
    dep = dep.join(pd.concat(parts))
    del parts
    gc.collect()

    # total surface occupancy - departures and arrivals share the taxiways
    dep['surface_total'] = (dep['apt_queue'] + dep['arr_on_surface']).astype(np.int16)
    dep['dep_share_of_surface'] = (
        dep['apt_queue'] / np.maximum(dep['surface_total'], 1)
    ).astype(np.float32)
    # how much of the airport's queue is on MY runway
    dep['rwy_share_of_apt_queue'] = (
        dep['rwy_queue'] / np.maximum(dep['apt_queue'], 1)
    ).astype(np.float32)

    # ---------- timing / schedule features ----------
    dep['implied_taxi_aobt'] = (dep['mvt_i'] - dep['aobt_i']).astype(np.float32)
    dep['has_aobt'] = dep['aobt_i'].notna().astype(np.int8)
    # how late the pushback was vs the filed off-block estimate
    dep['pushback_delay'] = (dep['aobt_i'] - dep['eobt_i']).astype(np.float32)
    dep['sched_dev'] = (dep['mvt_i'] - dep['sch_i']).astype(np.float32)

    sch = pd.to_datetime(dep['sch_i'], unit='s', utc=True)
    dep['hour'] = sch.dt.hour.astype(np.int8)
    dep['dow'] = sch.dt.dayofweek.astype(np.int8)
    dep['month'] = sch.dt.month.astype(np.int8)
    dep['doy'] = sch.dt.dayofyear.astype(np.int16)
    dep['minute_of_day'] = (sch.dt.hour * 60 + sch.dt.minute).astype(np.int16)
    dep['date'] = sch.dt.date.astype(str)
    dep['hour_sin'] = np.sin(2 * np.pi * dep['minute_of_day'] / 1440).astype(np.float32)
    dep['hour_cos'] = np.cos(2 * np.pi * dep['minute_of_day'] / 1440).astype(np.float32)
    dep['dow_sin'] = np.sin(2 * np.pi * dep['dow'] / 7).astype(np.float32)
    dep['dow_cos'] = np.cos(2 * np.pi * dep['dow'] / 7).astype(np.float32)

    # runway configuration in use right now - which runways are active in the
    # surrounding hour. Config changes are a known taxi-time driver.
    dep = dep.sort_values(['apt', 'date', 'RUNWAY_mvt'])
    cfg = (dep.groupby(['apt', 'date', 'hour'], observed=True)['RUNWAY_mvt']
           .nunique().rename('rwys_active_this_hour').reset_index())
    dep = dep.merge(cfg, on=['apt', 'date', 'hour'], how='left')
    dep['rwys_active_this_hour'] = dep['rwys_active_this_hour'].fillna(1).astype(np.int8)

    return dep


def run(name, csv_path, out_path):
    print(f"\n=== {name} ===")
    print("reading...")
    dep, arr = load_movements(csv_path)
    print(f"  DEP rows {len(dep):,}   ARR context rows {len(arr):,}")
    print("building queue features...")
    dep = build_features(dep, arr)
    del arr
    gc.collect()
    dep.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}   shape {dep.shape}")
    print(dep[['apt_queue', 'rwy_queue', 'arr_on_surface',
               'apt_wait_est_min', 'rwy_wait_est_min']].describe().round(2))
    del dep
    gc.collect()


if __name__ == "__main__":
    run("TRAINING 2025",
        os.path.join(DATA_FOLDER, "training_combined.csv"),
        os.path.join(DATA_FOLDER, "dep_q_train.parquet"))

    run("RANKING 2026",
        os.path.join(DATA_FOLDER, "ranking.csv"),
        os.path.join(DATA_FOLDER, "dep_q_rank.parquet"))

    print("\ndone. next script reads the two parquet files only.")
