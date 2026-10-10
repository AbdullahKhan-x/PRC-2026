"""
External, openly licensed context (documented in release/README.md):

  EUROCONTROL daily airport series (ansperformance.eu / eurocontrol.int performance data, free use
  with attribution), per airport and UTC day:
    ext_dep_n        IFR departures that day                              (atfm_slot_adherence FLT_DEP_1)
    ext_reg_share    share of departures under an ATFM regulation          (FLT_DEP_REG_1 / FLT_DEP_1)
    ext_late_share   regulated departures off slot, late                   (FLT_DEP_OUT_LATE_1 / FLT_DEP_REG_1)
    ext_pre_dly      all-causes pre-departure delay per IFR departure (min)(DLY_ALL_PRE_2 / FLT_DEP_IFR_2)
    ext_atc_dly      ATC pre-departure delay per departure (min)           (DLY_ATC_PRE_3 / FLT_DEP_3)
    ext_pre_dly_rel  ext_pre_dly minus that airport's 2025 mean (drift)

  OPDI v0.0.2 flight lists (PRC + OpenSky Network, open data with attribution), matched on
  callsign + departure airport, first_seen within [MVT - 2 h, MVT + 10 min]:
    ext_fs_gap       MVT - first_seen: ADS-B first contact before take-off (ground coverage only
                     at some airports, e.g. LSZH / EDDF; ~take-off elsewhere)
    ext_ground       MVT - last_seen of the same airframe's (icao24) previous flight INTO this
                     airport: time on the ground since landing (NaN if > 48 h or unknown)

writes ext_train.parquet / ext_rank.parquet  (MVT_ID_mvt + ext_* columns)
"""
import os, glob, gc
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X = os.path.join(D, 'data/ext')
A = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt']


def daily():
    def rd(n):
        d = pd.concat([pd.read_csv(os.path.join(X, f'{n}_{y}.csv')) for y in (2025, 2026)])
        d = d[d.APT_ICAO.isin(A)].copy(); d['day'] = pd.to_datetime(d.FLT_DATE).dt.date
        return d.drop_duplicates(['APT_ICAO', 'day']).set_index(['APT_ICAO', 'day'])
    s, p, c = rd('atfm_slot_adherence'), rd('all_pre_departure_delays'), rd('atc_pre_departure_delays')
    out = pd.DataFrame(index=s.index)
    out['ext_dep_n'] = s.FLT_DEP_1
    out['ext_reg_share'] = s.FLT_DEP_REG_1 / s.FLT_DEP_1.where(s.FLT_DEP_1 > 0)
    out['ext_late_share'] = s.FLT_DEP_OUT_LATE_1 / s.FLT_DEP_REG_1.where(s.FLT_DEP_REG_1 > 0)
    out = out.join((p.DLY_ALL_PRE_2 / p.FLT_DEP_IFR_2.where(p.FLT_DEP_IFR_2 > 0)).rename('ext_pre_dly'), how='outer')
    out = out.join((c.DLY_ATC_PRE_3 / c.FLT_DEP_3.where(c.FLT_DEP_3 > 0)).rename('ext_atc_dly'), how='outer')
    y25 = out[[d.year == 2025 for d in out.index.get_level_values(1)]]
    base = y25.groupby(level=0).ext_pre_dly.mean()
    out['ext_pre_dly_rel'] = out.ext_pre_dly - out.index.get_level_values(0).map(base).values
    return out.astype(np.float32)


def opdi(months):
    fs = [os.path.join(X, 'opdi', f'flight_list_{m}.parquet') for m in months]
    parts = []
    for f in fs:                                   # filter each month to our airports BEFORE concatenating (memory)
        if not os.path.exists(f):
            continue
        x = pd.read_parquet(f, columns=['icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen'])
        x = x[x.adep.isin(A) | x.ades.isin(A)]
        for c in ['icao24', 'flt_id', 'adep', 'ades']:
            x[c] = x[c].astype('category')
        parts.append(x); gc.collect()
    o = pd.concat(parts, ignore_index=True)
    for c in ['icao24', 'adep', 'ades']:
        o[c] = o[c].astype(str)
    o['fs'] = pd.to_datetime(o.first_seen, utc=True); o['ls'] = pd.to_datetime(o.last_seen, utc=True)
    o['cs'] = o.flt_id.astype(str).str.strip().str.upper()
    return o


def build(raw, days, o):
    d = raw[raw.PHASE_mvt == 'DEP'][COLS].copy()
    d['mvt'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    out = pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt.values})
    key = pd.MultiIndex.from_arrays([d.ADEP_mvt.astype(str).values, d.mvt.dt.date.values])
    for c in days.columns:
        out[c] = days[c].reindex(key).values
    # OPDI match: callsign + adep, first_seen in [mvt - 2 h, mvt + 10 min], closest
    d['cs'] = d.CALLSIGN_flt.astype(str).str.strip().str.upper()
    dep = o[o.adep.isin(A)][['cs', 'adep', 'fs', 'icao24']].copy()
    dep['day'] = dep.fs.dt.floor('D')
    # join on callsign + airport + DAY (same day, or first_seen on the previous day for after-midnight take-offs);
    # joining on callsign + airport alone matched every flight of that callsign over 14 months (OOM)
    left = d[['MVT_ID_mvt', 'cs', 'ADEP_mvt', 'mvt']].copy()
    left['day'] = left.mvt.dt.floor('D')
    m = pd.concat([left.merge(dep, left_on=['cs', 'ADEP_mvt', 'day'], right_on=['cs', 'adep', 'day']),
                   left.assign(day=left.day - pd.Timedelta('1D')).merge(dep, left_on=['cs', 'ADEP_mvt', 'day'], right_on=['cs', 'adep', 'day'])],
                  ignore_index=True)
    m['gap'] = (m.mvt - m.fs).dt.total_seconds()
    m = m[m.gap.between(-600, 7200)]
    m = m.loc[m.gap.abs().groupby(m.MVT_ID_mvt).idxmin()]
    out = out.merge(m[['MVT_ID_mvt', 'gap', 'icao24', 'fs']].rename(columns={'gap': 'ext_fs_gap'}), on='MVT_ID_mvt', how='left')
    # previous landing of the same airframe at this airport (OPDI flight with ades == airport, last_seen before my first_seen)
    arr = o[o.ades.isin(A)][['icao24', 'ades', 'ls']].dropna().sort_values('ls')
    q = out[['MVT_ID_mvt', 'icao24', 'fs']].merge(d[['MVT_ID_mvt', 'ADEP_mvt', 'mvt']], on='MVT_ID_mvt')
    q = q.dropna(subset=['icao24', 'fs']).sort_values('fs')
    q = pd.merge_asof(q, arr.rename(columns={'ades': 'ADEP_mvt'}), left_on='fs', right_on='ls',
                      by=['icao24', 'ADEP_mvt'], direction='backward', tolerance=pd.Timedelta('48h'))
    q['ext_ground'] = (q.mvt - q.ls).dt.total_seconds()
    out = out.merge(q[['MVT_ID_mvt', 'ext_ground']], on='MVT_ID_mvt', how='left').drop(columns=['icao24', 'fs'])
    for c in out.columns[1:]:
        out[c] = out[c].astype(np.float32)
    return out


if __name__ == '__main__':
    days = daily()
    m25 = [f'2025{m:02d}' for m in range(1, 13)]
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))],
                   ignore_index=True)
    o = opdi(['202412'] + m25 + ['202601'])
    e = build(tr, days, o); del tr, o; gc.collect()
    e.to_parquet(os.path.join(D, 'ext_train.parquet'), index=False)
    print('train', e.shape, e.isna().mean().round(3).to_dict(), flush=True)
    rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)
    o = opdi(['202512', '202601', '202602', '202605', '202606', '202607', '202608'])
    e = build(rk, days, o)
    e.to_parquet(os.path.join(D, 'ext_rank.parquet'), index=False)
    print('rank', e.shape, e.isna().mean().round(3).to_dict())
