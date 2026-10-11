"""
ADS-B take-off events WITH GEOMETRY (extends 62_adsb_features.py, same event definition and matcher).

Why: 73% of the remaining error is in the stand phase (recorded BLOCK -> first ADS-B movement). The time-only
features of 62 cannot tell WHERE the aircraft was first heard: at airports with poor ground coverage near the gates
the unobserved part before first contact varies with how far from the stand the aircraft already was.

Per take-off event (the continuous ground segment before it, as in 62):
  first point (lat, lon), movement-start point, last ground point (take-off roll start area), path length,
  largest time gap inside the segment.
Per departure (matched as in 62), label-free:
  adsb_f_x, adsb_f_y      first-contact position, metres east/north of the airport's median take-off point
  adsb_d_stand            first contact -> stand reference position (median first point of that stand's departures
                          heard PARKED: gs0 < 1 and >= 60 s before movement; NaN if < 3 such departures)
  adsb_d_stand_typ        first contact -> median first-contact position of that stand's departures (all)
  adsb_d_first_tko        straight line first contact -> last ground point
  adsb_path               path length of the ground segment
  adsb_maxgap             largest time gap between consecutive ground points of the segment
  adsb_d_move             first contact -> movement start
Stand reference tables are built from the given days only (label-free; pass all available days).

  python 62b_adsb_geo.py events DAYS_GLOB OUT.parquet        take-off events with geometry for every day file matching
  python 62b_adsb_geo.py build      training 2025 + ranking.parquet -> geo_train.parquet, geo_rank.parquet and
                                    geo_tables.pkl (airport / stand reference positions, reused by the final pipeline)
  (module use: events_for_days(), match(), stand_tables(), build_geo())
"""
import os, sys, glob, gc, pickle
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'STAND_mvt']
GEO = ['adsb_f_x', 'adsb_f_y', 'adsb_d_stand', 'adsb_d_stand_typ', 'adsb_d_first_tko', 'adsb_path', 'adsb_maxgap', 'adsb_d_move']


def _m(lat0, lon0, lat, lon):
    """metres (east, north) of (lat, lon) from (lat0, lon0) - local flat approximation, fine within an airport"""
    return (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 110_540.0


def takeoffs_geo(pts):
    pts = pts.sort_values(['icao', 't']).reset_index(drop=True)
    gr_all = (pts.alt < 0).to_numpy()
    ev = []
    for (ic, ap), idx in pts.groupby(['icao', 'apt'], observed=True).indices.items():
        g = pts.iloc[idx]
        t, gr = g.t.to_numpy(), gr_all[idx]
        la, lo, gs = g.lat.to_numpy(), g.lon.to_numpy(), g.gs.to_numpy()
        cs = g.cs.astype(str).to_numpy()
        for i in np.nonzero(gr[:-1] & ~gr[1:])[0]:
            if t[i + 1] - t[i] > 60:
                continue
            tko = t[i + 1]; j = i
            while j > 0 and gr[j - 1] and t[j] - t[j - 1] < 300 and tko - t[j - 1] < 3 * 3600:
                j -= 1
            x, y = _m(la[j], lo[j], la[j:i + 1], lo[j:i + 1])
            dist = np.hypot(x, y)
            mv = np.nonzero(dist > 30)[0]
            k = j + mv[0] if len(mv) else -1
            path = float(np.hypot(np.diff(x), np.diff(y)).sum()) if i > j else 0.0
            mg = float(np.diff(t[j:i + 1]).max()) if i > j else np.nan
            ev.append((ap, tko, t[j], t[k] if k >= 0 else np.nan, i - j + 1, gs[j], cs[i].strip().upper(),
                       la[j], lo[j], la[k] if k >= 0 else np.nan, lo[k] if k >= 0 else np.nan, la[i], lo[i], path, mg))
    return pd.DataFrame(ev, columns=['apt', 'tko', 'e_first', 'e_move', 'npts', 'gs0', 'cs', 'f_lat', 'f_lon',
                                     'm_lat', 'm_lon', 'l_lat', 'l_lon', 'path', 'maxgap'])


def _events(path):
    e = takeoffs_geo(pd.read_parquet(path))
    e['day'] = os.path.basename(path)[4:14]
    return e


def events_for_days(files, workers=int(os.environ.get('ADSB_WORKERS', '3'))):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as ex:
        out = list(ex.map(_events, files))
    return pd.concat(out, ignore_index=True)


def match(raw, ev):
    """62's matcher (callsign within 300 s, else nearest unused take-off within 90 s), returns the event row per DEP"""
    d = raw[raw.PHASE_mvt == 'DEP'][COLS].copy()
    d['mvt'] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    d['day'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')
    d['cs'] = d.CALLSIGN_flt.astype(str).str.strip().str.upper()
    ev = ev.reset_index(drop=True)
    rows = []
    evd = {k: v for k, v in ev.groupby('day')}
    for day, g in d.groupby('day'):
        if day not in evd:
            continue
        evday = evd[day]
        for ap, gi in g.reset_index(drop=True).groupby('ADEP_mvt', observed=True).indices.items():
            e = evday[evday.apt == ap]; used = set()
            if e.empty:
                continue
            sub = g.iloc[gi]
            tk, ecs, eid = e.tko.to_numpy(), e.cs.to_numpy(), e.index.to_numpy()
            mvts, css, ids = sub.mvt.to_numpy(), sub.cs.to_numpy(), sub.MVT_ID_mvt.to_numpy()
            for k_ in np.argsort(mvts):
                dt_ = np.abs(tk - mvts[k_])
                cand = np.nonzero((ecs == css[k_]) & (dt_ <= 300))[0]
                if not len(cand):
                    cand = np.nonzero(dt_ <= 90)[0]
                cand = [c for c in cand if c not in used]
                if not cand:
                    continue
                k = min(cand, key=lambda c: dt_[c]); used.add(k)
                rows.append((ids[k_], mvts[k_], eid[k]))
    m = pd.DataFrame(rows, columns=['MVT_ID_mvt', 'mvt', 'eid'])
    m = m.merge(ev[['tko', 'e_first', 'e_move', 'npts', 'gs0', 'f_lat', 'f_lon', 'm_lat', 'm_lon', 'l_lat', 'l_lon', 'path', 'maxgap']],
                left_on='eid', right_index=True, how='left').drop(columns='eid')
    return d[['MVT_ID_mvt', 'ADEP_mvt', 'STAND_mvt']].merge(m, on='MVT_ID_mvt', how='left')


def _key(x):
    """stand key, identical in every pandas version (pandas 2 turns None into 'None', pandas 3 keeps NaN): missing -> ''"""
    x = pd.Series(x).astype(object)
    return np.where(x.isna(), '', x.map(str)).astype(object)


def stand_tables(mt):
    """label-free reference positions from matched departures (mt = match() output, any days)"""
    mt = mt[mt.f_lat.notna()].copy()
    mt['apt'] = _key(mt.ADEP_mvt.values); mt['stand'] = _key(mt.STAND_mvt.values)
    ref = mt.groupby('apt')[['l_lat', 'l_lon']].median().rename(columns={'l_lat': 'a_lat', 'l_lon': 'a_lon'})
    mt = mt[mt.stand != '']
    parked = mt[(mt.gs0 < 1) & ((mt.e_move - mt.e_first) >= 60)]
    sp = parked.groupby(['apt', 'stand']).agg(s_lat=('f_lat', 'median'), s_lon=('f_lon', 'median'), s_n=('f_lat', 'size'))
    sp = sp[sp.s_n >= 3]
    st = mt.groupby(['apt', 'stand']).agg(t_lat=('f_lat', 'median'), t_lon=('f_lon', 'median'), t_n=('f_lat', 'size'))
    st = st[st.t_n >= 3]
    return ref, sp, st


def build_geo(mt, ref, sp, st):
    m = mt.copy()
    m['apt'] = _key(m.ADEP_mvt.values); m['stand'] = _key(m.STAND_mvt.values)
    m = m.merge(ref, left_on='apt', right_index=True, how='left')
    m = m.merge(sp, left_on=['apt', 'stand'], right_index=True, how='left').merge(st, left_on=['apt', 'stand'], right_index=True, how='left')
    fx, fy = _m(m.a_lat.values, m.a_lon.values, m.f_lat.values, m.f_lon.values)
    out = pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt.values})
    out['adsb_f_x'], out['adsb_f_y'] = fx, fy
    out['adsb_d_stand'] = np.hypot(*_m(m.f_lat.values, m.f_lon.values, m.s_lat.values, m.s_lon.values))
    out['adsb_d_stand_typ'] = np.hypot(*_m(m.f_lat.values, m.f_lon.values, m.t_lat.values, m.t_lon.values))
    out['adsb_d_first_tko'] = np.hypot(*_m(m.f_lat.values, m.f_lon.values, m.l_lat.values, m.l_lon.values))
    out['adsb_path'] = m.path.values
    out['adsb_maxgap'] = m.maxgap.values
    out['adsb_d_move'] = np.hypot(*_m(m.f_lat.values, m.f_lon.values, m.m_lat.values, m.m_lon.values))
    for c in GEO:
        out[c] = out[c].astype(np.float32)
    return out


def day_files():
    fs = glob.glob(os.path.join(D, 'data/adsb', '**', 'pts_*.parquet'), recursive=True)
    return {os.path.basename(f)[4:14]: f for f in fs}


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'build':
    files = day_files()
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))],
                   ignore_index=True)
    rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)
    days = sorted(set(pd.to_datetime(pd.concat([tr.MVT_TIME_UTC_mvt, rk.MVT_TIME_UTC_mvt]), utc=True).dt.strftime('%Y-%m-%d')) & set(files))
    print(f'{len(files)} ADS-B day files, {len(days)} used', flush=True)
    ev = events_for_days([files[d] for d in days])
    print(f'{len(ev):,} take-off events', flush=True)
    mtr, mrk = match(tr, ev), match(rk, ev)
    del tr, rk; gc.collect()
    tables = stand_tables(pd.concat([mtr, mrk], ignore_index=True))
    pickle.dump(tables, open(os.path.join(D, 'geo_tables.pkl'), 'wb'))
    print('stand references (parked):', tables[1].reset_index().groupby('apt').size().to_dict(), flush=True)
    for name, mt in [('train', mtr), ('rank', mrk)]:
        o = build_geo(mt, *tables)
        o.to_parquet(os.path.join(D, f'geo_{name}.parquet'), index=False)
        print(f'geo_{name}: {len(o):,} rows, first contact known {o.adsb_f_x.notna().mean():.3f}, stand ref known {o.adsb_d_stand.notna().mean():.3f}', flush=True)
        a = os.path.join(D, f'adsb_{name}.parquet')
        if os.path.exists(a):                        # consistency with 62: geometry exactly where 62 matched
            j = o.merge(pd.read_parquet(a, columns=['MVT_ID_mvt', 'adsb_matched']), on='MVT_ID_mvt', how='left')
            print(f'   agreement with {os.path.basename(a)} matched flag: {((j.adsb_f_x.notna()) == (j.adsb_matched == 1)).mean():.4f}', flush=True)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'events':
    fs = sorted(glob.glob(sys.argv[2], recursive=True))
    print(f'{len(fs)} day files', flush=True)
    ev = events_for_days(fs)
    ev.to_parquet(sys.argv[3], index=False)
    print(f'wrote {sys.argv[3]}: {len(ev):,} take-off events', flush=True)
