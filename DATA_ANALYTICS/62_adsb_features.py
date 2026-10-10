"""
ADS-B (adsb.lol, ODbL 1.0) features for every departure, from the per-day point files produced on Kaggle
(kaggle/adsb_kaggle.py -> data/adsb/<period>/pts/pts_YYYY-MM-DD.parquet). Label-free: uses only positions,
take-off times and callsigns, identical for training and ranking rows.

Per departure, ADS-B take-off = first airborne point after a ground point at that airport, matched to MVT
(callsign within 300 s, else nearest unused take-off within 90 s). From the continuous ground segment before it:
  adsb_matched     1 if a take-off was matched
  adsb_move_gap    MVT - first time the aircraft is > 30 m from where the segment starts (movement start)
  adsb_first_gap   MVT - first ground point of the segment (first heard on the ground)
  adsb_npts        points in the ground segment
  adsb_gs0         ground speed at the first ground point (0 = parked when first heard)
  adsb_cov_day     share of that airport-day's departures matched (coverage regime; NaN when no file that day)
Days without a point file -> all NaN (the model must not read "no file" as "no aircraft").

  python 62_adsb_features.py          writes adsb_train.parquet / adsb_rank.parquet
"""
import os, glob, gc
import numpy as np, pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt']
FEATS = ['adsb_matched', 'adsb_move_gap', 'adsb_first_gap', 'adsb_npts', 'adsb_gs0', 'adsb_cov_day']


def day_files():
    fs = glob.glob(os.path.join(D, 'data/adsb', '**', 'pts_*.parquet'), recursive=True)
    return {os.path.basename(f)[4:14]: f for f in fs}


def takeoffs(pts):
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
            dist = np.hypot((la[j:i + 1] - la[j]) * 111e3, (lo[j:i + 1] - lo[j]) * 111e3 * np.cos(np.radians(la[j])))
            mv = np.nonzero(dist > 30)[0]
            ev.append((ap, tko, t[j], t[j + mv[0]] if len(mv) else np.nan, i - j + 1, gs[j], cs[i].strip().upper()))
    return pd.DataFrame(ev, columns=['apt', 'tko', 'e_first', 'e_move', 'npts', 'gs0', 'cs'])


def _events(path):
    return takeoffs(pd.read_parquet(path))


EV = {}          # day -> take-off events, computed in parallel once per day


def precompute(days, files, workers=int(os.environ.get('ADSB_WORKERS', '4'))):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    todo = [d for d in sorted(set(days)) if d in files and d not in EV]
    # 'fork' explicitly: Python 3.14 defaults to forkserver, which re-imports the calling script in every worker.
    # Windows has no fork -> compute sequentially (same result, slower).
    if 'fork' in mp.get_all_start_methods() and workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as ex:
            for d, ev in zip(todo, ex.map(_events, [files[d] for d in todo])):
                EV[d] = ev
    else:
        for d in todo:
            EV[d] = _events(files[d])
    print(f'  take-off events computed for {len(todo)} days', flush=True)


def build(raw, files):
    d = raw[raw.PHASE_mvt == 'DEP'][COLS].copy()
    d['mvt'] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    d['day'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')
    d['cs'] = d.CALLSIGN_flt.astype(str).str.strip().str.upper()
    out = []
    for day, g in d.groupby('day'):
        res = pd.DataFrame({'MVT_ID_mvt': g.MVT_ID_mvt.values})
        for c in FEATS:
            res[c] = np.nan
        if day in files:
            # take-offs just after midnight may belong to traces of the previous/next day file: use the day's file only
            ev = EV[day] if day in EV else takeoffs(pd.read_parquet(files[day]))
            m = {c: np.full(len(g), np.nan) for c in FEATS}
            m['adsb_matched'][:] = 0
            for ap, gi in g.reset_index(drop=True).groupby('ADEP_mvt').indices.items():
                e = ev[ev.apt == ap].reset_index(drop=True); used = set()
                if e.empty:
                    continue
                sub = g.iloc[gi]
                order = np.argsort(sub.mvt.to_numpy())
                for k_ in order:
                    row_i = gi[k_]; mvt = sub.mvt.to_numpy()[k_]; cs = sub.cs.to_numpy()[k_]
                    dt_ = np.abs(e.tko.to_numpy() - mvt)
                    cand = np.nonzero((e.cs.to_numpy() == cs) & (dt_ <= 300))[0]
                    if not len(cand):
                        cand = np.nonzero(dt_ <= 90)[0]
                    cand = [c for c in cand if c not in used]
                    if not cand:
                        continue
                    k = min(cand, key=lambda c: dt_[c]); used.add(k)
                    m['adsb_matched'][row_i] = 1
                    m['adsb_move_gap'][row_i] = mvt - e.e_move[k]
                    m['adsb_first_gap'][row_i] = mvt - e.e_first[k]
                    m['adsb_npts'][row_i] = e.npts[k]
                    m['adsb_gs0'][row_i] = e.gs0[k]
                cov = m['adsb_matched'][gi].mean()
                m['adsb_cov_day'][gi] = cov
            for c in FEATS:
                res[c] = m[c]
        out.append(res)
    out = pd.concat(out, ignore_index=True)
    for c in FEATS:
        out[c] = out[c].astype(np.float32)
    return out


if __name__ == '__main__':
    files = day_files()
    print(f'{len(files)} ADS-B day files: {min(files) if files else None} .. {max(files) if files else None}', flush=True)
    tr = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))],
                   ignore_index=True)
    precompute(pd.to_datetime(tr.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').unique(), files)
    a = build(tr, files); del tr; gc.collect()
    a.to_parquet(os.path.join(D, 'adsb_train.parquet'), index=False)
    print('train', a.shape, 'matched share (days with files):', round(a.adsb_matched.mean(), 3), flush=True)
    rk = pd.read_parquet(os.path.join(D, 'data/raw/ranking.parquet'), columns=COLS)
    precompute(pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').unique(), files)
    b = build(rk, files)
    b.to_parquet(os.path.join(D, 'adsb_rank.parquet'), index=False)
    print('rank', b.shape, 'matched share (days with files):', round(b.adsb_matched.mean(), 3))
    if os.environ.get('MERGE_CTX'):          # also write ctxads_{train,rank} = ctxall + adsb (for harness runs)
        for s_ in ('train', 'rank'):
            pd.read_parquet(os.path.join(D, f'ctxall_{s_}.parquet')).merge(pd.read_parquet(os.path.join(D, f'adsb_{s_}.parquet')),
                on='MVT_ID_mvt', how='left').to_parquet(os.path.join(D, f'ctxads_{s_}.parquet'), index=False)
        print('ctxads written', flush=True)
