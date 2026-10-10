"""
Score post-processing rules on held-out 2025 months (out-of-month predictions from
41_oof_harness.py). Every rate a rule uses is estimated leave-one-month-out from the raw
2025 labels, so nothing is fitted on the month being scored.

  python 43_eval_postrules.py TAG          (reads candidates/oof_{TAG}_m*.parquet)
"""
import os, sys, glob
import numpy as np
import pandas as pd

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else 'v35'
WIN = 3606.0
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
        'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'AOBT_3_flt', 'LOBT_flt']
raw = pd.concat([pd.read_parquet(f, columns=cols) for f in
                 sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))], ignore_index=True)
raw = raw[raw.PHASE_mvt == 'DEP'].drop(columns='PHASE_mvt')
ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
raw['mvt'], raw['sch'], raw['blk'] = ep(raw.MVT_TIME_UTC_mvt), ep(raw.SCHED_TIME_UTC_mvt), ep(raw.BLOCK_TIME_UTC_mvt)
raw['aobt'], raw['lobt'] = ep(raw.AOBT_3_flt), ep(raw.LOBT_flt)
raw['y'] = raw.TAXITIME_SEC_mvt.astype(float)
raw['sd'] = raw.mvt - raw.sch
raw['mon'] = pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True).dt.month.values
loc = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
raw['cross'] = (loc(raw.MVT_TIME_UTC_mvt) - loc(raw.SCHED_TIME_UTC_mvt)).dt.days.values
raw['nonm'] = raw.aobt.isna()
raw['rome'] = (raw.ADEP_mvt == 'LIRF') & raw.nonm
raw['is_copy'] = (raw.y - raw.sd).abs() <= 60
raw['shift'] = (raw.y > 43200) & (raw.y - 86400 * raw.cross).between(0, 7200) & (raw.cross >= 1)
raw = raw.drop(columns=['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt'])

oof = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, 'candidates', f'oof_{TAG}_m*.parquet')))])
df = oof.merge(raw, on='MVT_ID_mvt', how='left', suffixes=('', '_r'))
assert df.y.notna().all()
W_APT = float(os.environ.get('APT_W', '0'))       # per-airport runs (PER_APT=1): base = (1-w)*global + w*per-airport
if W_APT > 0:
    df['base'] = (1 - W_APT) * df['base'] + W_APT * df['base_apt']
    print(f'base = {1-W_APT:.2f} * global + {W_APT:.2f} * per-airport')
print(f'OOF rows {len(df):,}  months {sorted(df.mon.unique())}')

sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import rome_rates as rr
RR = rr.load_2025(D); RR['mon'] = pd.to_datetime(RR.SCHED_TIME_UTC_mvt, utc=True).dt.month.values
fe = pd.concat([pd.read_parquet(f, columns=['MVT_ID_mvt', 'FLIGHT_mvt', 'ADES_mvt']) for f in
                sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))])
df = df.merge(fe, on='MVT_ID_mvt', how='left')

ROME_BANDS = {'R2': (3600, 7200), 'R1': (7200, 10800), 'C': (10800, 20000)}
SUMMER = [6, 7, 8, 9]


def apply(d, fit, rules):
    base, p, sd = d.base.values, d.p.values.copy(), d.sd.values
    pool = sd > 7200
    has_l = ~np.isnan(d.lobt.values)
    if 'D' in rules:
        p[pool & d.nonm.values & ~d.rome.values] = 0.0
    if 'A' in rules:
        p[has_l & pool & (np.abs(d.sch.values - d.lobt.values) > WIN)] = 0.0
    pc = d.pc.fillna(0).values if 'pc' in d.columns else 0.0          # copy-mixture runs (COPYMIX=1)
    pred = np.clip(np.where(pool, p * sd + (1 - p) * base, pc * sd + (1 - pc) * base), 60, 172800)
    for key, (lo, hi) in ROME_BANDS.items():
        if key in rules:
            f = fit[fit.rome & (fit.sd > lo) & (fit.sd <= hi)]
            pc = f.is_copy.mean(); tc = f.y[~f.is_copy & (f.y <= 10800)].median()
            m = d.rome.values & (sd > lo) & (sd <= hi)
            pred = np.where(m, pc * sd + (1 - pc) * tc, pred)
    if 'CS' in rules:      # seasonal version of C: copy rate fitted on same-season months only
        lo, hi = ROME_BANDS['C']
        summer = d.mon.values[0] in SUMMER
        f = fit[fit.rome & (fit.sd > lo) & (fit.sd <= hi) & (fit.mon.isin(SUMMER) == summer)]
        pc = f.is_copy.mean(); tc = f.y[~f.is_copy & (f.y <= 10800)].median()
        m = d.rome.values & (sd > lo) & (sd <= hi)
        pred = np.where(m, pc * sd + (1 - pc) * tc, pred)
    if 'E' in rules:       # Rome no-NM 1-5.5 h: band x airline x destination copy rate (rome_rates.py), fitted without month M
        M = int(d.mon.values[0])
        tab = rr.fit(RR[RR.mon != M])
        pe, _ = rr.predict(tab, sd, d.FLIGHT_mvt.values, d.ADES_mvt.values)
        m = d.rome.values & (sd > 3600) & (sd <= 20000)
        pred = np.where(m, pe, pred)
    if 'B' in rules:
        f = fit[fit.rome & (fit.cross >= 1) & (fit.sd > 55000) & (fit.sd <= 100000)]
        ws = (f['shift'].sum() + 1) / (len(f) + 2)
        m = d.rome.values & (d.cross.values >= 1) & (sd > 55000) & (sd <= 100000)
        pred = np.where(m, ws * (86400 * d.cross.values + 950) + (1 - ws) * pred, pred)
    if 'R3' in rules:      # NM AOBT more than an hour after LOBT: AOBT cannot be the airport off-block
        f = fit[(fit.aobt - fit.lobt) > WIN]
        c = np.median(f.blk - f.lobt)
        m = has_l & ~np.isnan(d.aobt.values) & ((d.aobt.values - d.lobt.values) > WIN)
        pred = np.where(m, d.mvt.values - d.lobt.values - c, pred)
    if 'R4' in rules:      # no NM record outside Rome: cell mean by airport x lateness band
        bands = [-1e9, 1800, 3600, 7200, 1e9]
        fb = pd.cut(fit.sd, bands, labels=False); db = pd.cut(d.sd, bands, labels=False)
        f = fit[fit.nonm & ~fit.rome].assign(b=fb[fit.nonm & ~fit.rome])
        cm = f[f.y <= 10800].groupby(['ADEP_mvt', 'b']).y.mean()
        key = pd.MultiIndex.from_arrays([d.ADEP_mvt.values, db.values])
        cellm = cm.reindex(key).values
        m = d.nonm.values & ~d.rome.values & (db.values >= 2) & ~np.isnan(cellm)
        pred = np.where(m, 0.5 * pred + 0.5 * cellm, pred)
    if 'A' in rules:
        lo = np.maximum(d.mvt.values - d.lobt.values - WIN, 0)
        hi = d.mvt.values - d.lobt.values + WIN
        pred = np.where(has_l, np.clip(pred, lo, np.maximum(hi, lo)), pred)
    return pred


lottery = (df.y > 10800) & ~df['is_copy'] & ~df['shift']
sets = os.environ.get('SETS', ',A,AC,ACD,ACDE,ADE,ADE R2,ACDB,ACDEB').split(',')
res = {s: {} for s in sets}
for M in sorted(df.mon.unique()):
    d, fit = df[df.mon == M], raw[raw.mon != M]
    for s in sets:
        toks = set()
        for t in s.split():
            toks |= {t} if (t.startswith('R') or t == 'CS') else set(t)
        pr = apply(d, fit, toks)
        e2 = (pr - d.y.values) ** 2
        res[s][M] = (e2.sum(), len(d), e2[~lottery[df.mon == M].values].sum(), int((~lottery[df.mon == M]).sum()))


def rm(s, ms, nl=False):
    a = sum(res[s][m][2 if nl else 0] for m in ms if m in res[s])
    n = sum(res[s][m][3 if nl else 1] for m in ms if m in res[s])
    return np.sqrt(a / n) if n else np.nan


mons = sorted(df.mon.unique())
print(f"\n{'rules':<14}" + ''.join(f'{"m"+str(m):>9}' for m in mons) + f"{'Jan+Jul':>9}{'Feb+Jun':>9}{'all':>9}{'all(no lottery)':>17}")
for s in sets:
    print(f"{s or 'v35-like':<14}" + ''.join(f'{rm(s, [m]):9.2f}' for m in mons)
          + f'{rm(s, [1, 7]):9.2f}{rm(s, [2, 6]):9.2f}{rm(s, mons):9.2f}{rm(s, mons, True):17.2f}')

# per-row predictions after the chosen rule set, for 48_error_map.py (env SAVE_RULES=ACD)
SAVE = os.environ.get('SAVE_RULES')
if SAVE:
    outs = []
    for M in sorted(df.mon.unique()):
        d, fit = df[df.mon == M], raw[raw.mon != M]
        outs.append(d[['MVT_ID_mvt', 'mon', 'base', 'p', 'sd', 'y_raw'] + (['pc'] if 'pc' in d.columns else [])].assign(pred=apply(d, fit, set(SAVE))))
    pd.concat(outs).to_parquet(os.path.join(D, 'candidates', f'oof_{TAG}{SAVE}_all.parquet'), index=False)
    print(f'saved candidates/oof_{TAG}{SAVE}_all.parquet')
