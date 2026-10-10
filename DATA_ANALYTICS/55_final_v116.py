"""
Final pipeline for the v116 recipe: a raw ranking parquet (any months) -> a verified submission.

  python 55_final_v116.py RANKING.parquet TEMPLATE.parquet OUT.parquet [WORKDIR]

Steps (every feature is rebuilt from the given file only - the organisers jitter MVT +-12 s between exports):
  1. queue features (12) + weather (15)                    -> dep_qw
  2. context 42 + 42b (ctxall), 42f (lf), 42d (ext: EUROCONTROL daily + OPDI), 62 (ADS-B, data/adsb/**)
  3. members (models trained on all of 2025, files in candidates/):
       ctx2id   ctxall + te_extra
       ctx2rep  ctxall + te_extra, without NM operator / ADES_flt (meta feats)
       ctx2res  ctxall + te_extra, residual target: + anchor (MVT - AOBT_3 if 60..3600 s, else airport median)
       ctx3     ctxall + lf + te_extra2
       kae      ctxall + ADS-B + ext + te_extra
       ctx2ida  global (= ctx2id recipe) + 10 per-airport models, 0.5 / 0.5
  4. blend (53b_make_blend4.py v116 0.25 0.3 0.5 kae):
       g    = 0.75 * (ctx2id + ctx2rep + ctx2res) / 3 + 0.25 * ctx3
       g    = 0.5 * g + 0.5 * kae
       apt  = 2 * ctx2ida - ctx2id
       base = g in the pool (MVT - SCHED > 7200), else 0.7 * g + 0.3 * apt
  5. v35 detector p, rules A C D (40_postrules.py), template-order gate.
Self-test: run on data/raw/ranking.parquet -> must equal candidates/logical-ladder_v116.parquet.
"""
import os, sys, gc, glob, pickle, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(D, 'DATA_ANALYTICS')
C = os.path.join(D, 'candidates')
RANK, TMPL, OUT = sys.argv[1:4]
WORK = sys.argv[4] if len(sys.argv) > 4 else os.path.join(C, 'final_work_v116')
os.makedirs(WORK, exist_ok=True)
WIN, P_ROME_3H, T_CLEAN = 3606.0, 107 / 173, 950.0
W3, W_APT, W_ADS = 0.25, 0.3, 0.5


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', '').replace('-', '_'), os.path.join(A, name))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                     # registered so worker processes can pickle its functions (62 precompute)
    spec.loader.exec_module(m); return m


q12, w15, c42, c42b, c42f, c42d, a62, v35 = (load(n) for n in ['12_queue_features.py', '15_weather.py', '42_context_features.py',
                                                              '42b_context_features2.py', '42f_labelfree.py', '42d_external.py',
                                                              '62_adsb_features.py', '24_v35.py'])
sys.path.insert(0, A); import te_extra, te_extra2
from pandas._libs.parsers import STR_NA_VALUES

raw = pd.read_parquet(RANK)
print(f'ranking file {RANK}: {len(raw):,} rows, DEP {int((raw.PHASE_mvt == "DEP").sum()):,}', flush=True)

# ---------- 1. queue + weather (as 50_final_pipeline.py; CSV-style NA strings for this step only) ----------
wpath = os.path.join(WORK, 'dep_qw.parquet')
if not os.path.exists(wpath):
    ch = raw[q12.USECOLS].copy()
    for c in ch.columns:
        if ch[c].dtype == object or pd.api.types.is_string_dtype(ch[c]):
            ch[c] = ch[c].where(~ch[c].isin(STR_NA_VALUES))
    for c in q12.CATCOLS:
        ch[c] = ch[c].astype('category')
    for src, dst in [('MVT_TIME_UTC_mvt', 'mvt_i'), ('BLOCK_TIME_UTC_mvt', 'blk_i'), ('SCHED_TIME_UTC_mvt', 'sch_i'),
                     ('AOBT_3_flt', 'aobt_i'), ('EOBT_1_flt', 'eobt_i')]:
        ch[dst] = q12._to_epoch(ch[src])
    ch = ch.drop(columns=['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt'])
    dep = ch[ch.PHASE_mvt == 'DEP'].copy(); dep['apt'] = dep['ADEP_mvt'].astype(str); dep = dep.drop(columns=['PHASE_mvt'])
    a = ch[ch.PHASE_mvt == 'ARR']
    arr = pd.DataFrame({'apt': a['ADES_mvt'].astype(str).values, 'land_i': a['mvt_i'].values, 'onblk_i': a['blk_i'].values})
    dep = q12.build_features(dep.reset_index(drop=True), arr)
    qpath = os.path.join(WORK, 'dep_q.parquet'); dep.to_parquet(qpath, index=False)
    del ch, dep, arr, a; gc.collect()
    w15.attach(qpath, w15.build_weather(w15.load_cache()), wpath, 'FINAL')
    gc.collect()
print('  queue + weather ready', flush=True)

# ---------- 2. context / lf / ext / ADS-B ----------
def cached(name, fn):
    p = os.path.join(WORK, name)
    if not os.path.exists(p):
        fn().to_parquet(p, index=False); gc.collect()
    return pd.read_parquet(p)

def _ctx():
    cx = raw[c42.COLS].copy()
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_flt']:
        cx[c] = cx[c].astype('category')
    ctx = c42.build(cx); del cx
    cx2 = raw[c42b.COLS].copy()
    for c in ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt']:
        cx2[c] = cx2[c].astype('category')
    return ctx.merge(c42b.build(cx2), on='MVT_ID_mvt', how='left')

ctxall = cached('ctxall.parquet', _ctx)
lf = cached('lf.parquet', lambda: c42f.build(raw[c42f.COLS].copy()))
months = sorted(set(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y%m')))
opdi_m = sorted({m for x in months for m in [x, (pd.Period(x, 'M') - 1).strftime('%Y%m'), (pd.Period(x, 'M') + 1).strftime('%Y%m')]})
ext = cached('ext.parquet', lambda: c42d.build(raw[c42d.COLS].copy(), c42d.daily(), c42d.opdi(opdi_m)))

def _adsb():
    files = a62.day_files()
    days = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').unique()
    print(f'  ADS-B: {sum(d in files for d in days)} of {len(days)} days have point files', flush=True)
    a62.precompute(days, files)
    return a62.build(raw[a62.COLS].copy(), files)

adsb = cached('adsb.parquet', _adsb)
print('  context / lf / ext / ADS-B ready', flush=True)

# ---------- 3. members ----------
base_df = v35.derive(pd.read_parquet(wpath), with_labels=False)
ids = base_df['MVT_ID_mvt'].values
apt_raw = base_df['apt'].astype(str).values
EXTRA_SRC = {'ctx2id': [ctxall], 'ctx2rep': [ctxall], 'ctx2res': [ctxall], 'ctx2ida': [ctxall],
             'ctx3': [ctxall, lf], 'kae': [ctxall, adsb, ext]}


def member(tag):
    meta = pickle.load(open(os.path.join(C, f'meta_{tag}.pkl'), 'rb'))
    r = base_df.copy()
    if meta['extra']:
        ex = EXTRA_SRC[tag][0]
        for more in EXTRA_SRC[tag][1:]:
            ex = ex.merge(more, on='MVT_ID_mvt', how='left')
        miss = [c for c in meta['extra'] if c not in ex.columns]
        assert not miss, f'{tag}: missing extra columns {miss}'
        r = r.merge(ex[['MVT_ID_mvt'] + meta['extra']], on='MVT_ID_mvt', how='left')
    r = v35.apply_encoders(r, meta['enc'])
    if meta['e2'] is not None:
        mod = te_extra2 if 'te_s00_stand' in meta['feats'] else te_extra
        r = mod.apply(r, meta['e2'])
    r = v35.as_cats(r, meta['cats_map'], meta['cats'])
    assert (r['MVT_ID_mvt'].values == ids).all(), f'{tag}: row order changed'
    models = [lgb.Booster(model_file=os.path.join(C, f'base_{tag}_s{i}.txt')) for i in range(meta['n_models'])]
    anc = 0.0
    if meta.get('resid'):
        imp = r['implied_taxi_aobt'].to_numpy(dtype=float)
        fb = r['apt'].astype(str).map(meta['med']).to_numpy(dtype=float)
        anc = np.where(np.isfinite(imp) & (imp > 60) & (imp < 3600), imp, fb)
    b = np.mean([np.clip(m.predict(r[meta['feats']]) + anc, v35.PRED_FLOOR, v35.PRED_CEIL) for m in models], axis=0)
    if meta.get('apt_w', 0) > 0:
        b_apt = b.copy()
        for ap in meta['apts']:
            mt = apt_raw == ap
            b_apt[mt] = np.clip(lgb.Booster(model_file=os.path.join(C, f'base_{tag}_apt_{ap}.txt')).predict(r.loc[mt, meta['feats']]),
                                v35.PRED_FLOOR, v35.PRED_CEIL)
        assert set(np.unique(apt_raw)) <= set(meta['apts'])
        b = (1 - meta['apt_w']) * b + meta['apt_w'] * b_apt
    print(f'  member {tag:8s} mean {b.mean():7.1f}', flush=True)
    del r; gc.collect()
    return b


B = {t: member(t) for t in ['ctx2id', 'ctx2rep', 'ctx2res', 'ctx3', 'kae', 'ctx2ida']}

# ---------- 4. detector p + blend ----------
v35cats = np.load(os.path.join(D, 'final_cats.npy'), allow_pickle=True).item()
det = lgb.Booster(model_file=os.path.join(D, 'final_detector.txt'))
din = base_df[list(v35.DET_COLS)].copy()
din['apt'] = pd.Categorical(apt_raw, categories=v35cats['apt'])
sd = base_df['sched_dev'].to_numpy()
pool = sd > v35.POOL_MIN
p = np.where(pool, det.predict(din[list(v35.DET_COLS) + ['apt']]), 0.0)
g = (1 - W3) * (B['ctx2id'] + B['ctx2rep'] + B['ctx2res']) / 3 + W3 * B['ctx3']
g = (1 - W_ADS) * g + W_ADS * B['kae']
apt_part = 2 * B['ctx2ida'] - B['ctx2id']
base = np.clip(np.where(pool, g, (1 - W_APT) * g + W_APT * apt_part), 60, 172800)

# ---------- 5. rules A C D (as 40_postrules.py) ----------
cols = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt']
rd = raw[raw.PHASE_mvt == 'DEP'][cols].set_index('MVT_ID_mvt').loc[ids].reset_index()
ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
mvt, sch, lobt = ep(rd.MVT_TIME_UTC_mvt), ep(rd.SCHED_TIME_UTC_mvt), ep(rd.LOBT_flt)
nonm = rd.AOBT_3_flt.isna().values
rome = (apt_raw == 'LIRF') & nonm
has = ~np.isnan(lobt)
p = np.where(pool & nonm & (apt_raw != 'LIRF'), 0.0, p)                      # D
p = np.where(has & pool & (np.abs(sch - lobt) > WIN), 0.0, p)                # A (p)
pred = np.clip(np.where(pool, p * sd + (1 - p) * base, base), 60, 172800)
m = rome & (sd > 10800) & (sd <= 20000)                                      # C
pred = np.where(m, P_ROME_3H * sd + (1 - P_ROME_3H) * T_CLEAN, pred)
lo, hi = np.maximum(mvt - lobt - WIN, 0), mvt - lobt + WIN                   # A (clip)
pred = np.where(has, np.clip(pred, lo, np.maximum(hi, lo)), pred)

# ---------- 6. gate ----------
tmpl = pd.read_parquet(TMPL)
sub = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': pred.astype(np.float64)})
sub['MVT_ID_mvt'] = sub['MVT_ID_mvt'].astype(tmpl['MVT_ID_mvt'].dtype)
out = tmpl[['MVT_ID_mvt']].merge(sub, on='MVT_ID_mvt', how='left')
assert len(out) == len(tmpl), 'row count changed'
assert (out.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all(), 'ROW ORDER WRONG'
assert out.TAXITIME_SEC_mvt.notna().all(), 'nulls (a template id is missing from the ranking file?)'
assert np.isfinite(out.TAXITIME_SEC_mvt).all() and (out.TAXITIME_SEC_mvt >= 0).all()
out.to_parquet(OUT, index=False)
chk = pd.read_parquet(OUT)
print(f'wrote {OUT}: rows {len(chk):,}  order ok {(chk.MVT_ID_mvt.values == tmpl.MVT_ID_mvt.values).all()}  '
      f'mean {chk.TAXITIME_SEC_mvt.mean():.1f}  >10800: {(chk.TAXITIME_SEC_mvt > 10800).sum()}', flush=True)
