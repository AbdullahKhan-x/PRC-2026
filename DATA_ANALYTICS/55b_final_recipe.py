"""
Recipe-driven final pipeline (generalises 55_final_v116.py): a raw ranking parquet -> a verified submission.

  python 55b_final_recipe.py RECIPE.json RANKING.parquet TEMPLATE.parquet OUT.parquet [WORKDIR]

RECIPE.json:
  {"global": [[["kae", "kaeS"], 1], [["krepae", "krepaeS"], 1], [["kresae", "kresaeS"], 1]],   groups of model tags
                                         (all saved models of a group are averaged equally = seed averaging), group weights
   "apt": {"tag": "ctx2ida", "global_tag": "ctx2id", "w": 0.3},                               per-airport layer (or null)
   "rules": "ACD",
   "outer": [[["kcat12f", "kcat12f2", "kcat12f3"], 0.5]]}     optional, after the per-airport layer, ALL rows:
                                         base = (1 - sum w) * base + sum w * group   (CatBoost members: meta learner 'cat')
All feature families are built from the given file (context 42/42b, provenance 42f, external 42d, ADS-B 62, stand
arrival-copy 42h, ADS-B geometry 62b when a member uses it - needs candidates/geo_tables.pkl) and each member takes the
columns listed in its meta_*.pkl. Everything else is identical to the
verified v116 pipeline (te_extra None-key fix, residual anchors, v35 detector, rules A/C/D, template gate).

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
import json
RECIPE = json.load(open(sys.argv[1]))
RANK, TMPL, OUT = sys.argv[2:5]
WORK = sys.argv[5] if len(sys.argv) > 5 else os.path.join(C, 'final_work_recipe')
os.makedirs(WORK, exist_ok=True)
WIN, P_ROME_3H, T_CLEAN = 3606.0, 107 / 173, 950.0


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', '').replace('-', '_'), os.path.join(A, name))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                     # registered so worker processes can pickle its functions (62 precompute)
    spec.loader.exec_module(m); return m


q12, w15, c42, c42b, c42f, c42d, a62, v35, c42h = (load(n) for n in ['12_queue_features.py', '15_weather.py', '42_context_features.py',
                                                              '42b_context_features2.py', '42f_labelfree.py', '42d_external.py',
                                                              '62_adsb_features.py', '24_v35.py', '42h_stand_arrcopy.py'])
g62b = load('62b_adsb_geo.py')
c42j = load('42j_next_arrival.py')
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
sac = cached('sac.parquet', lambda: c42h.build(raw[c42h.COLS].copy()))


def _recipe_tags(r):
    tags = [t for grp, _ in r['global'] for t in grp]
    ap = r.get('apt') or {}
    return tags + ap.get('pure', []) + [ap[k] for k in ('tag', 'global_tag') if k in ap] + [t for grp, _ in r.get('outer', []) for t in grp]


def _geo():
    # ADS-B first-contact geometry (62b): same events/matcher as 62; stand/airport references = geo_tables.pkl saved by the
    # Kaggle build (training 2025 + ranking.parquet), so the features are identical to the ones the members were trained on
    files = g62b.day_files()
    days = sorted(set(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')) & set(files))
    ev = g62b.events_for_days([files[d] for d in days])
    mt = g62b.match(raw[g62b.COLS].copy(), ev)
    return g62b.build_geo(mt, *pickle.load(open(os.path.join(C, 'geo_tables.pkl'), 'rb')))


NEED = {c for t in _recipe_tags(RECIPE) for c in pickle.load(open(os.path.join(C, f'meta_{t}.pkl'), 'rb'))['extra']}
MORE = [lf, ext, adsb, sac]
if NEED & set(c42j.NA):                         # stand-turnover witnesses (next arrival at my stand)
    MORE.append(cached('na.parquet', lambda: c42j.build(raw[c42j.COLS].copy())))
if NEED & set(g62b.GEO):
    MORE.append(cached('geo.parquet', _geo))
    print('  ADS-B geometry ready', flush=True)
WIDE = ctxall
for more in MORE:
    WIDE = WIDE.merge(more[['MVT_ID_mvt'] + [c for c in more.columns if c not in WIDE.columns]], on='MVT_ID_mvt', how='left')
print('  context / lf / ext / ADS-B ready', flush=True)

# ---------- 3. members ----------
base_df = v35.derive(pd.read_parquet(wpath), with_labels=False)
ids = base_df['MVT_ID_mvt'].values
apt_raw = base_df['apt'].astype(str).values


def member(tag):
    meta = pickle.load(open(os.path.join(C, f'meta_{tag}.pkl'), 'rb'))
    r = base_df.copy()
    if meta['extra']:
        ex = WIDE
        miss = [c for c in meta['extra'] if c not in ex.columns]
        assert not miss, f'{tag}: missing extra columns {miss}'
        r = r.merge(ex[['MVT_ID_mvt'] + meta['extra']], on='MVT_ID_mvt', how='left')
    r = v35.apply_encoders(r, meta['enc'])
    if meta['e2'] is not None:
        mod = te_extra2 if 'te_s00_stand' in meta['feats'] else te_extra
        r = mod.apply(r, meta['e2'])
    r = v35.as_cats(r, meta['cats_map'], meta['cats'])
    assert (r['MVT_ID_mvt'].values == ids).all(), f'{tag}: row order changed'
    if meta.get('learner', 'lgb') == 'cat':          # CatBoost member: string categoricals, missing -> 'nan' (cat_member.py)
        import cat_member
        from catboost import CatBoostRegressor
        xc = cat_member.cat_frame(r, meta['feats'], meta['cats'])
        anc_c = 0.0
        if meta.get('resid'):                         # residual-target CatBoost: + anchor (as the LightGBM residual members)
            imp = r['implied_taxi_aobt'].to_numpy(dtype=float)
            fb = r['apt'].astype(str).map(meta['med']).to_numpy(dtype=float)
            anc_c = np.where(np.isfinite(imp) & (imp > 60) & (imp < 3600), imp, fb)
        bs = []
        for i in range(meta['n_models']):
            cm = CatBoostRegressor(); cm.load_model(os.path.join(C, f'base_{tag}_s{i}.cbm'))
            bs.append(np.clip(cm.predict(xc) + anc_c, v35.PRED_FLOOR, v35.PRED_CEIL))
        b = np.mean(bs, axis=0)
        print(f'  member {tag:8s} mean {b.mean():7.1f}  (catboost)', flush=True)
        del r, xc; gc.collect()
        return b
    models = [lgb.Booster(model_file=os.path.join(C, f'base_{tag}_s{i}.txt')) for i in range(meta['n_models'])]
    anc = 0.0
    if meta.get('resid'):
        imp = r['implied_taxi_aobt'].to_numpy(dtype=float)
        fb = r['apt'].astype(str).map(meta['med']).to_numpy(dtype=float)
        anc = np.where(np.isfinite(imp) & (imp > 60) & (imp < 3600), imp, fb)
    b = (np.mean([np.clip(m.predict(r[meta['feats']]) + anc, v35.PRED_FLOOR, v35.PRED_CEIL) for m in models], axis=0)
         if models else np.zeros(len(r)))
    if meta.get('apt_w', 0) > 0:
        b_apt = b.copy()
        for ap in meta['apts']:
            mt = apt_raw == ap
            b_apt[mt] = np.mean([np.clip(lgb.Booster(model_file=os.path.join(C, f'base_{tag}_apt_{ap}' + (f'_s{j}' if j else '') + '.txt'))
                                         .predict(r.loc[mt, meta['feats']]), v35.PRED_FLOOR, v35.PRED_CEIL)
                                 for j in range(meta.get('apt_n', 1))], axis=0)
        assert set(np.unique(apt_raw)) <= set(meta['apts'])
        b = (1 - meta['apt_w']) * b + meta['apt_w'] * b_apt
    print(f'  member {tag:8s} mean {b.mean():7.1f}', flush=True)
    del r; gc.collect()
    return b


def group(tags):
    # equal average over every saved model of the tags (seed 42 file + 2-seed file = 3-seed average)
    tot, n = 0.0, 0
    for t in tags:
        k = pickle.load(open(os.path.join(C, f'meta_{t}.pkl'), 'rb'))['n_models']
        tot = tot + k * member(t); n += k
    return tot / n


# ---------- 4. detector p + blend ----------
v35cats = np.load(os.path.join(D, 'final_cats.npy'), allow_pickle=True).item()
det = lgb.Booster(model_file=os.path.join(D, 'final_detector.txt'))
din = base_df[list(v35.DET_COLS)].copy()
din['apt'] = pd.Categorical(apt_raw, categories=v35cats['apt'])
sd = base_df['sched_dev'].to_numpy()
pool = sd > v35.POOL_MIN
p = np.where(pool, det.predict(din[list(v35.DET_COLS) + ['apt']]), 0.0)
wsum = sum(w for _, w in RECIPE['global'])
g = sum(w * group(tags) for tags, w in RECIPE['global']) / wsum
AP = RECIPE.get('apt')
W_APT = AP['w'] if AP else 0.0
def apt_layer(ap):
    if 'pure' in ap:     # APT_ONLY members (44_train_full APT_ONLY=1): average over every per-airport seed model
        ns = [pickle.load(open(os.path.join(C, f'meta_{t}.pkl'), 'rb')).get('apt_n', 1) for t in ap['pure']]
        return sum(n * member(t) for n, t in zip(ns, ap['pure'])) / sum(ns)
    return 2 * member(ap['tag']) - member(ap['global_tag'])


apt_part = apt_layer(AP) if AP else g
base = np.clip(np.where(pool, g, (1 - W_APT) * g + W_APT * apt_part), 60, 172800)
OUTER = RECIPE.get('outer') or []
if OUTER:                    # outer mix (58_mix_oof.py on the held-out stack): all rows, after the per-airport layer
    base = np.clip((1 - sum(w for _, w in OUTER)) * base + sum(w * group(tags) for tags, w in OUTER), 60, 172800)
RULES = RECIPE.get('rules', 'ACD')
assert RULES in ('ACD', 'ADF', 'ADG', 'ACDS', 'ADFS', 'ADGS'), 'only rules ACD / ADF / ADG (+S) implemented here'

# ---------- 5. rules A C D (as 40_postrules.py) ----------
cols = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'LOBT_flt', 'FLIGHT_mvt', 'ADES_mvt', 'EOBT_1_flt']
rd = raw[raw.PHASE_mvt == 'DEP'][cols].set_index('MVT_ID_mvt').loc[ids].reset_index()
ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values
mvt, sch, lobt = ep(rd.MVT_TIME_UTC_mvt), ep(rd.SCHED_TIME_UTC_mvt), ep(rd.LOBT_flt)
nonm = rd.AOBT_3_flt.isna().values
rome = (apt_raw == 'LIRF') & nonm
has = ~np.isnan(lobt)
p = np.where(pool & nonm & (apt_raw != 'LIRF'), 0.0, p)                      # D
p = np.where(has & pool & (np.abs(sch - lobt) > WIN), 0.0, p)                # A (p)
pred = np.clip(np.where(pool, p * sd + (1 - p) * base, base), 60, 172800)
if 'C' in RULES:
    m = rome & (sd > 10800) & (sd <= 20000)                                  # C
    pred = np.where(m, P_ROME_3H * sd + (1 - P_ROME_3H) * T_CLEAN, pred)
elif 'G' in RULES:
    import rule_g                                                            # G (C rebuilt: mean(format, airline x dest) rate, ctx2id anchor)
    pred, m, _ = rule_g.apply(pred, sd, rome, rd.FLIGHT_mvt.values, rd.ADES_mvt.values, member('ctx2id'), rule_g.fit(D))
else:
    import rule_f                                                            # F (C split by flight-code format)
    pred, m, _ = rule_f.apply(pred, base, sd, rome, rd.FLIGHT_mvt.values, rule_f.fit(D))
if 'S' in RULES:                                                             # S (= B', Rome date stamps)
    import rule_s
    pred, _ = rule_s.apply(pred, sd, rome, rule_s.cross_days(rd.MVT_TIME_UTC_mvt.values, rd.SCHED_TIME_UTC_mvt.values))
lo, hi = np.maximum(mvt - lobt - WIN, 0), mvt - lobt + WIN                   # A (clip)
pred = np.where(has, np.clip(pred, lo, np.maximum(hi, lo)), pred)
POST = RECIPE.get('post', '')            # 'TP' = rules T/P (rule_tp.py), 'DD' = daily delay constraint (rule_dd.py), 'NX' = no-ADS-B corrector (rule_nx.py), in that order
if 'TP' in POST:
    import rule_tp
    ftp = rule_tp.features(raw).set_index('MVT_ID_mvt').loc[ids].reset_index()
    ftp['tp_pair'] = ftp.tp_pair.fillna(False).astype(bool)
    pred, mk = rule_tp.apply(pred, ftp)
    print('  rules T/P rows', {k: int(v.sum()) for k, v in mk.items()}, flush=True)
if 'FL' in POST:                         # FL = Rome flight-number copy habit (rule_fl.py), after T/P, before DD (held-out in stack −0.28, 4/4)
    import rule_fl
    pred, mfl, _ = rule_fl.apply(pred, sd, member('ctx2id'), rome, rd.FLIGHT_mvt.values, rule_fl.fit(rule_fl.load_2025(D)))
    print(f'  rule FL rows {int(mfl.sum()):,}', flush=True)
if 'DD' in POST:
    import rule_dd
    ddg = 'DDG' in POST                  # DDG = DDA + partial-day guard + adaptive cap 0.5
    pred, sdd = rule_dd.apply(pred, mvt, sd, apt_raw, rule_dd.load_daily(D), elig_max=1e9 if ('DDA' in POST or ddg) else 7200,
                              counts=rule_dd.load_counts(D) if ddg else None, adapt=0.5 if ddg else None)   # DDA = no delay limit
    print(f'  rule DD rows shifted {int((sdd != 0).sum()):,}', flush=True)
if 'NX' in POST:                         # NX = no-ADS-B corrector at LTFM/LFPG/EGLL (rule_nx.py), after T/P + DD
    import rule_nx
    rule_nx.ALL = 'NXA' in POST                                              # NXA = all airports, all rows
    rule_nx.LINEAR = 'NXL' in POST                                           # NXL = linear_tree corrector leaves
    na = cached('na.parquet', lambda: c42j.build(raw[c42j.COLS].copy()))
    dnx = pd.DataFrame({'MVT_ID_mvt': ids, 'ADEP_mvt': apt_raw, 'pred': pred, 'sd': sd, 'mvt': mvt, 'lobt': lobt,
                        'aobt': ep(rd.AOBT_3_flt), 'eobt': ep(rd.EOBT_1_flt)})
    dnx = (dnx.merge(adsb[['MVT_ID_mvt', 'adsb_matched']], on='MVT_ID_mvt', how='left')
              .merge(rule_nx.features(raw), on='MVT_ID_mvt', how='left')
              .merge(na[['MVT_ID_mvt'] + rule_nx.NA], on='MVT_ID_mvt', how='left'))
    assert (dnx.MVT_ID_mvt.values == ids).all()
    nxtr = pd.read_parquet(os.path.join(C, 'nx_train_ddg.parquet' if 'DDG' in POST else 'nx_train.parquet'))
    if os.environ.get('NXQ_TRAIN'):            # v162: direct sequence model q (70_seq_direct_kernel.py) as NX input (stacking)
        qt = pd.read_parquet(os.environ['NXQ_TRAIN']).set_index(['MVT_ID_mvt', 'mon']).q        # held-out 2025 q (LOMO)
        nxtr['q'] = qt.reindex(pd.MultiIndex.from_arrays([nxtr.MVT_ID_mvt.values, nxtr.mon.values])).values
        dnx['q'] = pd.read_parquet(os.environ['NXQ_FINAL']).set_index('MVT_ID_mvt').q.reindex(dnx.MVT_ID_mvt.values).values   # 71_seq_apply.py
        for f_ in (nxtr, dnx):
            f_['qd'] = (f_.q - f_.pred).clip(-1500, 1500).fillna(0.0); f_['hasq'] = f_.q.notna().astype(float)
        rule_nx.X = list(rule_nx.X) + ['qd', 'hasq']
        print(f'  NXQ stacking: train q {nxtr.q.notna().mean():.3f}, final q {dnx.q.notna().mean():.3f}', flush=True)
    pnx = rule_nx.apply(rule_nx.fit(nxtr), dnx)   # NX fitted on the same DD variant
    print(f'  rule NX rows {int(rule_nx.segment(dnx).sum()):,}  mean shift {np.mean(pnx - pred):+.2f} s', flush=True)
    pred = pnx

if 'MON' in POST:                        # MON = rule M (rule_km.apply_m): monthly mean shift to the EUROCONTROL ATXOT mean, last
    import rule_km
    atx = rule_km.load_atxot(sorted(glob.glob(os.path.join(D, 'data/ext/atxot/taxi_out_*.csv'))))
    pred, info = rule_km.apply_m(pred, sd, apt_raw, mvt, atx, json.load(open(os.path.join(D, 'recipes', 'constants', 'rule_m_tab.json'))))
    print('  rule M cells (apt, ym, raw s, weight):', info, flush=True)
if 'KCNT' in POST:                       # KCNT = rule K (rule_km.apply_k): LIRF monthly count of y > 2 h (ATXOT TF - VALID_FL), after M
    import rule_km
    atx = rule_km.load_atxot(sorted(glob.glob(os.path.join(D, 'data/ext/atxot/taxi_out_*.csv'))))
    pred, info = rule_km.apply_k(pred, sd, nonm, apt_raw, mvt, atx, json.load(open(os.path.join(D, 'recipes', 'constants', 'rule_m_tab.json'))))
    print('  rule K months (ym, K, expected before, lambda, rows):', info, flush=True)

if 'SP' in POST:                         # SP = rule S' (rule_sp.py): Rome date-wrap vs copy, exact E[y], applied LAST (overrides its rows)
    import rule_s, rule_sp
    pred, msp = rule_sp.apply(pred, sd, rome, rule_s.cross_days(rd.MVT_TIME_UTC_mvt.values, rd.SCHED_TIME_UTC_mvt.values))
    print(f"  rule S' rows {int(msp.sum()):,}", flush=True)

if 'CAP' in POST:                        # CAP = LEMD/LEBL taxi-out never exceeds 3,604 s (2025: 0 of 392k; maxima 3,598 / 3,604), applied last
    mcap = np.isin(apt_raw, ['LEMD', 'LEBL']) & (pred > 3604)
    pred = np.maximum(np.where(mcap, 3604.0, pred), 60.0)                   # + final 60 s floor (rules FL/M can dip below it)
    print(f'  rule CAP rows {int(mcap.sum()):,}', flush=True)

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
