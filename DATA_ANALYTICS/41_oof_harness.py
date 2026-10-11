"""
Out-of-month predictions for 2025 with the v35 recipe (imported, not re-implemented),
saved per row so post-rules / correctors can be scored on held-out months.
Faster than v35 (learning_rate 0.08) - compare arms within this harness only.

  python 41_oof_harness.py 1,2,6,7
writes candidates/oof_v35_m{M}.parquet  (MVT_ID, base, p, sd, y_raw, ...)
"""
import sys, os, gc, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('v35src', os.path.join(D, 'DATA_ANALYTICS/24_v35.py'))
v35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v35)
MONTHS = [int(x) for x in sys.argv[1].split(',')]
LR = float(os.environ.get('LR', '0.08'))
ROUNDS = int(os.environ.get('ROUNDS', '900'))
TAG = os.environ.get('TAG', 'v35')
T = v35.TARGET

df = pd.read_parquet(os.path.join(D, 'dep_qw_train.parquet'))
df['y_raw'] = df[T].astype(np.float64)
df['mvt_raw'] = df['mvt_i'].copy()          # the ranking file never has the training repair
y = df[T]
roll = (y > 43200) & ((y % 86400).between(60, 7200))          # v35's training repair, kept as-is
days = (df.loc[roll, T] // 86400).astype('int64')
df.loc[roll, 'mvt_i'] = df.loc[roll, 'mvt_i'] - 86400 * days
df.loc[roll, T] = df.loc[roll, T] % 86400
df = df[df[T] > 0].copy()
df = v35.derive(df, with_labels=True)
# for evaluation, sched_dev must be the raw one (as in the ranking file, no repair there)
df['sd_raw'] = np.where(roll.loc[df.index], df['sched_dev'] + 86400 * days.reindex(df.index).fillna(0), df['sched_dev'])
df['mon'] = pd.to_datetime(df['sch_i'], unit='s', utc=True).dt.month
feats = [c for c in (v35.CAT_COLS + v35.TIME_COLS + v35.AOBT_COLS + v35.ENC_COLS + v35.QUEUE_COLS
                     + v35.WX_COLS + v35.DEST_COLS + v35.SCHED_COLS) if c in df.columns]
EXTRA = os.environ.get('EXTRA', '')
DROP_BASE = [c for c in os.environ.get('DROP_BASE', '').split(',') if c]
feats = [c for c in feats if c not in DROP_BASE]          # e.g. replace NM operator/destination by op2/ades2
if EXTRA:                                   # extra feature file keyed by MVT_ID_mvt (e.g. ctx_train.parquet)
    ex = pd.read_parquet(os.path.join(D, EXTRA))
    drop = [c for c in os.environ.get('DROP_EXTRA', '').split(',') if c]
    keep = [c for c in ex.columns if c != 'MVT_ID_mvt' and c not in drop]
    df = df.merge(ex[['MVT_ID_mvt'] + keep], on='MVT_ID_mvt', how='left')
    for c in keep:
        if ex[c].dtype == object or pd.api.types.is_string_dtype(ex[c]):
            df[c] = df[c].astype('category')
    feats += keep
    print(f'extra features from {EXTRA}: {keep}', flush=True)
cats = [c for c in v35.CAT_COLS + v35.DEST_COLS if c in df.columns] + [c for c in feats if str(df[c].dtype) == 'category' and c not in v35.CAT_COLS + v35.DEST_COLS]
cats = [c for c in cats if c in feats]
for c in cats:
    df[c] = df[c].astype('category')
params = dict(v35.PARAMS); params['learning_rate'] = LR
if os.environ.get('LGB_PARAMS'):                 # e.g. LGB_PARAMS='{"extra_trees": true, "min_data_in_leaf": 300}'
    params.update(__import__('json').loads(os.environ['LGB_PARAMS']))
TE_EXTRA = os.environ.get('TE_EXTRA', '') == '1'
if TE_EXTRA:
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import te_extra
    if os.environ.get('TE_EXTRA2') == '1':        # + recording-regime fingerprints and copy-rate x lateness products
        import te_extra2 as te_extra
    feats += te_extra.COLS
    print('in-fold clock-offset / copy-rate encodings on', flush=True)
MINLEAF = os.environ.get('MINLEAF')
if MINLEAF: params['min_data_in_leaf'] = int(MINLEAF)
if os.environ.get('WITNESS') == '1':
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import te_extra as _tw0; feats += _tw0.WITNESS_COLS
if os.environ.get('LEAVES'): params['num_leaves'] = int(os.environ['LEAVES'])          # capacity sweep
TRAIN_FRAC = float(os.environ.get('TRAIN_FRAC', '1'))                                    # learning curve (row subsample of the training months)
INSAMPLE = os.environ.get('INSAMPLE') == '1'
WITNESS = os.environ.get('WITNESS') == '1'                                              # B8 witness-difference features
DET_EXTRA = [c for c in os.environ.get('DET_EXTRA', '').split(',') if c]                 # extra detector inputs (B3)
if os.environ.get('SEED'):                                                               # seed variation (bagging / feature-fraction / model seed)
    sd_ = int(os.environ['SEED']); params.update(seed=sd_, bagging_seed=sd_, feature_fraction_seed=sd_)
AUG_ADSB = float(os.environ.get('AUG_ADSB', '0'))                                        # share of ADS-B rows duplicated with ADS-B blanked
MASK_APTS = [a for a in os.environ.get('MASK_APTS', '').split(',') if a]                 # held-out airports whose ADS-B is blanked                                             # also report in-sample RMSE (overfit check)
if os.environ.get('OBJ'):                       # e.g. OBJ=huber ALPHA=300 (robust member for blending)
    params['objective'] = os.environ['OBJ']
    if os.environ.get('ALPHA'): params['alpha'] = float(os.environ['ALPHA'])
    print(f"objective {params['objective']} alpha {params.get('alpha')}", flush=True)
RESID = os.environ.get('RESID', '') == '1'
EXCL_APT = [a for a in os.environ.get('EXCL_APT', '').split(',') if a]


ANCHOR = os.environ.get('ANCHOR', 'aobt')       # RESID anchor: 'aobt' (MVT - AOBT_3) or 'eobt' (MVT - EOBT_1, needs ctx feature mvt_m_eobt)


def anchor(x, med):
    imp = (x['mvt_m_eobt'] if ANCHOR == 'eobt' else x['implied_taxi_aobt']).to_numpy(dtype=float)
    fb = x['apt'].astype(str).map(med).to_numpy(dtype=float)
    return np.where(np.isfinite(imp) & (imp > 60) & (imp < 3600), imp, fb)


COPYMIX = os.environ.get('COPYMIX', '') == '1'
CP_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, num_threads=-1, verbose=-1)
CP_ROUNDS = int(os.environ.get('CP_ROUNDS', '400'))


def copy_band(x, sd_):
    # NM-matched flights 30 min - 2 h late whose schedule lies inside the LOBT window (a copy is possible)
    feas = (x['sch_i'] - x['lobt_i']).abs().to_numpy() <= 3606
    return pd.Series((sd_ > 1800) & (sd_ <= 7200) & x['has_aobt'].to_numpy().astype(bool) & feas, index=x.index)


if COPYMIX:
    lob = pd.concat([pd.read_parquet(f, columns=['MVT_ID_mvt', 'LOBT_flt']) for f in
                     sorted(__import__('glob').glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))])
    lob['lobt_i'] = (pd.to_datetime(lob.LOBT_flt, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    df = df.merge(lob[['MVT_ID_mvt', 'lobt_i']], on='MVT_ID_mvt', how='left')
    print('copy mixture on (band 1800 < sd <= 7200, NM, SCHED within LOBT +- 3606)', flush=True)
print(f'rows {len(df):,}  feats {len(feats)}  lr {LR}', flush=True)

SEASON_TAU = float(os.environ.get('SEASON_TAU', '0'))
LEARNER = os.environ.get('LEARNER', 'lgb')        # lgb | cat | xgb
SEED_ALT = os.environ.get('SEED', '42')


class AltModel:
    def __init__(self, kind, X, y, cats, w, seed):
        self.kind, self.cats = kind, list(cats)
        if kind == 'cat':
            from catboost import CatBoostRegressor
            self.m = CatBoostRegressor(iterations=int(os.environ.get('CAT_ITERS', '3000')), learning_rate=float(os.environ.get('CAT_LR', '0.12')),
                                       depth=int(os.environ.get('CAT_DEPTH', '8')), loss_function='RMSE', random_seed=seed,
                                       thread_count=-1, verbose=500, border_count=254,
                                       task_type=os.environ.get('CAT_TASK', 'CPU'))       # CAT_TASK=GPU on Kaggle GPU kernels
            self.m.fit(self._cat(X), y, cat_features=self.cats, sample_weight=w)
        else:
            import xgboost as xgb
            self.xgb = xgb
            prm = dict(objective='reg:squarederror', tree_method='hist', grow_policy='lossguide', max_leaves=255, max_depth=0,
                       learning_rate=LR, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0, seed=seed,
                       max_cat_to_onehot=1, max_bin=255, nthread=-1)
            self.m = xgb.train(prm, xgb.DMatrix(self._xg(X), y, weight=w, enable_categorical=True), ROUNDS)

    def _cat(self, X):
        X = X.copy()
        for c in self.cats:
            X[c] = X[c].astype(str)
        return X

    def _xg(self, X):
        X = X.copy()
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)
        return X

    def predict(self, X):
        if self.kind == 'cat':
            return self.m.predict(self._cat(X))
        return self.m.predict(self.xgb.DMatrix(self._xg(X), enable_categorical=True))


def enc_safe(x, enc):
    # pandas 3: mapping a categorical key column then fillna() fails on unseen categories
    keep = {c: x[c] for c in ['AIRCRAFT_OPERATOR_flt'] if str(x[c].dtype) == 'category'}
    for c in keep: x[c] = x[c].astype(object)
    x = v35.apply_encoders(x, enc)
    for c, v in keep.items(): x[c] = v
    return x


for M in MONTHS:
    tr, te = df[df.mon != M].copy(), df[df.mon == M].copy()
    te['mvt_i'] = te['mvt_raw']; te = v35.derive(te, with_labels=False)    # held-out rows exactly as the ranking file shows them
    te['apt'] = pd.Categorical(te['apt'].astype(str), categories=df['apt'].cat.categories)
    enc = v35.fit_encoders(tr[tr[T] <= v35.TRAIN_CAP])
    tr = enc_safe(tr, enc); te = enc_safe(te, enc)
    if TE_EXTRA:
        e2 = te_extra.fit(tr); tr = te_extra.apply(tr, e2); te = te_extra.apply(te, e2)
    if WITNESS:
        import te_extra as _tw
        tr = _tw.add_witness(tr); te = _tw.add_witness(te)
    b = tr[tr[T] <= v35.TRAIN_CAP]
    med = b.groupby(b['apt'].astype(str))[T].median().to_dict()
    if EXCL_APT:          # train the base without these airports (their predictions are replaced later from another run)
        b = b[~b['apt'].astype(str).isin(EXCL_APT)]
    if COPYMIX:
        cp_tr = (tr['blk_i'] - tr['sch_i']).abs() <= 60
        band_tr = copy_band(tr, tr['sched_dev'].to_numpy())
        b = tr[(tr[T] <= v35.TRAIN_CAP) & ~(cp_tr & band_tr)]
    if TRAIN_FRAC < 1:
        b = b.sample(frac=TRAIN_FRAC, random_state=11)
    ADS_COLS = [c for c in feats if c.startswith('adsb_')]
    if AUG_ADSB > 0 and ADS_COLS:            # ADS-B dropout augmentation: duplicate covered rows with ADS-B blanked
        cov = b['adsb_matched'].fillna(0).to_numpy() == 1
        dup = b[cov].sample(frac=AUG_ADSB, random_state=13).copy()
        dup[ADS_COLS] = np.nan
        b = pd.concat([b, dup], ignore_index=True)
        print(f'   ADS-B dropout augmentation: +{len(dup):,} rows', flush=True)
    if MASK_APTS and ADS_COLS:               # stress test: held-out airports without ADS-B (e.g. LIRF in Jul 2026)
        mk = te['apt'].astype(str).isin(MASK_APTS).to_numpy()
        te.loc[mk, ADS_COLS] = np.nan
        print(f'   held-out ADS-B masked at {MASK_APTS}: {int(mk.sum()):,} rows', flush=True)
    best = ROUNDS                                  # fixed count (~ v35's 1783 at lr 0.04)
    wts = None
    if SEASON_TAU > 0:                             # seasonal weighting: training month m weighted exp(-circular |m - M| / tau)
        dm = np.abs(b['mon'].to_numpy() - M); dm = np.minimum(dm, 12 - dm)
        wts = np.exp(-dm / SEASON_TAU)
    if LEARNER != 'lgb':                           # diverse learner (blend member): CatBoost / XGBoost on the same features
        ya = (b[T] - anchor(b, med)).to_numpy() if RESID else b[T].to_numpy()          # RESID=1: residual target, as kresae
        mdl = AltModel(LEARNER, b[feats], ya, cats, wts, int(SEED_ALT))
        base = np.clip(mdl.predict(te[feats]) + (anchor(te, med) if RESID else 0.0), v35.PRED_FLOOR, v35.PRED_CEIL)
    elif RESID:          # residual target: y - anchor, anchor = MVT - AOBT when plausible, else the airport median
        mdl = lgb.train(params, lgb.Dataset(b[feats], b[T] - anchor(b, med), weight=wts, categorical_feature=cats), best)
        base = np.clip(mdl.predict(te[feats]) + anchor(te, med), v35.PRED_FLOOR, v35.PRED_CEIL)
    else:
        mdl = lgb.train(params, lgb.Dataset(b[feats], b[T], weight=wts, categorical_feature=cats), best)
        base = np.clip(mdl.predict(te[feats]), v35.PRED_FLOOR, v35.PRED_CEIL)
    pc = np.zeros(len(te))
    if COPYMIX:                                   # P(BLOCK == SCHED) for NM flights 30 min - 2 h late
        cb = tr[band_tr]
        clf = lgb.train(CP_PARAMS, lgb.Dataset(cb[feats], cp_tr[band_tr].astype(int), categorical_feature=cats), CP_ROUNDS)
        band_te = copy_band(te, te['sd_raw'].to_numpy())
        pc[band_te] = clf.predict(te.loc[band_te, feats])
        print(f'   copy band: train {len(cb):,} rows, copy rate {cp_tr[band_tr].mean():.3f};  held-out {band_te.sum():,} rows, mean p {pc[band_te].mean():.3f}', flush=True)
        del clf, cb; gc.collect()
    if INSAMPLE:
        bi = b.sample(n=min(300000, len(b)), random_state=5)
        pi = mdl.predict(bi[feats]) + (anchor(bi, med) if RESID else 0.0)
        print(f'   in-sample RMSE (y <= 3 h training rows, n={len(bi):,}): {np.sqrt(np.mean((pi - bi[T].values) ** 2)):.2f}', flush=True)
        ho = (te['y_raw'] <= 10800).to_numpy()
        print(f'   held-out RMSE on the same kind of rows (y <= 3 h): {np.sqrt(np.mean((base[ho] - te["y_raw"].to_numpy()[ho]) ** 2)):.2f}', flush=True)
    base_apt = np.full(len(te), np.nan)
    if os.environ.get('PER_APT') == '1':          # one extra model per airport, same features
        pa = dict(params); pa['num_leaves'] = int(os.environ.get('APT_LEAVES', '127'))
        for a in te['apt'].astype(str).unique():
            ma = (b['apt'].astype(str) == a).values; mt = (te['apt'].astype(str) == a).values
            ma_mdl = lgb.train(pa, lgb.Dataset(b.loc[ma, feats], b.loc[ma, T], categorical_feature=cats), ROUNDS)
            base_apt[mt] = np.clip(ma_mdl.predict(te.loc[mt, feats]), v35.PRED_FLOOR, v35.PRED_CEIL)
            del ma_mdl; gc.collect()
    dcols = [c for c in v35.DET_COLS if c in tr.columns] + ['apt'] + [c for c in DET_EXTRA if c in tr.columns]
    pool_tr = tr[tr['sched_dev'] > v35.POOL_MIN]
    det = lgb.train(v35.DET_PARAMS, lgb.Dataset(pool_tr[dcols], pool_tr['is_extreme'], categorical_feature=['apt']), v35.DET_ROUNDS)
    # detector input for the held-out month must use the RAW sched_dev (what the ranking file shows)
    te_d = te
    p = det.predict(te_d[dcols])
    sd = te['sd_raw'].to_numpy()
    p = np.where(sd > v35.POOL_MIN, p, 0.0)
    base_eff = pc * sd + (1 - pc) * base          # == base when COPYMIX is off
    pred = np.clip(np.where(sd > v35.POOL_MIN, p * sd + (1 - p) * base, base_eff), v35.PRED_FLOOR, v35.PRED_CEIL)
    out = pd.DataFrame({'MVT_ID_mvt': te['MVT_ID_mvt'].values, 'mon': M, 'base': base, 'base_apt': base_apt, 'p': p, 'pc': pc, 'sd': sd,
                        'pred': pred, 'y_raw': te['y_raw'].values})
    out.to_parquet(os.path.join(D, 'candidates', f'oof_{TAG}_m{M}.parquet'), index=False)
    r = np.sqrt(np.mean((out.pred - out.y_raw) ** 2))
    print(f'month {M}: best_iter {best}  rows {len(out):,}  uncapped RMSE {r:.2f}', flush=True)
    del tr, te, b, mdl, det, out; gc.collect()
print('done', flush=True)
