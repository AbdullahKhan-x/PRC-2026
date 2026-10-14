"""
Train the base regressor on all of 2025 with the v35 recipe (imported) plus optional extra
features, predict the ranking set, and write a parts file (base from this model, p and sd
from v35's saved detector) in template order, ready for 40_postrules.py.

  env EXTRA=ctx_train.parquet EXTRA_RANK=ctx_rank.parquet LR=0.04 ROUNDS=1800 TAG=ctx \
      python 44_train_full.py
writes candidates/parts_{TAG}.parquet
"""
import os, sys, gc, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('v35src', os.path.join(D, 'DATA_ANALYTICS/24_v35.py'))
v35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v35)
T = v35.TARGET
EXTRA, EXTRA_RANK = os.environ.get('EXTRA', ''), os.environ.get('EXTRA_RANK', '')
DROP = [c for c in os.environ.get('DROP_EXTRA', '').split(',') if c]
LR, ROUNDS, TAG = float(os.environ.get('LR', '0.04')), int(os.environ.get('ROUNDS', '1800')), os.environ.get('TAG', 'new')
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42').split(',')]
LEARNER = os.environ.get('LEARNER', 'lgb')        # lgb | cat (CatBoost member; same settings as 41_oof_harness LEARNER=cat)
CAT_PRM = dict(iterations=int(os.environ.get('CAT_ITERS', '2500')), learning_rate=float(os.environ.get('CAT_LR', '0.12')),
               depth=int(os.environ.get('CAT_DEPTH', '8')), loss_function='RMSE', thread_count=-1, verbose=500, border_count=254,
               task_type=os.environ.get('CAT_TASK', 'CPU'))
if os.environ.get('CAT_PARAMS'):                 # random-search overrides (same as 41_oof_harness)
    CAT_PRM.update(__import__('json').loads(os.environ['CAT_PARAMS']))

tr = v35.load_train()
feats = [c for c in (v35.CAT_COLS + v35.TIME_COLS + v35.AOBT_COLS + v35.ENC_COLS + v35.QUEUE_COLS
                     + v35.WX_COLS + v35.DEST_COLS + v35.SCHED_COLS) if c in tr.columns]
extra = []
DROP_BASE = [c for c in os.environ.get('DROP_BASE', '').split(',') if c]
feats = [c for c in feats if c not in DROP_BASE]          # e.g. replace NM operator/destination by op2/ades2
if EXTRA:
    ex = pd.read_parquet(os.path.join(D, EXTRA))
    extra = [c for c in ex.columns if c != 'MVT_ID_mvt' and c not in DROP]
    tr = tr.merge(ex[['MVT_ID_mvt'] + extra], on='MVT_ID_mvt', how='left')
    feats += extra
cats = [c for c in v35.CAT_COLS + v35.DEST_COLS if c in tr.columns] + \
       [c for c in extra if tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c])]
cats = [c for c in cats if c in feats]
enc = v35.fit_encoders(tr[tr[T] <= v35.TRAIN_CAP])
tr = v35.apply_encoders(tr, enc)
TE_EXTRA = os.environ.get('TE_EXTRA', '') == '1'
if TE_EXTRA:
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import te_extra
    if os.environ.get('TE_EXTRA2') == '1':        # + recording fingerprints and copy-rate x lateness products
        import te_extra2 as te_extra
    e2 = te_extra.fit(tr); tr = te_extra.apply(tr, e2); feats += te_extra.COLS
MINLEAF = os.environ.get('MINLEAF')
WITNESS = os.environ.get('WITNESS') == '1'                     # B8 witness-difference features (need ADS-B columns)
if WITNESS:
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import te_extra as _tw
    tr = _tw.add_witness(tr); feats += _tw.WITNESS_COLS
for c in cats:
    tr[c] = tr[c].astype('category')
cats_map = {c: tr[c].cat.categories for c in cats}
base_rows = tr[tr[T] <= v35.TRAIN_CAP]
if os.environ.get('TRAIN_SAMPLE'):              # smoke tests only
    base_rows = base_rows.sample(n=int(os.environ['TRAIN_SAMPLE']), random_state=1)
print(f'{len(feats)} features ({len(extra)} extra), {len(base_rows):,} rows, lr {LR}, rounds {ROUNDS}, seeds {SEEDS}', flush=True)
RESID = os.environ.get('RESID', '') == '1'      # residual target y - anchor (anchor = MVT - AOBT_3 if 60..3600 s, else airport median)
MED = base_rows.groupby(base_rows['apt'].astype(str))[T].median().to_dict()


def anchor(x):
    imp = x['implied_taxi_aobt'].to_numpy(dtype=float)
    fb = x['apt'].astype(str).map(MED).to_numpy(dtype=float)
    return np.where(np.isfinite(imp) & (imp > 60) & (imp < 3600), imp, fb)


label = base_rows[T] - anchor(base_rows) if RESID else base_rows[T]
if RESID:
    print('residual target on (anchor MVT - AOBT_3)', flush=True)


def cat_frame(x):          # CatBoost takes string categoricals (as 41_oof_harness AltModel._cat; missing -> 'nan')
    sys.path.insert(0, os.path.join(D, 'DATA_ANALYTICS')); import cat_member
    return cat_member.cat_frame(x, feats, cats)


models = []
if LEARNER == 'cat':
    from catboost import CatBoostRegressor
    assert os.environ.get('APT_ONLY') != '1' and float(os.environ.get('APT_W', '0')) == 0        # RESID=1 allowed (label = y - anchor)
    xc = cat_frame(base_rows)
    for i, sd_ in enumerate(SEEDS):
        m = CatBoostRegressor(random_seed=sd_, **CAT_PRM)
        m.fit(xc, np.asarray(label, dtype=float), cat_features=cats)
        m.save_model(os.path.join(D, 'candidates', f'base_{TAG}_s{i}.cbm'))
        models.append(m); print(f'  catboost seed {sd_} done', flush=True)
    del xc; gc.collect()
ds = None if LEARNER == 'cat' else lgb.Dataset(base_rows[feats], label, categorical_feature=cats, free_raw_data=False)
APT_ONLY = os.environ.get('APT_ONLY', '') == '1'     # train ONLY per-airport models (one per seed), base = per-airport
for sd_ in ([] if APT_ONLY or LEARNER == 'cat' else SEEDS):
    prm = dict(v35.PARAMS, learning_rate=LR, seed=sd_, bagging_seed=sd_, feature_fraction_seed=sd_)
    if os.environ.get('LGB_PARAMS'): prm.update(__import__('json').loads(os.environ['LGB_PARAMS']))
    if MINLEAF: prm['min_data_in_leaf'] = int(MINLEAF)
    models.append(lgb.train(prm, ds, ROUNDS))
    print(f'  seed {sd_} done', flush=True)
for i, m in enumerate(models if LEARNER == 'lgb' else []):
    m.save_model(os.path.join(D, 'candidates', f'base_{TAG}_s{i}.txt'))
APT_W = 1.0 if APT_ONLY else float(os.environ.get('APT_W', '0'))   # PER-AIRPORT models: base = (1-w)*global + w*per-airport
APT_SEEDS = SEEDS if APT_ONLY else SEEDS[:1]
apts = []
if APT_W > 0:
    apt_s = base_rows['apt'].astype(str)
    for a in sorted(apt_s.unique()):
        ma = (apt_s == a).values
        for j, sa in enumerate(APT_SEEDS):
            pa = dict(v35.PARAMS, learning_rate=LR, num_leaves=int(os.environ.get('APT_LEAVES', '127')), seed=sa,
                      bagging_seed=sa, feature_fraction_seed=sa)
            if MINLEAF: pa['min_data_in_leaf'] = int(MINLEAF)
            lgb.train(pa, lgb.Dataset(base_rows.loc[ma, feats], base_rows.loc[ma, T], categorical_feature=cats), ROUNDS) \
               .save_model(os.path.join(D, 'candidates', f'base_{TAG}_apt_{a}' + (f'_s{j}' if j else '') + '.txt'))
        apts.append(a); print(f'  airport model {a} done', flush=True); gc.collect()
import pickle
with open(os.path.join(D, 'candidates', f'meta_{TAG}.pkl'), 'wb') as fh:     # everything the final pipeline needs
    pickle.dump({'enc': enc, 'e2': e2 if TE_EXTRA else None, 'cats_map': cats_map, 'cats': cats,
                 'feats': feats, 'extra': extra, 'n_models': len(models), 'tag': TAG,
                 'apt_w': APT_W, 'apts': apts, 'apt_n': len(APT_SEEDS), 'resid': RESID, 'med': MED, 'witness': WITNESS, 'learner': LEARNER}, fh)
del ds, base_rows, tr; gc.collect()

rank = pd.read_parquet(os.path.join(D, 'dep_qw_rank.parquet'))
rank = v35.derive(rank, with_labels=False)
if EXTRA_RANK:
    exr = pd.read_parquet(os.path.join(D, EXTRA_RANK))
    rank = rank.merge(exr[['MVT_ID_mvt'] + extra], on='MVT_ID_mvt', how='left')
rank = v35.apply_encoders(rank, enc)
if TE_EXTRA:
    rank = te_extra.apply(rank, e2)
if WITNESS:
    rank = _tw.add_witness(rank)
rank = v35.as_cats(rank, cats_map, cats)
rx = cat_frame(rank) if LEARNER == 'cat' else rank[feats]
base = (np.mean([np.clip(m.predict(rx) + (anchor(rank) if RESID else 0.0), v35.PRED_FLOOR, v35.PRED_CEIL) for m in models], axis=0)
        if models else np.zeros(len(rank)))
if APT_W > 0:
    ra = rank['apt'].astype(str).values; b_apt = base.copy()
    assert set(np.unique(ra)) <= set(apts)
    for a in apts:
        mt = ra == a
        b_apt[mt] = np.mean([np.clip(lgb.Booster(model_file=os.path.join(D, 'candidates', f'base_{TAG}_apt_{a}' + (f'_s{j}' if j else '') + '.txt'))
                                     .predict(rank.loc[mt, feats]), v35.PRED_FLOOR, v35.PRED_CEIL) for j in range(len(APT_SEEDS))], axis=0)
    base = (1 - APT_W) * base + APT_W * b_apt
new = pd.DataFrame({'MVT_ID_mvt': rank['MVT_ID_mvt'].values, 'base_new': base})
parts = pd.read_parquet(os.path.join(D, 'candidates', 'v35_parts.parquet'))      # p, sd from v35's detector
parts = parts.merge(new, on='MVT_ID_mvt', how='left')
assert parts.base_new.notna().all()
pool = parts.sd.values > v35.POOL_MIN
parts['base'] = parts['base_new']
parts['pred'] = np.clip(np.where(pool, parts.p * parts.sd + (1 - parts.p) * parts.base, parts.base), v35.PRED_FLOOR, v35.PRED_CEIL)
parts = parts.drop(columns='base_new')
out = os.path.join(D, 'candidates', f'parts_{TAG}.parquet')
parts.to_parquet(out, index=False)
print(f'wrote {out}; mean |base_new - base_v35| = {np.abs(parts.base - pd.read_parquet(os.path.join(D, "candidates", "v35_parts.parquet")).base).mean():.1f}s', flush=True)
