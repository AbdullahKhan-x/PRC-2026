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
for c in cats:
    tr[c] = tr[c].astype('category')
cats_map = {c: tr[c].cat.categories for c in cats}
base_rows = tr[tr[T] <= v35.TRAIN_CAP]
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
ds = lgb.Dataset(base_rows[feats], label, categorical_feature=cats, free_raw_data=False)
models = []
for sd_ in SEEDS:
    prm = dict(v35.PARAMS, learning_rate=LR, seed=sd_, bagging_seed=sd_, feature_fraction_seed=sd_)
    if MINLEAF: prm['min_data_in_leaf'] = int(MINLEAF)
    models.append(lgb.train(prm, ds, ROUNDS))
    print(f'  seed {sd_} done', flush=True)
for i, m in enumerate(models):
    m.save_model(os.path.join(D, 'candidates', f'base_{TAG}_s{i}.txt'))
APT_W = float(os.environ.get('APT_W', '0'))         # PER-AIRPORT models blended in: base = (1-w)*global + w*per-airport
apts = []
if APT_W > 0:
    apt_s = base_rows['apt'].astype(str)
    for a in sorted(apt_s.unique()):
        pa = dict(v35.PARAMS, learning_rate=LR, num_leaves=int(os.environ.get('APT_LEAVES', '127')), seed=SEEDS[0],
                  bagging_seed=SEEDS[0], feature_fraction_seed=SEEDS[0])
        if MINLEAF: pa['min_data_in_leaf'] = int(MINLEAF)
        ma = (apt_s == a).values
        lgb.train(pa, lgb.Dataset(base_rows.loc[ma, feats], base_rows.loc[ma, T], categorical_feature=cats), ROUNDS) \
           .save_model(os.path.join(D, 'candidates', f'base_{TAG}_apt_{a}.txt'))
        apts.append(a); print(f'  airport model {a} done', flush=True); gc.collect()
import pickle
with open(os.path.join(D, 'candidates', f'meta_{TAG}.pkl'), 'wb') as fh:     # everything the final pipeline needs
    pickle.dump({'enc': enc, 'e2': e2 if TE_EXTRA else None, 'cats_map': cats_map, 'cats': cats,
                 'feats': feats, 'extra': extra, 'n_models': len(models), 'tag': TAG,
                 'apt_w': APT_W, 'apts': apts, 'resid': RESID, 'med': MED}, fh)
del ds, base_rows, tr; gc.collect()

rank = pd.read_parquet(os.path.join(D, 'dep_qw_rank.parquet'))
rank = v35.derive(rank, with_labels=False)
if EXTRA_RANK:
    exr = pd.read_parquet(os.path.join(D, EXTRA_RANK))
    rank = rank.merge(exr[['MVT_ID_mvt'] + extra], on='MVT_ID_mvt', how='left')
rank = v35.apply_encoders(rank, enc)
if TE_EXTRA:
    rank = te_extra.apply(rank, e2)
rank = v35.as_cats(rank, cats_map, cats)
base = np.mean([np.clip(m.predict(rank[feats]) + (anchor(rank) if RESID else 0.0), v35.PRED_FLOOR, v35.PRED_CEIL) for m in models], axis=0)
if APT_W > 0:
    ra = rank['apt'].astype(str).values; b_apt = base.copy()
    for a in apts:
        mt = ra == a
        b_apt[mt] = np.clip(lgb.Booster(model_file=os.path.join(D, 'candidates', f'base_{TAG}_apt_{a}.txt')).predict(rank.loc[mt, feats]),
                            v35.PRED_FLOOR, v35.PRED_CEIL)
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
