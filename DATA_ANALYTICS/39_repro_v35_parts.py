"""Rebuild v35's ranking predictions from its saved models (no refit) and split them into
base prediction and detector probability. Gate: must match logical-ladder_v35.parquet."""
import sys, os, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('v35src', os.path.join(D, 'DATA_ANALYTICS/24_v35.py'))
v35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v35)      # defs only (main guarded)
enc = np.load(os.path.join(D, 'final_enc.npy'), allow_pickle=True).item()
cats_map = np.load(os.path.join(D, 'final_cats.npy'), allow_pickle=True).item()
model = lgb.Booster(model_file=os.path.join(D, 'final_base.txt'))
det = lgb.Booster(model_file=os.path.join(D, 'final_detector.txt'))
rank = pd.read_parquet(os.path.join(D, 'dep_qw_rank.parquet'))
rank = v35.derive(rank, with_labels=False)
rank = v35.apply_encoders(rank, enc)
cats = [c for c in v35.CAT_COLS + v35.DEST_COLS if c in cats_map]
rank = v35.as_cats(rank, cats_map, cats)
rank['apt'] = rank['apt'].astype(str)
rank = v35.as_cats(rank, {'apt': cats_map['apt']}, ['apt'])
dcols = [c for c in v35.DET_COLS if c in rank.columns] + ['apt']
base = np.clip(model.predict(rank[model.feature_name()]), v35.PRED_FLOOR, v35.PRED_CEIL)
p = det.predict(rank[dcols])
sd = rank['sched_dev'].to_numpy()
pool = sd > v35.POOL_MIN
p = np.where(pool, p, 0.0)
pred = np.clip(np.where(pool, p*sd + (1-p)*base, base), v35.PRED_FLOOR, v35.PRED_CEIL)
out = pd.DataFrame({'MVT_ID_mvt': rank['MVT_ID_mvt'].values, 'base': base, 'p': p, 'sd': sd, 'pred': pred})
ref = pd.read_parquet(os.path.join(D, 'logical-ladder_v35.parquet'))
m = ref.merge(out, on='MVT_ID_mvt', how='left')
diff = np.abs(m['pred'] - m['TAXITIME_SEC_mvt'])
print(f'rows {len(m):,}  missing {int(m.pred.isna().sum())}  max |rebuilt - v35| = {diff.max():.9f}')
if not (diff.max() < 1e-6):
    sys.exit('DOES NOT REPRODUCE v35 - stop')
m[['MVT_ID_mvt','base','p','sd','pred']].to_parquet(os.path.join(sys.argv[1], 'v35_parts.parquet'), index=False)
print('REPRODUCED v35 exactly; wrote v35_parts.parquet (template order)')
