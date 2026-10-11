"""CatBoost member helpers shared by 44_train_full.py (Kaggle, pandas 2) and 55b_final_recipe.py (laptop, pandas 3).
Categoricals become strings with missing -> 'nan' in every pandas version (pandas 2 astype(str) gives 'nan',
pandas 3 keeps NaN, which CatBoost rejects) - the same strings at training and at prediction time.
"""
import numpy as np, pandas as pd


def _s(v):
    o = v.astype(object).to_numpy()
    miss = pd.isna(o)
    out = np.array([str(e) for e in o], dtype=object)
    out[miss] = 'nan'
    return out


def cat_frame(x, feats, cats):
    x = x[feats].copy()
    for c in cats:
        x[c] = _s(x[c])
    return x
