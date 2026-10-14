"""Merge feature files keyed by MVT_ID_mvt into a new pair:  python 57_merge_feats.py OUT BASE MORE [col_prefixes]
e.g. ctx3ae = ctx3 + the ADS-B/ext columns of ctxae:  python 57_merge_feats.py ctx3ae ctx3 ctxae adsb_,ext_"""
import os, sys
import pandas as pd
D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, base, more = sys.argv[1:4]
pref = tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 else None
for s in ('train', 'rank'):
    if not os.path.exists(os.path.join(D, f'{base}_{s}.parquet')):      # training-only inputs (held-out runs)
        print(f'{base}_{s}.parquet missing - skipped', flush=True); continue
    b = pd.read_parquet(os.path.join(D, f'{base}_{s}.parquet'))
    m = pd.read_parquet(os.path.join(D, f'{more}_{s}.parquet'))
    cols = [c for c in m.columns if c != 'MVT_ID_mvt' and c not in b.columns and (pref is None or c.startswith(pref))]
    b.merge(m[['MVT_ID_mvt'] + cols], on='MVT_ID_mvt', how='left').to_parquet(os.path.join(D, f'{out}_{s}.parquet'), index=False)
    print(f'{out}_{s}: +{len(cols)} cols from {more}', flush=True)
