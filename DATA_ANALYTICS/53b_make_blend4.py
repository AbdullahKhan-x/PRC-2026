"""v112 blend (validated on 4 held-out months, WORKLOG 9 Oct ~20:15 UTC):
  g    = (1 - W3) * (ctx2id + ctx2rep + ctx2res) / 3 + W3 * ctx3
  base = g                              where MVT - SCHED > 7200 (detector pool)
       = (1 - W_APT) * g + W_APT * apt  elsewhere, apt = 2 * ctx2ida - ctx2id (per-airport part)
  python 53b_make_blend4.py OUTTAG W3 W_APT     e.g. 53b_make_blend4.py v112 0.25 0.3
"""
import sys, os
import numpy as np, pandas as pd
C = os.path.join(os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'candidates')
OUT, W3, W_APT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
W_ADS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
ADS_TAG = sys.argv[5] if len(sys.argv) > 5 else 'kads'   # ADS-B member: kads (ADS-B) or kae (ADS-B + external daily data)
#     # v114: global = (1 - W_ADS) * v112-global + W_ADS * kads (ADS-B member)
P = {t: pd.read_parquet(os.path.join(C, f'parts_{t}.parquet')) for t in ['ctx2id', 'ctx2rep', 'ctx2res', 'ctx3', 'ctx2ida'] + ([ADS_TAG] if W_ADS > 0 else [])}
ref = P['ctx2id']
for t, p in P.items():
    assert (p.MVT_ID_mvt.values == ref.MVT_ID_mvt.values).all(), f'{t}: row order differs'
    assert np.allclose(p.p.values, ref.p.values) and np.allclose(p.sd.values, ref.sd.values), f'{t}: detector p / sd differ'
    print(f'{t:8s} base mean {p.base.mean():7.1f}   rms vs ctx2id {np.sqrt(((p.base - ref.base) ** 2).mean()):6.1f}')
g = (1 - W3) * (P['ctx2id'].base.values + P['ctx2rep'].base.values + P['ctx2res'].base.values) / 3 + W3 * P['ctx3'].base.values
if W_ADS > 0:
    g = (1 - W_ADS) * g + W_ADS * P[ADS_TAG].base.values
apt = 2 * P['ctx2ida'].base.values - P['ctx2id'].base.values
pool = ref.sd.values > 7200
base = np.clip(np.where(pool, g, (1 - W_APT) * g + W_APT * apt), 60, 172800)
out = ref.copy(); out['base'] = base
out['pred'] = np.clip(np.where(pool, out.p * out.sd + (1 - out.p) * base, base), 60, 172800)
out.to_parquet(os.path.join(C, f'parts_{OUT}.parquet'), index=False)
print(f'wrote parts_{OUT}.parquet  W3 {W3}  W_APT {W_APT}  W_ADS {W_ADS}  mean |base - ctx2id| {np.abs(base - ref.base.values).mean():.1f}s')
