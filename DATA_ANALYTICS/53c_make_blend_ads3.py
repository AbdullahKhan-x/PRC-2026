"""v117 blend (validated on 4 held-out months, WORKLOG 10 Oct): all members carry ADS-B + external features.
  g    = mean(kae, krepae, kresae)          (+ optional W_C3 * ctx3ae later)
  base = g in the pool (MVT - SCHED > 7200), else (1 - W_APT) * g + W_APT * apt,  apt = 2 * ctx2ida - ctx2id
  python 53c_make_blend_ads3.py OUTTAG W_APT [member,member,...]
  env APT_PURE=tag: apt = parts_{tag}.base (APT_ONLY member, 44_train_full APT_ONLY=1) instead of 2 * ctx2ida - ctx2id
"""
import sys, os
import numpy as np, pandas as pd
D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(D, 'candidates')
OUT, W_APT = sys.argv[1], float(sys.argv[2])
MEM = sys.argv[3].split(',') if len(sys.argv) > 3 else ['kae', 'krepae', 'kresae']
APT_PURE = os.environ.get('APT_PURE', '')
P = {t: pd.read_parquet(os.path.join(C, f'parts_{t}.parquet')) for t in MEM + ['ctx2id'] + ([APT_PURE] if APT_PURE else ['ctx2ida'])}
ref = P['ctx2id']
for t, p in P.items():
    assert (p.MVT_ID_mvt.values == ref.MVT_ID_mvt.values).all() and np.allclose(p.p.values, ref.p.values), t
    print(f'{t:8s} base mean {p.base.mean():7.1f}  rms vs ctx2id {np.sqrt(((p.base - ref.base) ** 2).mean()):6.1f}')
g = np.mean([P[t].base.values for t in MEM], axis=0)
apt = P[APT_PURE].base.values if APT_PURE else 2 * P['ctx2ida'].base.values - ref.base.values
pool = ref.sd.values > 7200
base = np.clip(np.where(pool, g, (1 - W_APT) * g + W_APT * apt), 60, 172800)
out = ref.copy(); out['base'] = base
out['pred'] = np.clip(np.where(pool, out.p * out.sd + (1 - out.p) * base, base), 60, 172800)
out.to_parquet(os.path.join(C, f'parts_{OUT}.parquet'), index=False)
print(f'wrote parts_{OUT}.parquet  members {MEM}  W_APT {W_APT}  apt {APT_PURE or "2*ctx2ida-ctx2id"}')
