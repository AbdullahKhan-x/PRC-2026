"""
te_extra (clock offsets, copy rates) + literature hypotheses H1 and H4, fitted on training rows only.

  H4  recording-regime fingerprints: share of training BLOCK stamps with seconds == 0 per (airport, stand)
      and (airport, operator), smoothed to the airport rate (K = 50). At 1-s-resolution airports a high share
      suggests manual entry; at 5-s airports the chance rate is 1/12.
  H1  products trees cannot form: copy rate x lateness  (te_cpy_stand * sched_dev, te_cpy_op * sched_dev),
      clipped lateness 0..20,000 s. The expected extra "taxi" from a schedule copy.
"""
import numpy as np
import pandas as pd
import te_extra

K = 50
COLS = te_extra.COLS + ['te_s00_stand', 'te_s00_op', 'te_cpyev_stand', 'te_cpyev_op']


def fit(tr):
    enc = te_extra.fit(tr)
    k = te_extra._keys(tr)
    s00 = (np.mod(tr['blk_i'], 60) == 0).astype(float)
    apt = tr['apt'].astype(str)
    enc['apt_s00'] = s00.groupby(apt).mean().to_dict()
    for name in ('stand', 'op'):
        g = s00.groupby(k[name]).agg(['sum', 'size'])
        pr = k[name].str.split('|').str[0].map(enc['apt_s00']).groupby(k[name]).first()
        enc['s00_' + name] = ((g['sum'] + K * pr) / (g['size'] + K)).to_dict()
    return enc


def apply(df, enc):
    df = te_extra.apply(df, enc)
    k = te_extra._keys(df)
    a = df['apt'].astype(str).map(enc['apt_s00']).astype(float)
    for name in ('stand', 'op'):
        df['te_s00_' + name] = k[name].map(enc['s00_' + name]).astype(float).fillna(a).astype(np.float32)
    late = np.clip(df['sched_dev'].astype(float), 0, 20000)
    df['te_cpyev_stand'] = (df['te_cpy_stand'] * late).astype(np.float32)
    df['te_cpyev_op'] = (df['te_cpy_op'] * late).astype(np.float32)
    return df
