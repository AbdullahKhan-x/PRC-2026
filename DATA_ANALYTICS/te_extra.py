"""
Target-side encodings fitted on training rows only (call inside each fold).

  te_dlt_*   median clock offset BLOCK - AOBT_3 per (airport, stand) / (airport, operator) /
             (airport, runway): how this stand's / handler's recorder sits relative to the
             Network Manager clock (clean rows: not a schedule copy, |offset| < 1 h)
  te_cpy_*   share of schedule copies (|BLOCK - SCHED| <= 60 s) per (airport, stand) and
             (airport, operator), smoothed towards the airport rate
"""
import numpy as np
import pandas as pd

MIN_N = 30
K = 50


def _s(x):
    # missing -> 'None' in every pandas version (pandas 2 object columns hold None -> 'None'; pandas 3 keeps NaN). Models trained on
    # Kaggle (pandas 2) learnt an 'apt|None' group (e.g. no-NM operator copy rate); this keeps their keys identical here.
    return x.astype(object).where(x.notna(), 'None').astype(str)


def _keys(df):
    apt = _s(df['apt'])
    return {
        'stand': apt + '|' + _s(df['STAND_mvt']),
        'op': apt + '|' + _s(df['AIRCRAFT_OPERATOR_flt']),
        'rwy': apt + '|' + _s(df['RUNWAY_mvt']),
    }


def fit(tr):
    k = _keys(tr)
    dlt = tr['blk_i'] - tr['aobt_i']
    cpy = ((tr['blk_i'] - tr['sch_i']).abs() <= 60).astype(float)
    clean = dlt.notna() & (dlt.abs() < 3600) & (cpy == 0)
    apt = tr['apt'].astype(str)
    enc = {'apt_dlt': dlt[clean].groupby(apt[clean]).median().to_dict(),
           'apt_cpy': cpy.groupby(apt).mean().to_dict()}
    for name, key in k.items():
        g = dlt[clean].groupby(key[clean])
        st = g.agg(['median', 'size'])
        enc['dlt_' + name] = st['median'][st['size'] >= MIN_N].to_dict()
        if name in ('stand', 'op'):
            g2 = cpy.groupby(key)
            s2 = g2.agg(['sum', 'size'])
            prior = key.str.split('|').str[0].map(enc['apt_cpy'])
            pr = prior.groupby(key).first()
            enc['cpy_' + name] = ((s2['sum'] + K * pr) / (s2['size'] + K)).to_dict()
    return enc


def apply(df, enc):
    k = _keys(df)
    apt = df['apt'].astype(str)
    a_dlt = apt.map(enc['apt_dlt']).astype(float)
    a_cpy = apt.map(enc['apt_cpy']).astype(float)
    for name in ('stand', 'op', 'rwy'):
        df['te_dlt_' + name] = k[name].map(enc['dlt_' + name]).astype(float).fillna(a_dlt).astype(np.float32)
    for name in ('stand', 'op'):
        df['te_cpy_' + name] = k[name].map(enc['cpy_' + name]).astype(float).fillna(a_cpy).astype(np.float32)
    # implied taxi corrected by the stand's clock offset: a direct estimate of MVT - BLOCK
    df['imp_minus_dlt_stand'] = (df['implied_taxi_aobt'] - df['te_dlt_stand']).astype(np.float32)
    return df


COLS = ['te_dlt_stand', 'te_dlt_op', 'te_dlt_rwy', 'te_cpy_stand', 'te_cpy_op', 'imp_minus_dlt_stand']


def add_witness(x):
    """B8 witness-difference features (need ADS-B adsb_move_gap): when the aircraft started moving relative to each
    other record of the off-block moment. Trees cannot subtract columns, so the differences are given explicitly."""
    g = x['adsb_move_gap'].astype('float32')
    x['w_move_m_aobt'] = (x['implied_taxi_aobt'] - g).astype('float32')          # t_move - AOBT_3
    x['w_move_m_eobt'] = (x['mvt_m_eobt'] - g).astype('float32')                 # t_move - EOBT_1
    x['w_move_m_lobt'] = (x['lobt_mvt_m_lobt'] - g).astype('float32')            # t_move - LOBT
    x['w_move_m_sch'] = (x['sched_dev'] - g).astype('float32')                    # t_move - SCHED
    return x


WITNESS_COLS = ['w_move_m_aobt', 'w_move_m_eobt', 'w_move_m_lobt', 'w_move_m_sch']
