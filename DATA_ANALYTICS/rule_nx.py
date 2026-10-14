"""
Rule NX = no-ADS-B segment corrector (agent seg_noads, 12 Oct 2026; held-out on the geo-cat stack + T/P + DD: −0.25/−2.74/−0.50/−0.68, all −0.87)
Training rows: candidates/nx_train.parquet (held-out 2025 Jan/Feb/Jun/Jul, oof_s6geocatADGS_all + T/P + DD, + EX + NA).
Original docstring: -- label-free features + LightGBM residual corrector.

Witness added (new): RELATIVE plan delay. Trees see my MVT - LOBT and the neighbours' mean AOBT - SCHED, but cannot subtract:
  ex_lobt  = (MVT - LOBT)_me - median (MVT - LOBT) of the same airport's departures taking off within +-15 min (excl. me)
  ex_eobt  = same with EOBT_1
  ov_before/ov_after/ov_net = share/count of departures (+-30 min of my take-off) planned LATER than me that took off before
             me / planned EARLIER that took off after me  (plan = LOBT, SCHED if missing)
  If I am late vs plan relative to my neighbours, the extra delay was mostly spent ON STAND -> shorter taxi than the model says.
Plus the existing stand-turnover witness na_g1, na_l_aobt, na_next, na_l_lobt (DATA_ANALYTICS/42j_next_arrival.build).

Segment (label-free): ADEP in LTFM/LFPG/EGLL, adsb_matched != 1, MVT - SCHED <= 7200 s.
Corrector: LightGBM l2 on clip(y - pred, +-1500), 300 rounds lr 0.05, 15 leaves, min leaf 500, l2 10; pred += 0.7 * f.
Apply AFTER rules A D G S + T/P + DD.

  features(raw)            raw = whole file (DEP + ARR) -> DataFrame MVT_ID_mvt + EX columns (departures)
  fit(df) / apply(model, df)   df needs: ADEP_mvt, pred, mvt, aobt, lobt, eobt, sd (+ y for fit) + EX + NA columns
  python noads_corrector.py    LOMO self-check on scratchpad base.parquet / feats.parquet (held-out 2025)
"""
import numpy as np, pandas as pd

APTS = ['LTFM', 'LFPG', 'EGLL']                      # LIRF dropped: Rome copy contamination (worse 3/4 months)
ALL_APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
ALL = False   # True = 'NXA': all 10 airports, all ADS-B states, sd <= 7200 (held-out −0.63/−3.07/−0.87/−1.06 = −1.24 vs −0.87 for 3 apts)
EX = ['ex_lobt', 'ex_eobt', 'ov_before', 'ov_after', 'ov_net']
NA = ['na_g1', 'na_l_aobt', 'na_next', 'na_l_lobt']
CTRL = ['apt', 'pred', 'ml', 'ma', 'al', 'sd', 'me', 'hr']
X = CTRL + EX + NA
P = dict(objective='l2', learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10, verbose=-1, seed=1, num_threads=2)
SHRINK, CLIP, ROUNDS = 0.7, 1500.0, 300
LINEAR = False   # True = 'NXL': linear_tree leaves (lambda 15, shrink 0.85: 283.421 vs 0.7 283.519, 4/4); held-out on DDG stack 283.519 vs 283.753 (−0.20/−0.26/−0.17/−0.33)


def _ep(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values


def features(raw):
    dep = raw[raw.PHASE_mvt == 'DEP']
    g0 = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.values, 'apt': dep.ADEP_mvt.astype(str).values, 'mvt': _ep(dep.MVT_TIME_UTC_mvt),
                       'sch': _ep(dep.SCHED_TIME_UTC_mvt), 'lobt': _ep(dep.LOBT_flt), 'eobt': _ep(dep.EOBT_1_flt)})
    res = []
    for _, g in g0.groupby('apt'):
        g = g.sort_values('mvt')
        mv = g.mvt.values; pl = np.where(np.isnan(g.lobt.values), g.sch.values, g.lobt.values)
        eo = np.where(np.isnan(g.eobt.values), g.sch.values, g.eobt.values)
        n = len(g)
        lo = np.searchsorted(mv, mv - 1800.); hi = np.searchsorted(mv, mv + 1800., 'right')
        lo15 = np.searchsorted(mv, mv - 900.); hi15 = np.searchsorted(mv, mv + 900., 'right')
        ovb, ova, nb, nn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        exl, exe = np.full(n, np.nan), np.full(n, np.nan)
        dl, de = mv - pl, mv - eo
        for i in range(n):
            b = pl[lo[i]:i]; c = pl[i + 1:hi[i]]
            ovb[i] = (b > pl[i]).sum(); nb[i] = len(b); ova[i] = (c < pl[i]).sum(); nn[i] = len(c)
            j = np.r_[lo15[i]:i, i + 1:hi15[i]]
            if len(j) >= 3:
                exl[i] = dl[i] - np.median(dl[j]); exe[i] = de[i] - np.median(de[j])
        res.append(pd.DataFrame({'MVT_ID_mvt': g.MVT_ID_mvt.values, 'ex_lobt': exl, 'ex_eobt': exe, 'ov_before': ovb / np.maximum(nb, 1),
                                 'ov_after': ova / np.maximum(nn, 1), 'ov_net': ovb - ova}))
    return pd.concat(res, ignore_index=True)


def _prep(df):
    s = df.copy()
    s['apt'] = pd.Categorical(s.ADEP_mvt.astype(str), categories=ALL_APTS if ALL else APTS)
    s['ml'] = s.mvt - s.lobt; s['ma'] = s.mvt - s.aobt; s['al'] = s.aobt - s.lobt
    s['hr'] = (s.mvt % 86400) / 3600; s['me'] = s.mvt - s.eobt
    return s


def segment(df):
    am = df['adsb_matched'] if 'adsb_matched' in df else pd.Series(np.nan, index=df.index)
    if ALL:
        return (df.sd <= 7200).values
    return (df.ADEP_mvt.isin(APTS) & (df.sd <= 7200) & (am.fillna(0) != 1)).values


def fit(df):
    import lightgbm as lgb
    s = _prep(df[segment(df) & (df.y <= 10800).values])
    prm = dict(P, linear_tree=True, linear_lambda=15.0) if LINEAR else P
    return lgb.train(prm, lgb.Dataset(s[X], np.clip(s.y - s.pred, -CLIP, CLIP)), ROUNDS)


def apply(model, df):
    p = df.pred.values.astype(float).copy(); m = segment(df)
    if m.any():
        p[m] = np.clip(p[m] + (0.85 if LINEAR else SHRINK) * model.predict(_prep(df[m])[X]), 60, None)
    return p
