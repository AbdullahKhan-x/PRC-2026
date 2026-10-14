"""
Rules K and M: monthly constraints from the EUROCONTROL ATXOT taxi-out file (agent3, 12 Oct 2026).
Source: https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_{YEAR}.csv
(ansperformance.eu; same portal as the daily delay file used by rule DD; monthly file ruled admissible by organisers 27 Sep / 8 Oct).

K (count): ATXOT excludes taxi-out > 120 min from VALID_FL. At LIRF, TF - VALID_FL equals the number of departures with
   y > 7200 s (2025: 12/12 months within -1..+2; TF equals our departure count in 2025 and 2026). These are schedule copies /
   date stamps (pool sd > 7200). One logit shift per LIRF month on the implied copy probability of no-NM pool rows,
   q = (pred - C)/(sd - C), C = 1000 s, makes the expected count match K_nonm = (TF - VALID_FL - off) * (1 - nm_share).
   pred = C + q'(sd - C), q' = sigmoid(logit q + w*lambda), |lambda| <= 3.
M (mean): mean taxi of valid flights = (REF + ADD)/REF_NB. Per airport-month (skipped when the invalid share > 2 %, i.e.
   de-icing months), shift rows with pred <= 7200 and sd <= 7200 by w*raw, raw = (mean_eu - c_apt) - mean(pred on them),
   c_apt = median 2025 offset (EU mean - label mean), w = max(0, 1 - var(c)/raw^2) (James-Stein).
Order: after ADGS (+FL) + TP + DD:  p = apply_k(...); p = apply_m(...).
"""
import numpy as np, pandas as pd

C, LMAX, W_K, SHARE = 1000.0, 3.0, 0.5, 0.02
MGUARD = True     # skip disruption months in rule M (set False to reproduce v146/v150/v151/v152)


def load_atxot(paths):
    e = pd.concat([pd.read_csv(p) for p in paths])
    e = e[e.TF.notna() & e.TOTAL_REF_NB_FL.notna()].copy()
    e['mean_eu'] = (e.TOTAL_REF_TIME_MIN + e.TOTAL_ADD_TIME_MIN) / e.TOTAL_REF_NB_FL * 60.0
    e['inval'] = e.TF - e.VALID_FL; e['share'] = e.inval / e.TF
    e['ym'] = e.YEAR.astype(int) * 100 + e.MONTH_NUM.astype(int)
    return e.set_index(['APT_ICAO', 'ym'])[['TF', 'VALID_FL', 'inval', 'share', 'mean_eu']]


def fit(dep, atx, exclude_month=None):
    """dep: 2025 labelled departures with ADEP_mvt, mvt, sch, aobt, y (epoch s). atx: load_atxot(2025 file).
    exclude_month: month (1-12) left out (LOMO evaluation). Returns the table of constants."""
    d = dep[['ADEP_mvt', 'mvt', 'sch', 'aobt', 'y']].copy(); d['ADEP_mvt'] = d.ADEP_mvt.astype(str)
    d['ym'] = 202500 + pd.to_datetime(d.mvt, unit='s').dt.month
    if exclude_month is not None:
        d = d[d.ym != 202500 + exclude_month]
    tab = {'M': {}}
    l = d[d.ADEP_mvt == 'LIRF']
    big = l.y > 7200
    tab['nm_share'] = float((big & l.aobt.notna()).sum() / max(big.sum(), 1))
    offs = []
    for ym, g in l.groupby('ym'):
        if ('LIRF', ym) in atx.index:
            offs.append(atx.loc[('LIRF', ym), 'inval'] - (g.y > 7200).sum())
    tab['k_off'] = float(np.mean(offs))
    for a, ga in d.groupby('ADEP_mvt'):
        cs = []
        for ym, g in ga.groupby('ym'):
            if (a, ym) not in atx.index or atx.loc[(a, ym), 'share'] > SHARE:
                continue
            v = g[(g.y <= 7200) & ((g.mvt - g.sch) <= 7200)]
            cs.append(atx.loc[(a, ym), 'mean_eu'] - v.y.mean())
        if len(cs) >= 3:
            tab['M'][a] = (float(np.median(cs)), float(np.var(cs, ddof=1)))
    return tab


def _ym(mvt):
    t = pd.to_datetime(mvt, unit='s')
    return (t.dt.year * 100 + t.dt.month).values


def apply_k(pred, sd, nonm, apt, mvt, atx, tab, w=W_K):
    pred = np.asarray(pred, float); out = pred.copy(); ym = _ym(pd.Series(mvt)); info = []
    apt = np.asarray(apt).astype(str); nonm = np.asarray(nonm).astype(bool); sd = np.asarray(sd, float)
    for k in np.unique(ym[apt == 'LIRF']):
        g = (apt == 'LIRF') & (ym == k)
        if g.sum() < 1000 or ('LIRF', k) not in atx.index:
            continue
        K = (atx.loc[('LIRF', k), 'inval'] - tab['k_off']) * (1 - tab['nm_share'])
        pool = g & nonm & (sd > 7200)
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.clip((pred - C) / (sd - C), 0, 1)
        act = pool & (q > 0) & (q < 1); fixed = (pool & (q >= 1)).sum()
        qq = np.clip(q[act], 1e-3, 1 - 1e-3); L = np.log(qq / (1 - qq)); t = K - fixed
        lo, hi = -LMAX, LMAX
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (1 / (1 + np.exp(-(L + mid)))).sum() < t: lo = mid
            else: hi = mid
        q2 = 1 / (1 + np.exp(-(L + w * mid)))
        out[act] = C + q2 * (sd[act] - C)
        info.append((int(k), round(float(K), 1), round(float(qq.sum() + fixed), 1), round(float(mid), 2), int(act.sum())))
    return out, info


def apply_m(pred, sd, apt, mvt, atx, tab, w='js'):
    pred = np.asarray(pred, float); out = pred.copy(); ym = _ym(pd.Series(mvt)); info = []
    apt = np.asarray(apt).astype(str); sd = np.asarray(sd, float)
    for a in np.unique(apt):
        if a not in tab['M']:
            continue
        c, s2 = tab['M'][a]
        for k in np.unique(ym[apt == a]):
            g = (apt == a) & (ym == k)
            if g.sum() < 1000 or (a, k) not in atx.index or atx.loc[(a, k), 'share'] > SHARE:
                continue
            if MGUARD and (sd[g] > 7200).mean() > 0.06:     # disruption month (2025 max 0.0504; EHAM 2026-01 0.088): the published
                continue                                     # mean includes its many >2 h rows, the comparable set excludes them (audit 12 Oct)
            comp = g & (pred <= 7200) & (sd <= 7200)
            raw = (atx.loc[(a, k), 'mean_eu'] - c) - pred[comp].mean()
            ww = max(0.0, 1 - s2 / raw ** 2) if w == 'js' else float(w)
            out[comp] += ww * raw
            info.append((a, int(k), round(float(raw), 1), round(ww, 2)))
    return out, info
