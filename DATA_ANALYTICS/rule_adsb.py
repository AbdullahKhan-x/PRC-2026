"""
Rule ADSB (own idea, 13 Oct 2026): blend toward a direct ADS-B observation of the taxi.
Grade rows (label-free): ADS-B take-off matched, aircraft parked when first heard (gs0 <= 1), 0 <= move_gap <= max(sd, 3600).
  y_e = adsb_move_gap + off[airport]   (off = 2025 median of y - move_gap on grade rows)
  pred += w(|y_e - pred|) * (y_e - pred),  w by disagreement band (through-origin slope, clipped [0,1]):
  agree < 60 s -> ~0.78, 60-180 s -> ~0.42, larger -> ~0 (mismatched tracks ignored).
Held-out on the v150 stack (LOMO constants): -0.106 RMSE (-0.091/-0.140/-0.122/-0.086), 50k rows (7.5 %).
Apply after K, before S' and CAP.
"""
import json, numpy as np

def grade(sd, matched, move_gap, gs0):
    sd = np.asarray(sd, float); mg = np.asarray(move_gap, float)
    return (np.asarray(matched) == 1) & (np.nan_to_num(np.asarray(gs0, float), nan=99) <= 1) & (mg >= 0) & (mg <= np.maximum(sd, 3600))

def apply(pred, sd, apt, matched, move_gap, gs0, tab, exclude=None):
    pred = np.asarray(pred, float).copy(); g = grade(sd, matched, move_gap, gs0)
    if exclude is not None: g &= ~np.asarray(exclude, bool)
    off = np.array([tab['off'].get(a, np.nan) for a in np.asarray(apt).astype(str)])
    g &= ~np.isnan(off)
    dif = np.asarray(move_gap, float) + off - pred
    w = np.zeros(len(pred)); e = tab['edges']
    for i in range(len(e) - 1):
        m = g & (np.abs(dif) > e[i]) & (np.abs(dif) <= e[i + 1]); w[m] = tab['w'][i]
    pred[g] = pred[g] + w[g] * dif[g]
    return pred, g
