"""
Rule G = rule C rebuilt (Rome, no NM record, 3-5.5 h late vs schedule):
  P(copy) = mean of two estimators: flight-code format (rule_f) and band x airline x destination (rome_rates, rule E's table);
  clean anchor = ctx2id base (a clean-taxi model; the ADS-B stack's base already hedges these rows -> double count).
  pred = P * sd + (1 - P) * ctx2id_base.
  Evidence (dSSE, date-stamp rows excluded): 12-month ctx2id LOMO -1.79e9, 11/12 months (259.32 -> 258.05);
  held-out ADS-B stack -7.6e8, 3/4 months (Jan +7e6). Rule E outside this zone loses every month -> not used.
"""
import numpy as np
import rule_f, rome_rates

LO, HI = 10800, 20000


def fit(D):
    return {'f': rule_f.fit(D), 'e': rome_rates.fit(rome_rates.load_2025(D))}


def apply(pred, sd, rome, flight, ades, anchor, tab):
    m = rome & (sd > LO) & (sd <= HI)
    fr = rule_f.fmt_long(flight)
    pf = np.where(fr, tab['f'][True], tab['f'][False])
    _, pe = rome_rates.predict(tab['e'], sd, flight, ades)
    r = 0.5 * (pf + pe)
    return np.where(m, r * sd + (1 - r) * anchor, pred), m, r
