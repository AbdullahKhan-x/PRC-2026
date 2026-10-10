"""
Rule E - Rome (LIRF) departures without a Network Manager record, 1 h to 5.5 h late.

At LIRF the airport's off-block field is sometimes the schedule (a "copy"), which makes the
recorded taxi equal MVT - SCHED. For flights without an NM record the copy rate depends on
the lateness band AND on who operates the flight: in 2025, ITA Airways rows were 96% clean,
Israir / T'way / AlbaStar rows 0% clean (handling agents report differently). Destination
carries a similar, weaker signal.

  P(copy) = band_rate(lateness) * airline_ratio * destination_ratio     (clipped to [0.01, 0.99])
  ratio   = observed copies / expected copies for that airline (destination), shrunk towards 1
            with K pseudo-rows
  pred    = P * (MVT - SCHED) + (1 - P) * median clean taxi

Leave-one-month-out on 2025 (bands 3.6-20 ks): SSE 16.19e9 with band rates only,
11.63e9 with the airline ratio (better in 12/12 months), 10.78e9 adding destination
(better in 11/12). Fitted on all 2025 rows for the submission.
"""
import glob, os
import numpy as np
import pandas as pd

BANDS = [3600, 7200, 10800, 20000]
K = 10


def load_2025(D):
    cols = ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
            'TAXITIME_SEC_mvt', 'AOBT_3_flt']
    raw = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(D, 'data/raw/training_2025-*.parquet')))])
    r = raw[(raw.PHASE_mvt == 'DEP') & (raw.ADEP_mvt == 'LIRF') & raw.AOBT_3_flt.isna()].copy()
    r['sd'] = (pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(r.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    r['y'] = r.TAXITIME_SEC_mvt.astype(float)
    return r[(r.sd > BANDS[0]) & (r.sd <= BANDS[-1]) & (r.y <= 43200)]


def fit(r):
    r = r.copy()
    r['cp'] = (r.y - r.sd).abs() <= 60
    r['band'] = pd.cut(r.sd, BANDS, labels=False)
    r['al'] = r.FLIGHT_mvt.astype(str).str[:3]
    r['ad'] = r.ADES_mvt.astype(str)
    pb = r.groupby('band').cp.mean()
    r['pb'] = r.band.map(pb)

    def ratio(col):
        g = r.groupby(col).agg(n=('cp', 'size'), c=('cp', 'sum'), e=('pb', 'sum'))
        return ((g.c + K * (g.e / g.n)) / (g.e + K * (g.e / g.n))).to_dict()

    return {'band_rate': pb.to_dict(), 'al': ratio('al'), 'ad': ratio('ad'),
            't_clean': float(r.y[~r.cp & (r.y <= 10800)].median())}


def predict(tab, sd, flight, ades):
    band = pd.cut(pd.Series(sd), BANDS, labels=False).values
    pb = pd.Series(band).map(tab['band_rate']).values.astype(float)
    al = pd.Series(pd.Series(flight).astype(str).str[:3]).map(tab['al']).fillna(1.0).values
    ad = pd.Series(pd.Series(ades).astype(str)).map(tab['ad']).fillna(1.0).values
    p = np.clip(pb * al * ad, 0.01, 0.99)
    return p * sd + (1 - p) * tab['t_clean'], p
