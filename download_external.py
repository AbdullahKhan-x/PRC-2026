"""
Download the openly licensed external data used by the model (any OS, standard library only).

  python download_external.py

  data/ext/<series>_<year>.csv          EUROCONTROL daily airport series (free use with attribution):
                                        atfm_slot_adherence, all_pre_departure_delays, atc_pre_departure_delays
  data/ext/opdi/flight_list_YYYYMM.parquet   OPDI v0.0.2 flight lists (PRC + OpenSky Network, open data with
                                        attribution), filtered to flights touching the 10 challenge airports
Files that already exist are skipped; partial downloads are retried and never left behind as good files.
"""
import os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, 'data', 'ext')
SERIES = ['atfm_slot_adherence', 'all_pre_departure_delays', 'atc_pre_departure_delays']
YEARS = [2025, 2026]
CSV_URL = 'https://www.eurocontrol.int/performance/data/download/csv/{n}_{y}.csv'
OPDI_URL = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{m}.parquet'
OPDI_MONTHS = ['202412'] + [f'2025{m:02d}' for m in range(1, 13)] + ['202601', '202602', '202605', '202606', '202607']
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']


def fetch(url, dest, tries=8):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'exists  {os.path.relpath(dest, ROOT)}'); return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, 'wb') as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(part, dest)
            print(f'ok      {os.path.relpath(dest, ROOT)}  {os.path.getsize(dest) / 1e6:.1f} MB'); return True
        except Exception as e:
            print(f'retry {k + 1}/{tries} {url}: {e}'); time.sleep(10 * (k + 1))
    if os.path.exists(part):
        os.remove(part)
    print(f'FAILED  {url}'); return False


def filter_opdi(path):
    """keep only flights from/to the 10 airports (700 MB -> ~100 MB in total); verifies the file is complete"""
    import pandas as pd
    import pyarrow.parquet as pq
    pq.ParquetFile(path).metadata                     # raises on a truncated file
    x = pd.read_parquet(path, columns=['icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen'])
    if len(x) and not (x.adep.isin(AIRPORTS) | x.ades.isin(AIRPORTS)).all():
        x = x[x.adep.isin(AIRPORTS) | x.ades.isin(AIRPORTS)]
        x.to_parquet(path + '.tmp', index=False); os.replace(path + '.tmp', path)


if __name__ == '__main__':
    ok = all(fetch(CSV_URL.format(n=n, y=y), os.path.join(EXT, f'{n}_{y}.csv')) for n in SERIES for y in YEARS)
    for m in OPDI_MONTHS:
        dest = os.path.join(EXT, 'opdi', f'flight_list_{m}.parquet')
        if fetch(OPDI_URL.format(m=m), dest):
            filter_opdi(dest)
        else:
            ok = False
    sys.exit(0 if ok else 1)
