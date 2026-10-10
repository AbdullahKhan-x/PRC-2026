# ADS-B extraction on Kaggle (team logical-ladder, PRC Data Challenge 2026). Self-contained, no inputs needed.
# Data: adsb.lol globe_history (ODbL 1.0), https://github.com/adsblol/globe_history_2025 / _2026 releases.
#
# HOW TO USE (Kaggle notebook):
#   1. New Notebook -> Settings: Internet ON (needs a phone-verified account), Accelerator: None (CPU).
#   2. Paste this whole file into ONE code cell. Edit PERIOD below (one notebook per period).
#   3. "Save Version" -> "Save & Run All (Commit)". It runs in the background (<= 12 h); you can close the tab.
#   4. When finished: open the version -> Output -> download pts_<PERIOD>.zip (or "Download all").
#
# Output: /kaggle/working/pts/pts_YYYY-MM-DD.parquet (ground / <=3000 ft points within 0.10 deg of the
#         10 airports; ~10-20 MB per day) + a zip of them + progress.log.
import os, sys, json, gzip, tarfile, subprocess, time, datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

PERIOD = 'A'          # 'A' = 2026 Jan+Feb, 'B' = 2026 Jun+Jul, 'C' = 2025 Jan+Feb, 'D' = 2025 Jun+Jul
WORKERS = 4

PERIODS = {'A': [(2026, 1), (2026, 2)], 'B': [(2026, 6), (2026, 7)], 'C': [(2025, 1), (2025, 2)], 'D': [(2025, 6), (2025, 7)],
           'E': [(2025, 3), (2025, 4)], 'G': [(2025, 5), (2025, 8)], 'H': [(2025, 9), (2025, 10)], 'I': [(2025, 11), (2025, 12)]}
OUT = '/kaggle/working/pts'
os.makedirs(OUT, exist_ok=True)

APT = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4700, -0.4543), 'EHAM': (52.3086, 4.7639),
       'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626), 'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389),
       'LSZH': (47.4647, 8.5492), 'LTFM': (41.2753, 28.7519)}
BOX = 0.10


def log(msg):
    line = f'{dt.datetime.utcnow():%H:%M:%S} {msg}'
    print(line, flush=True)
    with open('/kaggle/working/progress.log', 'a') as fh:
        fh.write(line + '\n')


def asset_urls(day):
    """URLs of the split tar parts (aa, ab, ...) of the day's 'prod' release, built directly (no API rate limit)."""
    for variant in ('prod-0', 'prod-0tmp', 'prod-1', 'prod-1tmp'):   # some days carry a 'tmp' / '-1' tag (e.g. 2025-06-01..10)
        res = _asset_urls(day, variant)
        if res:
            return res
    return []


def _asset_urls(day, variant):
    tag = f'v{day:%Y.%m.%d}-planes-readsb-{variant}'
    base = f'https://github.com/adsblol/globe_history_{day.year}/releases/download/{tag}/{tag}.tar'
    out = []
    for suf in ['aa', 'ab', 'ac', 'ad', 'ae', 'af']:
        url = f'{base}.{suf}'
        code = '000'
        for attempt in range(4):
            r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '-I', '-L', url], capture_output=True, text=True)
            code = r.stdout.strip()
            if code in ('200', '404'):
                break
            time.sleep(5 * (attempt + 1))
        if code != '200':
            break
        out.append((f'{tag}.tar.{suf}', url))
    if not out:                     # some days may be a single un-split .tar
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '-I', '-L', base], capture_output=True, text=True)
        if r.stdout.strip() == '200':
            out = [(f'{tag}.tar', base)]
    return out


def extract_day(day_str):
    import numpy as np, pandas as pd
    day = dt.date.fromisoformat(day_str)
    out = f'{OUT}/pts_{day_str}.parquet'
    if os.path.exists(out):
        return day_str, 'exists'
    parts = asset_urls(day)
    if not parts:
        return day_str, 'NO RELEASE FOUND'
    # stream all parts back-to-back into tarfile (split tar = plain byte split)
    cmd = ' ; '.join(f"curl -sfL --retry 10 --retry-all-errors '{u}'" for _, u in parts)
    proc = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE, bufsize=1 << 20)
    names = list(APT); lat0 = np.array([APT[a][0] for a in names]); lon0 = np.array([APT[a][1] for a in names])
    rows = {k: [] for k in ['icao', 't', 'lat', 'lon', 'alt', 'gs', 'apt', 'cs']}
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
            for m in tf:
                if not m.isfile() or 'trace_full' not in m.name:
                    continue
                n += 1
                raw = tf.extractfile(m).read()
                try:
                    js = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
                except Exception:
                    continue
                tr = js.get('trace') or []
                if not tr:
                    continue
                lat = np.array([p[1] for p in tr], dtype=float); lon = np.array([p[2] for p in tr], dtype=float)
                near = (np.abs(lat[:, None] - lat0[None, :]) < BOX) & (np.abs(lon[:, None] - lon0[None, :]) < BOX)
                anyn = near.any(1)
                if not anyn.any():
                    continue
                t0 = js.get('timestamp', 0); cs = ''
                for i in np.nonzero(anyn)[0]:
                    p = tr[i]; alt = p[3]; ground = alt == 'ground'
                    if not ground and (alt is None or alt > 3000):
                        continue
                    if len(p) > 8 and isinstance(p[8], dict) and p[8].get('flight'):
                        cs = p[8]['flight'].strip()
                    rows['icao'].append(js.get('icao', m.name[-11:-5])); rows['t'].append(t0 + p[0])
                    rows['lat'].append(p[1]); rows['lon'].append(p[2]); rows['alt'].append(-1 if ground else alt)
                    rows['gs'].append(p[4] if p[4] is not None else np.nan)
                    rows['apt'].append(names[int(np.argmax(near[i]))]); rows['cs'].append(cs)
    finally:
        proc.stdout.close(); proc.wait()
    df = pd.DataFrame(rows)
    df['t'] = df.t.astype(float); df['alt'] = df.alt.astype(float); df['gs'] = pd.to_numeric(df.gs, errors='coerce')
    for c in ['icao', 'apt', 'cs']:
        df[c] = df[c].astype('category')
    df.to_parquet(out + '.part', index=False); os.replace(out + '.part', out)
    return day_str, f'{n:,} traces, {len(df):,} points, parts {[p for p, _ in parts]}'


DAYS = []            # fix-up runs: explicit list of 'YYYY-MM-DD' (overrides PERIOD)
days = list(DAYS)
for y, mth in ([] if DAYS else PERIODS[PERIOD]):
    d = dt.date(y, mth, 1)
    while d.month == mth:
        days.append(d.isoformat()); d += dt.timedelta(days=1)
log(f'period {PERIOD}: {len(days)} days, {WORKERS} workers')
t0 = time.time()
with ProcessPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(extract_day, d): d for d in days}
    for f in as_completed(futs):
        try:
            d, msg = f.result()
        except Exception as e:
            d, msg = futs[f], f'FAILED {e!r}'
        log(f'{d}: {msg}   (elapsed {(time.time() - t0) / 60:.0f} min)')
subprocess.run(['bash', '-c', f'cd /kaggle/working && zip -q -r pts_{PERIOD}.zip pts progress.log'])
log('ALL DONE')
