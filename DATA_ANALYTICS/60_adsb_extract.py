"""
adsb.lol globe_history (ODbL 1.0) -> ground / low-altitude points near the 10 airports.

Streams the day's split tar parts, so the archive is never unpacked to disk. Either
  python 60_adsb_extract.py OUT.parquet day.tar.aa day.tar.ab       (any OS)
  cat day.tar.aa day.tar.ab | python 60_adsb_extract.py OUT.parquet (Linux / macOS)
so the
3-4 GB archive is never unpacked to disk. Each member traces/xx/trace_full_<icao>.json is gzip JSON:
  {"icao": hex, "timestamp": t0, "trace": [[dt, lat, lon, alt|"ground", gs, track, flags, vrate, info, ...], ...]}
Kept: points within +-0.10 deg of an airport reference point with alt == "ground" or alt <= 3000 ft.
"""
import sys, tarfile, gzip, json, io
import numpy as np, pandas as pd

APT = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4700, -0.4543), 'EHAM': (52.3086, 4.7639),
       'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626), 'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389),
       'LSZH': (47.4647, 8.5492), 'LTFM': (41.2753, 28.7519)}
BOX = 0.10
names = list(APT)
lat0 = np.array([APT[a][0] for a in names]); lon0 = np.array([APT[a][1] for a in names])

rows = {k: [] for k in ['icao', 't', 'lat', 'lon', 'alt', 'gs', 'apt', 'cs']}
n_mem = n_keep = 0
class _Chain(io.RawIOBase):
    """read several files back-to-back as one stream (split tar parts)"""
    def __init__(self, paths):
        self.fs = [open(p, 'rb') for p in paths]
    def readable(self):
        return True
    def readinto(self, b):
        while self.fs:
            n = self.fs[0].readinto(b)
            if n:
                return n
            self.fs.pop(0).close()
        return 0


src = io.BufferedReader(_Chain(sys.argv[2:])) if len(sys.argv) > 2 else sys.stdin.buffer
with tarfile.open(fileobj=src, mode='r|') as tf:
    for m in tf:
        if not m.isfile() or 'trace_full' not in m.name:
            continue
        n_mem += 1
        raw = tf.extractfile(m).read()
        try:
            js = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
        except Exception:
            continue
        tr = js.get('trace') or []
        if not tr:
            continue
        t0 = js.get('timestamp', 0)
        lat = np.array([p[1] for p in tr], dtype=float); lon = np.array([p[2] for p in tr], dtype=float)
        near = (np.abs(lat[:, None] - lat0[None, :]) < BOX) & (np.abs(lon[:, None] - lon0[None, :]) < BOX)
        anyn = near.any(1)
        if not anyn.any():
            continue
        idx = np.nonzero(anyn)[0]
        cs = ''
        for i in idx:
            p = tr[i]
            alt = p[3]
            ground = alt == 'ground'
            if not ground and (alt is None or alt > 3000):
                continue
            if len(p) > 8 and isinstance(p[8], dict) and p[8].get('flight'):
                cs = p[8]['flight'].strip()
            rows['icao'].append(js.get('icao', m.name[-11:-5])); rows['t'].append(t0 + p[0])
            rows['lat'].append(p[1]); rows['lon'].append(p[2]); rows['alt'].append(-1 if ground else alt)
            rows['gs'].append(p[4] if p[4] is not None else np.nan); rows['apt'].append(names[int(np.argmax(near[i]))])
            rows['cs'].append(cs)
            n_keep += 1
        if n_mem % 20000 == 0:
            print(f'members {n_mem:,}  points kept {n_keep:,}', flush=True)
df = pd.DataFrame(rows)
df['t'] = df.t.astype(float); df['alt'] = df.alt.astype(float); df['gs'] = pd.to_numeric(df.gs, errors='coerce')
for c in ['icao', 'apt', 'cs']:
    df[c] = df[c].astype('category')
df.to_parquet(sys.argv[1], index=False)
print(f'done: members {n_mem:,}, points {len(df):,} -> {sys.argv[1]}')
