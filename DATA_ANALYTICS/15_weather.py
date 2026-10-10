"""
STEP 4 - Weather features from METAR.

Targets the three things that are now most of your error:
  LIRF      34.1% of squared error, variance not structure, worst in July
  the tail  42.6% of squared error sits in taxi > 30 min
  summer    45 RMSE worse than winter at constant training volume

Downloads from the Iowa Environmental Mesonet ASOS archive (free, no key).
Caches every raw download to disk, so a re-run costs nothing and a crash
mid-way loses nothing.

IMPORTANT: p01i (precipitation) is US-only. All ten of our airports are
European so that column is empty for us. Weather type comes from wxcodes
and the raw METAR string instead. Do not build a precip feature off p01i.

Writes dep_qw_train.parquet and dep_qw_rank.parquet, which are the queue
parquets plus weather columns. 12/13/14 are untouched - v2 stays reproducible.
"""
import os
import re
import time
import gc
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATA_FOLDER = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(DATA_FOLDER, "metar_cache")

STATIONS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL',
            'LEMD', 'LFPG', 'LIRF', 'LTFM', 'LSZH']

# training covers 2025, ranking covers Jan-Jul 2026. Pad both ends.
START = pd.Timestamp("2024-12-28")
END = pd.Timestamp("2026-08-05")

SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
MAX_ATTEMPTS = 6


# ----------------------------------------------------------------------
# download
# ----------------------------------------------------------------------
def fetch(uri):
    """IEM rate-limits inbound requests, so back off rather than hammer."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data and not data.startswith("ERROR"):
                return data
        except Exception as exc:
            print(f"    attempt {attempt + 1} failed: {exc}")
        time.sleep(5 * (attempt + 1))
    return ""


def download_all():
    os.makedirs(CACHE, exist_ok=True)
    # quarterly chunks keep each request modest and each cache file resumable
    edges = pd.date_range(START, END, freq="QS").tolist()
    if edges[0] > START:
        edges = [START] + edges
    if edges[-1] < END:
        edges = edges + [END]

    for st in STATIONS:
        for a, b in zip(edges[:-1], edges[1:]):
            path = os.path.join(CACHE, f"{st}_{a:%Y%m%d}_{b:%Y%m%d}.txt")
            if os.path.exists(path) and os.path.getsize(path) > 500:
                continue
            uri = (SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=no"
                   + f"&station={st}"
                   + a.strftime("&year1=%Y&month1=%m&day1=%d")
                   + b.strftime("&year2=%Y&month2=%m&day2=%d")
                   + "&missing=null&trace=null"
                   + "&report_type=3&report_type=4")
            print(f"  {st}  {a:%Y-%m-%d} -> {b:%Y-%m-%d}")
            txt = fetch(uri)
            if not txt:
                print(f"    EMPTY for {st} {a:%Y-%m-%d}, will retry next run")
                continue
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
            time.sleep(1)


def load_cache():
    frames = []
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith(".txt"):
            continue
        try:
            d = pd.read_csv(os.path.join(CACHE, fn), comment='#',
                            low_memory=False)
        except Exception:
            continue
        if len(d):
            frames.append(d)
    if not frames:
        raise SystemExit("no METAR cached - run download_all() first")
    m = pd.concat(frames, ignore_index=True)
    m = m.drop_duplicates(subset=['station', 'valid'])
    present = set(m['station'].astype(str).unique())
    absent = [s for s in STATIONS if s not in present]
    if absent:
        print(f"  WARNING: zero observations downloaded for {absent}")
        print("  re-run to retry those stations before building features")
    return m


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def num(s):
    return pd.to_numeric(s, errors='coerce')


def build_weather(m):
    w = pd.DataFrame()
    w['apt'] = m['station'].astype(str)
    w['t'] = pd.to_datetime(m['valid'], errors='coerce', utc=True)

    tmpf = num(m.get('tmpf'))
    dwpf = num(m.get('dwpf'))
    w['temp_c'] = ((tmpf - 32) * 5 / 9).astype(np.float32)
    w['dewp_c'] = ((dwpf - 32) * 5 / 9).astype(np.float32)
    w['spread_c'] = (w['temp_c'] - w['dewp_c']).astype(np.float32)

    w['wind_kt'] = num(m.get('sknt')).astype(np.float32)
    w['gust_kt'] = num(m.get('gust')).fillna(0).astype(np.float32)
    w['wind_dir'] = num(m.get('drct')).astype(np.float32)
    # visibility arrives in statute miles; metres is what ATC thresholds use
    w['vis_m'] = (num(m.get('vsby')) * 1609.34).astype(np.float32)

    ceil = num(m.get('skyl1'))
    cover = m.get('skyc1', pd.Series(index=m.index, dtype=object)).astype('string').str.upper()
    # only BKN/OVC count as a ceiling; SCT and FEW do not. A BKN/OVC report
    # with no altitude is NOT "no ceiling" - it is unknown, so leave it NaN.
    is_ceil = cover.isin(['BKN', 'OVC']).fillna(False).to_numpy()
    no_ceil = cover.isin(['CLR', 'SKC', 'NSC', 'NCD', 'FEW',
                          'SCT']).fillna(False).to_numpy()
    ceiling = np.full(len(m), np.nan, dtype=np.float32)
    ceiling[no_ceil] = 9999.0
    have = is_ceil & ceil.notna().to_numpy()
    ceiling[have] = ceil.to_numpy()[have]
    w['ceiling_ft'] = ceiling

    # ---- present weather -------------------------------------------------
    # p01i is US-only, so wxcodes is the primary source. Fall back to the raw
    # METAR only when wxcodes is absent, and then match proper weather GROUPS
    # rather than bare substrings: 'GRADU' contains GR and RA, 'RERA' contains
    # RA, and neither is weather.
    wx_s = m.get('wxcodes', pd.Series(index=m.index, dtype=object)).astype('string')
    raw_s = m.get('metar', pd.Series(index=m.index, dtype=object)).astype('string')
    has_wx = (wx_s.notna() & (wx_s.str.strip() != '')).fillna(False)
    has_raw = (raw_s.notna() & (raw_s.str.strip() != '')).fillna(False)

    PRECIP = (r'(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|SQ|'
              r'PO|FC|SS|DS)')
    DESC = r'(?:MI|PR|BC|DR|BL|SH|TS|FZ)'
    # A descriptor may stand alone (TS, VCTS, VCSH) or be followed by
    # precipitation (TSRA, FZRA, +SHRA). Precipitation may also stand alone
    # (RA, BR, FG). The trailing lookahead is what rejects GRADU and RERA.
    WX_GROUP = re.compile(
        r'(?:^|\s)([+-]?(?:VC)?(?:' + DESC + PRECIP + r'*|' + PRECIP
        + r'+))(?=\s|$)')

    def groups_from_raw(s):
        if not isinstance(s, str):
            return ''
        return ' '.join(WX_GROUP.findall(s.upper()))

    tokens = wx_s.fillna('').str.upper()
    need_raw = (~has_wx) & has_raw
    if need_raw.any():
        tokens.loc[need_raw] = raw_s.loc[need_raw].map(groups_from_raw)
    tokens = tokens.fillna('')

    # a row with neither wxcodes nor a raw METAR tells us nothing about
    # weather - that must stay missing, not silently become "clear"
    wx_known = (has_wx | has_raw).to_numpy()

    for flag, pat in [('wx_ts', 'TS'), ('wx_sn', 'SN'), ('wx_fz', 'FZ'),
                      ('wx_ra', 'RA'), ('wx_fg', 'FG'), ('wx_br', 'BR'),
                      ('wx_gr', 'GR'), ('wx_shra', 'SHRA')]:
        hit = tokens.str.contains(pat, regex=False).fillna(False).to_numpy()
        w[flag] = np.where(wx_known, hit.astype(np.float32),
                           np.nan).astype(np.float32)

    # ---- derived operational flags, missing-aware ------------------------
    # NaN < x evaluates False in pandas, which would code a station outage
    # during fog as confirmed good weather. Every branch below is explicit.
    vis = w['vis_m'].to_numpy()
    cf = w['ceiling_ft'].to_numpy()
    vis_k, cf_k = np.isfinite(vis), np.isfinite(cf)
    trig = (vis_k & (vis < 1500)) | (cf_k & (cf < 300))
    lvp = np.full(len(w), np.nan, dtype=np.float32)
    lvp[trig] = 1.0                       # one confirmed trigger is enough
    lvp[(~trig) & vis_k & cf_k] = 0.0     # both known and neither fired
    w['lvp'] = lvp                        # otherwise: cannot rule it out

    wet = w[['wx_sn', 'wx_fz', 'wx_ra', 'wx_br', 'wx_fg']].max(axis=1).to_numpy()
    temp = w['temp_c'].to_numpy()
    t_k = np.isfinite(temp)
    cold = t_k & (temp <= 3)
    deice = np.full(len(w), np.nan, dtype=np.float32)
    deice[t_k & ~cold] = 0.0              # too warm: no de-icing regardless
    known_wet = cold & np.isfinite(wet)
    deice[known_wet] = wet[known_wet]
    w['deice_risk'] = deice

    w['freezing'] = np.where(t_k, (temp <= 0).astype(np.float32),
                             np.nan).astype(np.float32)

    w = w.dropna(subset=['t']).sort_values(['apt', 't']).reset_index(drop=True)

    # persistence matters more than the instantaneous reading - a thunderstorm
    # that cleared 20 minutes ago still has a queue backed up behind it
    out = []
    for apt, g in w.groupby('apt', sort=False):
        g = g.set_index('t')
        r = pd.DataFrame(index=g.index)
        r['ts_prev3h'] = g['wx_ts'].rolling('3h').max()
        r['ts_prev6h'] = g['wx_ts'].rolling('6h').max()
        r['lvp_prev3h'] = g['lvp'].rolling('3h').mean()
        r['deice_prev3h'] = g['deice_risk'].rolling('3h').max()
        r['gust_max3h'] = g['gust_kt'].rolling('3h').max()
        r['vis_min3h'] = g['vis_m'].rolling('3h').min()
        r['apt'] = apt
        out.append(r.reset_index())
    roll = pd.concat(out, ignore_index=True)
    w = w.merge(roll, on=['apt', 't'], how='left')
    return w


def runway_heading(rwy):
    """'25' -> 250 deg, '16R' -> 160 deg. Returns NaN when unparseable."""
    s = rwy.astype(str).str.extract(r'^(\d{1,2})')[0]
    return pd.to_numeric(s, errors='coerce') * 10.0


def attach(dep_path, w, out_path, tag):
    print(f"\n=== {tag} ===")
    d = pd.read_parquet(dep_path)
    d['apt'] = d['apt'].astype(str)
    # Join on pushback time. Where AOBT is absent, fall back to scheduled time
    # but never later than actual takeoff - on 3.3% of fallback rows the
    # schedule sits after the aircraft was already airborne.
    fallback = np.fmin(d['sch_i'].to_numpy(), d['mvt_i'].to_numpy())
    ref = np.where(d['aobt_i'].notna(), d['aobt_i'].fillna(0), fallback)
    # both keys must carry the same datetime resolution or merge_asof refuses
    d['_t'] = pd.to_datetime(ref, unit='s', utc=True).astype('datetime64[us, UTC]')

    w = w.copy()
    w['t'] = w['t'].astype('datetime64[us, UTC]')
    d = d.sort_values('_t')
    w = w.sort_values('t')
    wcols = [c for c in w.columns if c not in ('apt', 't')]
    # backward as-of: only observations already published at pushback time.
    # a forward join would leak weather the crew could not have known.
    d = pd.merge_asof(d, w[['apt', 't'] + wcols],
                      left_on='_t', right_on='t', by='apt',
                      direction='backward',
                      tolerance=pd.Timedelta('2h'))

    d['wx_age_min'] = ((d['_t'] - d['t']).dt.total_seconds() / 60).astype(np.float32)
    d = d.drop(columns=['_t', 't'])

    # crosswind relative to the runway actually in use
    hd = runway_heading(d['RUNWAY_mvt'])
    ang = np.radians(d['wind_dir'] - hd)
    d['crosswind_kt'] = (d['wind_kt'] * np.sin(ang)).abs().astype(np.float32)
    d['headwind_kt'] = (d['wind_kt'] * np.cos(ang)).astype(np.float32)

    cov = d['temp_c'].notna().mean() * 100
    print(f"  rows {len(d):,}   weather matched {cov:.2f}% overall")
    coverage_report(w, d, tag)
    d.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    del d
    gc.collect()


def coverage_report(w, d, tag):
    """
    Per-ICAO coverage. An aggregate number hides a dead station: if LTFM
    returned nothing, overall coverage still reads ~86% and you would have
    no idea which airport vanished.
    """
    obs = w.groupby('apt').size()
    print(f"\n  per-airport coverage [{tag}]")
    print(f"    {'ICAO':<6s}{'wx_obs':>10s}{'flights':>10s}"
          f"{'matched':>10s}{'match%':>9s}{'age_med':>9s}")
    problems = []
    for a in STATIONS:
        n_obs = int(obs.get(a, 0))
        sub = d[d['apt'] == a]
        n_f = len(sub)
        ok = sub['temp_c'].notna()
        n_m = int(ok.sum())
        pct = (n_m / n_f * 100) if n_f else np.nan
        age = sub.loc[ok, 'wx_age_min'].median() if n_m else np.nan
        print(f"    {a:<6s}{n_obs:>10,d}{n_f:>10,d}{n_m:>10,d}"
              f"{pct:>8.2f}%{age:>9.0f}")
        if n_obs == 0 or (n_f and pct < 90):
            problems.append((a, n_obs, pct))

    if problems:
        print("\n    *** COVERAGE PROBLEM - DO NOT TRUST CV ON THESE ***")
        for a, n_obs, pct in problems:
            print(f"    {a}: {n_obs:,} observations, {pct:.2f}% of flights matched")
        print("    LTFM is the usual suspect: Istanbul Airport opened in 2019")
        print("    replacing LTBA, and IEM coverage of newer stations is thin.")
        print("    It is also the largest airport here by volume.")
    else:
        print("    all 10 stations above 90% - safe to proceed")
    return problems


if __name__ == "__main__":
    print("downloading METAR (cached, safe to re-run)...")
    download_all()
    print("\nparsing...")
    m = load_cache()
    print(f"  {len(m):,} observations, {m['station'].nunique()} stations")
    w = build_weather(m)
    del m
    gc.collect()
    print(f"  {len(w):,} usable observations")

    attach(os.path.join(DATA_FOLDER, "dep_q_train.parquet"), w,
           os.path.join(DATA_FOLDER, "dep_qw_train.parquet"), "TRAIN")
    attach(os.path.join(DATA_FOLDER, "dep_q_rank.parquet"), w,
           os.path.join(DATA_FOLDER, "dep_qw_rank.parquet"), "RANK")
    print("\ndone. v2 pipeline untouched - 12/13/14 still reproduce the baseline.")
