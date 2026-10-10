"""
Base features (queue counts 12_queue_features.py + METAR weather 15_weather.py) from the challenge PARQUET files.

  python DATA_ANALYTICS/10_build_base_features.py train            -> dep_q_train.parquet, dep_qw_train.parquet
  python DATA_ANALYTICS/10_build_base_features.py rank             -> dep_q_rank.parquet,  dep_qw_rank.parquet
  python DATA_ANALYTICS/10_build_base_features.py rank data/raw/final_ranking.parquet final   -> dep_qw_final.parquet

The original project built these from CSV exports of the same files; pandas reads strings such as "NA" as missing
in a CSV (e.g. RUNWAY_mvt == "NA"), so this script applies the same rule to reproduce those features exactly.
METAR comes from the cached IEM files in metar_cache/ (15_weather.py download_all() fills it).
"""
import os, sys, glob, gc, importlib.util
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

D = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(D, 'DATA_ANALYTICS')


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), os.path.join(A, name))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def build(raw, name, q12, w15):
    ch = raw[q12.USECOLS].copy()
    for c in ch.columns:
        if ch[c].dtype == object or pd.api.types.is_string_dtype(ch[c]):
            ch[c] = ch[c].where(~ch[c].isin(STR_NA_VALUES))
    for c in q12.CATCOLS:
        ch[c] = ch[c].astype('category')
    for src, dst in [('MVT_TIME_UTC_mvt', 'mvt_i'), ('BLOCK_TIME_UTC_mvt', 'blk_i'), ('SCHED_TIME_UTC_mvt', 'sch_i'),
                     ('AOBT_3_flt', 'aobt_i'), ('EOBT_1_flt', 'eobt_i')]:
        ch[dst] = q12._to_epoch(ch[src])
    keep_target = 'TAXITIME_SEC_mvt' in ch.columns
    ch = ch.drop(columns=['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt'])
    dep = ch[ch.PHASE_mvt == 'DEP'].copy(); dep['apt'] = dep['ADEP_mvt'].astype(str); dep = dep.drop(columns=['PHASE_mvt'])
    a = ch[ch.PHASE_mvt == 'ARR']
    arr = pd.DataFrame({'apt': a['ADES_mvt'].astype(str).values, 'land_i': a['mvt_i'].values, 'onblk_i': a['blk_i'].values})
    del ch, a; gc.collect()
    dep = q12.build_features(dep.reset_index(drop=True), arr)
    qpath = os.path.join(D, f'dep_q_{name}.parquet'); dep.to_parquet(qpath, index=False)
    del dep, arr; gc.collect()
    w15.attach(qpath, w15.build_weather(w15.load_cache()), os.path.join(D, f'dep_qw_{name}.parquet'), name.upper())
    print(f'wrote dep_q_{name}.parquet, dep_qw_{name}.parquet (target kept: {keep_target})', flush=True)


if __name__ == '__main__':
    q12, w15 = load('12_queue_features.py'), load('15_weather.py')
    which = sys.argv[1]
    if which == 'train':
        fs = sorted(glob.glob(os.path.join(D, 'data', 'raw', 'training_2025-*.parquet')))
        assert len(fs) == 12, f'expected 12 training files in data/raw, found {len(fs)} (run download.py)'
        raw = pd.concat([pd.read_parquet(f, columns=q12.USECOLS) for f in fs], ignore_index=True)
        build(raw, 'train', q12, w15)
    else:
        path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'data', 'raw', 'ranking.parquet')
        build(pd.read_parquet(path), sys.argv[3] if len(sys.argv) > 3 else 'rank', q12, w15)
