"""
STEP 13 - Final pipeline. Base model + extreme detector, refit on all of 2025,
blended at prediction time, written with the row-order gate.

WHY THIS SHAPE
Three submissions so far:
    v1 546.4895   v2 545.5379   v3 521.4497
against locals of 255-272. Three independent pipelines, same band. The
capped local of 272.50 corresponds to an SSE of 2.55e10; reaching 521 needs
roughly 70-80 additional rows at ~30,000s of error each. That is the entire
gap, and it lives in a few dozen rows none of these models ever predicted,
because every one of them clips at 10800.

On 2025 Jan/Jul, measured on the UNCAPPED metric:
    base (no intervention)   417.32
    soft blend (pool only)   336.59   (+80.73)
    oracle                   316.72   (+100.60)
The detector recovers ~80% of what perfect knowledge would give.

THE MECHANISM
A subset of rows has no measured off-block time: BLOCK_TIME is filled with
SCHED_TIME, so the target equals MVT_TIME - SCHED_TIME to within 5 seconds.
When such a flight departs hours late, the "taxi time" is the whole departure
delay - tens of thousands of seconds. Both columns are 100% present in the
ranking file, so those rows are exactly predictable once identified.

SAFETY
  - Only rows inside the candidate pool (sched_dev > POOL_MIN) are ever
    modified. Everything else keeps the base prediction untouched. A global
    blend was tried and cost 3.75 RMSE by injecting bias into clean rows.
  - Predictions are NOT clipped at 10800. That clip made the extremes
    unwinnable by construction.
  - The submission is reindexed onto the template and asserted element-wise.
"""
import warnings
import gc
import os
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_FOLDER = os.environ.get('PRC_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GLOB = os.path.join(DATA_FOLDER, "data_raw_training_*.parquet")
TEAM = "logical-ladder"
VERSION = 30
TEMPLATE = "submitting.csv"

TARGET = 'TAXITIME_SEC_mvt'
TRAIN_CAP = 10800          # base regressor only; measured correct (-25.18 without)
POOL_MIN = 7200            # candidate pool floor
BLEND = 'soft'             # 'soft' won locally (336.59 vs 336.63 at hard p>0.2)
HARD_T = 0.20              # used only when BLEND == 'hard'
PRED_FLOOR = 60
PRED_CEIL = 86400 * 2      # a sane absolute ceiling, far above any real value
EARLY_STOP = 100
MAX_ROUNDS = 3000
HOLDOUT_FRAC = 0.05        # random slice used ONLY to pick the iteration count

CAT_COLS = ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt',
            'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt',
            'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt', 'apt']
QUEUE_COLS = [
    'apt_queue', 'apt_push_prev15', 'apt_push_prev60', 'apt_push_next15',
    'apt_push_next30', 'apt_tko_prev15', 'apt_tko_prev60', 'apt_wait_est_min',
    'rwy_queue', 'rwy_push_prev15', 'rwy_push_prev60', 'rwy_push_next15',
    'rwy_push_next30', 'rwy_tko_prev15', 'rwy_tko_prev60', 'rwy_wait_est_min',
    'arr_on_surface', 'arr_land_prev15', 'arr_land_prev60', 'arr_land_next30',
    'surface_total', 'dep_share_of_surface', 'rwy_share_of_apt_queue',
    'rwys_active_this_hour',
]
TIME_COLS = ['hour', 'dow', 'month', 'minute_of_day',
             'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
AOBT_COLS = ['implied_taxi_aobt', 'has_aobt', 'pushback_delay']
ENC_COLS = ['stand_unimp', 'rwy_stand_unimp', 'rwy_stand_delay',
            'stand_te', 'operator_te', 'apt_rwy_unimp']
WX_COLS = [
    'temp_c', 'dewp_c', 'spread_c', 'wind_kt', 'gust_kt', 'wind_dir',
    'vis_m', 'ceiling_ft', 'wx_ts', 'wx_sn', 'wx_fz', 'wx_ra', 'wx_fg',
    'wx_br', 'wx_gr', 'wx_shra', 'lvp', 'deice_risk', 'freezing',
    'ts_prev3h', 'ts_prev6h', 'lvp_prev3h', 'deice_prev3h', 'gust_max3h',
    'vis_min3h', 'wx_age_min', 'crosswind_kt', 'headwind_kt',
]
DEST_COLS = ['ADES_flt']
SCHED_COLS = ['sched_dev_clip', 'sched_dev_abs', 'sched_dev_pos']
DET_COLS = ['sched_dev_log', 'implied_taxi_aobt', 'has_aobt',
            'imp_minus_sched', 'imp_is_odd']

PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.04, 'num_leaves': 255, 'min_data_in_leaf': 60,
    'feature_fraction': 0.60, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'lambda_l2': 1.0, 'verbose': -1, 'num_threads': -1,
    'seed': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42,
}
DET_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.02, 'num_leaves': 31, 'min_data_in_leaf': 20,
    'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
    'verbose': -1, 'num_threads': -1, 'seed': 42, 'is_unbalance': True,
}
DET_ROUNDS = 400


def derive(df, with_labels):
    """Shared feature construction. Identical for training and ranking."""
    df['apt'] = df['apt'].astype(str)
    df['rwy_stand'] = (df['apt'] + '|' + df['RUNWAY_mvt'].astype(str)
                       + '|' + df['STAND_mvt'].astype(str))
    df['apt_stand'] = df['apt'] + '|' + df['STAND_mvt'].astype(str)
    df['apt_rwy'] = df['apt'] + '|' + df['RUNWAY_mvt'].astype(str)

    df['sched_dev'] = (df['mvt_i'] - df['sch_i']).astype(np.float32)
    df['sched_dev_clip'] = df['sched_dev'].clip(-3600, 10800).astype(np.float32)
    df['sched_dev_abs'] = df['sched_dev_clip'].abs().astype(np.float32)
    df['sched_dev_pos'] = df['sched_dev_clip'].clip(lower=0).astype(np.float32)

    df['sched_dev_log'] = np.log1p(df['sched_dev'].clip(lower=0)).astype(np.float32)
    df['imp_minus_sched'] = (df['implied_taxi_aobt']
                             - df['sched_dev']).astype(np.float32)
    df['imp_is_odd'] = ((df['implied_taxi_aobt'].isna())
                        | (df['implied_taxi_aobt'] > 7200)
                        | (df['implied_taxi_aobt'] < 0)).astype(np.int8)
    if with_labels:
        df['is_extreme'] = (df[TARGET] > TRAIN_CAP).astype(np.int8)
    return df


def load_train():
    df = pd.read_parquet(os.path.join(DATA_FOLDER, "dep_qw_train.parquet"))
    # date-rollover repair on mvt_i, so every derived column inherits it
    y = df[TARGET]
    roll = (y > 43200) & ((y % 86400).between(60, 7200))
    days = (df.loc[roll, TARGET] // 86400).astype('int64')
    df.loc[roll, 'mvt_i'] = df.loc[roll, 'mvt_i'] - 86400 * days
    df.loc[roll, TARGET] = df.loc[roll, TARGET] % 86400
    df = df[df[TARGET] > 0].copy()
    df = derive(df, with_labels=True)
    print(f"  train rows {len(df):,}   rollover repaired {int(roll.sum())}   "
          f"extreme {int(df.is_extreme.sum())}")
    return df


def fit_encoders(fit_df):
    g_p10 = fit_df[TARGET].quantile(0.10)
    g_mean = fit_df[TARGET].mean()
    enc = {'g_p10': g_p10, 'g_mean': g_mean}
    MIN_N = 30
    for key, name in [('apt_stand', 'stand_unimp'),
                      ('rwy_stand', 'rwy_stand_unimp'),
                      ('apt_rwy', 'apt_rwy_unimp')]:
        grp = fit_df.groupby(key, observed=True)[TARGET]
        enc[name] = grp.quantile(0.10)[grp.size() >= MIN_N].to_dict()
    grp = fit_df.groupby('rwy_stand', observed=True)[TARGET]
    enc['rwy_stand_delay'] = ((grp.mean() - grp.quantile(0.10))
                              [grp.size() >= MIN_N]).to_dict()
    sm = 20
    for key, name in [('apt_stand', 'stand_te'),
                      ('AIRCRAFT_OPERATOR_flt', 'operator_te')]:
        s = fit_df.groupby(key, observed=True)[TARGET].agg(['mean', 'count'])
        enc[name] = ((s['mean'] * s['count'] + g_mean * sm)
                     / (s['count'] + sm)).to_dict()
    return enc


def apply_encoders(df, enc):
    for name, key, fb in [
            ('stand_unimp', 'apt_stand', enc['g_p10']),
            ('rwy_stand_unimp', 'rwy_stand', enc['g_p10']),
            ('apt_rwy_unimp', 'apt_rwy', enc['g_p10']),
            ('rwy_stand_delay', 'rwy_stand', 0.0),
            ('stand_te', 'apt_stand', enc['g_mean']),
            ('operator_te', 'AIRCRAFT_OPERATOR_flt', enc['g_mean'])]:
        df[name] = df[key].map(enc[name]).fillna(fb).astype(np.float32)
    return df


def as_cats(df, cats_map, cols):
    for c in cols:
        df[c] = pd.Categorical(df[c].where(df[c].isin(cats_map[c])),
                               categories=cats_map[c])
    return df


if __name__ == "__main__":
    tr = load_train()
    feats = [c for c in (CAT_COLS + TIME_COLS + AOBT_COLS + ENC_COLS
                         + QUEUE_COLS + WX_COLS + DEST_COLS + SCHED_COLS)
             if c in tr.columns]
    cats = [c for c in CAT_COLS + DEST_COLS if c in tr.columns]
    dcols = [c for c in DET_COLS if c in tr.columns] + ['apt']
    print(f"  {len(feats)} model features, {len(dcols)} detector features")

    enc = fit_encoders(tr[tr[TARGET] <= TRAIN_CAP])
    tr = apply_encoders(tr, enc)
    for c in cats:
        tr[c] = tr[c].astype('category')
    cats_map = {c: tr[c].cat.categories for c in cats}

    # ---------- base regressor: all 12 months, capped target ----------
    base_rows = tr[tr[TARGET] <= TRAIN_CAP]
    rng = np.random.default_rng(42)
    hold_mask = rng.random(len(base_rows)) < HOLDOUT_FRAC
    b_tr, b_va = base_rows[~hold_mask], base_rows[hold_mask]
    print(f"\n  base regressor: fit {len(b_tr):,}  early-stop slice {len(b_va):,}")
    ds = lgb.Dataset(b_tr[feats], b_tr[TARGET], categorical_feature=cats)
    vs = lgb.Dataset(b_va[feats], b_va[TARGET], categorical_feature=cats,
                     reference=ds)
    tuned = lgb.train(PARAMS, ds, MAX_ROUNDS, valid_sets=[vs],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(0)])
    best = tuned.best_iteration
    print(f"    best_iteration {best}; refitting on all {len(base_rows):,} rows")
    del ds, vs, tuned, b_tr, b_va
    gc.collect()
    ds = lgb.Dataset(base_rows[feats], base_rows[TARGET],
                     categorical_feature=cats)
    model = lgb.train(PARAMS, ds, best)
    model.save_model(os.path.join(DATA_FOLDER, "final_base.txt"))
    del ds, base_rows
    gc.collect()

    # ---------- detector: candidate pool, UNCAPPED labels ----------
    det_pool = tr[tr['sched_dev'] > POOL_MIN]
    print(f"\n  detector: pool {len(det_pool):,} rows, "
          f"{int(det_pool.is_extreme.sum())} positives "
          f"({det_pool.is_extreme.mean()*100:.2f}%)")
    dds = lgb.Dataset(det_pool[dcols], det_pool['is_extreme'],
                      categorical_feature=['apt'])
    det = lgb.train(DET_PARAMS, dds, DET_ROUNDS)
    det.save_model(os.path.join(DATA_FOLDER, "final_detector.txt"))
    np.save(os.path.join(DATA_FOLDER, "final_enc.npy"), enc, allow_pickle=True)
    np.save(os.path.join(DATA_FOLDER, "final_cats.npy"), cats_map,
            allow_pickle=True)
    del dds, det_pool, tr
    gc.collect()
    print("  saved final_base.txt / final_detector.txt / final_enc.npy / final_cats.npy")

    # ---------- score the ranking set ----------
    rank = pd.read_parquet(os.path.join(DATA_FOLDER, "dep_qw_rank.parquet"))
    rank = derive(rank, with_labels=False)
    rank = apply_encoders(rank, enc)
    rank = as_cats(rank, cats_map, cats)
    rank['apt'] = rank['apt'].astype(str)      # detector needs the raw value
    rank = as_cats(rank, {'apt': cats_map['apt']}, ['apt']) \
        if 'apt' in cats_map else rank

    miss = [f for f in model.feature_name() if f not in rank.columns]
    if miss:
        raise SystemExit(f"ranking parquet missing: {miss}")

    base_pred = np.clip(model.predict(rank[model.feature_name()]),
                        PRED_FLOOR, PRED_CEIL)
    p = det.predict(rank[dcols])
    pool = rank['sched_dev'].to_numpy() > POOL_MIN
    p = np.where(pool, p, 0.0)
    sd = rank['sched_dev'].to_numpy()

    if BLEND == 'soft':
        pred = np.where(pool, p * sd + (1 - p) * base_pred, base_pred)
    else:
        pred = np.where(p > HARD_T, sd, base_pred)
    pred = np.clip(pred, PRED_FLOOR, PRED_CEIL)

    n_flag = int((p > HARD_T).sum())
    print(f"\n  ranking rows {len(rank):,}")
    print(f"    candidate pool      : {int(pool.sum()):,}")
    print(f"    p > {HARD_T}            : {n_flag:,}")
    print(f"    predictions above 10800: {int((pred > 10800).sum()):,}  "
          f"(v3 had 0 by construction)")
    print(f"    range {pred.min():.0f} - {pred.max():.0f}, mean {pred.mean():.0f}")

    # ---------- write with the row-order gate ----------
    tp = os.path.join(DATA_FOLDER, TEMPLATE)
    tmpl = pd.read_parquet(tp) if tp.endswith('.parquet') else pd.read_csv(tp)
    sub = pd.DataFrame({'MVT_ID_mvt': rank['MVT_ID_mvt'].values,
                        TARGET: pred.astype(np.float64)})
    sub['MVT_ID_mvt'] = sub['MVT_ID_mvt'].astype(tmpl['MVT_ID_mvt'].dtype)
    ordered = tmpl[['MVT_ID_mvt']].merge(sub, on='MVT_ID_mvt', how='left')

    same = (ordered['MVT_ID_mvt'].values == tmpl['MVT_ID_mvt'].values)
    print(f"\n  --- submission gate ---")
    print(f"    rows {len(ordered):,} / {len(tmpl):,}")
    print(f"    ID order element-wise: {same.sum():,}/{len(tmpl):,} "
          f"({same.mean()*100:.4f}%)")
    print(f"    nulls: {int(ordered[TARGET].isna().sum())}")
    assert len(ordered) == len(tmpl), "row count changed"
    assert same.all(), "ROW ORDER WRONG - refusing to write"
    assert ordered[TARGET].notna().all(), "nulls present"
    assert np.isfinite(ordered[TARGET]).all(), "non-finite predictions"
    assert (ordered[TARGET] >= 0).all(), "negative predictions"

    out = os.path.join(DATA_FOLDER, f"{TEAM}_v{VERSION}.parquet")
    ordered.to_parquet(out, index=False)
    print(f"\n  wrote {out}")
    chk = pd.read_parquet(out)
    print(f"  re-read order check: "
          f"{(chk['MVT_ID_mvt'].values == tmpl['MVT_ID_mvt'].values).all()}")
    print(chk.head(3).to_string(index=False))
