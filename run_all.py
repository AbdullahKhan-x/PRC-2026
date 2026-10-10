"""
Run the whole logical-ladder pipeline, in order, on any OS (Linux, macOS, Windows).

  python run_all.py --list                 show the steps and whether their outputs exist
  python run_all.py                        run every step whose outputs are missing (stops at the first failure)
  python run_all.py --only members         run one step (outputs are rebuilt even if present)
  python run_all.py --from context         run from a step onwards

Memory: the member-training steps need ~6 GB RAM (LightGBM on 2.08 M rows x ~120 features). On a smaller machine run
them on a larger machine (e.g. a Kaggle notebook). LightGBM results depend slightly on the thread count, so a model retrained on
another machine is not byte-identical to ours; the trained model files we submitted with can be shared separately.
"""
import os, sys, subprocess, importlib.util, argparse, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, 'DATA_ANALYTICS')
C = os.path.join(ROOT, 'candidates')
PY = sys.executable
os.makedirs(C, exist_ok=True)


def p(*parts):
    return os.path.join(ROOT, *parts)


def run(script, *args, env=None):
    e = {**os.environ, 'PRC_ROOT': ROOT, **(env or {})}
    path = os.path.join(A, script) if os.path.exists(os.path.join(A, script)) else p(script)
    cmd = [PY, '-u', path] + list(args)
    print('  $', ' '.join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd), {k: v for k, v in (env or {}).items()}, flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=e)
    if r.returncode:
        sys.exit(f'step failed: {script} (exit {r.returncode})')


def merge(out, *srcs):
    import pandas as pd
    for s in ('train', 'rank'):
        df = pd.read_parquet(p(f'{srcs[0]}_{s}.parquet'))
        for more in srcs[1:]:
            df = df.merge(pd.read_parquet(p(f'{more}_{s}.parquet')), on='MVT_ID_mvt', how='left')
        df.to_parquet(p(f'{out}_{s}.parquet'), index=False)
        print(f'  wrote {out}_{s}.parquet {df.shape}', flush=True)


def metar():
    spec = importlib.util.spec_from_file_location('w15', os.path.join(A, '15_weather.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.download_all()


FULL = {'LR': '0.04', 'ROUNDS': '1800', 'TE_EXTRA': '1'}
MEMBERS = {   # tag: extra env for 44_train_full.py   (v116 recipe; see docs and 53b_make_blend4.py)
    'ctx2id': {'EXTRA': 'ctxall_train.parquet', 'EXTRA_RANK': 'ctxall_rank.parquet'},
    'ctx2rep': {'EXTRA': 'ctxall_train.parquet', 'EXTRA_RANK': 'ctxall_rank.parquet', 'DROP_BASE': 'AIRCRAFT_OPERATOR_flt,ADES_flt'},
    'ctx2res': {'EXTRA': 'ctxall_train.parquet', 'EXTRA_RANK': 'ctxall_rank.parquet', 'RESID': '1'},
    'ctx3': {'EXTRA': 'ctx3_train.parquet', 'EXTRA_RANK': 'ctx3_rank.parquet', 'TE_EXTRA2': '1'},
    'kae': {'EXTRA': 'ctxae_train.parquet', 'EXTRA_RANK': 'ctxae_rank.parquet'},
    'ctx2ida': {'EXTRA': 'ctxall_train.parquet', 'EXTRA_RANK': 'ctxall_rank.parquet', 'APT_W': '0.5'},
}


def members():
    for tag, env in MEMBERS.items():
        if not os.path.exists(os.path.join(C, f'parts_{tag}.parquet')):
            run('44_train_full.py', env={**FULL, **env, 'TAG': tag})


def adsb():
    if not glob.glob(p('data', 'adsb', '**', 'pts_*.parquet'), recursive=True):
        sys.exit('no ADS-B point files in data/adsb/ - extract them on Kaggle first (kaggle/make_adsb_kernels.py)')
    run('62_adsb_features.py')


STEPS = [
    ('download', [p('data', 'raw', 'ranking.parquet'), p('data', 'raw', 'final_ranking.parquet')],
     lambda: run('download.py')),
    ('external', [p('data', 'ext', 'atfm_slot_adherence_2026.csv'), p('data', 'ext', 'opdi', 'flight_list_202607.parquet')],
     lambda: run('download_external.py')),
    ('metar', [p('metar_cache')], metar),
    ('base', [p('dep_qw_train.parquet'), p('dep_qw_rank.parquet')],
     lambda: (run('10_build_base_features.py', 'train'), run('10_build_base_features.py', 'rank'))),
    ('v35', [p('final_detector.txt'), p('logical-ladder_v35.parquet'), os.path.join(C, 'v35_parts.parquet')],
     lambda: (run('24_v35.py'), run('39_repro_v35_parts.py', C))),
    ('context', [p('ctxall_train.parquet'), p('ctx3_train.parquet'), p('ext_train.parquet')],
     lambda: (run('42_context_features.py'), run('42b_context_features2.py'), merge('ctxall', 'ctx', 'ctx2'),
              run('42f_labelfree.py'), merge('ctx3', 'ctxall', 'lf'), run('42d_external.py'))),
    ('adsb', [p('adsb_train.parquet'), p('ctxae_train.parquet')],
     lambda: (adsb(), merge('ctxae', 'ctxall', 'adsb', 'ext'))),
    ('members', [os.path.join(C, f'parts_{t}.parquet') for t in MEMBERS], members),
    ('blend', [os.path.join(C, 'logical-ladder_v116.parquet')],
     lambda: (run('53b_make_blend4.py', 'v116', '0.25', '0.3', '0.5', 'kae'),
              run('40_postrules.py', 'ACD', '116', os.path.join(C, 'parts_v116.parquet')))),
    ('final', [os.path.join(C, 'FINAL_v116_4month.parquet')],
     lambda: run('55_final_v116.py', p('data', 'raw', 'final_ranking.parquet'), p('data', 'raw', 'final_submitting.parquet'),
                 os.path.join(C, 'FINAL_v116_4month.parquet'))),
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true'); ap.add_argument('--only'); ap.add_argument('--from', dest='start')
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for n, outs, _ in STEPS:
            print(f"{n:9s} {'done   ' if all(os.path.exists(o) for o in outs) else 'pending'}  " + ', '.join(os.path.relpath(o, ROOT) for o in outs))
        sys.exit()
    todo = [s for s in STEPS if (a.only is None or s[0] == a.only) and (a.start is None or names.index(s[0]) >= names.index(a.start))]
    for n, outs, fn in todo:
        if a.only is None and all(os.path.exists(o) for o in outs):
            print(f'[skip] {n} (outputs exist)'); continue
        print(f'[run ] {n}', flush=True); fn()
    print('all requested steps done')
