"""
Push the ADS-B extraction notebooks to Kaggle (any OS; needs `pip install kaggle` and a Kaggle API token).

  python kaggle/make_adsb_kernels.py A B C D            periods defined in adsb_kaggle.py (A/B = 2026, C..I = 2025)
  python kaggle/make_adsb_kernels.py --days 2025-06-01 2025-06-02      fix-up run for explicit days

Each notebook downloads adsb.lol daily archives (ODbL 1.0) at datacenter speed and keeps only ground / low points
near the 10 airports (~15 MB per day). Collect the outputs with:
  kaggle kernels output <user>/prc-adsb-<p> -p data/adsb/<p>      then unzip pts_<P>.zip there
"""
import os, sys, json, re, subprocess

K = os.path.dirname(os.path.abspath(__file__))
USER = os.environ.get('KAGGLE_USERNAME', 'mohammadabdullahnazi')
src = open(os.path.join(K, 'adsb_kaggle.py')).read()
args = sys.argv[1:]
jobs = []
if args and args[0] == '--days':
    jobs.append(('fix', src.replace("DAYS = []", f"DAYS = {args[1:]!r}", 1).replace("PERIOD = 'A'", "PERIOD = 'F'", 1)))
else:
    for p in args or ['A', 'B', 'C', 'D']:
        jobs.append((p.lower(), re.sub(r"^PERIOD = 'A'", f"PERIOD = '{p}'", src, count=1, flags=re.M)))
for name, code in jobs:
    d = os.path.join(K, f'adsb_{name}')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f'adsb_{name}.py'), 'w').write(code)
    json.dump({'id': f'{USER}/prc-adsb-{name}', 'title': f'prc-adsb-{name}', 'code_file': f'adsb_{name}.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True,
               'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    subprocess.run(['kaggle', 'kernels', 'push', '-p', d])
