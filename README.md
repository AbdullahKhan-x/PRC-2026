# PRC Data Challenge 2026 — Team logical-ladder

Predicting taxi-out time at 10 European airports.
Deadline: 11 October 2026.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Data

Not in this repo (300 MB, and access is credentialed).

Put `credentials.json` in the project root, then:

```bash
python download.py
```

Files land in `data/raw/` — 12 monthly training files, plus
`ranking.parquet` and `submitting.parquet`.

## Files

- `download.py` — pulls the dataset from the challenge bucket
- `peek.py` — prints schema and summary stats

## License

GPLv3, as required for prize eligibility.
