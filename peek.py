import polars as pl

df = pl.read_parquet("data/raw/training_2025-01-01_2025-02-01.parquet")

dep = df.filter(pl.col("PHASE_mvt") == "DEP")
print(f"departures: {len(dep):,} of {len(df):,}")
print()

print(dep.select([
    "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
    "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
]).head(10))
print()

print(dep.select("TAXITIME_SEC_mvt").describe())
