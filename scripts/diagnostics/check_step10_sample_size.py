"""
check_step10_sample_size.py

Audit script, not production. Step 10 reported 100 objects where the
sample cascade says 266. This locates where the other 166 went.

Three candidate explanations, tested in order:
  (a) sn_catalog_final.csv itself holds fewer than 266 objects
  (b) it holds 266 but ned_resolved_name is null for most of them,
      so the dropna() in step 10 silently removed them
  (c) it holds a non-random subset -- e.g. the alphabetically first
      100, which would mean a --limit or .head() upstream

(c) is checked explicitly because an alphabetical --limit is the known
historical failure mode in this repository.
"""

from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

FINAL = RESULTS / "sn_catalog_final.csv"
V2 = RESULTS / "sn_catalog_v2.csv"
EXCLUDED = RESULTS / "excluded_objects_log.csv"
PROGRESS = RESULTS / "sn_coordinates_progress.csv"

print("=" * 70)
print("FILES IN results\\")
print("=" * 70)
for p in sorted(RESULTS.glob("*.csv")):
    print(f"  {p.name:<52} {p.stat().st_size:>10,} bytes")

def describe(path, label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  rows            : {len(df)}")
    print(f"  columns         : {list(df.columns)}")
    if "object" in df.columns:
        print(f"  unique objects  : {df['object'].nunique()}")
    if "ned_resolved_name" in df.columns:
        n_null = df["ned_resolved_name"].isna().sum()
        print(f"  ned_resolved_name null : {n_null}")
        print(f"  ned_resolved_name set  : {len(df) - n_null}")
    if "redshift" in df.columns:
        n_null = df["redshift"].isna().sum()
        print(f"  redshift null   : {n_null}")
        print(f"  redshift range  : {df['redshift'].min():.5f} "
              f"to {df['redshift'].max():.5f}")
    return df

final = describe(FINAL, "sn_catalog_final.csv  (input to step 10)")
v2 = describe(V2, "sn_catalog_v2.csv  (upstream, before redshift cut)")
excl = describe(EXCLUDED, "excluded_objects_log.csv")
prog = describe(PROGRESS, "sn_coordinates_progress.csv  (step 10 output)")

# ------------------------------------------------------------------
# Test (c): is the surviving set alphabetically truncated?
# ------------------------------------------------------------------
if final is not None and v2 is not None and "object" in final.columns \
        and "object" in v2.columns:
    print("\n" + "=" * 70)
    print("ALPHABETICAL TRUNCATION TEST")
    print("=" * 70)
    kept = sorted(final["object"].dropna().unique())
    universe = sorted(v2["object"].dropna().unique())
    head = universe[:len(kept)]
    if kept == head:
        print("  *** POSITIVE: the kept objects are exactly the "
              "alphabetically first")
        print(f"  {len(kept)} of {len(universe)}. This is a --limit or "
              f".head(), not science.")
    else:
        print("  Negative: kept set is not an alphabetical prefix.")
        print(f"  first 5 kept     : {kept[:5]}")
        print(f"  last 5 kept      : {kept[-5:]}")
        print(f"  last 5 in v2     : {universe[-5:]}")
        missing = [o for o in universe if o not in set(kept)]
        print(f"  missing from final: {len(missing)}")
        print(f"  e.g. {missing[:10]}")

print("\n" + "=" * 70)
print("EXPECTED FROM HANDOVER: 266 objects after redshift cut "
      "and ASAS14ad exclusion.")
print("=" * 70)