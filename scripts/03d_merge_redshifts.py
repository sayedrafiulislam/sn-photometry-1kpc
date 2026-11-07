"""
03d_merge_redshifts.py

Write the recovered NED redshifts back into the catalogue.

Step 02 was run on an unstable connection and left only 101 redshifts in
sn_catalog_v2.csv. Script 03b recovered the rest into ned_progress.csv.
Step 04 reads the catalogue, so the recovered values must be merged in
first or the redshift cut would discard 166 objects for a network fault
rather than a scientific reason.

The original file is backed up rather than overwritten, per the project's
never-overwrite convention.
"""

from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "results" / "sn_catalog_v2.csv"
PROGRESS = REPO / "results" / "ned_progress.csv"

cat = pd.read_csv(CATALOG)
prog = pd.read_csv(PROGRESS)

before = int(cat.drop_duplicates("object")["redshift"].notna().sum())

zmap = dict(zip(prog["object"], prog["redshift"]))
filled = cat["object"].map(zmap)

# Prefer the existing value where present; fill the gaps from the retry run.
cat["redshift"] = cat["redshift"].where(cat["redshift"].notna(), filled)

after = int(cat.drop_duplicates("object")["redshift"].notna().sum())

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(CATALOG, CATALOG.with_suffix(f".bak_{stamp}.csv"))
cat.to_csv(CATALOG, index=False)

print("=" * 60)
print(f"  objects with redshift before : {before}")
print(f"  objects with redshift after  : {after}")
print(f"  recovered                    : {after - before}")
print(f"\n  backup: {CATALOG.name}.bak_{stamp}.csv")
print("=" * 60)