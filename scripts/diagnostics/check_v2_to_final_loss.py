"""
check_v2_to_final_loss.py

Audit script, not production.

sn_catalog_v2.csv contains 282 objects with a du Pont B/V pair, which
matches the documented 338 -> 282 cut. sn_catalog_final.csv contains
only 222. The redshift cut accounts for 16 objects, not 60, so roughly
44 objects lost their pair status for some other reason.

The catalogue holds only 223 B frames in total, so the loss is
specifically of B rows. This script locates which rows were dropped and
tests whether their disappearance correlates with a null
ned_resolved_name -- which would indicate the redshift merge at step 04
joined on a field that is null for two thirds of the rows.

Reads only.
"""

from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

v2 = pd.read_csv(RESULTS / "sn_catalog_v2.csv")
final = pd.read_csv(RESULTS / "sn_catalog_final.csv")

print(f"v2    : {len(v2):4d} rows, {v2['object'].nunique()} objects")
print(f"final : {len(final):4d} rows, {final['object'].nunique()} objects\n")

print("Frames by filter and telescope")
print("-" * 46)
for name, df in (("v2", v2), ("final", final)):
    counts = df.groupby(["telescope", "filter"]).size()
    print(f"\n{name}:")
    print(counts.to_string())

# Which rows in v2 are absent from final?
key = ["object", "file"]
merged = v2.merge(final[key].assign(_in_final=True), on=key, how="left")
lost = merged[merged["_in_final"].isna()].copy()

print("\n" + "=" * 60)
print(f"ROWS IN v2 BUT NOT IN final: {len(lost)}")
print("=" * 60)
print(lost.groupby(["telescope", "filter"]).size().to_string())

# Restrict to objects that DID survive into final -- these are the
# rows lost without their object being cut.
survivors = set(final["object"])
lost_from_survivors = lost[lost["object"].isin(survivors)]

print(f"\nOf those, belonging to objects still present in final: "
      f"{len(lost_from_survivors)}")
if len(lost_from_survivors):
    print(lost_from_survivors.groupby(["telescope", "filter"]).size().to_string())

    print("\nWas ned_resolved_name null on the lost rows?")
    null_lost = lost_from_survivors["ned_resolved_name"].isna().sum()
    print(f"  null : {null_lost} / {len(lost_from_survivors)}")

    kept = v2[v2.set_index(key).index.isin(final.set_index(key).index)]
    print(f"  for comparison, null on kept rows: "
          f"{kept['ned_resolved_name'].isna().sum()} / {len(kept)}")

    print("\nExample lost rows:")
    cols = ["object", "file", "filter", "telescope", "redshift",
            "ned_resolved_name"]
    print(lost_from_survivors[cols].head(15).to_string(index=False))