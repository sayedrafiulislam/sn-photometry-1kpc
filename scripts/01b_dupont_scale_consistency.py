"""
01b_dupont_scale_consistency.py

Re-run the plate-scale consistency check on du Pont frames only.

Script 01 groups frames by object across ALL telescopes and reports 92
mismatches. Inspection shows every one has the same signature: B at 0.23,
V at "0.23,0.43". The 0.43 frames are exclusively Swope (Q1 confirms no
du Pont frame sits at 0.43), so those objects have a consistent du Pont
B/V pair plus a Swope V that never enters a colour -- the analysis pairs
within one telescope.

This script restricts to du Pont and asks the question that actually
matters for the science: within du Pont alone, do B and V share a scale?

It also isolates the 12 frames at 0.159 arcsec/pixel, three of which are
du Pont B and four du Pont V. Those are the only frames that could produce
a genuine same-telescope mismatch.

READS ONLY.
"""

from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "results" / "header_summary_full.csv"

df = pd.read_csv(SUMMARY)
df = df[df["pixscale_arcsec"].notna()].copy()
df["scale"] = df["pixscale_arcsec"].astype(float).round(4)
df["object"] = (df["file"].apply(lambda p: Path(p).stem.split("_")[0]))

dup = df[(df["telescope_from_name"] == "du Pont") &
         (df["filter_from_name"].isin(["B", "V"]))]

print("=" * 70)
print("DU PONT ONLY -- do B and V share a plate scale?")
print("=" * 70)
print(f"  du Pont B/V frames : {len(dup)}")
print(f"  distinct objects   : {dup['object'].nunique()}\n")

rows = []
for obj, g in dup.groupby("object"):
    sB = sorted(g[g["filter_from_name"] == "B"]["scale"].unique())
    sV = sorted(g[g["filter_from_name"] == "V"]["scale"].unique())
    if not sB or not sV:
        status = "missing one filter"
    elif set(sB) == set(sV) and len(sB) == 1:
        status = "consistent"
    else:
        status = "MISMATCH"
    rows.append({"object": obj, "scales_B": sB, "scales_V": sV,
                 "status": status})

res = pd.DataFrame(rows)
for status, n in res["status"].value_counts().items():
    print(f"  {status:>20s}: {n}")

bad = res[res["status"] == "MISMATCH"]
print(f"\n  Genuine same-telescope mismatches: {len(bad)}")
if len(bad):
    print(bad.to_string(index=False))
else:
    print("  None. Every du Pont B/V pair shares one plate scale.")

print("\n" + "=" * 70)
print("The 0.159 arcsec/pixel frames")
print("=" * 70)
odd = df[df["scale"] == 0.159]
print(odd[["object", "telescope_from_name",
           "filter_from_name", "scale"]].to_string(index=False))

odd_objs = set(odd["object"])
print(f"\n  objects involved: {len(odd_objs)}")
print(f"  their status    :")
print(res[res["object"].isin(odd_objs)].to_string(index=False))
print("=" * 70)