"""
03c_compare_redshifts.py

Compare this run's NED redshifts against the earlier run.

NED is a live service. Between runs, objects can gain a redshift, lose one,
or have a published value revised. A revised redshift is the consequential
case: it changes the angular diameter distance, hence the aperture size in
pixels, hence the photometry. This must be checked rather than assumed.

READS ONLY.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

new = pd.read_csv(REPO / "results" / "ned_progress.csv")
old = pd.read_csv(REPO / "inputs" / "sn_catalog_final.csv")

old_z = (old.drop_duplicates("object")[["object", "redshift"]]
            .rename(columns={"redshift": "z_old"}))
new_z = new.rename(columns={"redshift": "z_new"})

m = new_z.merge(old_z, on="object", how="outer")

gained = m[m["z_old"].isna() & m["z_new"].notna()]
lost = m[m["z_old"].notna() & m["z_new"].isna()]
both = m[m["z_old"].notna() & m["z_new"].notna()].copy()

print("=" * 68)
print("REDSHIFT COMPARISON: this run vs earlier run")
print("=" * 68)
print(f"  resolved now      : {int(m['z_new'].notna().sum())}")
print(f"  resolved earlier  : {int(m['z_old'].notna().sum())}")
print(f"  in both           : {len(both)}\n")

print(f"  Newly resolved ({len(gained)}):")
print("    " + (", ".join(sorted(gained['object'])) if len(gained) else "none"))

print(f"\n  Lost since earlier run ({len(lost)}):")
print("    " + (", ".join(sorted(lost['object'])) if len(lost) else "none"))

both["diff"] = both["z_new"] - both["z_old"]
changed = both[~np.isclose(both["z_new"], both["z_old"], rtol=1e-6, atol=1e-8)]

print(f"\n  Values changed ({len(changed)}):")
if len(changed):
    for _, r in changed.sort_values("diff", key=abs, ascending=False).iterrows():
        pct = 100 * r["diff"] / r["z_old"]
        print(f"    {r['object']:<12s} {r['z_old']:.5f} -> {r['z_new']:.5f}"
              f"   ({pct:+.2f}%)")
    print("\n    A changed redshift changes the angular diameter distance and")
    print("    therefore the physical aperture size. These need attention.")
else:
    print("    none -- every shared object has an identical redshift.")

print("=" * 68)