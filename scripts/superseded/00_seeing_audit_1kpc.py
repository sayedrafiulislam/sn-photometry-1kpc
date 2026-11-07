"""
00_seeing_audit_1kpc.py

Does a 1.0 kpc aperture resolve anything?

At the 5.0 kpc fiducial radius the seeing floor was never binding, so no
downstream script filtered on it. At 1.0 kpc it may be binding for a
substantial and NON-RANDOM part of the sample: the floor scales with angular
diameter distance, so it removes high-redshift objects preferentially.

This script quantifies that before any photometry is re-extracted.
It READS ONLY. It writes nothing and changes nothing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INPUTS = REPO / "inputs"

RADIUS_KPC = 1.0
TELESCOPE = "dup"

cog = pd.read_csv(INPUTS / "curve_of_growth_ann20-30.csv")

sel = cog[np.isclose(cog["radius_kpc"], RADIUS_KPC) &
          (cog["telescope"] == TELESCOPE)].copy()

if sel.empty:
    raise SystemExit(f"No {TELESCOPE} rows at {RADIUS_KPC} kpc.")

print("=" * 74)
print(f"SEEING AUDIT AT {RADIUS_KPC:g} kpc  ({TELESCOPE} only)")
print("=" * 74)
print(f"  rows at this radius        : {len(sel)}")
print(f"  distinct objects           : {sel['object'].nunique()}")

safe = sel[sel["seeing_safe"]]
unsafe = sel[~sel["seeing_safe"]]

print(f"  rows passing seeing floor  : {len(safe)}  "
      f"({100 * len(safe) / len(sel):.0f}%)")

# An object needs BOTH B and V above the floor to give a colour.
if len(safe):
    both = (safe.groupby("object")["filter"].apply(set)
                .apply(lambda s: {"B", "V"}.issubset(s)))
    print(f"  objects with safe B AND V  : {int(both.sum())}")
else:
    print("  objects with safe B AND V  : 0")

print("\n  redshift of resolved vs unresolved:")
for label, frame in (("resolved  ", safe), ("unresolved", unsafe)):
    if len(frame):
        z = frame.drop_duplicates("object")["z"]
        print(f"    {label} n={z.size:3d}  median z={z.median():.4f}  "
              f"range {z.min():.4f}-{z.max():.4f}")
    else:
        print(f"    {label} n=  0")

if len(unsafe) and len(safe):
    print(f"\n  lowest z failing the floor : "
          f"{unsafe.drop_duplicates('object')['z'].min():.4f}")
    print(f"  highest z passing it       : "
          f"{safe.drop_duplicates('object')['z'].max():.4f}")

print("=" * 74)