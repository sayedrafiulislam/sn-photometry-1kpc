"""
check_guard_failures.py  (diagnostics, read-only)

Step 15 logs 35 photometry losses between 221 and 186 objects:
  11  non-positive flux at 1 kpc
  24  no du Pont flux at 1 kpc (every row removed by a step-13 guard)

For the 24, report WHICH guard failed and why, compare their redshifts
with the other groups, and name the objects absent from step 14's output
(the 218-vs-221 question). Writes nothing. Place in scripts/diagnostics/.

Guards (step 13):
  annulus_ok  : annulus_frac_on_chip >= 0.80
                and annulus_frac_expected_pix >= 0.30 and bkg finite
  aperture_ok : aperture_frac_on_chip >= 0.98 and aperture_n_nonfinite == 0
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
COG = REPO / "results" / "phase4_aperture" / "curve_of_growth_ann20-30.csv"
S14 = REPO / "results" / "phase4_aperture" / "local_colour_vs_radius_ann20-30.csv"
EXCL = REPO / "results" / "phase5_calibration" / "calibrated_colour_1kpc_exclusions.csv"
R_FID = 1.0

for p in (COG, S14, EXCL):
    if not p.exists():
        raise SystemExit(f"STOP: missing input {p}")

cog = pd.read_csv(COG)
need = ["object", "telescope", "filter", "z", "radius_kpc", "radius_pix",
        "annulus_frac_on_chip", "annulus_frac_expected_pix", "annulus_ok",
        "aperture_frac_on_chip", "aperture_n_nonfinite", "aperture_ok",
        "bkg_per_pixel"]
missing = [c for c in need if c not in cog.columns]
if missing:
    raise SystemExit(f"STOP: columns missing from {COG.name}: {missing}")

excl = pd.read_csv(EXCL)
phot = excl[excl["stage"] == "photometry"]
guard_objs = set(phot.loc[phot["reason"] == "no du Pont flux at the fiducial radius", "object"])
nonpos_objs = set(phot.loc[phot["reason"] == "non-positive flux at the fiducial radius", "object"])
print(f"Guard failures in log: {len(guard_objs)}   non-positive: {len(nonpos_objs)}")

dup = cog[cog["telescope"].str.lower().str.contains("dup")]
fid = dup[np.isclose(dup["radius_kpc"], R_FID)].copy()

rows = fid[fid["object"].isin(guard_objs)].copy()
absent = sorted(guard_objs - set(rows["object"]))
print(f"Of the {len(guard_objs)}, with NO du Pont rows in step 13 at all: {absent}")


def why(rec):
    r = []
    if rec["annulus_frac_on_chip"] < 0.80:
        r.append("annulus off chip")
    if rec["annulus_frac_expected_pix"] < 0.30:
        r.append("annulus few finite px")
    if not np.isfinite(rec["bkg_per_pixel"]):
        r.append("bkg non-finite")
    if rec["aperture_frac_on_chip"] < 0.98:
        r.append("aperture off chip")
    if rec["aperture_n_nonfinite"] > 0:
        r.append(f"aperture NaN px={int(rec['aperture_n_nonfinite'])}")
    return "; ".join(r) if r else "none (?)"


rows["why"] = [why(rec) for _, rec in rows.iterrows()]
show = ["object", "filter", "z", "radius_pix", "annulus_frac_on_chip",
        "annulus_frac_expected_pix", "aperture_frac_on_chip",
        "aperture_n_nonfinite", "why"]
print("\nPer object and band at 1 kpc:")
print(rows.sort_values(["object", "filter"])[show].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))

print("\nFailure reasons (object-band rows):")
print(rows["why"].value_counts().to_string())

z_obj = dup.groupby("object")["z"].first()
kept = set(z_obj.index) - guard_objs - nonpos_objs
print("\nRedshift by group (median, range, n):")
for name, s in (("guard failures", guard_objs), ("non-positive", nonpos_objs),
                ("rest (kept)", kept)):
    zz = z_obj[z_obj.index.isin(s)]
    if len(zz):
        print(f"  {name:15s} {zz.median():.4f}  [{zz.min():.4f}, {zz.max():.4f}]  n={len(zz)}")

s14_objs = set(pd.read_csv(S14)["object"])
both = set(dup[dup["filter"] == "B"]["object"]) & set(dup[dup["filter"] == "V"]["object"])
print(f"\ndu Pont B+V objects in step 13: {len(both)}   in step 14 output: {len(s14_objs)}")
print(f"In step 13 pair but absent from step 14: {sorted(both - s14_objs)}")