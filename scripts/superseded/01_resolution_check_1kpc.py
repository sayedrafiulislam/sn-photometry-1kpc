"""
01_resolution_check_1kpc.py

How well is a 1.0 kpc aperture actually resolved?

Script 00 applied the inherited pass/fail seeing floor, r >= sigma_min, where
sigma = FWHM / 2.355. That criterion permits an aperture whose DIAMETER is
0.85 x FWHM -- narrower than the seeing disc itself. It is a lower bound on
sanity, not a statement that the region is resolved.

This script reports the distribution behind the pass/fail, in two forms:

    r / sigma      how many Gaussian widths the aperture spans
    2r / FWHM      aperture diameter in units of the seeing disc

The second is the honest one for the paper. 2r/FWHM < 1 means the aperture is
smaller than the PSF and the measurement is a blurred average over a region
larger than the nominal radius.

READS ONLY. Writes nothing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INPUTS = REPO / "inputs"

RADIUS_KPC = 1.0
TELESCOPE = "dup"
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

floor = pd.read_csv(INPUTS / "aperture_floor_per_object_corrected.csv")

print("Columns available in the aperture-floor file:")
print("   ", list(floor.columns))
print()

if "sigma_min_kpc" not in floor.columns:
    raise SystemExit("No sigma_min_kpc column -- paste the column list above to me.")

sub = floor.copy()
if "telescope" in sub.columns:
    sub = sub[sub["telescope"] == TELESCOPE]
if "filter" in sub.columns:
    sub = sub[sub["filter"].isin(["B", "V"])]

sub = sub[np.isfinite(sub["sigma_min_kpc"]) & (sub["sigma_min_kpc"] > 0)].copy()

sub["r_over_sigma"] = RADIUS_KPC / sub["sigma_min_kpc"]
sub["fwhm_kpc"] = sub["sigma_min_kpc"] / FWHM_TO_SIGMA
sub["diam_over_fwhm"] = (2.0 * RADIUS_KPC) / sub["fwhm_kpc"]

print("=" * 74)
print(f"RESOLUTION OF A {RADIUS_KPC:g} kpc APERTURE  ({TELESCOPE}, B and V)")
print("=" * 74)
print(f"  measurements : {len(sub)}   objects : {sub['object'].nunique()}")
print()

for col, label in (("r_over_sigma", "r / sigma"),
                   ("diam_over_fwhm", "2r / FWHM  <-- the honest one")):
    q = sub[col].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"  {label}")
    print(f"      5th pct {q[0.05]:6.2f}   25th {q[0.25]:6.2f}   "
          f"median {q[0.50]:6.2f}   75th {q[0.75]:6.2f}   95th {q[0.95]:6.2f}")
    print()

n = len(sub)
print("  How many measurements fall in each regime (by 2r / FWHM):")
for lo, hi, note in ((0.0, 1.0, "aperture SMALLER than the seeing disc"),
                     (1.0, 2.0, "comparable to the seeing disc"),
                     (2.0, 3.0, "moderately resolved"),
                     (3.0, np.inf, "comfortably resolved")):
    m = int(((sub["diam_over_fwhm"] >= lo) & (sub["diam_over_fwhm"] < hi)).sum())
    print(f"      {lo:.0f} - {hi if np.isfinite(hi) else 999:>3.0f}  :  "
          f"{m:4d}  ({100 * m / n:5.1f}%)   {note}")

print()
print(f"  median seeing FWHM in kpc : {sub['fwhm_kpc'].median():.3f}")
print("=" * 74)
