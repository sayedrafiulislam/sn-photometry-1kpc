"""
02_resolution_cut_options.py

What does each candidate resolution cut cost?

Script 01 showed the 1 kpc aperture is typically 3 seeing discs across, but
that ~2 per cent of measurements have an aperture NARROWER than the PSF while
still passing the inherited r >= sigma_min floor.

This script prices three candidate criteria so the choice is made on evidence
rather than by inheritance. It reports, for each: measurements kept, objects
with BOTH B and V surviving, and the redshift distribution of what remains --
because any resolution cut removes high-z objects preferentially, and a cut
that reshapes the sample is a different thing from one that trims it.

READS ONLY. Writes nothing. Decides nothing.
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

sub = floor[(floor["telescope"] == TELESCOPE) &
            (floor["filter"].isin(["B", "V"]))].copy()
sub = sub[np.isfinite(sub["sigma_min_kpc"]) & (sub["sigma_min_kpc"] > 0)]

sub["fwhm_kpc"] = sub["sigma_min_kpc"] / FWHM_TO_SIGMA
sub["diam_over_fwhm"] = (2.0 * RADIUS_KPC) / sub["fwhm_kpc"]

CRITERIA = [
    ("A. inherited floor   r >= sigma",      RADIUS_KPC >= sub["sigma_min_kpc"]),
    ("B. aperture >= seeing disc  2r >= FWHM", sub["diam_over_fwhm"] >= 1.0),
    ("C. properly resolved  2r >= 2 x FWHM",   sub["diam_over_fwhm"] >= 2.0),
]

print("=" * 78)
print(f"COST OF EACH RESOLUTION CRITERION AT {RADIUS_KPC:g} kpc")
print("=" * 78)
print(f"  starting point : {len(sub)} measurements, "
      f"{sub['object'].nunique()} objects\n")

for label, mask in CRITERIA:
    kept = sub[mask]
    if len(kept) == 0:
        print(f"  {label}\n      nothing survives\n")
        continue

    both = (kept.groupby("object")["filter"].apply(set)
                .apply(lambda s: {"B", "V"}.issubset(s)))
    n_obj = int(both.sum())
    names = set(both[both].index)
    z = kept[kept["object"].isin(names)].drop_duplicates("object")["z"]

    print(f"  {label}")
    print(f"      measurements kept   : {len(kept):4d}  "
          f"({100 * len(kept) / len(sub):5.1f}%)")
    print(f"      objects with B AND V: {n_obj:4d}")
    print(f"      redshift            : median {z.median():.4f}   "
          f"range {z.min():.4f} - {z.max():.4f}")
    print(f"      z 90th percentile   : {z.quantile(0.90):.4f}")
    print()

print("  Compare the 90th percentiles. If a stricter cut pulls it down sharply,")
print("  the cut is reshaping the redshift reach of the sample, not just")
print("  trimming its edges -- and that must be declared, not absorbed.")
print("=" * 78)