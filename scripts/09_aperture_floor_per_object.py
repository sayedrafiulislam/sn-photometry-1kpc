"""
09_aperture_floor_per_object.py

Corrected replacement for 07_aperture_floor_per_object.py.

Purpose (unchanged)
-------------------
For every image in the working sample, compute the seeing-limited minimum
physical aperture radius in kpc, using that image's own measured PSF and that
object's own redshift.

This is the analogue of Kelsey et al. (2021), Fig. 2 and Section 2.2.3, with
one methodological difference. Their DES stacks are seeing-selected (a 1.3
arcsec FWHM ceiling), so seeing is effectively constant and a single global
floor of sigma_min ~ 0.55 arcsec describes the whole sample; only redshift
varies. In this sample neither term is constant:

    floor_kpc = sigma_min(arcsec) x (kpc per arcsec at z)
                ^ varies 0.36-1.94"   ^ varies over a factor of 38 in z

and the two are uncorrelated (r = -0.22). Testing a single fixed floor against
the per-image result misclassifies 13-36 of 536 images depending on the value
chosen, and the errors are asymmetric: a conservative floor discards up to 34
usable measurements while still passing 2 genuinely seeing-limited ones. No
single threshold separates two populations that differ along two axes.

What was wrong with script 07
-----------------------------
1. WRONG PLATE SCALE. It reads the `median` column of per_file_summary.csv,
   which script 04 computed as fwhm_pixels x 0.23 for every frame. 131 of 716
   frames are not at 0.23 arcsec/pixel (see C2 in PAPER_CORRECTIONS.md), so
   every Swope floor was too small by a factor of 1.87 and the twelve 0.159
   arcsec frames were too large by 1.45.

   Consequence: the paper's claim that 529 of 536 images (98.7 per cent) have
   a seeing floor below the smallest tested aperture is too optimistic. With
   correct scales the figure is nearer 96 per cent.

2. SUPERSEDED EXCLUSION LIST. It reads image_quality_flags.csv, whose flags
   were assigned by comparing wrongly-scaled FWHM values against thresholds in
   arcsec (see C7). The corrected list differs: 13 images excluded rather than
   20, with 8 Swope frames added and 15 removed.

3. CONTAMINATED PSF MEASUREMENTS. The per-image medians in per_file_summary.csv
   include cosmic rays and hot pixels that passed script 04's fit sanity check
   (see C6). 876 detections, 5.1 per cent, are narrower than half the median
   width of their own frame.

This script recomputes the per-image FWHM directly from the per-star table,
applying each frame's own plate scale and rejecting the artifacts, so none of
the three problems propagate.

Cosmology
---------
FlatLambdaCDM(H0=70, Om0=0.3), matching Kelsey et al. Consistency with the
reference matters more here than the particular parameter values, since the
comparison is relative.

Usage
-----
    python 09_aperture_floor_per_object.py ^
        --per-star results\\phase1_psf\\psf_fwhm_per_star.csv ^
        --header-summary results\\header_summary_full.csv ^
        --flags results\\phase1_psf\\image_quality_flags_corrected.csv ^
        --redshifts results\\sn_catalog_final.csv ^
        --out-csv results\\phase4_aperture\\aperture_floor_per_object_corrected.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# sigma = FWHM / (2 sqrt(2 ln 2)). The same relation Kelsey et al. use to turn
# a seeing cut into a minimum aperture radius.
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Candidate aperture grid tested by the curve-of-growth step.
APERTURE_GRID_KPC = np.arange(1.0, 10.5, 0.5)

# A detection narrower than this fraction of its own frame's median cannot be a
# star: nothing can be that much sharper than everything else under the same
# atmosphere. Relative rather than absolute, so genuinely sharp frames are not
# penalised. See C6.
SHARP_REJECT_FRACTION = 0.5

SCALE_DP = 4

# <object>_<filter>_comb_<telescope>.fits
# Deliberately matches B and V only. The single r-band frame
# (SN05gj_r_comb_dup.fits) is not part of the colour analysis and is dropped
# here rather than silently failing to parse later.
FNAME_RE = re.compile(r"^(?P<object>.+?)_(?P<filter>[BV])_comb_(?P<tel>dup|swo)\.fits$")


def basename(p) -> str:
    return Path(str(p).replace("\\", "/")).name


def arcsec_to_kpc(theta_arcsec, z, cosmo=COSMO):
    """
    Physical size subtended by an angle at redshift z.

    Uses the angular diameter distance, which is the distance measure relating
    a physical transverse size to an observed angle. Note this is the inverse
    of the conversion used when placing apertures (kpc -> arcsec); getting the
    direction wrong is a common error, so it is stated explicitly.
    """
    d_a_mpc = np.asarray(cosmo.angular_diameter_distance(z).value)
    theta_rad = np.deg2rad(np.asarray(theta_arcsec) / 3600.0)
    return theta_rad * d_a_mpc * 1000.0


def load_scales(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path)
    if "pixscale_arcsec" not in h.columns:
        raise SystemExit(f"{path} has no pixscale_arcsec column. Re-run "
                         "00_inspect_headers.py without --limit.")
    h = h[h["pixscale_arcsec"].notna()].copy()
    h["fname"] = h["file"].apply(basename)
    h["plate_scale"] = h["pixscale_arcsec"].round(SCALE_DP)
    return h[["fname", "plate_scale"]].drop_duplicates("fname")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-star", required=True, type=Path)
    ap.add_argument("--header-summary", required=True, type=Path)
    ap.add_argument("--flags", required=True, type=Path,
                    help="image_quality_flags_corrected.csv from 05b.")
    ap.add_argument("--redshifts", required=True, type=Path,
                    help="sn_catalog_final.csv from script 02.")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--keep-sharp", action="store_true",
                    help="Retain sub-half-median detections (uncorrected behaviour).")
    args = ap.parse_args()

    stars = pd.read_csv(args.per_star)
    stars["fname"] = stars["file"].apply(basename)

    # --- correction 1: per-frame plate scale ------------------------------
    scales = load_scales(args.header_summary)
    # The corrected 06_measure_psf_fwhm.py now records each frame's plate
    # scale, read from its own WCS. The original did not, so this script
    # merges the scale in from the header summary. With both present, a merge
    # on the same column name yields plate_scale_x and plate_scale_y.
    #
    # The step-06 value is set aside and re-merged from the header summary,
    # keeping one authority for the number, then the two are cross-checked.
    # They read the same FITS files by different routes and must agree.
    scale_from_06 = None
    if "plate_scale" in stars.columns:
        scale_from_06 = (stars[["fname", "plate_scale"]]
                         .drop_duplicates("fname")
                         .rename(columns={"plate_scale": "scale_06"}))
        stars = stars.drop(columns=["plate_scale"])
    if "plate_scale_from_wcs" in stars.columns:
        stars = stars.drop(columns=["plate_scale_from_wcs"])

    stars = stars.merge(scales, on="fname", how="left")

    if scale_from_06 is not None:
        chk = scales.merge(scale_from_06, on="fname", how="inner")
        disagree = chk[~np.isclose(chk["plate_scale"], chk["scale_06"],
                                   rtol=1e-3, atol=1e-4)]
        print(f"Plate-scale cross-check: {len(chk)} frames compared, "
              f"{len(disagree)} disagree between step 00 and step 06.")
        if len(disagree):
            print(disagree.head(10).to_string(index=False))
            raise SystemExit("Plate scales disagree between steps 00 and 06. "
                             "Both read the same FITS files and must give the "
                             "same answer -- resolve before proceeding.")
    if stars["plate_scale"].isna().any():
        n = int(stars["plate_scale"].isna().sum())
        print(f"WARNING: {n} stars have no plate scale and are dropped.")
        stars = stars[stars["plate_scale"].notna()].copy()

    # --- correction 2: reject non-stellar detections ----------------------
    stars["img_median_pix"] = stars.groupby("fname")["fwhm_avg_pix"].transform("median")
    artifacts = stars["fwhm_avg_pix"] < SHARP_REJECT_FRACTION * stars["img_median_pix"]
    print(f"Sub-half-median detections: {int(artifacts.sum())} of {len(stars)} "
          f"({100 * artifacts.mean():.2f}%)")
    if not args.keep_sharp:
        stars = stars[~artifacts].copy()
        print("  rejected before computing per-image FWHM.")
    else:
        print("  RETAINED (--keep-sharp).")

    stars["fwhm_arcsec"] = stars["fwhm_avg_pix"] * stars["plate_scale"]

    per_file = (stars.groupby(["fname", "telescope", "filter", "plate_scale"])
                ["fwhm_arcsec"]
                .agg(n_stars="count", fwhm_arcsec="median")
                .reset_index())

    # --- correction 3: corrected exclusion list ---------------------------
    flags = pd.read_csv(args.flags)
    flag_name_col = "fname" if "fname" in flags.columns else "file"
    flagged = set(flags.loc[flags["any_flag"], flag_name_col].apply(basename))
    before = len(per_file)
    per_file = per_file[~per_file["fname"].isin(flagged)].copy()
    print(f"Excluded {before - len(per_file)} flagged images "
          f"({len(flagged)} in the corrected list).")

    # --- object names and redshifts ---------------------------------------
    parsed = per_file["fname"].str.extract(FNAME_RE)
    per_file["object"] = parsed["object"]
    n_unparsed = int(per_file["object"].isna().sum())
    if n_unparsed:
        print(f"{n_unparsed} file(s) did not match the B/V naming convention "
              f"and are dropped (expected: the single r-band frame).")
        per_file = per_file[per_file["object"].notna()].copy()

    redshifts = pd.read_csv(args.redshifts)
    z_by_object = (redshifts[["object", "redshift"]]
                   .drop_duplicates(subset="object")
                   .rename(columns={"redshift": "z"}))

    merged = per_file.merge(z_by_object, on="object", how="inner")
    n_lost = per_file["object"].nunique() - merged["object"].nunique()
    if n_lost > 0:
        print(f"{n_lost} objects with PSF measurements have no redshift and are "
              f"dropped (expected: objects excluded in script 02).")

    # --- the calculation ---------------------------------------------------
    merged["sigma_min_arcsec"] = merged["fwhm_arcsec"] * FWHM_TO_SIGMA
    merged["sigma_min_kpc"] = arcsec_to_kpc(merged["sigma_min_arcsec"].values,
                                            merged["z"].values)

    grid = APERTURE_GRID_KPC
    merged["smallest_safe_grid_radius_kpc"] = merged["sigma_min_kpc"].apply(
        lambda s: grid[grid >= s][0] if np.any(grid >= s) else np.nan)

    out_cols = ["object", "telescope", "filter", "fname", "plate_scale", "z",
                "n_stars", "fwhm_arcsec", "sigma_min_arcsec", "sigma_min_kpc",
                "smallest_safe_grid_radius_kpc"]
    result = merged[out_cols].rename(columns={"fname": "file"})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)

    # --- report ------------------------------------------------------------
    print(f"\nWrote {len(result)} rows ({result['object'].nunique()} objects) "
          f"-> {args.out_csv}\n")

    print("Seeing floor in kpc, by group:")
    print(result.groupby(["telescope", "filter"])["sigma_min_kpc"]
          .describe()[["count", "min", "50%", "max"]].round(4).to_string())

    r_min = grid.min()
    n_safe = int((result["sigma_min_kpc"] <= r_min).sum())
    print(f"\n{n_safe} / {len(result)} images ({100 * n_safe / len(result):.1f}%) "
          f"have a seeing floor below the smallest grid radius ({r_min} kpc), "
          f"i.e. the whole aperture grid is seeing-safe for them.")

    over = result[result["sigma_min_kpc"] > r_min].sort_values(
        "sigma_min_kpc", ascending=False)
    if len(over):
        print(f"\nImages whose floor exceeds {r_min} kpc:")
        print(over[["object", "telescope", "filter", "plate_scale", "z",
                    "fwhm_arcsec", "sigma_min_kpc",
                    "smallest_safe_grid_radius_kpc"]].round(3).to_string(index=False))

    # Per-object view: an object is only usable at a radius where every one of
    # its images is seeing-safe, since B and V must be measured at the same
    # physical radius for the colour to mean anything.
    per_obj = result.groupby("object")["sigma_min_kpc"].max()
    print(f"\nPer-object worst-case floor (the binding constraint for a colour):")
    print(f"  median {per_obj.median():.3f} kpc, "
          f"90th pct {per_obj.quantile(0.9):.3f} kpc, max {per_obj.max():.3f} kpc")
    print(f"  objects whose worst image exceeds {r_min} kpc: "
          f"{int((per_obj > r_min).sum())} of {len(per_obj)}")


if __name__ == "__main__":
    main()