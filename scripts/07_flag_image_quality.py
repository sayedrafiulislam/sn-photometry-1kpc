"""
07_flag_image_quality.py

Corrected replacement for 05_flag_image_quality.py.

What was wrong with the original
--------------------------------
Script 05 reads per_file_summary.csv, whose FWHM values were computed by
script 04 using a single hard-coded plate scale of 0.23 arcsec/pixel. Two
consequences follow, both documented as C6 and C7 in PAPER_CORRECTIONS.md:

  1. THE THRESHOLDS WERE NOT APPLIED UNIFORMLY. Both cuts compare a value
     in arcsec:

         SCATTER_THRESHOLD_ARCSEC = 0.6
         SOFT_THRESHOLD_ARCSEC    = 2.0

     A Swope frame whose true scale is 0.43 arcsec/pixel has its FWHM
     understated by a factor of 1.87, so the effective soft-seeing
     threshold on those frames was 1.07 arcsec rather than 2.0 arcsec.
     Approximately 27 genuinely soft Swope frames passed a cut they
     should have failed, while the twelve 0.159 arcsec frames had their
     FWHM overstated by 1.45x and were flagged despite ordinary seeing.

  2. NON-STELLAR DETECTIONS WERE COUNTED AS STARS. The fit sanity check
     in script 04 accepts anything with sigma > 0.5 px, i.e. FWHM >
     1.18 px, so single hot pixels and cosmic rays survive. 876 of 17060
     detections (5.1%) are narrower than half the median width of their
     own image. These pull the per-image median down and inflate the
     per-image scatter, which affects both flags.

What this script does differently
---------------------------------
  - Recomputes the per-image statistics directly from psf_fwhm_per_star.csv
    rather than trusting per_file_summary.csv.
  - Converts each star's FWHM using the plate scale of the frame it was
    measured in, read from the WCS by the full 00_inspect_headers.py run.
  - Optionally rejects detections narrower than half their own image's
    median before computing statistics (on by default; see --keep-sharp).
  - Reports the FEW_STARS rule explicitly, since with MIN_STARS = 8 and a
    minimum of 13 fitted stars per image it can never fire.
  - Prints a before/after comparison so the effect of each correction is
    visible rather than assumed.

Nothing existing is overwritten: output goes to new filenames.

Usage
-----
    python 07_flag_image_quality.py ^
        --per-star results\\phase1_psf\\psf_fwhm_per_star.csv ^
        --header-summary results\\header_summary_full.csv ^
        --out-flags results\\phase1_psf\\image_quality_flags_corrected.csv ^
        --out-excluded results\\phase1_psf\\excluded_images_phase1_corrected.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Thresholds unchanged from script 05, so that the only difference in the
# result is the correction, not a re-tuning.
SCATTER_THRESHOLD_ARCSEC = 0.6
SOFT_THRESHOLD_ARCSEC = 2.0
MIN_STARS = 8

# A detection narrower than this fraction of its own image's median width
# cannot be a star: nothing in one frame can be that much sharper than
# everything else under the same atmosphere. Relative rather than absolute
# so that genuinely sharp frames are not penalised.
SHARP_REJECT_FRACTION = 0.5

SCALE_DP = 4


def basename(p) -> str:
    return Path(str(p).replace("\\", "/")).name


def mad_std(x: np.ndarray) -> float:
    """
    Robust scatter estimator: median absolute deviation, scaled so that for
    Gaussian data it estimates the same quantity as the standard deviation.

    The 1.4826 factor is 1 / Phi^-1(0.75). Implemented locally rather than
    imported from astropy so this script has no extra dependency, and so the
    definition is visible at the point of use.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def load_scales(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path)
    if "pixscale_arcsec" not in h.columns:
        raise SystemExit(f"{path} has no pixscale_arcsec column. Re-run "
                         "00_inspect_headers.py without --limit.")
    h = h[h["pixscale_arcsec"].notna()].copy()
    h["fname"] = h["file"].apply(basename)
    h["plate_scale"] = h["pixscale_arcsec"].round(SCALE_DP)
    return h[["fname", "plate_scale"]].drop_duplicates("fname")


def summarise(stars: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-image statistics from the per-star table."""
    g = stars.groupby(["fname", "telescope", "filter", "plate_scale"])[col]
    out = g.agg(n_stars="count", median="median", mean="mean",
                std="std", max="max").reset_index()
    out["mad_sigma"] = g.apply(lambda s: mad_std(s.values)).values
    # Percentile-based scatter, kept only so the comparison script and the
    # paper can refer to it. Not used for flagging -- see the companion
    # script quality_statistic_comparison.py for why.
    out["iqr_p16_p84"] = g.apply(
        lambda s: np.percentile(s, 84) - np.percentile(s, 16)).values
    return out


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_high_scatter"] = df["mad_sigma"] > SCATTER_THRESHOLD_ARCSEC
    df["flag_very_soft"] = df["median"] > SOFT_THRESHOLD_ARCSEC
    df["flag_few_stars"] = df["n_stars"] < MIN_STARS
    df["any_flag"] = df[["flag_high_scatter", "flag_very_soft",
                         "flag_few_stars"]].any(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-star", required=True, type=Path)
    ap.add_argument("--header-summary", required=True, type=Path)
    ap.add_argument("--out-flags", required=True, type=Path)
    ap.add_argument("--out-excluded", required=True, type=Path)
    ap.add_argument("--keep-sharp", action="store_true",
                    help="Do NOT reject sub-half-median detections. Use only "
                         "to reproduce the uncorrected behaviour.")
    args = ap.parse_args()

    stars = pd.read_csv(args.per_star)
    stars["fname"] = stars["file"].apply(basename)

    # The corrected 06_measure_psf_fwhm.py now reads each frame's plate scale
    # from its own WCS and records it. The original did not, so this script
    # merged the scale in from the header summary. Both sources exist now, and
    # a merge on the same column name would silently produce plate_scale_x and
    # plate_scale_y rather than the expected column.
    #
    # The scale is dropped here and re-merged from the header summary, so the
    # value used stays traceable to a single authority -- the full
    # 00_inspect_headers.py run. The two are cross-checked below; they read the
    # same FITS files by different routes and must agree.
    scale_from_06 = None
    if "plate_scale" in stars.columns:
        scale_from_06 = (stars[["fname", "plate_scale"]]
                         .drop_duplicates("fname")
                         .rename(columns={"plate_scale": "scale_06"}))
        stars = stars.drop(columns=["plate_scale"])
    if "plate_scale_from_wcs" in stars.columns:
        stars = stars.drop(columns=["plate_scale_from_wcs"])

    scales = load_scales(args.header_summary)
    stars = stars.merge(scales, on="fname", how="left")

    if scale_from_06 is not None:
        chk = (scales.merge(scale_from_06, on="fname", how="inner"))
        disagree = chk[~np.isclose(chk["plate_scale"], chk["scale_06"],
                                   rtol=1e-3, atol=1e-4)]
        print(f"Plate-scale cross-check: {len(chk)} frames compared, "
              f"{len(disagree)} disagree between step 00 and step 06.")
        if len(disagree):
            print(disagree.head(10).to_string(index=False))
            raise SystemExit("Plate scales disagree between steps 00 and 06. "
                             "Resolve before proceeding -- both read the same "
                             "FITS files and must give the same answer.")

    missing = stars["plate_scale"].isna()
    if missing.any():
        print(f"WARNING: {int(missing.sum())} stars in "
              f"{stars.loc[missing,'fname'].nunique()} frames have no plate "
              f"scale and are dropped.")
        stars = stars[~missing].copy()

    # --- correction 1: per-file plate scale -------------------------------
    stars["fwhm_arcsec_wrong"] = stars["fwhm_avg_pix"] * 0.23
    stars["fwhm_arcsec"] = stars["fwhm_avg_pix"] * stars["plate_scale"]

    # --- correction 2: reject non-stellar detections ----------------------
    stars["img_median_pix"] = stars.groupby("fname")["fwhm_avg_pix"].transform("median")
    stars["is_sharp_artifact"] = (
        stars["fwhm_avg_pix"] < SHARP_REJECT_FRACTION * stars["img_median_pix"]
    )
    n_artifacts = int(stars["is_sharp_artifact"].sum())
    n_imgs_affected = stars.loc[stars["is_sharp_artifact"], "fname"].nunique()
    print(f"Sub-half-median detections identified: {n_artifacts} of "
          f"{len(stars)} ({100 * n_artifacts / len(stars):.2f}%) "
          f"across {n_imgs_affected} images.")

    if args.keep_sharp:
        print("  --keep-sharp set: they are RETAINED (uncorrected behaviour).")
        clean = stars
    else:
        clean = stars[~stars["is_sharp_artifact"]].copy()
        print("  They are rejected before computing per-image statistics.")

    # --- summarise and flag, before and after ----------------------------
    old = apply_flags(summarise(stars, "fwhm_arcsec_wrong"))
    new = apply_flags(summarise(clean, "fwhm_arcsec"))

    print("\nPlate scales present:")
    for s, n in new.groupby("plate_scale")["fname"].nunique().items():
        print(f"  {s}\"/pix : {n} images")

    print("\n" + "=" * 72)
    print("Flag comparison: original (0.23 assumed, artifacts kept) vs corrected")
    print("=" * 72)
    print(f"{'group':10s} {'imgs':>5s} | {'soft old':>9s} {'soft new':>9s} | "
          f"{'scat old':>9s} {'scat new':>9s} | {'any old':>8s} {'any new':>8s}")
    for (tel, filt) in sorted(set(zip(new["telescope"], new["filter"]))):
        o = old[(old["telescope"] == tel) & (old["filter"] == filt)]
        n = new[(new["telescope"] == tel) & (new["filter"] == filt)]
        print(f"{tel + '-' + filt:10s} {len(n):5d} | "
              f"{int(o['flag_very_soft'].sum()):9d} {int(n['flag_very_soft'].sum()):9d} | "
              f"{int(o['flag_high_scatter'].sum()):9d} {int(n['flag_high_scatter'].sum()):9d} | "
              f"{int(o['any_flag'].sum()):8d} {int(n['any_flag'].sum()):8d}")

    print(f"\nTotal flagged: {int(old['any_flag'].sum())} -> "
          f"{int(new['any_flag'].sum())} of {len(new)} images")

    # The FEW_STARS rule, stated explicitly rather than left implicit.
    print(f"\nFEW_STARS rule (n_stars < {MIN_STARS}): "
          f"{int(new['flag_few_stars'].sum())} images flagged. "
          f"Minimum n_stars in the data set is {int(new['n_stars'].min())}, "
          f"so this rule cannot fire.")

    # Which images changed status, and in which direction.
    merged = old[["fname", "any_flag"]].rename(columns={"any_flag": "old"}).merge(
        new[["fname", "any_flag", "telescope", "filter", "plate_scale",
             "n_stars", "median", "mad_sigma"]].rename(columns={"any_flag": "new"}),
        on="fname", how="outer")
    newly = merged[(~merged["old"].fillna(False)) & (merged["new"].fillna(False))]
    rescued = merged[(merged["old"].fillna(False)) & (~merged["new"].fillna(False))]

    print(f"\nNewly flagged (passed before, fail now): {len(newly)}")
    if len(newly):
        print(newly[["fname", "telescope", "filter", "plate_scale",
                     "n_stars", "median", "mad_sigma"]].round(3).to_string(index=False))
    print(f"\nNo longer flagged (failed before, pass now): {len(rescued)}")
    if len(rescued):
        print(rescued[["fname", "telescope", "filter", "plate_scale",
                       "n_stars", "median", "mad_sigma"]].round(3).to_string(index=False))

    args.out_flags.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.out_flags, index=False)
    new[new["any_flag"]].to_csv(args.out_excluded, index=False)
    print(f"\nWrote {args.out_flags}")
    print(f"Wrote {args.out_excluded} ({int(new['any_flag'].sum())} images)")


if __name__ == "__main__":
    main()