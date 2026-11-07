"""
08_summarise_psf.py

Purpose
-------
Rebuild the PSF characterisation table (Table 2 of the paper) with two
corrections to the original 04/06 output:

  1. PLATE SCALE PER FILE. Scripts 04 and 10 hard-code
     PLATE_SCALE_ARCSEC_PER_PIX = 0.23. The full reconnaissance run showed
     three distinct scales in the data set:

         0.230 arcsec/pix : 585 frames  (du Pont B/V, some Swope V)
         0.430 arcsec/pix : 119 frames  (Swope V -- the SITe3 direct CCD)
         0.159 arcsec/pix :  12 frames  (mixed)

     Every FWHM in arcsec produced with the single constant is wrong for
     the 131 frames that are not at 0.23. The Swope V group is worst
     affected: its reported median of 0.875 arcsec implies the 1 m Swope
     achieved better seeing than the 2.5 m du Pont under the same sky,
     which is not physically plausible. This script converts each star's
     FWHM using the plate scale of the frame it was measured in.

  2. CLUSTERED UNCERTAINTY ON THE MEDIAN. The original summary reports
     median, mean, std, p16 and p84 but no uncertainty on the median.
     The naive standard error, std/sqrt(n_stars), is wrong here: the
     stars are not independent samples of the seeing. Up to 25 stars come
     from each image and share one atmosphere, so the effective sample
     size is closer to the number of images than the number of stars.
     Using stars as independent understates the error by a factor of
     roughly 1.7-2.5.

     The uncertainty is therefore estimated by a cluster (block)
     bootstrap: whole images are resampled with replacement, and the
     median is recomputed over the pooled stars of the resampled images.
     This is the same resampling principle used for the aperture
     selection in script 14/18, applied at the image level.

Note on interpretation
----------------------
The bootstrap error and the percentile range answer different questions
and both belong in the table:

    median +/- SE   how precisely the CENTRE of the seeing distribution
                    is located. Small, because there are many images.
    p16 - p84       how much the seeing actually VARIES between images.
                    Large, because this is a decade of archival data with
                    no uniform seeing selection.

Reporting only the first would imply a homogeneity this data set does not
have.

Usage
-----
    python 08_summarise_psf.py ^
        --per-star results\\phase1_psf\\psf_fwhm_per_star.csv ^
        --header-summary results\\header_summary_full.csv ^
        --excluded results\\phase1_psf\\excluded_images_phase1.csv ^
        --out-csv results\\phase1_psf\\psf_fwhm_summary_corrected.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOTSTRAP = 5000
RANDOM_SEED = 42          # fixed so the quoted errors are reproducible
SCALE_DP = 4              # WCS values carry float noise (0.23000000000000403)


def basename(p) -> str:
    """Filename only, so paths from different machines still match."""
    return Path(str(p).replace("\\", "/")).name


def load_scales(header_summary: Path) -> pd.DataFrame:
    """Per-frame plate scale, keyed on bare filename."""
    h = pd.read_csv(header_summary)
    if "pixscale_arcsec" not in h.columns:
        raise SystemExit(f"{header_summary} has no pixscale_arcsec column. "
                         "Re-run 00_inspect_headers.py (the version without --limit).")
    h = h[h["pixscale_arcsec"].notna()].copy()
    h["fname"] = h["file"].apply(basename)
    h["plate_scale"] = h["pixscale_arcsec"].round(SCALE_DP)
    return h[["fname", "plate_scale"]].drop_duplicates("fname")


def cluster_bootstrap_median(df: pd.DataFrame, value_col: str,
                             cluster_col: str = "fname",
                             n_boot: int = N_BOOTSTRAP,
                             seed: int = RANDOM_SEED):
    """
    Bootstrap the median by resampling CLUSTERS (images), not rows (stars).

    Returns (se, lo95, hi95). Resampling whole images preserves the
    within-image correlation that makes the naive standard error too small.
    """
    rng = np.random.default_rng(seed)
    groups = {k: v[value_col].to_numpy() for k, v in df.groupby(cluster_col)}
    keys = np.array(list(groups.keys()))
    n = len(keys)
    if n < 2:
        return np.nan, np.nan, np.nan

    medians = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, n, size=n)
        medians[i] = np.median(np.concatenate([groups[keys[j]] for j in picked]))

    return float(np.std(medians, ddof=1)), \
        float(np.percentile(medians, 2.5)), \
        float(np.percentile(medians, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-star", required=True, type=Path,
                    help="psf_fwhm_per_star.csv from script 04.")
    ap.add_argument("--header-summary", required=True, type=Path,
                    help="header_summary_full.csv from the full 00 run.")
    ap.add_argument("--excluded", type=Path, default=None,
                    help="excluded_images_phase1.csv, to reproduce the cleaned table.")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    stars = pd.read_csv(args.per_star)
    stars["fname"] = stars["file"].apply(basename)

    if args.excluded is not None and args.excluded.exists():
        # Script 07 writes its excluded-frame list with the filename in a
        # column called `fname`, having already stripped the directory. Older
        # outputs use `file` with a full path. Accept either, and fail loudly
        # rather than silently excluding nothing if neither is present -- an
        # empty exclusion set would quietly put the 13 rejected frames back
        # into the table.
        _exc = pd.read_csv(args.excluded)
        _col = next((c for c in ("fname", "file", "filename")
                     if c in _exc.columns), None)
        if _col is None:
            raise SystemExit(
                f"{args.excluded} has no fname/file column. "
                f"Columns present: {list(_exc.columns)}")
        dropped = set(_exc[_col].apply(basename))
        print(f"  excluded-frame list: {len(dropped)} frames "
              f"(from column '{_col}')")
        before = stars["fname"].nunique()
        stars = stars[~stars["fname"].isin(dropped)].copy()
        print(f"Applied phase-1 image quality cut: "
              f"{before - stars['fname'].nunique()} images removed.")

    # The corrected 06_measure_psf_fwhm.py now reads each frame's plate scale
    # from its own WCS and records it in the per-star table. The original did
    # not, so this script merges the scale in from the header summary. With
    # both present, a merge on the same column name yields plate_scale_x and
    # plate_scale_y rather than the expected column.
    #
    # The scale from step 06 is set aside and re-merged from the header
    # summary, keeping a single authority for the value, then the two are
    # cross-checked. They read the same FITS files by different routes and
    # must agree; if they do not, that is a real inconsistency and the script
    # stops rather than silently choosing one.
    scale_from_06 = None
    if "plate_scale" in stars.columns:
        scale_from_06 = (stars[["fname", "plate_scale"]]
                         .drop_duplicates("fname")
                         .rename(columns={"plate_scale": "scale_06"}))
        stars = stars.drop(columns=["plate_scale"])
    if "plate_scale_from_wcs" in stars.columns:
        stars = stars.drop(columns=["plate_scale_from_wcs"])

    scales = load_scales(args.header_summary)
    merged = stars.merge(scales, on="fname", how="left")

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

    n_missing = int(merged["plate_scale"].isna().sum())
    if n_missing:
        missing_files = merged.loc[merged["plate_scale"].isna(), "fname"].nunique()
        print(f"WARNING: {n_missing} stars in {missing_files} frames have no "
              f"plate scale and are excluded from the corrected table.")
        merged = merged[merged["plate_scale"].notna()].copy()

    # The correction itself. Everything else in this script is bookkeeping.
    merged["fwhm_arcsec_corr"] = merged["fwhm_avg_pix"] * merged["plate_scale"]

    print("\nPlate scales in use, by group:")
    for (tel, filt), g in merged.groupby(["telescope", "filter"]):
        counts = g.groupby("plate_scale")["fname"].nunique().to_dict()
        print(f"  {tel}-{filt}: " +
              ", ".join(f"{k}\" x{v} frames" for k, v in sorted(counts.items())))

    rows = []
    for (tel, filt), g in merged.groupby(["telescope", "filter"]):
        se, lo, hi = cluster_bootstrap_median(g, "fwhm_arcsec_corr",
                                              n_boot=args.n_boot)
        med = float(np.median(g["fwhm_arcsec_corr"]))
        naive_se = 1.2533 * g["fwhm_arcsec_corr"].std() / np.sqrt(len(g))

        rows.append({
            "telescope": tel,
            "filter": filt,
            "n_stars": len(g),
            "n_images": g["fname"].nunique(),
            "median_arcsec": med,
            "se_median_arcsec": se,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "p16_arcsec": float(np.percentile(g["fwhm_arcsec_corr"], 16)),
            "p84_arcsec": float(np.percentile(g["fwhm_arcsec_corr"], 84)),
            "median_pix": float(np.median(g["fwhm_avg_pix"])),
            "sigma_min_p84_arcsec": float(np.percentile(g["fwhm_arcsec_corr"], 84))
                                    / (2.0 * np.sqrt(2.0 * np.log(2.0))),
            "naive_se_arcsec": float(naive_se),
            "se_understated_by": float(se / naive_se) if naive_se > 0 else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["telescope", "filter"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 78)
    print("Corrected PSF summary")
    print("=" * 78)
    print(f"{'group':10s} {'imgs':>5s} {'stars':>6s} {'median':>16s} "
          f"{'p16-p84':>15s} {'sigma_min':>10s}")
    for r in out.itertuples(index=False):
        print(f"{r.telescope + '-' + r.filter:10s} {r.n_images:5d} {r.n_stars:6d} "
              f"{r.median_arcsec:8.3f} +/- {r.se_median_arcsec:.3f}  "
              f"{r.p16_arcsec:6.2f}-{r.p84_arcsec:<6.2f} "
              f"{r.sigma_min_p84_arcsec:9.3f}\"")

    print("\nThe quoted error is a cluster bootstrap over images. The naive "
          "std/sqrt(n_stars)")
    print("understates it by: " +
          ", ".join(f"{r.telescope}-{r.filter} {r.se_understated_by:.1f}x"
                    for r in out.itertuples(index=False)))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()