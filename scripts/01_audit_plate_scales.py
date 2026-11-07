"""
01_audit_plate_scales.py

Purpose
-------
Quantify the plate-scale problem found by the full run of 00_inspect_headers.py.

Background
----------
The original reconnaissance run used --limit and, because the file listing is
sorted alphabetically, sampled only du Pont frames -- all of which reported
0.23 arcsec/pixel. That value was hard-coded into:

    06_measure_psf_fwhm.py   PLATE_SCALE_ARCSEC_PER_PIX = 0.23
    10_curve_of_growth.py    PLATE_SCALE_ARCSEC_PER_PIX = 0.23
    check_aperture_overlay.py PLATE_SCALE = 0.23

The full run shows three distinct scales in the data set (0.159, 0.23, 0.43),
distributed across both telescopes. Every frame whose true scale is not 0.23
has therefore had its aperture placed at the wrong physical radius.

This script does NOT fix anything. It answers four questions so we know the
size of the problem before deciding how to respond:

    Q1. How many frames sit at each scale, broken down by telescope and filter?
    Q2. For each object, do its B and V frames share the same scale? A mismatch
        means the two filters sampled different physical regions of the same
        galaxy -- a direct, uncontrolled B-V colour error rather than a
        harmless overall offset.
    Q3. What is the true physical radius that a nominal 5.0 kpc aperture
        actually corresponded to, per scale?
    Q4. How many objects in the final calibrated catalogue are affected?

Usage
-----
    python 01_audit_plate_scales.py --summary-csv results\\header_summary_full.csv
    python 01_audit_plate_scales.py --summary-csv results\\header_summary_full.csv --catalog-csv results\\calibrated_color_5kpc_flagged.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

# The value that was assumed everywhere downstream.
ASSUMED_SCALE = 0.23

# The fiducial aperture radius the final catalogue was built on.
FIDUCIAL_RADIUS_KPC = 5.0

# Rounding used when comparing scales. The raw WCS values carry floating-point
# noise (0.23000000000000403), so an exact comparison reports spurious
# distinct values.
SCALE_DP = 4


def object_from_filename(fname: str) -> str:
    """
    Recover the object name from <object>_<filter>_comb_<telescope>.fits.

    This mirrors the parsing in 02_build_catalog.py. Kept local rather than
    imported so this audit stays a standalone diagnostic.
    """
    stem = Path(str(fname)).stem
    parts = stem.split("_")
    return parts[0] if parts else ""


def filter_from_filename(fname: str) -> str:
    stem = Path(str(fname)).stem
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unreadable frames and frames with no WCS -- neither can contribute
    # a scale, and both are separate problems worth counting on their own.
    n_all = len(df)
    n_unreadable = int((~df["readable"].astype(bool)).sum())
    df = df[df["readable"].astype(bool)].copy()

    no_scale = df["pixscale_arcsec"].isna()
    n_no_scale = int(no_scale.sum())
    df = df[~no_scale].copy()

    print(f"Frames in summary CSV:        {n_all}")
    print(f"  unreadable:                 {n_unreadable}")
    print(f"  readable but no WCS scale:  {n_no_scale}")
    print(f"  usable for this audit:      {len(df)}")

    df["scale"] = df["pixscale_arcsec"].round(SCALE_DP)

    # Prefer the columns written by the v2 reconnaissance script; fall back to
    # parsing the filename if this CSV came from the older version.
    if "telescope_from_name" not in df.columns:
        df["telescope_from_name"] = df["file"].apply(
            lambda f: "Swope" if "_swo" in str(f).lower() else
                      ("du Pont" if "_dup" in str(f).lower() else "unknown")
        )
    if "filter_from_name" not in df.columns:
        df["filter_from_name"] = df["file"].apply(filter_from_filename)

    df["object"] = df["file"].apply(object_from_filename)
    return df


def q1_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Q1. Frame counts by scale, telescope and filter")
    print("=" * 70)

    table = (df.groupby(["scale", "telescope_from_name", "filter_from_name"])
               .size()
               .rename("n_frames")
               .reset_index()
               .sort_values(["scale", "telescope_from_name", "filter_from_name"]))
    print(table.to_string(index=False))

    print("\nTotals per scale:")
    for scale, n in df["scale"].value_counts().sort_index().items():
        frac = 100.0 * n / len(df)
        marker = "  <- assumed value" if abs(scale - ASSUMED_SCALE) < 1e-6 else ""
        print(f"  {scale:>6.3f} arcsec/pix : {n:4d} frames ({frac:5.1f}%){marker}")

    n_wrong = int((df["scale"] != ASSUMED_SCALE).sum())
    print(f"\nFrames processed with the wrong scale: {n_wrong}/{len(df)} "
          f"({100.0 * n_wrong / len(df):.1f}%)")


def q2_bv_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each object, check whether its B and V frames share a plate scale.

    This is the question that determines whether the problem is a colour
    error or merely a radius error. If B and V share a scale, both filters
    sampled the same (wrong) physical radius, and the colour is measured
    over a consistent region -- recoverable. If they differ, the colour
    itself is contaminated.
    """
    print("\n" + "=" * 70)
    print("Q2. Do B and V frames of the same object share a plate scale?")
    print("=" * 70)

    bv = df[df["filter_from_name"].isin(["B", "V"])]

    rows = []
    for obj, grp in bv.groupby("object"):
        scales_b = sorted(set(grp.loc[grp["filter_from_name"] == "B", "scale"]))
        scales_v = sorted(set(grp.loc[grp["filter_from_name"] == "V", "scale"]))
        if not scales_b or not scales_v:
            status = "missing one filter"
        elif set(scales_b) == set(scales_v) and len(scales_b) == 1:
            status = "consistent"
        else:
            status = "MISMATCH"
        rows.append({
            "object": obj,
            "scales_B": ",".join(f"{s:g}" for s in scales_b) or "-",
            "scales_V": ",".join(f"{s:g}" for s in scales_v) or "-",
            "status": status,
        })

    out = pd.DataFrame(rows)
    counts = out["status"].value_counts()
    for status, n in counts.items():
        print(f"  {status:>20s}: {n}")

    mismatches = out[out["status"] == "MISMATCH"]
    if len(mismatches):
        print(f"\nObjects where B and V were measured at different plate scales "
              f"({len(mismatches)}):")
        print(mismatches.to_string(index=False))
        print("\n  These are the objects whose B-V colour is directly affected.")
    else:
        print("\n  No B/V mismatches. The error is a radius error, not a colour error.")

    return out


def q3_true_radii(df: pd.DataFrame) -> None:
    """
    Translate the scale error into the physical radius actually measured.

    The pipeline computed a pixel radius as
        r_pix = r_arcsec / ASSUMED_SCALE
    but the frame's true angular radius is
        r_true_arcsec = r_pix * true_scale
    so the true physical radius scales as true_scale / ASSUMED_SCALE.
    """
    print("\n" + "=" * 70)
    print(f"Q3. What a nominal {FIDUCIAL_RADIUS_KPC:.1f} kpc aperture actually measured")
    print("=" * 70)
    print(f"{'true scale':>12s} {'ratio':>8s} {'true radius':>13s} {'area ratio':>11s}")
    for scale in sorted(df["scale"].unique()):
        ratio = scale / ASSUMED_SCALE
        print(f"{scale:>12.3f} {ratio:>8.3f} "
              f"{FIDUCIAL_RADIUS_KPC * ratio:>10.2f} kpc {ratio ** 2:>11.2f}")
    print("\n  A ratio of 1.00 means the frame was handled correctly.")


def q4_catalog_impact(df: pd.DataFrame, bv_status: pd.DataFrame,
                      catalog_csv: Path) -> None:
    print("\n" + "=" * 70)
    print("Q4. Impact on the final calibrated catalogue")
    print("=" * 70)

    cat = pd.read_csv(catalog_csv)
    if "object" not in cat.columns:
        print(f"  No 'object' column in {catalog_csv} -- skipping.")
        return

    merged = cat.merge(bv_status, on="object", how="left")
    n_cat = len(merged)
    n_unmatched = int(merged["status"].isna().sum())
    print(f"  Objects in catalogue:            {n_cat}")
    if n_unmatched:
        print(f"  Not matched to a frame:          {n_unmatched} "
              f"(check for name-format differences)")

    for status, n in merged["status"].value_counts().items():
        print(f"  {status:>30s}: {n}")

    # Of the consistent ones, how many were nonetheless at the wrong scale?
    consistent = merged[merged["status"] == "consistent"]
    wrong_scale = consistent[consistent["scales_B"].astype(str) != f"{ASSUMED_SCALE:g}"]
    print(f"\n  Consistent B/V pair but wrong scale: {len(wrong_scale)}")
    print("    -> radius error only; colour internally consistent, "
          "physical radius mislabelled.")
    mismatch = merged[merged["status"] == "MISMATCH"]
    print(f"  B/V scale mismatch:                  {len(mismatch)}")
    print("    -> colour itself affected; these need recomputation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-csv", required=True, type=Path,
                    help="Output of the full 00_inspect_headers.py run.")
    ap.add_argument("--catalog-csv", type=Path, default=None,
                    help="Optional: final calibrated catalogue, to scope the impact.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional: write the per-object B/V scale status table.")
    args = ap.parse_args()

    df = load_summary(args.summary_csv)
    q1_distribution(df)
    bv_status = q2_bv_consistency(df)
    q3_true_radii(df)

    if args.catalog_csv:
        q4_catalog_impact(df, bv_status, args.catalog_csv)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        bv_status.to_csv(args.out_csv, index=False)
        print(f"\nWrote per-object scale status to {args.out_csv}")


if __name__ == "__main__":
    main()