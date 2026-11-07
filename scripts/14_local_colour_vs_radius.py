"""
14_local_colour_vs_radius.py

Local instrumental B-V as a function of aperture radius, from
same-telescope du Pont pairs.

WHY COLOUR AND NOT FLUX
-----------------------
Flux from an extended host has no reason to converge: galaxies have no
edge, and most of these curves are still climbing at 10 kpc. A colour is
a ratio between two bands, so it can stabilise even while both bands
keep growing, provided the local stellar population is roughly uniform
in colour at that scale. If it does not stabilise, that is a colour
gradient -- an astrophysical result, not a fault in the measurement.

WHY DU PONT ONLY
----------------
Forming B-V across two telescopes means differencing measurements made
through different optics, different detectors and different sky. Any
instrumental difference enters the colour directly and is
indistinguishable from a real one. Using one telescope for both bands
makes every instrumental term common to B and V, so it cancels in the
subtraction.

This is not a preference. The delivered data contain 223 du Pont B
frames and zero Swope B frames, so a same-telescope Swope colour cannot
be formed at all.

It is also why the plate-scale error never reached the colours. Swope is
the telescope whose scale was wrongly assumed -- 0.430 arcsec/pixel in
reality against 0.230 assumed. This measurement discards Swope entirely.

NOT PHOTOMETRICALLY CALIBRATED
------------------------------
The quantity here is -2.5*log10(flux_B/flux_V), which differs from true
B-V by a constant per frame. Correct for the SHAPE of colour against
radius; not an absolute colour. Zero points are applied at step 17.

TARGET RADIUS
-------------
1.0 kpc, this project's defining scale. The upstream 5 kpc pipeline used
5.0 as its fiducial; that value appears here only as a secondary
comparison point against Kelsey et al. (2021), who work at 4 kpc.

THREE WAYS A COLOUR CAN BE UNDEFINED
------------------------------------
They are counted separately because they mean different things:
  band absent      - the frame failed a guard at that radius, so there
                     is nothing to divide
  non-positive flux- background-subtracted flux at or below zero. This
                     is the measurement floor, not a fault: where local
                     surface brightness sits at sky level the value is
                     noise, and noise is negative half the time
  both present     - a colour is defined

OUTPUT
------
  local_colour_vs_radius_{TAG}.csv   canonical, provenance in the name
  local_colour_vs_radius.csv         identical copy for downstream steps

The annulus tag is written into a column as well as the filename, so
provenance travels with the data rather than depending on file
modification times.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
ANNULUS_TAG = "ann20-30"          # ann10-15 | ann15-25 | ann20-30

REPO = Path(__file__).resolve().parent.parent
PHASE4 = REPO / "results" / "phase4_aperture"
COG_CSV = PHASE4 / f"curve_of_growth_{ANNULUS_TAG}.csv"

OUT_TAGGED = PHASE4 / f"local_colour_vs_radius_{ANNULUS_TAG}.csv"
OUT_COMPAT = PHASE4 / "local_colour_vs_radius.csv"

TARGET_RADIUS = 1.0
COMPARISON_RADII = [1.0, 2.0, 4.0, 5.0]
KEYS = ["object", "telescope", "filter"]


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def is_dupont(series: pd.Series) -> pd.Series:
    """Telescope test by substring, not equality.

    The catalogue writes 'duPont'; an earlier version of this script
    tested equality against 'dup' and would have selected nothing.
    """
    return series.astype(str).str.lower().str.contains("dup")


def band_sets(frame: pd.DataFrame) -> pd.Series:
    """object -> set of du Pont filters present."""
    sub = frame[is_dupont(frame["telescope"])]
    if len(sub) == 0:
        return pd.Series(dtype=object)
    return sub.groupby("object")["filter"].apply(set)


def main() -> None:
    if not COG_CSV.exists():
        raise SystemExit(f"Input not found:\n  {COG_CSV}\n"
                         f"Run 13_curve_of_growth.py first, or change "
                         f"ANNULUS_TAG at the top of this script.")

    print(f"Annulus setting : {ANNULUS_TAG}")
    print(f"Reading         : {COG_CSV.name}\n")

    df = pd.read_csv(COG_CSV)
    raw = df.copy()
    n_obj_raw = raw["object"].nunique()

    print("=" * 74)
    print("SAMPLE FUNNEL")
    print("=" * 74)
    print(f"  rows in                                 : {len(raw)}")
    print(f"  distinct objects                        : {n_obj_raw}")

    # Guard 1 -- the background annulus. A property of the FRAME, so it
    # is all radii or none.
    per_frame = df.drop_duplicates(subset=KEYS)
    n_bad = int((~per_frame["annulus_ok"]).sum())
    df = df[df["annulus_ok"]]
    print(f"  annulus guard                           : "
          f"{len(per_frame) - n_bad}/{len(per_frame)} frames kept "
          f"({n_bad} excluded)")

    # Guard 2 -- the aperture. A property of a SINGLE radius: a frame can
    # be sound at 1 kpc and off the chip at 9 kpc, so this is applied per
    # row and a frame may survive with a partial radius grid.
    n_ap_bad = int((~df["aperture_ok"]).sum())
    df = df[df["aperture_ok"]]
    print(f"  aperture guard                          : {n_ap_bad} rows "
          f"excluded (off-chip or non-finite pixels)")

    dup = df[is_dupont(df["telescope"])]
    if len(dup) == 0:
        raise RuntimeError("No du Pont rows selected -- cannot form a colour. "
                           f"Telescope values present: "
                           f"{sorted(df['telescope'].unique())}")

    bands_pre = band_sets(raw)
    bands_post = band_sets(df)

    def has_both(s):
        return {"B", "V"} <= s

    obj_both_pre = set(bands_pre[bands_pre.apply(has_both)].index)
    obj_both = set(bands_post[bands_post.apply(has_both)].index)
    obj_one_band = set(bands_post.index) - obj_both
    obj_no_dup = set(raw["object"]) - set(bands_pre.index)
    obj_lost = sorted(obj_both_pre - obj_both)

    z_by_obj = raw.groupby("object")["z"].first()

    print()
    print(f"  objects with du Pont B and V, pre-guard : {len(obj_both_pre)}")
    print(f"  objects lost to the guards              : {len(obj_lost)}")
    for o in obj_lost:
        print(f"      {o:<14} z = {z_by_obj[o]:.4f}")
    print(f"  objects with du Pont B and V, post-guard: {len(obj_both)}")
    print(f"  objects with no du Pont frames          : {len(obj_no_dup)}")
    print(f"  objects with du Pont in one band only   : {len(obj_one_band)}")
    for o in sorted(obj_one_band):
        print(f"      {o:<14} has only {sorted(bands_post[o])}")

    checksum = len(obj_both) + len(obj_lost) + len(obj_no_dup) + len(obj_one_band)
    print(f"  -> accounted for                        : {checksum} of {n_obj_raw}")
    if checksum != n_obj_raw:
        print(f"  [WARN] FUNNEL DOES NOT BALANCE -- "
              f"{n_obj_raw - checksum} object(s) unaccounted for. Every drop "
              f"must be traceable to a named reason; investigate before "
              f"trusting anything downstream.")

    # pivot_table defaults to aggfunc='mean' and would average duplicate
    # frames silently.
    dup_counts = dup.groupby(KEYS + ["radius_kpc"]).size()
    n_dupes = int((dup_counts > 1).sum())
    if n_dupes:
        print(f"\n  [WARN] {n_dupes} (object, telescope, filter, radius) "
              f"combinations appear more than once. pivot_table will AVERAGE "
              f"them silently.")
        print(dup_counts[dup_counts > 1].head(10).to_string())
    else:
        print(f"  duplicate check                         : none")
    print("=" * 74 + "\n")

    # ----------------------------------------------------------------
    pivot = dup.pivot_table(index=["object", "z", "radius_kpc"],
                            columns="filter", values="flux_bkgsub").reset_index()

    for band in ("B", "V"):
        if band not in pivot.columns:
            raise RuntimeError(f"No {band} column after pivoting -- the "
                               f"curve-of-growth file lacks {band} frames for "
                               f"every du Pont object.")

    missing = pivot["B"].isna() | pivot["V"].isna()
    nonpos = (~missing) & ((pivot["B"] <= 0) | (pivot["V"] <= 0))
    valid = (~missing) & (~nonpos)

    print(f"colour defined for {int(valid.sum())} / {len(pivot)} "
          f"(object, radius) points")
    print(f"  undefined, band absent       : {int(missing.sum())}")
    print(f"  undefined, flux non-positive : {int(nonpos.sum())}")

    with np.errstate(invalid="ignore", divide="ignore"):
        pivot["instrumental_B_minus_V"] = np.where(
            valid,
            -2.5 * np.log10(pivot["B"].to_numpy() / pivot["V"].to_numpy()),
            np.nan)

    pivot["annulus_tag"] = ANNULUS_TAG

    PHASE4.mkdir(parents=True, exist_ok=True)
    print()
    for path in (OUT_TAGGED, OUT_COMPAT):
        backup(path)
        pivot.to_csv(path, index=False)

    print(f"\nWrote {len(pivot)} rows ({pivot['object'].nunique()} objects)")
    print(f"  canonical : {OUT_TAGGED.name}")
    print(f"  compat    : {OUT_COMPAT.name}")

    print("\n" + "=" * 74)
    print("INSTRUMENTAL COLOUR BY RADIUS")
    print("=" * 74)
    print(f"  {'radius':>8}  {'objects':>8}  {'median':>9}  {'16th':>9}  {'84th':>9}")
    for r in COMPARISON_RADII:
        at_r = pivot[np.isclose(pivot["radius_kpc"], r)]
        col = at_r["instrumental_B_minus_V"].dropna()
        if len(col) == 0:
            print(f"  {r:8.2f}  {'-':>8}  {'-':>9}")
            continue
        marker = "  <- target" if np.isclose(r, TARGET_RADIUS) else ""
        print(f"  {r:8.2f}  {len(col):8d}  {col.median():9.4f}  "
              f"{col.quantile(0.16):9.4f}  {col.quantile(0.84):9.4f}{marker}")

    print("\nNOTE: instrumental colour. Zero points are one additive constant "
          "\n      per frame and have NOT been applied. Absolute values here "
          "\n      are meaningless; only the radial trend is interpretable.")


if __name__ == "__main__":
    main()