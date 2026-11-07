"""
15_apply_zero_points.py

Convert instrumental flux at the 1.0 kpc aperture into calibrated
magnitudes and a calibrated local B-V colour, using the supplied
du Pont zero-point files.

    mag   = ZP - 2.5*log10(flux)
    B-V   = mag_B - mag_V
          = (ZP_B - ZP_V) - 2.5*log10(flux_B/flux_V)
          = (ZP_B - ZP_V) + instrumental_(B-V)

The calibrated colour is therefore step 14's instrumental colour shifted
by one additive constant per object. Any analysis performed WITHIN an
object across radii is unaffected by this step entirely, which is why
the gradient and selection tests remain valid on instrumental colours.

Zero-point file format, space-delimited, no header:

    <object>  <zero_point_mag>  <zero_point_err>  <n_ref_stars>

DU PONT ONLY
------------
Consistent with step 14. Swope has 103 V frames and zero B frames in
this data set, so a same-telescope Swope colour does not exist.

THE LARGEST CUT IN THE PIPELINE
-------------------------------
The supplied calibration files were derived for CSP-II and do not cover
CSP-I. Objects from 2004-2009 have no zero point and cannot be
calibrated at all. The break falls at 2010, the gap between the two CSP
campaigns.

The consequence is that the published catalogue is a CSP-II sample.
CSP-I and CSP-II differ in target selection and redshift reach, so this
reshapes the redshift distribution and host population rather than
thinning the sample evenly. It must be stated in the paper's sample
description, and it bears on comparability with Kelsey et al. (2021).

If CSP-I zero points are later supplied, add the filenames to ZP_FILES
below; the structure takes a list per (telescope, filter) and
concatenates, so no other change is needed.

INVALID ZERO POINTS
-------------------
A zero point built from no reference stars is not a measurement.
LSQ12gef carries zp = inf, zp_err = inf and n_ref_stars = 0 in both
du Pont files. Left unscreened this produces mag = inf, then
B-V = inf - inf = NaN, and the object disappears through NaN arithmetic
rather than by any stated rule. Zero points are therefore rejected
explicitly if the value or its error is non-finite, or if n_ref_stars is
below MIN_REF_STARS, and every rejection is logged.

UNCERTAINTY: A LOWER BOUND, KNOWN TO BE
---------------------------------------
B_minus_V_err propagates the two zero-point uncertainties in quadrature
and nothing else. It excludes photon noise on the aperture flux,
background-estimation uncertainty, flat-field error, and the
flux-dependent annulus systematic. It must be described as a lower
bound.

Closing this gap does not require detector gain or read noise:
injection-recovery photometry after Mowla et al. (2022) section 3.1 --
injecting synthetic sources of known flux near each target and
recovering them through this same pipeline -- measures the total
uncertainty empirically.

OUTPUT
------
  calibrated_colour_1kpc.csv             the catalogue
  calibrated_colour_1kpc_exclusions.csv  every object dropped, with a reason

Existing files are renamed to *.bak_YYYYmmdd_HHMMSS, never deleted.
"""

from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Must match the tag used in steps 13 and 14.
ANNULUS_TAG = "ann20-30"

REPO = Path(__file__).resolve().parent.parent
ZP_DIR = REPO / "inputs"
PHASE4 = REPO / "results" / "phase4_aperture"
OUT_DIR = REPO / "results" / "phase5_calibration"

COG_CSV = PHASE4 / f"curve_of_growth_{ANNULUS_TAG}.csv"
OUT_PATH = OUT_DIR / "calibrated_colour_1kpc.csv"
EXCL_PATH = OUT_DIR / "calibrated_colour_1kpc_exclusions.csv"

# A LIST per (telescope, filter), concatenated in order. Add CSP-I files
# here if they are ever supplied; nothing else needs to change.
ZP_FILES = {
    ("dup", "B"): ["B_ZP_dup.dat"],
    ("dup", "V"): ["V_ZP_dup.dat"],
}

FIDUCIAL_RADIUS_KPC = 1.0
KEYS = ["object", "telescope", "filter"]

# Set to 1 so only the unambiguous case is rejected. The full
# distribution of n_ref_stars is printed so a stricter rule can be
# chosen later on stated grounds rather than by whichever value
# flatters the result.
MIN_REF_STARS = 1

CSP1_LAST_YEAR = 2009


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def is_dupont(series: pd.Series) -> pd.Series:
    """Substring test. The catalogue writes 'duPont', not 'dup'."""
    return series.astype(str).str.lower().str.contains("dup")


def object_year(name) -> float:
    """
    Discovery year from the designation. Handles four-digit (SN2012fr)
    and two-digit (SN04ef, ASAS14ad, CSP13aam) conventions; the
    four-digit alternative is tried first so SN2011iy does not parse
    as 20.
    """
    m = re.search(r"(\d{4}|\d{2})", str(name))
    if not m:
        return np.nan
    v = int(m.group(1))
    return float(v if v > 1000 else 2000 + v)


def epoch(year) -> str:
    if not np.isfinite(year):
        return "unknown"
    return "CSP-I" if year <= CSP1_LAST_YEAR else "CSP-II"


def load_zp(tel: str, filt: str):
    frames = []
    for fname in ZP_FILES.get((tel, filt), []):
        path = ZP_DIR / fname
        if not path.exists():
            print(f"  [warn] not found, skipping: {path}")
            continue
        t = pd.read_csv(path, sep=r"\s+", header=None,
                        names=["object", "zp", "zp_err", "n_ref_stars"])
        t["zp_source"] = fname
        frames.append(t)
        print(f"  loaded {fname:16s} {len(t):4d} objects")
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    n_dup = int(out["object"].duplicated().sum())
    if n_dup:
        print(f"  [warn] {n_dup} objects appear in more than one {tel} {filt} "
              f"file; keeping the first occurrence.")
        out = out.drop_duplicates(subset="object", keep="first")
    return out


def main() -> None:
    excl = []

    def drop(objects, stage, reason):
        for o in sorted(set(objects)):
            excl.append({"object": o, "stage": stage, "reason": reason})

    print("=" * 74)
    print("ZERO POINTS")
    print("=" * 74)
    zp_b = load_zp("dup", "B")
    zp_v = load_zp("dup", "V")
    if zp_b is None or zp_v is None:
        raise RuntimeError(f"Need both du Pont B and V zero-point files. "
                           f"Check ZP_DIR:\n  {ZP_DIR}")

    def screen(t, label):
        bad = (~np.isfinite(t["zp"]) | ~np.isfinite(t["zp_err"])
               | (t["n_ref_stars"] < MIN_REF_STARS))
        if bad.any():
            print(f"\n  {label}: rejecting {int(bad.sum())} zero point(s)")
            for _, r in t[bad].iterrows():
                why = []
                if not np.isfinite(r["zp"]):
                    why.append("zp non-finite")
                if not np.isfinite(r["zp_err"]):
                    why.append("zp_err non-finite")
                if r["n_ref_stars"] < MIN_REF_STARS:
                    why.append(f"n_ref_stars={r['n_ref_stars']:g}")
                print(f"      {r['object']:<14} {', '.join(why)}")
                drop([r["object"]], "zero point",
                     f"{label} invalid: {', '.join(why)}")
        return t[~bad]

    zp_b = screen(zp_b, "B")
    zp_v = screen(zp_v, "V")

    print(f"\n  n_ref_stars (B, after screening): "
          f"{zp_b['n_ref_stars'].describe().round(1).to_dict()}")
    print(f"  zp_err (B): median {zp_b['zp_err'].median():.4f}, "
          f"max {zp_b['zp_err'].max():.4f} mag")

    # ----------------------------------------------------------------
    if not COG_CSV.exists():
        raise SystemExit(f"\nInput not found:\n  {COG_CSV}\n"
                         f"Run 13_curve_of_growth.py first, or change "
                         f"ANNULUS_TAG.")

    print("\n" + "=" * 74)
    print(f"PHOTOMETRY  (annulus {ANNULUS_TAG}, radius "
          f"{FIDUCIAL_RADIUS_KPC:g} kpc)")
    print("=" * 74)
    cog = pd.read_csv(COG_CSV)
    all_objects = set(cog["object"])
    print(f"  rows in                          : {len(cog)}")
    print(f"  distinct objects                 : {len(all_objects)}")

    per_frame = cog.drop_duplicates(subset=KEYS)
    n_bad = int((~per_frame["annulus_ok"]).sum())
    cog = cog[cog["annulus_ok"]]
    print(f"  annulus guard                    : "
          f"{len(per_frame) - n_bad}/{len(per_frame)} frames kept "
          f"({n_bad} excluded)")

    n_ap_bad = int((~cog["aperture_ok"]).sum())
    cog = cog[cog["aperture_ok"]]
    print(f"  aperture guard                   : {n_ap_bad} rows excluded")

    fid = cog[np.isclose(cog["radius_kpc"], FIDUCIAL_RADIUS_KPC)
              & is_dupont(cog["telescope"])]
    if len(fid) == 0:
        raise RuntimeError(
            f"No du Pont rows at {FIDUCIAL_RADIUS_KPC} kpc. Telescope values "
            f"present: {sorted(cog['telescope'].unique())}")

    dupes = fid.groupby(["object", "filter"]).size()
    if (dupes > 1).any():
        print(f"\n  [WARN] {int((dupes > 1).sum())} (object, filter) pairs "
              f"appear more than once; the merge below would duplicate rows.")
        print(dupes[dupes > 1].head(10).to_string())
    else:
        print(f"  duplicate check                  : none")

    flux_b = (fid[fid["filter"] == "B"][["object", "z", "flux_bkgsub"]]
              .rename(columns={"flux_bkgsub": "flux_B"}))
    flux_v = (fid[fid["filter"] == "V"][["object", "flux_bkgsub"]]
              .rename(columns={"flux_bkgsub": "flux_V"}))

    drop(set(flux_b["object"]) - set(flux_v["object"]), "photometry",
         "du Pont B only at the fiducial radius")
    drop(set(flux_v["object"]) - set(flux_b["object"]), "photometry",
         "du Pont V only at the fiducial radius")
    drop(all_objects - set(flux_b["object"]) - set(flux_v["object"]),
         "photometry", "no du Pont flux at the fiducial radius")

    flux = flux_b.merge(flux_v, on="object", how="inner")
    print(f"  objects with du Pont B and V     : {len(flux)}")

    nonpos = (flux["flux_B"] <= 0) | (flux["flux_V"] <= 0)
    drop(flux.loc[nonpos, "object"], "photometry",
         "non-positive flux at the fiducial radius")
    print(f"  of those, non-positive flux      : {int(nonpos.sum())}")

    usable = flux[~nonpos].copy()

    # ----------------------------------------------------------------
    usable["year"] = usable["object"].map(object_year)
    usable["epoch"] = usable["year"].map(epoch)
    have_zp = set(zp_b["object"]) & set(zp_v["object"])
    usable["has_zp"] = usable["object"].isin(have_zp)

    print("\n" + "=" * 74)
    print("ZERO-POINT COVERAGE BY SURVEY EPOCH")
    print("=" * 74)
    tab = pd.crosstab(usable["epoch"], usable["has_zp"])
    for col in (False, True):
        if col not in tab.columns:
            tab[col] = 0
    tab = tab[[False, True]]
    tab.columns = ["no zero point", "has zero point"]
    print(tab.to_string())

    missing = usable[~usable["has_zp"]]
    drop(missing["object"], "zero point",
         "no du Pont zero point available for this object")

    n_csp1 = int((missing["epoch"] == "CSP-I").sum())
    if n_csp1:
        print(f"""
  {n_csp1} CSP-I objects are excluded here for want of a zero point.
  This is the largest single cut in the pipeline and it is by SURVEY
  EPOCH, not by data quality.

  The resulting catalogue is a CSP-II sample. CSP-I and CSP-II differ in
  target selection and redshift reach, so this reshapes the redshift
  distribution rather than thinning the sample evenly. It must be stated
  in the paper's sample description.

  SUPERVISOR QUESTION: do du Pont zero points exist for the CSP-I
  objects (2004-2009)? If they are supplied, add the filenames to
  ZP_FILES at the top of this script; the calibrated sample roughly
  doubles.""")

    late = missing[missing["epoch"] != "CSP-I"]
    if len(late):
        print(f"\n  Also lacking a zero point, outside CSP-I: "
              f"{sorted(late['object'])}")

    # ----------------------------------------------------------------
    merged = usable[usable["has_zp"]].merge(
        zp_b[["object", "zp", "zp_err", "n_ref_stars"]].rename(
            columns={"zp": "zp_B", "zp_err": "zp_err_B",
                     "n_ref_stars": "n_ref_B"}), on="object", how="inner")
    merged = merged.merge(
        zp_v[["object", "zp", "zp_err", "n_ref_stars"]].rename(
            columns={"zp": "zp_V", "zp_err": "zp_err_V",
                     "n_ref_stars": "n_ref_V"}), on="object", how="inner")

    merged["mag_B"] = merged["zp_B"] - 2.5 * np.log10(merged["flux_B"])
    merged["mag_V"] = merged["zp_V"] - 2.5 * np.log10(merged["flux_V"])
    merged["B_minus_V"] = merged["mag_B"] - merged["mag_V"]
    merged["B_minus_V_err"] = np.sqrt(merged["zp_err_B"] ** 2
                                      + merged["zp_err_V"] ** 2)
    merged["annulus_tag"] = ANNULUS_TAG
    merged["aperture_kpc"] = FIDUCIAL_RADIUS_KPC

    out_cols = ["object", "z", "flux_B", "flux_V", "mag_B", "mag_V",
                "B_minus_V", "B_minus_V_err", "aperture_kpc", "annulus_tag",
                "epoch", "zp_err_B", "zp_err_V", "n_ref_B", "n_ref_V"]
    result = merged[out_cols].sort_values("object").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    backup(OUT_PATH)
    result.to_csv(OUT_PATH, index=False)

    excl_df = pd.DataFrame(excl).drop_duplicates(subset=["object", "stage"])
    backup(EXCL_PATH)
    excl_df.to_csv(EXCL_PATH, index=False)

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  calibrated objects : {len(result)}")
    print(f"  by epoch           : {result['epoch'].value_counts().to_dict()}")
    print(f"  written to         : {OUT_PATH.name}")
    print(f"  exclusions logged  : {len(excl_df)} -> {EXCL_PATH.name}")
    print()
    print(f"  median B-V           : {result['B_minus_V'].median():.4f} mag")
    print(f"  16th-84th percentile : "
          f"{result['B_minus_V'].quantile(0.16):.4f} - "
          f"{result['B_minus_V'].quantile(0.84):.4f} mag")
    print(f"  full range           : {result['B_minus_V'].min():.4f} - "
          f"{result['B_minus_V'].max():.4f} mag")
    print()
    print(f"  median B-V uncertainty : "
          f"{result['B_minus_V_err'].median():.4f} mag")
    print(f"  worst                  : {result['B_minus_V_err'].max():.4f} mag "
          f"({result.loc[result['B_minus_V_err'].idxmax(), 'object']})")
    print("""
  This uncertainty is the two zero-point terms in quadrature and NOTHING
  ELSE. It omits photon noise, background-estimation uncertainty,
  flat-field error and the flux-dependent annulus systematic. Report it
  as a LOWER BOUND.

  NOTE: this is the observed colour. Galactic extinction has NOT been
  applied. Next: 16_flag_unreliable_colours.py, then
  17_apply_galactic_extinction.py.""")
    print("=" * 74)


if __name__ == "__main__":
    main()