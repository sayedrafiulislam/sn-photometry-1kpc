"""
16_flag_unreliable_colours.py

Flag catalogue entries whose colour cannot be trusted, using three
automated criteria plus one named manual exclusion.

Nothing is dropped. Flags are written; downstream scripts decide.

THE THREE CRITERIA
------------------

A. PHOTOMETRIC -- background fraction at the fiducial radius

       bkg_frac = (bkg_per_pixel * aperture area) / raw enclosed counts

   A ratio, so exposure time, units and survey campaign all cancel.
   Directly interpretable: 0.10 means a tenth of what was measured was
   sky rather than galaxy.

   Tested on |bkg_frac|, not bkg_frac. A large NEGATIVE background
   fraction is as wrong as a large positive one: it means flux is being
   added rather than removed. Small negative values are legitimate,
   since these stacks have already had a sky level subtracted and the
   annulus median can fall below zero.

B. CALIBRATION -- the object's own quoted colour uncertainty

   B_minus_V_err is the two zero-point errors in quadrature. An object
   whose stated uncertainty exceeds MAX_COLOUR_ERR cannot constrain the
   population it belongs to.

   Chosen over a zero-point-difference outlier test because it is not
   circular (it never inspects the colour VALUE, only how well that
   value was determined), self-declared (the calibration states its own
   reliability rather than having it inferred from other objects), and
   conservative (B_minus_V_err omits photon noise, background
   uncertainty and flat-field error, so it is a lower bound; if even the
   lower bound is too large, the truth is worse).

C. SIGNAL -- minimum enclosed flux in either band

   Criteria A and B test how well a measurement was made. Neither tests
   whether there was enough signal for the ratio B/V to mean anything.

   THE THRESHOLD MUST BE SCALED TO THE APERTURE. The 5 kpc pipeline used
   1000 counts. Enclosed flux scales as r^2 over this range -- the
   measured growth-curve slope at 1 kpc is 1.98 -- so a 1 kpc aperture
   encloses about 25 times less light. Carried across unchanged, 1000
   counts sits just below the 1 kpc sample median of 2050 and removes 41
   per cent of the catalogue: a threshold on scale, not on data quality.
   This is the same error the project has caught repeatedly, a threshold
   in fixed units applied across a change of scale.

   Scaled by the aperture ratio: 1000 / (5/1)^2 = 40 counts. The value
   is derived from the earlier threshold and the measured slope, not
   chosen by inspecting which objects it removes.

   An absolute count threshold is defensible for THIS catalogue in a way
   it was not for the 5 kpc one. No CSP-I zero points exist, so the
   calibrated sample is CSP-II only and du Pont only: homogeneous in
   telescope, campaign and depth.

WHY ZP_B - ZP_V IS REPORTED BUT NOT CUT
---------------------------------------
An earlier version excluded objects whose ZP_B - ZP_V deviated by more
than 3 MAD, on the reasoning that a zero point depends on the instrument
rather than the galaxy. That reasoning was tested and does not hold: the
distribution is intrinsically broad, with 6.8 per cent beyond 3 MAD
against 0.3 per cent for a Gaussian, so the cut slices into the body
rather than trimming a tail. Validation against the instrumental
colours, which never touch a zero point, showed that of 8 flagged
objects only 3 had a genuine calibration failure; one had an
instrumental colour of +3.1 sigma and a normal calibrated colour, so the
zero point was CORRECTING it and flagging it would have discarded real
data.

It remains a real observation about the calibration files and is
reported as a diagnostic.

WHY PLAUSIBILITY IS A CHECK AND NOT A CRITERION
-----------------------------------------------
Integrated stellar populations span roughly B-V = 0.0 to 1.3 (Fukugita,
Shimasaku & Ichikawa 1995, PASP 107, 945), and reddening extends the red
end. Excluding objects because their colour falls outside that range
would be circular: it removes measurements for giving an answer we
dislike, which is not a property of the measurement.

flag_implausible_bv is therefore computed and reported, never used to
exclude. Its value is as an independent check on criterion C: the two
are logically unrelated -- one tests the measurement, the other the
result -- so agreement between them is evidence that C removes the right
objects for the right reason rather than by coincidence.

THE LIMIT OF AN ENCLOSED-FLUX CRITERION
---------------------------------------
Criterion C tests the SIZE of the enclosed signal, not whether that
signal is a source. Seventy-two pixels of sky noise sum to a number, and
that number can exceed any fixed floor.

LSQ14foj demonstrates this. It passes every automated criterion with
B-V = -1.01 and B brighter than V, which no stellar population produces.
Visual inspection shows the aperture falling on blank sky roughly 30
pixels -- about 6 kpc at z = 0.046 -- from its host, whose light does
not reach the supernova position. Median pixel values inside the
aperture are 2.6 counts in B and 0.1 in V against a frame median of 0.0.
Both "fluxes" are summed noise, and the ratio of two noise measurements
is arbitrary.

Separating a faint source from accumulated noise requires a per-aperture
noise estimate, which this pipeline does not have. Obtaining one does
not need detector gain or read noise: injection-recovery photometry
after Mowla et al. (2022) section 3.1 -- injecting synthetic sources of
known flux near each target and recovering them through this same
pipeline -- measures it empirically. That is future work.

Until then the object is excluded BY NAME with a stated physical reason,
rather than by raising a threshold until it disappears. A named
exclusion is auditable; a tuned threshold is not.

This also implies a limitation that must be stated in the paper:
LSQ14foj was identified because its colour was absurd. Objects whose
apertures fall on faint sky but whose noise happens to produce an
ordinary-looking colour would not be caught by any criterion here.

ON CHOOSING THE THRESHOLDS
--------------------------
All are configuration below, and the diagnostic distributions are
printed BEFORE the cuts are applied. Fix the values on stated grounds,
record the reasoning, THEN look at what they cost. Selecting the
threshold that produces the most agreeable catalogue is how a null
result becomes an apparent discovery.

OUTPUT
------
  calibrated_colour_1kpc_flagged.csv

Columns added:
  bkg_frac_B, bkg_frac_V   background as a fraction of enclosed flux
  zp_diff, zp_diff_n_mad   ZP_B - ZP_V and its deviation -- DIAGNOSTIC
  flag_high_bkg            fails criterion A
  flag_large_err           fails criterion B
  flag_low_flux            fails criterion C
  flag_manual              excluded by name, reason in MANUAL_EXCLUSIONS
  flag_zp_outlier          beyond 3 MAD in ZP_B - ZP_V -- REPORTED, NOT CUT
  flag_implausible_bv      colour outside the plausible range -- CHECK ONLY
  flag_exclude             fails A, B, C or manual. Filter on THIS.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Configuration -- fix on stated grounds BEFORE examining the effect
# --------------------------------------------------------------------
# Criterion A: reject if background exceeds this fraction of enclosed
# flux in either band.
MAX_BKG_FRAC = 0.10

# Criterion B: reject if the quoted colour uncertainty exceeds this.
# The retained 16th-84th spread is about 0.5 mag, so this is half the
# spread the measurement is meant to constrain.
MAX_COLOUR_ERR = 0.25

# Criterion C: minimum enclosed flux in either band, in counts.
# 1000 counts at 5 kpc scaled by the aperture ratio (5/1)^2 = 25.
# See the docstring: this MUST be scaled, not inherited unchanged.
MIN_FLUX_COUNTS = 40.0

# Excluded by name, with a stated physical reason. Used only where an
# automated criterion cannot express the failure. Each entry must be
# justified in the docstring above.
MANUAL_EXCLUSIONS = {
    "LSQ14foj": ("no detectable host flux at the SN position; aperture "
                 "falls on blank sky ~6 kpc from the host"),
}

# Reported as a cross-check on criterion C, never used to exclude.
PLAUSIBLE_BV_RANGE = (0.0, 1.5)

# Reported only, never used to exclude.
REPORT_ZP_DIFF_N_MAD = 3.0

ANNULUS_TAG = "ann20-30"
FIDUCIAL_RADIUS_KPC = 1.0

# Walk up until the results directory is found, so this works wherever
# the script sits.
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
CAL_DIR = REPO / "results" / "phase5_calibration"
COG_DIR = REPO / "results" / "phase4_aperture"
ZP_DIR = REPO / "inputs"

IN_PATH = CAL_DIR / "calibrated_colour_1kpc.csv"
OUT_PATH = CAL_DIR / "calibrated_colour_1kpc_flagged.csv"
COG_CSV = COG_DIR / f"curve_of_growth_{ANNULUS_TAG}.csv"
ZP_B_FILE = ZP_DIR / "B_ZP_dup.dat"
ZP_V_FILE = ZP_DIR / "V_ZP_dup.dat"


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def is_dupont(series: pd.Series) -> pd.Series:
    """Substring test. The catalogue writes 'duPont', not 'dup'."""
    return series.astype(str).str.lower().str.contains("dup")


def mad_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def main() -> None:
    for p in (IN_PATH, COG_CSV, ZP_B_FILE, ZP_V_FILE):
        if not p.exists():
            raise SystemExit(f"\nInput not found:\n  {p}\n")

    df = pd.read_csv(IN_PATH)
    print(f"Catalogue in : {IN_PATH.name}  ({len(df)} objects)")
    if "annulus_tag" in df.columns:
        tags = list(df["annulus_tag"].unique())
        print(f"Annulus tag  : {tags}")
        if len(tags) == 1 and tags[0] != ANNULUS_TAG:
            print(f"  [WARN] catalogue built with {tags[0]} but this script is "
                  f"configured for {ANNULUS_TAG}. The background fractions "
                  f"below will not correspond to the catalogue fluxes.")

    # ----------------------------------------------------------------
    # Criterion A -- background fraction at the fiducial radius
    # ----------------------------------------------------------------
    cog = pd.read_csv(COG_CSV)
    fid = cog[np.isclose(cog["radius_kpc"], FIDUCIAL_RADIUS_KPC)
              & is_dupont(cog["telescope"])].copy()
    if len(fid) == 0:
        raise SystemExit(f"No du Pont rows at {FIDUCIAL_RADIUS_KPC} kpc. "
                         f"Telescope values: {sorted(cog['telescope'].unique())}")

    fid["area"] = np.pi * fid["radius_pix"] ** 2
    fid["raw"] = fid["flux_bkgsub"] + fid["bkg_per_pixel"] * fid["area"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fid["bkg_frac"] = (fid["bkg_per_pixel"] * fid["area"]) / fid["raw"]

    for band in ("B", "V"):
        s = (fid[fid["filter"] == band]
             .set_index("object")["bkg_frac"].rename(f"bkg_frac_{band}"))
        df = df.merge(s, left_on="object", right_index=True, how="left")

    # ----------------------------------------------------------------
    # Diagnostic -- ZP_B - ZP_V
    # ----------------------------------------------------------------
    def read_zp(path):
        return pd.read_csv(path, sep=r"\s+", header=None,
                           names=["object", "zp", "zp_err", "n_ref"])

    zb, zv = read_zp(ZP_B_FILE), read_zp(ZP_V_FILE)
    zpd = (zb[["object", "zp"]].rename(columns={"zp": "zp_B"})
           .merge(zv[["object", "zp"]].rename(columns={"zp": "zp_V"}),
                  on="object", how="inner"))
    zpd = zpd[np.isfinite(zpd["zp_B"]) & np.isfinite(zpd["zp_V"])]
    zpd["zp_diff"] = zpd["zp_B"] - zpd["zp_V"]

    # Reference distribution over ALL valid du Pont objects, not only
    # those in the catalogue, so the spread does not shift as cuts are
    # applied.
    med = float(np.median(zpd["zp_diff"]))
    mad = mad_sigma(zpd["zp_diff"])
    zpd["zp_diff_n_mad"] = (zpd["zp_diff"] - med) / mad

    df = df.merge(zpd[["object", "zp_diff", "zp_diff_n_mad"]],
                  on="object", how="left")

    # ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("DIAGNOSTIC DISTRIBUTIONS  (before any cut)")
    print("=" * 74)
    print(f"  background fraction at {FIDUCIAL_RADIUS_KPC:g} kpc:")
    for band in ("B", "V"):
        s = df[f"bkg_frac_{band}"].dropna()
        print(f"    {band}: median {s.median():.4f}   75th {s.quantile(.75):.4f}"
              f"   90th {s.quantile(.90):.4f}   max {s.max():.4f}  (n={len(s)})")

    print(f"\n  enclosed flux at {FIDUCIAL_RADIUS_KPC:g} kpc (counts):")
    for band in ("B", "V"):
        s = df[f"flux_{band}"].dropna()
        print(f"    {band}: median {s.median():10.1f}   "
              f"10th {s.quantile(.10):9.1f}   min {s.min():9.1f}  (n={len(s)})")

    print(f"\n  ZP_B - ZP_V across {len(zpd)} valid du Pont objects:")
    print(f"    median {med:+.4f}   MAD-sigma {mad:.4f}   "
          f"range {zpd['zp_diff'].min():+.3f} to {zpd['zp_diff'].max():+.3f}")
    s = df["zp_diff_n_mad"].abs().dropna()
    print(f"    |deviation| in catalogue: median {s.median():.2f} MAD, "
          f"90th {s.quantile(.90):.2f}, max {s.max():.2f}")

    e = df["B_minus_V_err"].dropna()
    print(f"\n  quoted colour uncertainty (zero-point terms only):")
    print(f"    median {e.median():.4f}   90th {e.quantile(.90):.4f}   "
          f"max {e.max():.4f} mag")

    # ----------------------------------------------------------------
    # Apply
    # ----------------------------------------------------------------
    df["flag_high_bkg"] = ((df["bkg_frac_B"].abs() > MAX_BKG_FRAC)
                           | (df["bkg_frac_V"].abs() > MAX_BKG_FRAC)
                           | df["bkg_frac_B"].isna()
                           | df["bkg_frac_V"].isna())

    df["flag_large_err"] = (df["B_minus_V_err"] > MAX_COLOUR_ERR).fillna(True)

    df["flag_low_flux"] = ((df["flux_B"] < MIN_FLUX_COUNTS)
                           | (df["flux_V"] < MIN_FLUX_COUNTS))

    df["flag_manual"] = df["object"].isin(MANUAL_EXCLUSIONS)

    df["flag_zp_outlier"] = (df["zp_diff_n_mad"].abs()
                             > REPORT_ZP_DIFF_N_MAD).fillna(False)

    lo, hi = PLAUSIBLE_BV_RANGE
    df["flag_implausible_bv"] = ((df["B_minus_V"] < lo)
                                 | (df["B_minus_V"] > hi)).fillna(True)

    df["flag_exclude"] = (df["flag_high_bkg"] | df["flag_large_err"]
                          | df["flag_low_flux"] | df["flag_manual"])

    print("\n" + "=" * 74)
    print("FLAGS APPLIED")
    print("=" * 74)
    print(f"  criterion A, |background fraction| > {MAX_BKG_FRAC:.2f}   : "
          f"{int(df['flag_high_bkg'].sum()):3d} objects")
    print(f"  criterion B, colour error > {MAX_COLOUR_ERR:.2f} mag        : "
          f"{int(df['flag_large_err'].sum()):3d} objects")
    print(f"  criterion C, flux < {MIN_FLUX_COUNTS:g} counts either band   : "
          f"{int(df['flag_low_flux'].sum()):3d} objects")
    print(f"  manual exclusion, by name                 : "
          f"{int(df['flag_manual'].sum()):3d} objects")
    for obj, why in MANUAL_EXCLUSIONS.items():
        present = "" if obj in set(df["object"]) else "  [not in catalogue]"
        print(f"      {obj}: {why}{present}")
    print(f"  excluded by any of the above              : "
          f"{int(df['flag_exclude'].sum()):3d} objects")
    print(f"  RETAINED                                  : "
          f"{int((~df['flag_exclude']).sum()):3d} objects")

    print("\n  overlap between automated criteria:")
    for a, b in (("flag_high_bkg", "flag_large_err"),
                 ("flag_high_bkg", "flag_low_flux"),
                 ("flag_large_err", "flag_low_flux")):
        print(f"    {a[5:]:14s} & {b[5:]:14s} : "
              f"{int((df[a] & df[b]).sum()):3d}")

    print(f"\n  ZP_B - ZP_V beyond {REPORT_ZP_DIFF_N_MAD:.0f} MAD "
          f"(DIAGNOSTIC ONLY) : {int(df['flag_zp_outlier'].sum()):3d} objects")
    zpo = df[df["flag_zp_outlier"]]
    if len(zpo):
        print("    " + ", ".join(f"{r.object}({r.zp_diff_n_mad:+.1f})"
                                 for _, r in zpo.iterrows()))
        also = int(zpo["flag_exclude"].sum())
        print(f"    of these, {also} excluded anyway; {len(zpo) - also} retained.")

    for name, col in [("high background", "flag_high_bkg"),
                      ("large quoted uncertainty", "flag_large_err"),
                      ("insufficient flux", "flag_low_flux")]:
        sub = df[df[col]]
        if len(sub):
            print(f"\n  --- {name} ---")
            cols = ["object", "B_minus_V", "B_minus_V_err", "flux_B", "flux_V",
                    "bkg_frac_B", "bkg_frac_V"]
            print(sub[cols].sort_values("object").round(4).to_string(index=False))

    # ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("EFFECT ON THE SAMPLE")
    print("=" * 74)
    kept = df[~df["flag_exclude"]]
    for label, s in [("all", df), ("retained", kept)]:
        c = s["B_minus_V"].dropna()
        if len(c):
            print(f"  {label:9s} n={len(c):3d}  median {c.median():.4f}  "
                  f"16-84 {c.quantile(.16):.4f} - {c.quantile(.84):.4f}  "
                  f"min {c.min():+.4f}  max {c.max():+.4f}")

    # ----------------------------------------------------------------
    # Cross-check: criterion C against physical plausibility. The two
    # are logically independent -- one tests the measurement, the other
    # the result -- so agreement is evidence that C removes the right
    # objects for the right reason rather than by coincidence.
    # ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("CROSS-CHECK: CRITERION C AGAINST PHYSICAL PLAUSIBILITY")
    print("=" * 74)
    print(pd.crosstab(df["flag_low_flux"].rename("flux below floor"),
                      df["flag_implausible_bv"].rename(
                          f"B-V outside {lo:g}-{hi:g}")).to_string())

    disagree = df[df["flag_implausible_bv"] & ~df["flag_exclude"]]
    print(f"\n  implausible colours surviving ALL criteria : {len(disagree)}")
    if len(disagree):
        cols = ["object", "z", "B_minus_V", "B_minus_V_err", "flux_B", "flux_V",
                "bkg_frac_B", "bkg_frac_V"]
        print(disagree[cols].sort_values("B_minus_V").round(4)
              .to_string(index=False))
        print("\n    Not explained by low flux, high background or a large")
        print("    zero-point error. Investigate individually and exclude by")
        print("    name with a stated reason, or state them in the paper.")
        print("    Do NOT adjust a threshold until they disappear.")
    else:
        print("    None. Every implausible colour is independently explained")
        print("    by a failure of the measurement.")

    print()
    backup(OUT_PATH)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} objects -> {OUT_PATH.name}")

    print(f"""
Filter downstream on `flag_exclude`.

Thresholds: background fraction {MAX_BKG_FRAC:g}, colour error
{MAX_COLOUR_ERR:g} mag, minimum flux {MIN_FLUX_COUNTS:g} counts, plus
{len(MANUAL_EXCLUSIONS)} named exclusion(s). Record the values agreed with
your supervisor and the reasoning, then report what they cost -- not the
reverse.

flag_implausible_bv and flag_zp_outlier are diagnostics and do NOT
affect flag_exclude.""")


if __name__ == "__main__":
    main()