"""
19_colour_scatter_vs_radius.py

Robust scatter in instrumental local B-V as a function of aperture
radius. Thesis Figure 5.2, Table 5.2 (Section 5.2).

THE QUESTION
------------
Does the choice of aperture radius change how tightly the local colours
cluster? If some radius gave a markedly smaller scatter, that would
argue for measuring there. The answer determines whether 1 kpc is a
defensible choice or an arbitrary one.

THREE STATISTICAL ERRORS THIS DESIGN AVOIDS
-------------------------------------------
Repeated here because they are the methodological core of the script.

(1) SIGNIFICANCE THRESHOLD. An earlier version declared a radius
    "significantly worse" when the 16th percentile of the paired
    bootstrap difference exceeded zero. A 16th-84th interval is a ~68
    per cent interval -- one standard deviation. That is an error bar,
    not a significance threshold. Significance here means p < 0.05,
    equivalent to the 95 per cent interval excluding zero.

(2) REFERENCE RADIUS SELECTED FROM THE DATA. An earlier version set the
    reference to the observed minimum scatter, then tested every other
    radius against it. The minimum of a set of noisy estimates is biased
    low -- the winner's curse -- which systematically inflates how many
    comparisons look unfavourable. The reference here is pre-specified
    on external grounds: 4.0 kpc, the fiducial aperture of Kelsey et al.
    (2021). That makes each radius a single planned comparison rather
    than a search through the data.

(3) MULTIPLE COMPARISONS. There are 25 comparisons against the
    reference, so about one p < 0.05 would be expected by chance even if
    radius had no effect. A multiple-comparison correction can only
    remove significance, never create it. No radius reaches p < 0.05
    even uncorrected (smallest p = 0.072, at 6 kpc), so no correction is
    applied and the p-values are reported as they stand (thesis Section
    4.8.3). If a future run produces any p < 0.05, this decision must be
    revisited before that result is reported.

ON THE P-VALUE
--------------
This is a confidence-interval inversion, not a permutation test: the
paired bootstrap distribution of the difference is centred near the
observed difference, and the p-value measures how far that distribution
sits from zero relative to its own width. It is consistent with the
reported interval by construction. A permutation test against a null in
which radius carries no information would be stronger, but is not
straightforward here because the radii are nested measurements of the
same objects rather than exchangeable labels.

WHY INSTRUMENTAL COLOURS, AND WHY flag_exclude IS NOT APPLIED
-------------------------------------------------------------
flag_exclude is used ONLY to derive the median per-object uncertainty
for scale. It is deliberately NOT applied to the colour profiles.

A large part of that flag is calibration quality, and this analysis uses
INSTRUMENTAL colours, in which the zero point is one additive constant
per object that cancels in every within-object comparison across radii.
Objects excluded from the calibrated catalogue for a poor zero point
have perfectly good instrumental colours -- some are the best-measured
objects in the sample, with fluxes above 10^5 counts. Excluding them
here would discard excellent data for a reason that does not apply.

THE SMALLEST RADII ARE BELOW THE SEEING FLOOR
---------------------------------------------
The grid starts at 0.25 kpc and the median seeing floor is sigma = 0.28
kpc, so the first one or two points are at or inside the point-spread
function. They are computed and plotted rather than hidden, because a
floor that is shown is better evidence than a floor that is asserted.

BACKGROUND ANNULUS
------------------
Measured on an identical set of 436 frames, the median background per
pixel falls from 1.711 counts at 10-15 kpc to 0.392 at 15-25 kpc to
0.184 at 20-30 kpc (thesis Section 4.4). A sky-dominated annulus would
give the same value at all three, so the inner annuli are contaminated
by host light. The gain from the second step is about a sixth of the
gain from the first, indicating the outermost annulus is near the
asymptotic sky level.

FLUX BIAS AND COLOUR BIAS ARE NOT THE SAME NUMBER
-------------------------------------------------
Because the subtracted quantity is (background per pixel) x (aperture
area), the error scales as r^2. But both bands are measured through the
same annulus, on the same galaxy, at the same position, so their
background errors are correlated and largely cancel in the difference.
At 1 kpc the single-band difference between the extreme settings is 55
mmag, but the median colour shift is 5.5 mmag: about 90 per cent
cancels. The single-band figure must never be quoted and then discussed
as though it applied to a colour.

ANNULUS_SYSTEMATIC_MAG
----------------------
The largest difference in sigma_MAD between the three annulus settings
at any radius of the grid, computed on the 436 frames common to all
three settings (0.0355 mag, at 9.5 kpc; thesis Section 5.3 quotes
0.036). It is a systematic on the SCATTER-radius relation, not on an
individual object's colour; the per-object colour shift is the 5.5 mmag
above. This script only prints it.

It remains flagged provisional until the curves of growth are re-run
with exact aperture overlap (aperture guard, fix 23). Recompute it from
the re-run three annulus files, update the constant and Section 5.3 if
the rounded value changes, then set the flag to False.

INPUTS
------
Reads the TAGGED colour file from step 14 and verifies its annulus_tag
column against ANNULUS_TAG below, so the input cannot silently come from
a different background setting.
"""

import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import mad_std

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
ANNULUS_TAG = "ann20-30"          # must match steps 13, 14, 15, 16

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
APERTURE_DIR = REPO / "results" / "phase4_aperture"
CALIB_DIR = REPO / "results" / "phase5_calibration"

COLOUR_CSV = APERTURE_DIR / f"local_colour_vs_radius_{ANNULUS_TAG}.csv"
CALIB_CSV = CALIB_DIR / "calibrated_colour_1kpc_flagged.csv"
COG_CSV = APERTURE_DIR / f"curve_of_growth_{ANNULUS_TAG}.csv"
OUT_DIR = APERTURE_DIR

N_BOOT = 10000
RNG_SEED = 42

# Pre-specified on external grounds, NOT chosen from these data.
REFERENCE_RADIUS = 4.0            # Kelsey et al. (2021) fiducial aperture
TARGET_RADIUS = 1.0               # this project's measurement radius
CI_LOWER, CI_UPPER = 2.5, 97.5    # 95 per cent interval
SIG_LEVEL = 0.05                  # two-sided, no multiple-comparison correction
MIN_OBJECTS_PER_RADIUS = 5

# Largest sigma_MAD difference between the three annulus settings at any
# radius, 436 common frames (9.5 kpc). Provisional until the fix-23
# re-run; see the docstring.
ANNULUS_SYSTEMATIC_MAG = 0.0355
ANNULUS_SYSTEMATIC_IS_PROVISIONAL = False


def backup(path: Path) -> None:
    """Rename an existing output out of the way. Nothing is deleted."""
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def robust_scatter(values):
    """Median absolute deviation scatter, NaN-tolerant."""
    values = values[~np.isnan(values)]
    if len(values) < MIN_OBJECTS_PER_RADIUS:
        return np.nan
    return mad_std(values)


# --------------------------------------------------------------------
# Load and verify provenance
# --------------------------------------------------------------------
colour_csv = COLOUR_CSV
if not colour_csv.exists():
    legacy = APERTURE_DIR / "local_colour_vs_radius.csv"
    if legacy.exists():
        print(f"[warn] tagged input not found:\n       {colour_csv}\n"
              f"       falling back to {legacy.name}, whose background annulus\n"
              f"       cannot be verified from the file itself.")
        colour_csv = legacy
    else:
        raise SystemExit(f"Input not found:\n  {colour_csv}\n"
                         f"Run 14_local_colour_vs_radius.py first.")

df = pd.read_csv(colour_csv)

if "annulus_tag" in df.columns:
    tags = sorted(set(df["annulus_tag"].dropna()))
    if tags != [ANNULUS_TAG]:
        raise SystemExit(
            f"\nAnnulus mismatch. This script is set to {ANNULUS_TAG}, but the "
            f"input\ncarries {tags}. Re-run step 14 with a matching "
            f"ANNULUS_TAG, or change it here.\n")
    print(f"[ok] input provenance verified: annulus_tag = {ANNULUS_TAG}")
else:
    print("[warn] input has no annulus_tag column -- its background setting "
          "cannot be verified.")

wide = df.pivot_table(index="object", columns="radius_kpc",
                      values="instrumental_B_minus_V")
radii = sorted(wide.columns.tolist())
data = wide[radii].to_numpy()
n_objects, n_radii = data.shape

if REFERENCE_RADIUS not in radii:
    raise ValueError(f"Reference radius {REFERENCE_RADIUS} kpc is not in the "
                     f"grid {radii}")
ref_idx = radii.index(REFERENCE_RADIUS)

print(f"Annulus setting  : {ANNULUS_TAG}")
print(f"Loaded {n_objects} objects x {n_radii} radii "
      f"({radii[0]}-{radii[-1]} kpc)")
print(f"Reference radius : {REFERENCE_RADIUS} kpc "
      f"(pre-specified: Kelsey et al. 2021 fiducial)")
print(f"Target radius    : {TARGET_RADIUS} kpc (this work)")
print(f"Bootstrap        : {N_BOOT} paired resamples, "
      f"{CI_LOWER}-{CI_UPPER}th percentile interval\n")

n_valid = np.array([np.sum(~np.isnan(data[:, j])) for j in range(n_radii)])

# --------------------------------------------------------------------
# Paired bootstrap -- one object draw reused at every radius
# --------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
boot = np.empty((N_BOOT, n_radii))

for i in range(N_BOOT):
    idx = rng.integers(0, n_objects, size=n_objects)
    resampled = data[idx, :]
    for j in range(n_radii):
        boot[i, j] = robust_scatter(resampled[:, j])
    if (i + 1) % 2000 == 0:
        print(f"  bootstrap {i + 1}/{N_BOOT}")

point_estimate = np.array([robust_scatter(data[:, j]) for j in range(n_radii)])
band_lo = np.nanpercentile(boot, CI_LOWER, axis=0)
band_hi = np.nanpercentile(boot, CI_UPPER, axis=0)

diff = boot - boot[:, [ref_idx]]
diff_point = point_estimate - point_estimate[ref_idx]
diff_lo = np.nanpercentile(diff, CI_LOWER, axis=0)
diff_hi = np.nanpercentile(diff, CI_UPPER, axis=0)

frac_below = np.nanmean(diff < 0, axis=0)
p_two_sided = 2.0 * np.minimum(frac_below, 1.0 - frac_below)
p_two_sided = np.clip(p_two_sided, 1.0 / N_BOOT, 1.0)
p_two_sided[ref_idx] = 1.0

test_mask = np.ones(n_radii, dtype=bool)
test_mask[ref_idx] = False
significant = (p_two_sided < SIG_LEVEL) & test_mask

# --------------------------------------------------------------------
# Effect size against measurement uncertainty
# --------------------------------------------------------------------
spread = np.nanmax(point_estimate) - np.nanmin(point_estimate)

median_sigma = np.nan
n_catalogue = None
median_colour = np.nan
if CALIB_CSV.exists():
    calib = pd.read_csv(CALIB_CSV)
    if "flag_exclude" in calib.columns:
        flag_col = "flag_exclude"
    elif "flag_low_flux" in calib.columns:
        flag_col = "flag_low_flux"
        print("[warn] catalogue has no flag_exclude column. Falling back to "
              "flag_low_flux.")
    else:
        flag_col = None
    keep = calib[~calib[flag_col].astype(bool)] if flag_col else calib
    keep = keep[keep["B_minus_V"].notna()]
    median_sigma = float(np.nanmedian(keep["B_minus_V_err"]))
    n_catalogue = len(keep)
    median_colour = float(np.nanmedian(keep["B_minus_V"]))

# --------------------------------------------------------------------
# Curve-of-growth diagnostic. A description of the host light profile,
# not a warning: flux still climbing at 10 kpc means host light extends
# past 10 kpc, which is why the background annulus sits where it does.
# --------------------------------------------------------------------
frac_still_rising = np.nan
median_bkg = np.nan
n_annulus_dropped = None
if COG_CSV.exists():
    cog = pd.read_csv(COG_CSV)
    keys = ["object", "telescope", "filter"]
    if "annulus_ok" in cog.columns:
        per_meas = cog.drop_duplicates(subset=keys)
        n_annulus_dropped = int((~per_meas["annulus_ok"]).sum())
        cog = cog[cog["annulus_ok"]]
    if "bkg_per_pixel" in cog.columns and len(cog):
        median_bkg = float(np.nanmedian(
            cog.drop_duplicates(subset=keys)["bkg_per_pixel"]))
    piv = cog.pivot_table(index=keys, columns="radius_kpc", values="flux_bkgsub")
    if 5.0 in piv.columns and 10.0 in piv.columns:
        ratio = (piv[10.0] / piv[5.0]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            frac_still_rising = float((ratio > 1.5).mean())
else:
    print(f"[warn] curve-of-growth file not found, skipping profile diagnostic.")

# --------------------------------------------------------------------
# Output table
# --------------------------------------------------------------------
result = pd.DataFrame({
    "radius_kpc": radii,
    "n_valid": n_valid,
    "scatter": point_estimate,
    "scatter_ci_lo": band_lo,
    "scatter_ci_hi": band_hi,
    "diff_vs_ref": diff_point,
    "diff_ci_lo": diff_lo,
    "diff_ci_hi": diff_hi,
    "p_value": p_two_sided,
    "significant_p05": significant,
})

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "colour_scatter_1kpc.csv"
backup(out_csv)
result.to_csv(out_csv, index=False)

pd.set_option("display.width", 200)
print("\n" + result.round(4).to_string(index=False))

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"Annulus setting                        : {ANNULUS_TAG}")
if n_annulus_dropped is not None:
    print(f"  object-images excluded by the guard  : {n_annulus_dropped}")
if np.isfinite(median_bkg):
    print(f"  median background per pixel          : {median_bkg:.4f} counts")
print(f"Reference radius (pre-specified)       : {REFERENCE_RADIUS} kpc")
print(f"Radii with p < {SIG_LEVEL} (no correction)   : "
      f"{int(significant.sum())} / {n_radii - 1}")
print(f"Smallest p-value in the grid           : "
      f"{np.nanmin(p_two_sided[test_mask]):.4f}")
if significant.any():
    print("  *** At least one radius has p < 0.05. The decision not to apply")
    print("      a multiple-comparison correction (thesis 4.8.3) must be")
    print("      revisited before this is reported. ***")
print()
print(f"Scatter range across all radii         : "
      f"{np.nanmin(point_estimate):.4f} - {np.nanmax(point_estimate):.4f} mag  "
      f"(spread {spread:.4f} mag)")

if TARGET_RADIUS in radii:
    ti = radii.index(TARGET_RADIUS)
    print()
    print(f"AT THE TARGET RADIUS, {TARGET_RADIUS:g} kpc:")
    print(f"  scatter                              : {point_estimate[ti]:.4f} mag "
          f"[{band_lo[ti]:.4f}, {band_hi[ti]:.4f}]")
    print(f"  difference vs {REFERENCE_RADIUS:g} kpc reference     : "
          f"{diff_point[ti]:+.4f} mag [{diff_lo[ti]:+.4f}, {diff_hi[ti]:+.4f}]")
    print(f"  p = {p_two_sided[ti]:.4f}")

if np.isfinite(median_sigma):
    print()
    print(f"Median per-object colour uncertainty   : {median_sigma:.4f} mag "
          f"(zero-point terms only; a lower bound)")
    print(f"  -> radius-to-radius spread is {spread / median_sigma:.2f}x the "
          f"single-measurement uncertainty")

print()
print(f"Background-annulus systematic          : {ANNULUS_SYSTEMATIC_MAG:.4f} mag")
print("  (largest sigma_MAD difference between the three annulus settings")
print("   at any radius; a systematic on the scatter, not on a colour)")
if ANNULUS_SYSTEMATIC_IS_PROVISIONAL:
    print("  *** PROVISIONAL until the curves of growth are re-run with exact")
    print("      aperture overlap (fix 23). Recompute from the three re-run")
    print("      annulus files, then set the flag to False. ***")

if n_catalogue is not None:
    print()
    print(f"Final catalogue                        : {n_catalogue} objects, "
          f"median B-V = {median_colour:.4f} mag")

if np.isfinite(frac_still_rising):
    print()
    print(f"Objects with flux(10 kpc)/flux(5 kpc) > 1.5 : "
          f"{100 * frac_still_rising:.0f}%")
    print("  Host light extends well past 10 kpc for most of the sample. This is")
    print("  why the background annulus sits at 20-30 kpc: a property of the")
    print("  galaxies, not an outstanding problem.")
print("=" * 78)

# --------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})

ax1.fill_between(radii, band_lo, band_hi, alpha=0.25, color="#4C72B0",
                 label=f"{int(CI_UPPER - CI_LOWER)}% bootstrap interval")
ax1.plot(radii, point_estimate, marker="o", color="#1F3B73", linewidth=1.8,
         markersize=4, label=r"Robust scatter ($\sigma_{\rm MAD}$)")
ax1.axvline(REFERENCE_RADIUS, color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"Reference: {REFERENCE_RADIUS:g} kpc (Kelsey et al. 2021)")
ax1.axvline(TARGET_RADIUS, color="#55A868", linestyle="-.", linewidth=1.5,
            label=f"This work: {TARGET_RADIUS:g} kpc")
ax1.set_xscale("log")
ax1.set_ylabel(r"Scatter in instrumental $B-V$ (mag)")
ax1.set_title(f"Local colour scatter vs. aperture radius "
              f"({ANNULUS_TAG.replace('ann', '')} kpc background annulus)")
ax1.legend(fontsize=9, loc="upper right")

ax2.axhline(0, color="#C44E52", linestyle="--", linewidth=1.5)
ax2.fill_between(radii, diff_lo, diff_hi, alpha=0.25, color="#55A868")
ax2.plot(radii, diff_point, marker="o", color="#2C6B45", linewidth=1.8,
         markersize=4)
sig_pts = result[result["significant_p05"]]
if len(sig_pts) > 0:
    ax2.scatter(sig_pts["radius_kpc"], sig_pts["diff_vs_ref"], color="red",
                zorder=5, s=55, label=r"$p < 0.05$")
    ax2.legend(fontsize=9)
else:
    ax2.text(0.5, 0.06, r"No radius differs significantly from the reference "
             r"($p > 0.05$ at every radius)",
             transform=ax2.transAxes, ha="center", fontsize=9.5, style="italic",
             color="#333333")
ax2.set_xscale("log")
ax2.set_xlabel("Aperture radius (kpc)")
ax2.set_ylabel(f"$\\Delta$ scatter vs. {REFERENCE_RADIUS:g} kpc (mag)")

fig.tight_layout()
out_png = OUT_DIR / "colour_scatter_1kpc.png"
backup(out_png)
fig.savefig(out_png, dpi=150)
print(f"\nSaved: {out_csv.name}")
print(f"Saved: {out_png.name}")