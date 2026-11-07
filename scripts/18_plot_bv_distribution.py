"""
18_plot_bv_distribution.py

The local B-V colour distribution at 1 kpc. Paper Figure 2.

WHAT THE FIGURE SHOWS
---------------------
Main panel: the dereddened distribution, with the observed distribution
as a step outline behind it, so the size and direction of the extinction
correction are visible in one image rather than requiring two figures.

The measurement resolution is drawn as a horizontal bar of width equal
to the median per-object uncertainty. Structure narrower than that bar
is not resolved.

DO NOT BIN FINER THAN THE MEASUREMENT CAN RESOLVE
-------------------------------------------------
An earlier version used bins=15 over a range of about 1.24 mag, giving
bins 0.083 mag wide against a median per-object uncertainty of 0.117.
Every object was therefore smeared across more than one bin, and any
peak or gap in that histogram was structure the data cannot resolve --
which a reader would nonetheless see and interpret.

Bin width here is the LARGER of the Freedman-Diaconis width and the
median per-object uncertainty, and the adopted width is annotated on the
figure so the resolution limit is explicit rather than implied.

NO KELSEY COMPARISON LINE IS POSSIBLE, AND NONE SHOULD BE ADDED
---------------------------------------------------------------
Kelsey et al. (2021) measure rest-frame U-R, not B-V. There is no Kelsey
B-V median to plot. Drawing their U-R value on a B-V axis would compare
two different quantities that happen to share units, and a reader would
take the offset between them to mean something.

The comparison with Kelsey et al. is methodological -- local aperture
photometry at a fixed physical radius, and the choice of that radius --
and belongs in prose, not on this figure.

READ THIS BEFORE QUOTING THE NUMBERS
------------------------------------
The quoted uncertainties are the two zero-point terms in quadrature plus
the 16 per cent SFD map term. They omit photon noise, background
uncertainty, flat-field error, and the flux-dependent annulus
systematic. They are LOWER BOUNDS.

The sample is CSP-II only. The supplied du Pont zero points do not cover
CSP-I, so every CSP-I object with a measured colour is lost at the
calibration stage. That is a cut by survey epoch, not by data quality,
and must be stated in the sample description. The number lost is counted
at run time from calibrated_colour_1kpc_exclusions.csv, not typed in,
so it cannot fall out of step with the pipeline.

OUTPUT
------
  bv_distribution_1kpc.png
  bv_distribution_1kpc_summary.txt   the numbers, ready for the paper
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
CALIB_DIR = REPO / "results" / "phase5_calibration"

IN_PATH = CALIB_DIR / "calibrated_colour_1kpc_dered.csv"
EXCL_PATH = CALIB_DIR / "calibrated_colour_1kpc_exclusions.csv"
OUT_PNG = CALIB_DIR / "bv_distribution_1kpc.png"
OUT_TXT = CALIB_DIR / "bv_distribution_1kpc_summary.txt"

FIDUCIAL_RADIUS_KPC = 1.0
N_BOOT = 5000
RNG_SEED = 42


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def boot_median(values, n_boot=N_BOOT, seed=RNG_SEED):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                  for _ in range(n_boot)])
    return (float(np.median(v)), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def n_lost_to_zero_points() -> int | None:
    """Objects that had a colour at the fiducial radius but no usable
    zero point: logged at the 'zero point' stage and not already lost at
    the 'photometry' stage."""
    if not EXCL_PATH.exists():
        return None
    ex = pd.read_csv(EXCL_PATH)
    zp = set(ex.loc[ex["stage"] == "zero point", "object"])
    phot = set(ex.loc[ex["stage"] == "photometry", "object"])
    return len(zp - phot)


def mad_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"\nInput not found:\n  {IN_PATH}\n"
                         f"Run 17_apply_galactic_extinction.py first.\n")

    df = pd.read_csv(IN_PATH)

    if "B_minus_V_dered" not in df.columns:
        raise SystemExit("\nNo B_minus_V_dered column -- this file has not been "
                         "through the extinction step. Run step 17 first.\n")

    if "flag_exclude" in df.columns:
        flag_col = "flag_exclude"
    elif "flag_low_flux" in df.columns:
        flag_col = "flag_low_flux"
        print("[WARN] no flag_exclude column -- falling back to flag_low_flux. "
              "The calibration-quality and manual criteria are NOT applied.")
    else:
        flag_col = None
        print("[WARN] no exclusion flag found -- plotting the full sample.")

    keep = (~df[flag_col].astype(bool) if flag_col
            else pd.Series(True, index=df.index))
    clean = df[keep & df["B_minus_V_dered"].notna()].copy()

    obs = clean["B_minus_V"].to_numpy()
    ded = clean["B_minus_V_dered"].to_numpy()
    err = clean["B_minus_V_dered_err"].to_numpy()
    n = len(clean)

    med, lo, hi = boot_median(ded)
    med_obs, _, _ = boot_median(obs)
    med_err = float(np.nanmedian(err))

    # ----------------------------------------------------------------
    q75, q25 = np.percentile(ded, [75, 25])
    fd_width = 2.0 * (q75 - q25) / np.cbrt(n) if n > 1 else med_err
    bin_width = max(fd_width, med_err)
    lo_edge = np.floor(min(ded.min(), obs.min()) / bin_width) * bin_width
    hi_edge = np.ceil(max(ded.max(), obs.max()) / bin_width) * bin_width
    bins = np.arange(lo_edge, hi_edge + bin_width, bin_width)

    print(f"Sample            : {n} objects (flag: {flag_col})")
    print(f"Freedman-Diaconis : {fd_width:.4f} mag")
    print(f"median per-object : {med_err:.4f} mag")
    print(f"bin width adopted : {bin_width:.4f} mag  ({len(bins) - 1} bins)")
    if np.isclose(bin_width, med_err):
        print("  -> limited by the measurement uncertainty, not by sample size")
    else:
        print("  -> limited by sample size, not by measurement uncertainty")

    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.hist(obs, bins=bins, histtype="step", linewidth=1.4,
            color="#999999", label=f"observed (median {med_obs:.3f})")
    ax.hist(ded, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6,
            alpha=0.85, label=f"dereddened (median {med:.3f})")

    ax.axvspan(lo, hi, color="#C44E52", alpha=0.18, zorder=0)
    ax.axvline(med, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"median = {med:.3f} [{lo:.3f}, {hi:.3f}]")

    ymax = ax.get_ylim()[1]
    xbar = np.percentile(ded, 92)
    ax.errorbar([xbar], [ymax * 0.88], xerr=[med_err / 2], fmt="none",
                ecolor="black", capsize=3, linewidth=1.2)
    ax.text(xbar, ymax * 0.91, "median\nuncertainty", ha="center", va="bottom",
            fontsize=8)

    ax.set_xlabel(r"Local $B-V$ colour (mag)")
    ax.set_ylabel("Number of objects")
    ax.set_title(f"Local $B-V$ within {FIDUCIAL_RADIUS_KPC:g} kpc  "
                 f"($n={n}$, bin width {bin_width:.3f} mag)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    backup(OUT_PNG)
    fig.savefig(OUT_PNG, dpi=150)

    # ----------------------------------------------------------------
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out()
    out("=" * 74)
    out(f"LOCAL B-V WITHIN {FIDUCIAL_RADIUS_KPC:g} kpc")
    out("=" * 74)
    out(f"  sample                     : {n} objects")
    out(f"  exclusion flag             : {flag_col}")
    out()
    out(f"  median, dereddened         : {med:.4f} mag   "
        f"95% CI [{lo:.4f}, {hi:.4f}]")
    out(f"  median, observed           : {med_obs:.4f} mag")
    out(f"  extinction correction      : {med - med_obs:+.4f} mag")
    out()
    out(f"  16th-84th percentile       : {np.percentile(ded, 16):.4f} - "
        f"{np.percentile(ded, 84):.4f} mag")
    out(f"  interquartile range        : {np.percentile(ded, 25):.4f} - "
        f"{np.percentile(ded, 75):.4f} mag")
    out(f"  robust scatter (sigma_MAD) : {mad_sigma(ded):.4f} mag")
    out(f"  full range                 : {ded.min():+.4f} to {ded.max():+.4f} mag")
    out()
    out(f"  median per-object error    : {med_err:.4f} mag  (LOWER BOUND --")
    out(f"    zero points and the SFD map term only. Omits photon noise,")
    out(f"    background uncertainty, flat-field error, and the flux-dependent")
    out(f"    annulus systematic.)")
    out()
    ratio = mad_sigma(ded) / med_err
    out(f"  scatter/error ratio        : {ratio:.2f}")
    if ratio < 1.5:
        out(f"    Below ~1.5 the observed spread is comparable to the")
        out(f"    measurement errors, so little of it can be claimed as")
        out(f"    intrinsic variation between host environments.")
    else:
        out(f"    Above ~1.5 the spread exceeds the errors, so some of it is")
        out(f"    plausibly intrinsic. Quantifying how much requires the full")
        out(f"    error budget, which is not yet available.")
    out()
    n_zp = n_lost_to_zero_points()
    n_zp_txt = f"{n_zp} objects" if n_zp is not None else "the CSP-I objects"
    out("  SAMPLE CAVEAT: CSP-II only. The supplied du Pont zero points do not")
    out(f"  cover CSP-I, so {n_zp_txt} with a measured colour are lost at the")
    out("  calibration stage (all CSP-I except ASAS14lq; see")
    out("  calibrated_colour_1kpc_exclusions.csv). That is a cut by survey")
    out("  epoch, not by data quality.")
    out("=" * 74)

    backup(OUT_TXT)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {OUT_PNG.name}")
    print(f"Saved: {OUT_TXT.name}")


if __name__ == "__main__":
    main()