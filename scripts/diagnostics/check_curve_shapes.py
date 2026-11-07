"""
check_curve_shapes.py

Audit script, not production.

The summary statistics establish that the background is well behaved.
They say nothing about whether 1 kpc is a resolved radius, which is the
question the project rests on.

Below the seeing width the enclosed flux stops responding to aperture
radius: the aperture is sampling the point-spread function rather than
the galaxy, so the curve of growth flattens. Above it, flux grows with
radius as real structure is admitted. The radius at which that
transition happens is the seeing floor, measured from the data rather
than derived from the PSF width at step 09.

Two things are computed here:

  1. The median curve of growth, normalised per frame so that objects of
     wildly different brightness can be averaged. Normalisation is at
     5 kpc, which is well clear of the seeing floor everywhere in the
     sample and well inside the frame for all but the nearest hosts.

  2. The local logarithmic slope d(log F)/d(log r) at 1 kpc. For a
     point source inside the seeing disc the enclosed flux is nearly
     constant, so the slope tends to zero. For an extended source of
     roughly uniform surface brightness the slope tends to 2. A slope
     near 2 at 1 kpc means the measurement is dominated by real
     structure; a slope near 0 means it is dominated by the PSF.

The slope also quantifies sensitivity to centring error: at slope s, a
fractional radius error e produces a fractional flux error of about s*e.

Uses the 20-30 kpc background, the setting the catalogue is built from.
Reads only; writes one PNG and one CSV.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _find_repo() -> Path:
    """Walk upward from this script until a folder containing results/ is found."""
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "results").is_dir():
            return folder
    raise SystemExit("FATAL: no results/ folder found above this script.")


REPO = _find_repo()
PHASE4 = REPO / "results" / "phase4_aperture"
COG = PHASE4 / "curve_of_growth_ann20-30.csv"
OUT_PNG = PHASE4 / "curve_of_growth_median.png"
OUT_CSV = PHASE4 / "curve_slope_at_1kpc.csv"

NORM_KPC = 5.0
TARGET_KPC = 1.0


def main() -> None:
    if not COG.exists():
        raise SystemExit(f"FATAL: {COG.name} not found. Run step 13 first.")

    df = pd.read_csv(COG)
    good = df[df["annulus_ok"] & df["aperture_ok"]].copy()
    print(f"{len(good)} usable rows, {good['file'].nunique()} frames.\n")

    # ----------------------------------------------------------------
    # Normalised median curve
    # ----------------------------------------------------------------
    norm = (good[np.isclose(good["radius_kpc"], NORM_KPC)]
            .set_index("file")["flux_bkgsub"])
    norm = norm[norm > 0]
    print(f"{len(norm)} frames have positive flux at {NORM_KPC:g} kpc "
          f"and can be normalised.")

    sub = good[good["file"].isin(norm.index)].copy()
    sub["flux_norm"] = sub["flux_bkgsub"] / sub["file"].map(norm)

    curve = (sub.groupby("radius_kpc")["flux_norm"]
             .agg(median="median",
                  q16=lambda s: s.quantile(0.16),
                  q84=lambda s: s.quantile(0.84),
                  n="count")
             .reset_index())

    print("\nMedian normalised curve of growth:")
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ----------------------------------------------------------------
    # Local logarithmic slope at the target radius, per frame
    # ----------------------------------------------------------------
    slopes = []
    for fname, grp in sub.groupby("file"):
        grp = grp.sort_values("radius_kpc")
        r = grp["radius_kpc"].values
        f = grp["flux_bkgsub"].values
        # Centred difference using the grid points either side of the
        # target. All three radii must be present and positive; no
        # substitution of more distant radii.
        lo, hi = 0.75, 1.25
        m_lo = np.isclose(r, lo)
        m_t = np.isclose(r, TARGET_KPC)
        m_hi = np.isclose(r, hi)
        if not (m_lo.any() and m_t.any() and m_hi.any()):
            continue
        f_lo, f_t, f_hi = f[m_lo][0], f[m_t][0], f[m_hi][0]
        if not all(np.isfinite(v) and v > 0 for v in (f_lo, f_t, f_hi)):
            continue
        slope = np.log(f_hi / f_lo) / np.log(hi / lo)

        row = grp.iloc[0]
        slopes.append({"file": fname, "object": row["object"],
                       "filter": row["filter"], "z": row["z"],
                       "sigma_min_kpc": row["sigma_min_kpc"],
                       "slope_at_1kpc": float(slope)})

    sl = pd.DataFrame(slopes)
    sl.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"LOGARITHMIC SLOPE AT {TARGET_KPC:g} kpc")
    print("=" * 70)
    print(f"  frames measured : {len(sl)}")
    print(f"  median slope    : {sl['slope_at_1kpc'].median():.3f}")
    print(f"  16th-84th       : {sl['slope_at_1kpc'].quantile(0.16):.3f} "
          f"to {sl['slope_at_1kpc'].quantile(0.84):.3f}")
    print(f"  slope < 0.5 (PSF-dominated) : "
          f"{int((sl['slope_at_1kpc'] < 0.5).sum())}")
    print(f"  slope > 1.5 (well resolved) : "
          f"{int((sl['slope_at_1kpc'] > 1.5).sum())}")

    if sl["sigma_min_kpc"].notna().any():
        print(f"\n  seeing floor sigma_min_kpc, median : "
              f"{sl['sigma_min_kpc'].median():.3f} kpc")
        below = sl[sl["sigma_min_kpc"] >= TARGET_KPC]
        print(f"  frames whose floor exceeds {TARGET_KPC:g} kpc : {len(below)}")

    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.fill_between(curve["radius_kpc"], curve["q16"], curve["q84"],
                    alpha=0.25, color="tab:blue", label="16th-84th percentile")
    ax.plot(curve["radius_kpc"], curve["median"], color="tab:blue", lw=2,
            label="median")
    ax.plot(curve["radius_kpc"], (curve["radius_kpc"] / NORM_KPC) ** 2,
            color="0.35", ls="--", lw=1,
            label="slope 2 (uniform surface brightness)")
    ax.axvline(TARGET_KPC, color="crimson", ls="--", lw=1.5,
               label=f"{TARGET_KPC:g} kpc aperture")
    if sl["sigma_min_kpc"].notna().any():
        ax.axvline(sl["sigma_min_kpc"].median(), color="grey", ls=":", lw=1.5,
                   label="median seeing floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("aperture radius (kpc)")
    ax.set_ylabel(f"enclosed flux / flux at {NORM_KPC:g} kpc")
    ax.set_title("Median curve of growth")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(sl["slope_at_1kpc"].clip(-0.5, 3.0), bins=40, color="tab:blue",
            alpha=0.8)
    ax.axvline(2.0, color="grey", ls=":", label="uniform surface brightness")
    ax.axvline(sl["slope_at_1kpc"].median(), color="crimson", ls="--",
               label="median")
    ax.set_xlabel(f"d log F / d log r at {TARGET_KPC:g} kpc")
    ax.set_ylabel("frames")
    ax.set_title("Local slope at the target radius")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nWrote {OUT_PNG.name} and {OUT_CSV.name}.")


if __name__ == "__main__":
    main()