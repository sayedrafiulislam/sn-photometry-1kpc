"""
20_annulus_sensitivity.py

Run the full downstream chain once per background-annulus setting and
report the answers side by side. Thesis Sections 4.4 and 5.3, Figure 5.3.

THE QUESTION
------------
Not "what is the scatter", but "does the answer depend on a choice
nobody validated". Three things are watched:

  1. Does the median background per pixel fall as the annulus moves
     outward? If it does, the inner annulus was sitting in host light.

  2. How large is the induced bias ON COLOUR, which is what the thesis
     reports? This is NOT the same as the bias on flux -- see Q1b.

  3. Does the SHAPE of scatter-versus-radius survive? If it keeps its
     shape across settings it is a property of the data. If it moves by
     more than the bootstrap uncertainty it was a property of the
     background, and no aperture conclusion could be drawn.

The colour calculation is lifted from step 14 and the statistics from
step 19. Both are reimplemented inline rather than imported, so a single
run cannot half-succeed with mismatched inputs. As in step 19,
significance means a two-sided p < 0.05 with no multiple-comparison
correction (thesis Section 4.8.3).

WHY THE COMMON SUBSET MATTERS
-----------------------------
The guard removes different objects at different annulus widths -- the
lowest-redshift hosts fail first, since their annuli are largest in
pixels. Comparing settings on their own samples therefore conflates a
change of background with a change of sample: median flux can fall as
the background falls purely because different objects are being
averaged. Every cross-setting number below is computed on the subset
usable in every setting.

FLUX BIAS IS NOT COLOUR BIAS
----------------------------
Q1 reports the bias on FLUX in a single band, measured frame by frame:
for each frame, the enclosed flux at the fiducial radius under the
innermost and outermost annulus settings is compared directly, and the
median fractional change is reported. (An earlier version multiplied the
median background change by the median aperture area and divided by the
median flux. Those three medians come from different frames, so the
product described no actual frame and overstated the bias: 4.9 per cent
against 3.4 per cent measured directly.)

The thesis reports B-V, a DIFFERENCE of magnitudes. Both bands are
measured from the same annulus geometry on the same galaxy, so their
background errors are correlated and largely cancel. The single-band
figure is therefore an upper bound on the colour bias, never the colour
bias itself.

Q1b measures the colour shift directly: for every object with a valid
colour at the fiducial radius under both the first and last annulus
setting, take the difference and bootstrap the median. No model, no
assumption about how much cancels.

THE ANNULUS SYSTEMATIC FOR STEP 19
----------------------------------
Q2 prints the largest difference in sigma_MAD between the settings at
any radius. This is a systematic on the SCATTER-radius relation, not on
an individual colour, so it does not scale with aperture area the way
the flux bias does. Copy it into ANNULUS_SYSTEMATIC_MAG in step 19.
checks/annulus_systematic_check.py reproduces the same number
independently.

TWO GUARDS, DIFFERENT GRAIN
---------------------------
annulus_ok is a property of the FRAME: one background per frame per
setting, so it is all radii or none. aperture_ok is a property of a
SINGLE measurement: a frame can be sound at 1 kpc and off the chip by
9 kpc. It is applied per row, and a frame may survive with a partial
radius grid.

OUTPUT
------
  annulus_sensitivity_summary.csv
  annulus_sensitivity_comparison.png
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import mad_std

# --------------------------------------------------------------------
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
IN_DIR = REPO / "results" / "phase4_aperture"
OUT_DIR = IN_DIR
INPUT_GLOB = "curve_of_growth_ann*.csv"

N_BOOT = 10000                    # same as step 19
RNG_SEED = 42
REFERENCE_RADIUS = 4.0            # pre-specified: Kelsey et al. (2021)
FIDUCIAL_RADIUS = 1.0             # radius at which the catalogue is built
CI_LOWER, CI_UPPER = 2.5, 97.5    # 95 per cent
SIG_LEVEL = 0.05                  # two-sided, no multiple-comparison correction
MIN_OBJECTS_PER_RADIUS = 5


def is_dupont(series: pd.Series) -> pd.Series:
    """Substring test. The catalogue writes 'duPont', not 'dup'."""
    return series.astype(str).str.lower().str.contains("dup")


def robust_scatter(values):
    values = values[~np.isnan(values)]
    if len(values) < MIN_OBJECTS_PER_RADIUS:
        return np.nan
    return mad_std(values)


def frac_to_millimag(f):
    """Exact conversion of a fractional flux error to millimagnitudes."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(1.0 - f) * 1000.0


def compute_colours(cog):
    """Step 14's logic: same-telescope du Pont B/V instrumental colour."""
    dup = cog[is_dupont(cog["telescope"])]
    if len(dup) == 0:
        raise RuntimeError(f"No du Pont rows -- cannot form a colour. "
                           f"Telescope values present: "
                           f"{sorted(cog['telescope'].unique())}")

    pivot = dup.pivot_table(index=["object", "z", "radius_kpc"],
                            columns="filter", values="flux_bkgsub").reset_index()

    if "B" not in pivot.columns or "V" not in pivot.columns:
        raise RuntimeError("Expected both B and V after pivoting.")

    valid = (pivot["B"] > 0) & (pivot["V"] > 0)
    # Non-positive flux is the measurement floor, not an error. Counted,
    # set to NaN, warning suppressed rather than printed.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, pivot["B"].to_numpy() / pivot["V"].to_numpy(),
                         np.nan)
        pivot["instrumental_B_minus_V"] = np.where(valid,
                                                   -2.5 * np.log10(ratio),
                                                   np.nan)
    return pivot, int((~valid).sum()), len(pivot)


def scatter_analysis(colours):
    """Step 19's logic: paired bootstrap, pre-specified reference,
    two-sided p < 0.05 without correction."""
    wide = colours.pivot_table(index="object", columns="radius_kpc",
                               values="instrumental_B_minus_V")
    radii = sorted(wide.columns.tolist())
    data = wide[radii].to_numpy()
    n_objects, n_radii = len(wide), len(radii)

    if REFERENCE_RADIUS not in radii:
        raise ValueError(f"Reference radius {REFERENCE_RADIUS} not in grid.")
    ref_idx = radii.index(REFERENCE_RADIUS)

    rng = np.random.default_rng(RNG_SEED)
    boot = np.empty((N_BOOT, n_radii))
    for i in range(N_BOOT):
        idx = rng.integers(0, n_objects, size=n_objects)
        resampled = data[idx, :]
        for j in range(n_radii):
            boot[i, j] = robust_scatter(resampled[:, j])

    point = np.array([robust_scatter(data[:, j]) for j in range(n_radii)])
    diff = boot - boot[:, [ref_idx]]

    frac_below = np.nanmean(diff < 0, axis=0)
    p = 2.0 * np.minimum(frac_below, 1.0 - frac_below)
    p = np.clip(p, 1.0 / N_BOOT, 1.0)
    p[ref_idx] = 1.0

    test_mask = np.ones(n_radii, dtype=bool)
    test_mask[ref_idx] = False
    sig = (p < SIG_LEVEL) & test_mask

    return {"radii": np.array(radii), "scatter": point,
            "ci_lo": np.nanpercentile(boot, CI_LOWER, axis=0),
            "ci_hi": np.nanpercentile(boot, CI_UPPER, axis=0),
            "n_objects": n_objects, "n_sig": int(sig.sum()),
            "min_p": float(np.nanmin(p[test_mask]))}


def main():
    paths = sorted(glob.glob(str(IN_DIR / INPUT_GLOB)))
    if not paths:
        raise SystemExit(f"No files matching {INPUT_GLOB} in {IN_DIR}. "
                         f"Run 13_curve_of_growth.py first.")

    print(f"Found {len(paths)} annulus setting(s):")
    for p in paths:
        print(f"   {Path(p).name}")
    print()

    KEYS = ["object", "telescope", "filter"]

    # ----------------------------------------------------------------
    # PASS 1 -- find the object-images usable in EVERY setting
    # ----------------------------------------------------------------
    loaded, guard_stats = {}, {}
    common = None

    for path in paths:
        name = Path(path).name.replace("curve_of_growth_", "").replace(".csv", "")
        cog = pd.read_csv(path)
        per_meas = cog.drop_duplicates(subset=KEYS)
        n_meas, n_rows_before = len(per_meas), len(cog)

        if "annulus_ok" in cog.columns:
            n_bad = int((~per_meas["annulus_ok"]).sum())
            cog = cog[cog["annulus_ok"]]
        else:
            print(f"  [warn] {name}: no annulus_ok column. Proceeding without "
                  f"the guard.")
            n_bad = 0

        if "aperture_ok" in cog.columns:
            n_ap_bad = int((~cog["aperture_ok"]).sum())
            cog = cog[cog["aperture_ok"]]
        else:
            print(f"  [warn] {name}: no aperture_ok column. Aperture guard NOT "
                  f"applied; truncated and NaN-contaminated apertures included.")
            n_ap_bad = 0

        loaded[name] = cog
        guard_stats[name] = (n_meas, n_bad, n_ap_bad, n_rows_before, len(cog))
        keys_here = set(map(tuple, cog[KEYS].drop_duplicates().to_numpy()))
        common = keys_here if common is None else (common & keys_here)

    print(f"Object-images usable in every setting: {len(common)}")
    for nm, (n_meas, n_bad, n_ap_bad, n_before, n_after) in guard_stats.items():
        print(f"   {nm:<12} annulus {n_meas - n_bad}/{n_meas} frames "
              f"({n_bad} dropped) | aperture {n_ap_bad} rows dropped | "
              f"rows {n_after}/{n_before}")
    print("\nAll cross-setting comparisons below use the common subset, so a\n"
          "change of sample cannot masquerade as a change of background.\n")

    common_idx = pd.MultiIndex.from_tuples(sorted(common), names=KEYS)

    # ----------------------------------------------------------------
    # PASS 2 -- analyse each setting on the common subset
    # ----------------------------------------------------------------
    rows, curves, colours_at_fid, flux_at_fid = [], {}, {}, {}

    for name, cog_full in loaded.items():
        print("=" * 72)
        print(f"SETTING: {name}")
        print("=" * 72)

        cog = cog_full.set_index(KEYS).loc[common_idx].reset_index()

        med_bkg = (float(np.nanmedian(
            cog.drop_duplicates(subset=KEYS)["bkg_per_pixel"]))
            if "bkg_per_pixel" in cog.columns else np.nan)

        colours, n_invalid, n_points = compute_colours(cog)
        print(f"  colour: {n_invalid}/{n_points} (object, radius) points have "
              f"non-positive flux -> NaN")

        res = scatter_analysis(colours)
        curves[name] = res

        at_fid = colours[np.isclose(colours["radius_kpc"], FIDUCIAL_RADIUS)]
        med_colour_fid = float(np.nanmedian(at_fid["instrumental_B_minus_V"]))
        n_valid_fid = int(at_fid["instrumental_B_minus_V"].notna().sum())

        colours_at_fid[name] = (at_fid.groupby("object")["instrumental_B_minus_V"]
                                .first().dropna())

        cog_fid = cog[np.isclose(cog["radius_kpc"], FIDUCIAL_RADIUS)]
        flux_at_fid[name] = cog_fid.set_index(KEYS)["flux_bkgsub"]
        med_flux_fid = float(np.nanmedian(cog_fid["flux_bkgsub"]))
        med_area_fid = float(np.pi * np.nanmedian(cog_fid["radius_pix"]) ** 2)

        n_meas, n_bad, n_ap_bad, _, _ = guard_stats[name]
        print(f"  scatter analysis: {res['n_objects']} objects, "
              f"{res['n_sig']}/{len(res['radii']) - 1} radii with p < "
              f"{SIG_LEVEL}, min p = {res['min_p']:.3f}\n")

        rows.append({
            "setting": name,
            "n_object_images_total": n_meas,
            "n_dropped_by_annulus_guard": n_bad,
            "n_rows_dropped_by_aperture_guard": n_ap_bad,
            "n_in_common_subset": len(common),
            "median_bkg_per_pixel": med_bkg,
            "median_flux_at_fid": med_flux_fid,
            "median_aperture_area_px_at_fid": med_area_fid,
            "n_objects_in_scatter": res["n_objects"],
            "n_valid_colours_at_fid": n_valid_fid,
            "median_instr_colour_at_fid": med_colour_fid,
            "scatter_min": float(np.nanmin(res["scatter"])),
            "scatter_max": float(np.nanmax(res["scatter"])),
            "scatter_spread": float(np.nanmax(res["scatter"])
                                    - np.nanmin(res["scatter"])),
            "n_radii_significant": res["n_sig"],
            "min_p_value": res["min_p"],
        })

    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 240)
    print("=" * 72)
    print(f"SIDE-BY-SIDE COMPARISON  (fiducial radius {FIDUCIAL_RADIUS:g} kpc)")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    print()

    names = list(curves.keys())

    # ----------------------------------------------------------------
    # Q1: does the background change matter, in FLUX terms?
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Q1  Was the inner annulus contaminated by host light?")
    print("-" * 72)
    bkgs = summary["median_bkg_per_pixel"].to_numpy()
    flux_bias_mmag = np.nan
    if np.all(np.isfinite(bkgs)) and len(bkgs) > 1:
        for nm, b in zip(summary["setting"], bkgs):
            print(f"     {nm:<12} median background/pixel = {b:.4f}")
        delta_bkg = bkgs[0] - bkgs[-1]
        print()
        print(f"     change in median background     : {delta_bkg:+.4f} "
              f"counts/pixel ({100 * delta_bkg / bkgs[0]:+.2f}% of the "
              f"{names[0]} level)")

        first, last = names[0], names[-1]
        pair = pd.concat([flux_at_fid[first].rename("inner"),
                          flux_at_fid[last].rename("outer")],
                         axis=1, join="inner").dropna()
        pair = pair[(pair["inner"] > 0) & (pair["outer"] > 0)]
        if len(pair):
            frac = (pair["outer"] - pair["inner"]) / pair["outer"]
            med_frac = float(np.median(frac))
            flux_bias_mmag = float(frac_to_millimag(med_frac))
            print(f"     frames with positive flux at {FIDUCIAL_RADIUS:g} kpc "
                  f"under both {first} and {last}: {len(pair)}")
            print(f"     median single-band flux change  : {100 * med_frac:.2f}% "
                  f"= {flux_bias_mmag:.1f} mmag")
            p16, p84 = np.percentile(frac, [16, 84])
            print(f"       16th-84th percentile          : {100 * p16:.1f}% "
                  f"to {100 * p84:.1f}%")
            for band in ("B", "V"):
                fb = frac[pair.index.get_level_values("filter") == band]
                if len(fb):
                    print(f"       {band}: {len(fb)} frames, median "
                          f"{frac_to_millimag(np.median(fb)):.1f} mmag")
        print()
        print("     The subtracted background is (background per pixel) x "
              "(aperture")
        print("     area), and area scales as r^2, so the same background "
              "error is ~25x")
        print(f"     less damaging at {FIDUCIAL_RADIUS:g} kpc than at 5 kpc.")
    else:
        print("     No bkg_per_pixel column found; cannot assess.")
    print()

    # ----------------------------------------------------------------
    # Q1b: the bias ON COLOUR, measured directly
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Q1b How large is the bias on COLOUR, which is what the thesis reports?")
    print("-" * 72)
    if len(names) > 1:
        first, last = names[0], names[-1]
        joined = pd.concat([colours_at_fid[first].rename("a"),
                            colours_at_fid[last].rename("b")],
                           axis=1, join="inner").dropna()
        if len(joined) >= MIN_OBJECTS_PER_RADIUS:
            d = (joined["a"] - joined["b"]).to_numpy()
            med = float(np.median(d))
            rng = np.random.default_rng(RNG_SEED)
            bs = np.array([np.median(rng.choice(d, size=len(d), replace=True))
                           for _ in range(N_BOOT)])
            lo, hi = np.percentile(bs, [CI_LOWER, CI_UPPER])

            print(f"     Objects with a valid colour at {FIDUCIAL_RADIUS:g} kpc "
                  f"under both")
            print(f"     {first} and {last} : {len(joined)}")
            print()
            print(f"     median colour shift ({first} - {last})")
            print(f"       = {med * 1000:+.1f} mmag   "
                  f"95% CI [{lo * 1000:+.1f}, {hi * 1000:+.1f}]")
            print(f"     mean {np.mean(d) * 1000:+.1f} mmag, "
                  f"scatter (MAD) {mad_std(d) * 1000:.1f} mmag")
            print()
            if np.isfinite(flux_bias_mmag) and flux_bias_mmag > 0:
                pct_cancel = 100 * (1 - abs(med) * 1000 / flux_bias_mmag)
                print(f"     Single-band flux bias (Q1, upper bound) : "
                      f"{flux_bias_mmag:.1f} mmag")
                print(f"     Measured colour bias                    : "
                      f"{abs(med) * 1000:.1f} mmag")
                print(f"     Cancellation between B and V removes about "
                      f"{pct_cancel:.0f}% of it.")
                print()
            if lo < 0 < hi:
                print("     The interval spans zero: no detectable colour bias "
                      "from the")
                print("     annulus choice. Report the flux bias as a bound and "
                      "say the")
                print("     colour is consistent with unaffected.")
            else:
                print("     The colour shift is resolved. THIS number, not the "
                      "flux bias,")
                print("     is the annulus bias on B-V.")
        else:
            print(f"     Only {len(joined)} objects in common -- too few.")
    else:
        print("     Only one setting present; nothing to compare.")
    print()

    # ----------------------------------------------------------------
    # Q2: is the shape of scatter-vs-radius stable?
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Q2  Does the scatter-versus-radius shape survive the change?")
    print("-" * 72)
    if len(names) > 1:
        ref_curve = curves[names[0]]
        grid = ref_curve["radii"]
        stack = np.vstack([np.interp(grid, curves[n]["radii"],
                                     curves[n]["scatter"]) for n in names])
        dev = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        max_dev = float(np.nanmax(dev))
        typical_ci = float(np.nanmedian(ref_curve["ci_hi"] - ref_curve["ci_lo"]))
        print(f"     Largest disagreement between settings at any radius : "
              f"{max_dev:.4f} mag (at {grid[np.nanargmax(dev)]:g} kpc)")
        print(f"     Typical bootstrap 95% interval width                : "
              f"{typical_ci:.4f} mag")
        print()
        print(f"     -> Set ANNULUS_SYSTEMATIC_MAG = {max_dev:.4f} in")
        print(f"        19_colour_scatter_vs_radius.py. Once the curves of growth")
        print(f"        are final (fix 23), set ANNULUS_SYSTEMATIC_IS_PROVISIONAL")
        print(f"        = False there.")
        print()
        if max_dev < 0.5 * typical_ci:
            print("     The curves agree to well within their own uncertainty. "
                  "The shape is")
            print("     a property of the data, not of the background choice. "
                  "The null")
            print("     result stands, and the spread can be quoted as a "
                  "systematic.")
        elif max_dev < typical_ci:
            print("     The curves differ by less than the bootstrap interval, "
                  "but not")
            print("     negligibly. Report the spread as a systematic and say "
                  "so explicitly.")
        else:
            print("     The curves differ by MORE than the bootstrap "
                  "uncertainty. The")
            print("     shape is driven by the background, not the data. No "
                  "aperture")
            print("     conclusion can be drawn until the background is settled.")
    else:
        print("     Only one setting present. Re-run step 13 with at least two "
              "ANNULUS_SETTINGS.")
    print()

    # ----------------------------------------------------------------
    # Q3: does the significance verdict change?
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Q3  Does the significance verdict change?")
    print("-" * 72)
    for nm, ns, mp in zip(summary["setting"], summary["n_radii_significant"],
                          summary["min_p_value"]):
        print(f"     {nm:<12} {ns} radii with p < {SIG_LEVEL}, min p = {mp:.3f}")
    if (summary["n_radii_significant"].nunique() == 1
            and summary["n_radii_significant"].iloc[0] == 0):
        print("\n     Null result across every setting. Robust to the "
              "background choice.")
        print("     min p is expected to drift between settings; it is the "
              "smallest of")
        print("     25 correlated comparisons and is not itself a result.")
    else:
        print("\n     At least one setting has a radius with p < 0.05. The "
              "verdict is not")
        print("     stable, and the decision not to correct for multiple "
              "comparisons")
        print("     (thesis 4.8.3) must be revisited before anything is "
              "reported.")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "annulus_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    palette = ["#1F3B73", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
    for k, nm in enumerate(names):
        cv = curves[nm]
        c = palette[k % len(palette)]
        ax.fill_between(cv["radii"], cv["ci_lo"], cv["ci_hi"], color=c,
                        alpha=0.15, linewidth=0)
        ax.plot(cv["radii"], cv["scatter"], marker="o", markersize=4,
                linewidth=1.8, color=c, label=f"{nm}  (n={cv['n_objects']})")
    ax.axvline(REFERENCE_RADIUS, color="#888888", linestyle="--", linewidth=1.4,
               label=f"Reference: {REFERENCE_RADIUS:g} kpc")
    ax.axvline(FIDUCIAL_RADIUS, color="#333333", linestyle="-.", linewidth=1.4,
               label=f"This work: {FIDUCIAL_RADIUS:g} kpc")
    ax.set_xscale("log")
    ax.set_xlabel("Aperture radius (kpc)")
    ax.set_ylabel(r"Scatter in instrumental $B-V$ (mag)")
    ax.set_title("Sensitivity of the scatter-radius relation to background annulus")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = OUT_DIR / "annulus_sensitivity_comparison.png"
    fig.savefig(fig_path, dpi=150)

    print(f"\nSaved: {summary_path.name}")
    print(f"Saved: {fig_path.name}")


if __name__ == "__main__":
    main()