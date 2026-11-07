"""
check_selection_bias.py

Audit script, not production.

Step 14 measures a local colour for 186 of 218 objects at 1 kpc. The 32
that drop out sit at more than twice the median redshift of those that
survive (0.063 against 0.030, KS D = 0.52, p < 1e-4). The cause is
geometric: 1 kpc subtends a smaller angle at greater distance, so the
aperture shrinks toward the seeing disc and the enclosed flux falls
toward the noise floor.

WHY A REDSHIFT BIAS MAY BE A COLOUR BIAS
----------------------------------------
CSP is magnitude limited, so at higher redshift only intrinsically
brighter supernovae are detected, and these sit preferentially in more
luminous and more massive hosts. Host stellar mass correlates with
colour through the red sequence and the mass-metallicity relation
(Tremonti et al. 2004, ApJ 613, 898). Preferentially removing
high-redshift objects may therefore preferentially remove red hosts,
biasing the measured colour distribution blue.

That is a hypothesis, and this script tests it rather than assuming it
either way.

THE TEST
--------
The 32 dropped objects are unmeasurable at 1 kpc but most are
measurable at 5 kpc, where the aperture is five times larger in angle.
If their 5 kpc colours match the survivors' 5 kpc colours, the selection
operates in redshift but not in colour, and the bias on the reported
distribution is benign. If they are systematically redder, the bias is
real and must be stated prominently rather than in a footnote.

THE ROBUSTNESS THRESHOLD
------------------------
Chosen on GEOMETRY, before any colour is examined: the redshift below
which essentially every object yields a 1 kpc measurement. Selecting a
threshold by scanning for the value that makes the result look best
would tune an analysis choice against the answer, which is not a
defensible procedure.

Instrumental colours throughout. Zero points are one additive constant
per frame, so comparisons BETWEEN objects at one radius are affected by
them and comparisons WITHIN an object between radii are not. The
survivor-versus-dropped comparison below is between objects, so it is
indicative rather than final; step 17 repeats it on calibrated
magnitudes.

Reads only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
PHASE4 = REPO / "results" / "phase4_aperture"
COLOUR = PHASE4 / "local_colour_vs_radius_ann20-30.csv"
OUT = PHASE4 / "selection_bias_audit.csv"

R_TARGET = 1.0
R_COMPARE = 5.0
COL = "instrumental_B_minus_V"


def main() -> None:
    if not COLOUR.exists():
        raise SystemExit(f"FATAL: {COLOUR.name} not found. Run step 14 first.")

    df = pd.read_csv(COLOUR)
    z_by_obj = df.groupby("object")["z"].first()

    at_target = df[np.isclose(df["radius_kpc"], R_TARGET)]
    measured = set(at_target.loc[at_target[COL].notna(), "object"])
    dropped = set(z_by_obj.index) - measured

    print("=" * 72)
    print(f"COMPLETENESS AT {R_TARGET:g} kpc AS A FUNCTION OF REDSHIFT")
    print("=" * 72)
    print(f"  {'z bin':>14} {'n':>5} {'measured':>9} {'fraction':>9}")

    edges = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = z_by_obj[(z_by_obj >= lo) & (z_by_obj < hi)]
        if len(in_bin) == 0:
            continue
        n_ok = len(set(in_bin.index) & measured)
        print(f"  {lo:6.3f}-{hi:6.3f} {len(in_bin):5d} {n_ok:9d} "
              f"{n_ok / len(in_bin):9.2f}")

    # Highest threshold at which completeness stays at or above 95%.
    # Derived from geometry alone; no colour has been examined yet.
    threshold = None
    for z_try in np.arange(0.02, 0.121, 0.005):
        below = z_by_obj[z_by_obj < z_try]
        if len(below) < 20:
            continue
        frac = len(set(below.index) & measured) / len(below)
        if frac >= 0.95:
            threshold = z_try
    print(f"\n  Highest z with >=95% completeness at {R_TARGET:g} kpc: "
          f"{threshold if threshold else float('nan'):.3f}")

    # ----------------------------------------------------------------
    # The colour test
    # ----------------------------------------------------------------
    at_cmp = df[np.isclose(df["radius_kpc"], R_COMPARE)].dropna(subset=[COL])
    c_meas = at_cmp[at_cmp["object"].isin(measured)][COL]
    c_drop = at_cmp[at_cmp["object"].isin(dropped)][COL]

    print("\n" + "=" * 72)
    print(f"COLOUR AT {R_COMPARE:g} kpc: SURVIVORS vs DROPPED")
    print("=" * 72)
    print(f"  measurable at {R_TARGET:g} kpc : n = {len(c_meas):3d}  "
          f"median {c_meas.median():.4f}  "
          f"16-84 {c_meas.quantile(0.16):.4f} to {c_meas.quantile(0.84):.4f}")
    print(f"  dropped at {R_TARGET:g} kpc    : n = {len(c_drop):3d}  "
          f"median {c_drop.median():.4f}  "
          f"16-84 {c_drop.quantile(0.16):.4f} to {c_drop.quantile(0.84):.4f}")

    if len(c_drop) >= 5:
        u, p_u = stats.mannwhitneyu(c_meas, c_drop, alternative="two-sided")
        ks, p_ks = stats.ks_2samp(c_meas, c_drop)
        print(f"\n  difference in medians : "
              f"{c_drop.median() - c_meas.median():+.4f} mag")
        print(f"  Mann-Whitney U        : p = {p_u:.4f}")
        print(f"  KS                    : D = {ks:.3f}, p = {p_ks:.4f}")
        print("\n  A large p means the dropped objects are not distinguishable")
        print("  in colour, so the selection acts in redshift alone and the")
        print("  reported distribution is not biased in the quantity measured.")
    else:
        print("\n  too few dropped objects measurable at the comparison "
              "radius for a test.")

    # ----------------------------------------------------------------
    # The robustness subsample
    # ----------------------------------------------------------------
    if threshold:
        print("\n" + "=" * 72)
        print(f"ROBUSTNESS SUBSAMPLE: z < {threshold:.3f}")
        print("=" * 72)
        full = at_target.dropna(subset=[COL])
        sub = full[full["z"] < threshold]
        print(f"  full sample     : n = {len(full):3d}  "
              f"median {full[COL].median():.4f}  "
              f"16-84 {full[COL].quantile(0.16):.4f} to "
              f"{full[COL].quantile(0.84):.4f}")
        print(f"  z-cut subsample : n = {len(sub):3d}  "
              f"median {sub[COL].median():.4f}  "
              f"16-84 {sub[COL].quantile(0.16):.4f} to "
              f"{sub[COL].quantile(0.84):.4f}")
        print(f"\n  shift in median : "
              f"{sub[COL].median() - full[COL].median():+.4f} mag")
        ks, p_ks = stats.ks_2samp(full[COL], sub[COL])
        print(f"  KS             : D = {ks:.3f}, p = {p_ks:.4f}")
        print("\n  These samples are nested, so the KS p-value is indicative")
        print("  only. The number that matters is the shift in the median")
        print("  relative to the measurement uncertainty.")

    pd.DataFrame({
        "object": sorted(z_by_obj.index),
        "z": [z_by_obj[o] for o in sorted(z_by_obj.index)],
        f"measured_at_{R_TARGET:g}kpc": [o in measured
                                         for o in sorted(z_by_obj.index)],
    }).to_csv(OUT, index=False)
    print(f"\nWrote {OUT.name}.")


if __name__ == "__main__":
    main()