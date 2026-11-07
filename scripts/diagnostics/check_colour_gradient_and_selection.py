"""
check_colour_gradient_and_selection.py

Audit script, not production. Two questions the step 14 summary raises
but cannot answer.

(1) IS THE FLAT MEDIAN COLOUR A REAL ABSENCE OF GRADIENT?
    The medians at 1, 2, 4 and 5 kpc are computed on different numbers
    of objects (186 to 211), so part of the 0.023 mag variation is
    sample composition rather than radial change. Restricting to
    objects measured at every radius removes that ambiguity.

    A flat MEDIAN also does not imply flat colour per object: galaxies
    reddening outward and galaxies bluing outward cancel in a median.
    The per-object difference colour(5 kpc) - colour(1 kpc) is therefore
    computed and its distribution examined, not just its centre.

(2) IS THE 1 kpc SAMPLE BIASED IN REDSHIFT?
    The seeing floor scales with angular diameter distance, so it
    removes distant objects preferentially. In a magnitude-limited
    survey more distant hosts are also more luminous and more massive,
    and host mass correlates with colour. A redshift-biased sample is
    therefore potentially a colour-biased sample, which would matter for
    any comparison against Kelsey et al. (2021).

Instrumental colours. Zero points are one additive constant per frame,
so per-object DIFFERENCES between radii are unaffected by them, while
absolute values are meaningless. The gradient test is valid on
instrumental colour; the distribution's centre is not.

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

R_INNER, R_OUTER = 1.0, 5.0
RADII = [1.0, 2.0, 3.0, 4.0, 5.0]


def main() -> None:
    if not COLOUR.exists():
        raise SystemExit(f"FATAL: {COLOUR.name} not found. Run step 14 first.")

    df = pd.read_csv(COLOUR)
    col = "instrumental_B_minus_V"

    # ----------------------------------------------------------------
    # (1) Common-sample radial comparison
    # ----------------------------------------------------------------
    wide = (df[df["radius_kpc"].isin(RADII)]
            .pivot_table(index=["object", "z"], columns="radius_kpc",
                         values=col))
    common = wide.dropna()

    print("=" * 70)
    print("RADIAL COLOUR, COMMON SAMPLE")
    print("=" * 70)
    print(f"  objects measured at all of {RADII}: {len(common)}")
    print(f"\n  {'radius':>8} {'median':>9} {'16th':>9} {'84th':>9}")
    for r in RADII:
        s = common[r]
        print(f"  {r:8.1f} {s.median():9.4f} {s.quantile(0.16):9.4f} "
              f"{s.quantile(0.84):9.4f}")

    # ----------------------------------------------------------------
    # Per-object gradient
    # ----------------------------------------------------------------
    delta = (common[R_OUTER] - common[R_INNER]).dropna()

    print("\n" + "=" * 70)
    print(f"PER-OBJECT GRADIENT: colour({R_OUTER:g} kpc) - colour({R_INNER:g} kpc)")
    print("=" * 70)
    print(f"  n              : {len(delta)}")
    print(f"  median         : {delta.median():+.4f} mag")
    print(f"  mean           : {delta.mean():+.4f} mag")
    print(f"  std            : {delta.std():.4f} mag")
    print(f"  16th to 84th   : {delta.quantile(0.16):+.4f} to "
          f"{delta.quantile(0.84):+.4f}")
    print(f"  redder outward : {int((delta > 0).sum())} "
          f"({100 * (delta > 0).mean():.1f}%)")
    print(f"  bluer  outward : {int((delta < 0).sum())} "
          f"({100 * (delta < 0).mean():.1f}%)")

    # Is the centre distinguishable from zero?
    t_stat, p_t = stats.ttest_1samp(delta, 0.0)
    w_stat, p_w = stats.wilcoxon(delta)
    print(f"\n  t-test  vs zero : t = {t_stat:+.3f}, p = {p_t:.4f}")
    print(f"  Wilcoxon vs zero: W = {w_stat:.1f}, p = {p_w:.4f}")
    print("\n  A median near zero with a WIDE distribution means individual")
    print("  gradients exist in both directions and cancel. A median near")
    print("  zero with a NARROW distribution means colour really is flat")
    print("  from 1 to 5 kpc in most hosts. The standard deviation above")
    print("  distinguishes these.")

    # ----------------------------------------------------------------
    # (2) Redshift selection
    # ----------------------------------------------------------------
    at1 = df[np.isclose(df["radius_kpc"], R_INNER)]
    measured = set(at1.loc[at1[col].notna(), "object"])
    z_all = df.groupby("object")["z"].first()

    z_in = z_all[z_all.index.isin(measured)]
    z_out = z_all[~z_all.index.isin(measured)]

    print("\n" + "=" * 70)
    print(f"REDSHIFT SELECTION AT {R_INNER:g} kpc")
    print("=" * 70)
    print(f"  with a colour    : n = {len(z_in):3d}  median z = {z_in.median():.4f}"
          f"  range {z_in.min():.4f}-{z_in.max():.4f}")
    if len(z_out):
        print(f"  without          : n = {len(z_out):3d}  "
              f"median z = {z_out.median():.4f}"
              f"  range {z_out.min():.4f}-{z_out.max():.4f}")
        ks, p_ks = stats.ks_2samp(z_in, z_out)
        print(f"\n  KS test on the two redshift distributions: "
              f"D = {ks:.3f}, p = {p_ks:.4f}")
        print("  A small p means the objects that drop out at 1 kpc are NOT a")
        print("  random subset in redshift, and the surviving sample is")
        print("  biased. That must be stated in the paper whether or not it")
        print("  affects the conclusion.")
    else:
        print("  every object has a colour at this radius; no selection.")


if __name__ == "__main__":
    main()