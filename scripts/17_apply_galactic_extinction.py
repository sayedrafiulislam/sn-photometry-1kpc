"""
17_apply_galactic_extinction.py

Remove Milky Way dust reddening from the calibrated 1 kpc colours.

WHAT THIS DOES
--------------
Light from a host galaxy passes through the Milky Way's dust on its way
to Las Campanas. Dust scatters blue light more than red, so everything
looks redder than it is. The SFD98 map gives the reddening along any
line of sight; Schlafly & Finkbeiner (2011) give the extinction per
band.

    A_B = 3.626 * E(B-V)_SFD        A_V = 2.742 * E(B-V)_SFD
    corrected colour = observed - (A_B - A_V)

This matters here because the sample spans the whole southern sky and
therefore sits behind very different dust columns. Without the
correction, part of the measured colour scatter is Milky Way geography
rather than host astrophysics.

THE SF11 RECALIBRATION ENTERS THROUGH THE COEFFICIENTS
------------------------------------------------------
This is subtle and a reader will look for something that is not there.

The coefficients above are Schlafly & Finkbeiner (2011) Table 6 for
R_V = 3.1, applied to the UNMODIFIED SFD98 map value. Their difference
is

    3.626 - 2.742 = 0.884

so the colour excess applied is 0.884 x E(B-V)_SFD, not 1.0 x E(B-V).

That is deliberate. SF11 established that SFD98 overestimates E(B-V) by
roughly 14 per cent, a recalibration factor of 0.86. Applying SF11's
per-band coefficients to the raw SFD map reproduces that recalibration
automatically, so no separate 0.86 scaling appears anywhere in this
script. Cite both papers, and say in the text that the recalibration
enters through the coefficients -- otherwise a reader will search for a
factor that was never written down.

Schlegel, Finkbeiner & Davis (1998), ApJ 500, 525 -- the map, and the
16 per cent uncertainty adopted below.
Schlafly & Finkbeiner (2011), ApJ 737, 103, Table 6 -- the coefficients.

FILTER ON flag_exclude
----------------------
An earlier version filtered on flag_low_flux, which was one criterion of
an older scheme. Step 16 writes the combined result as flag_exclude.
Filtering on the old flag means the calibration-quality and manual
criteria never reach the final sample. This script prefers flag_exclude
and falls back only with a loud warning.

NAMING
------
colour_excess_BV holds A_B - A_V = 0.884 x E(B-V), NOT E(B-V) itself. An
earlier version called this ebv_applied, which contradicted its own
printed explanation. The legacy name is retained as a duplicate column
so nothing downstream breaks.

STEP 16 DOES NOT NEED RE-RUNNING AFTER THIS
-------------------------------------------
Galactic extinction is a single additive constant per object, so it
cancels in any within-object comparison across radii; and step 16
operates partly on instrumental colours, which never receive the
correction.

OUTPUT
------
  calibrated_colour_1kpc_dered.csv

All rows are written with flags preserved. Filtering is left to the
plotting steps.
"""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# sfdmap refers to np.int / np.float / np.bool, removed in numpy 1.24.
# Restored here ONLY so the import succeeds. This is a global patch and
# affects every module imported after this point.
# --------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

# --------------------------------------------------------------------
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
CALIB_DIR = REPO / "results" / "phase5_calibration"

IN_PATH = CALIB_DIR / "calibrated_colour_1kpc_flagged.csv"
COORDS_CSV = REPO / "results" / "sn_coordinates.csv"
OUT_PATH = CALIB_DIR / "calibrated_colour_1kpc_dered.csv"

# The SFD98 maps live outside the repository: they are 128 MB of static
# reference data shared between projects, not a product of this pipeline.
DUST_DIR = Path(r"E:\dustmaps")

# Schlafly & Finkbeiner (2011) Table 6, R_V = 3.1, per unit E(B-V)_SFD.
COEFF_B = 3.626
COEFF_V = 2.742

# SFD98 quote a 16 per cent uncertainty on the map value.
EBV_FRAC_ERR = 0.16


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def load_dust_query():
    """
    sfdmap2 is the maintained fork; sfdmap is the original and no longer
    updated. Either is accepted.
    """
    try:
        import sfdmap2.sfdmap as sfdmap
    except ImportError:
        try:
            import sfdmap
        except ImportError:
            sys.exit("\n  No SFD dust map package installed.\n\n"
                     "  Run: pip install sfdmap2\n")

    try:
        return sfdmap.SFDMap(str(DUST_DIR))
    except Exception as exc:
        sys.exit(f"\n  SFD maps not found in:\n    {DUST_DIR}\n\n"
                 f"  ({type(exc).__name__}: {exc})\n\n"
                 f"  Download the SFD98 dust maps and place the .fits files\n"
                 f"  there, then re-run. They are about 150 MB.\n")


def mad(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def main() -> None:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    for path, what in [(IN_PATH, "flagged colours (run step 16 first)"),
                       (COORDS_CSV, "SN coordinates (run step 10 first)")]:
        if not path.exists():
            sys.exit(f"\n  Input not found -- {what}:\n    {path}\n")

    calib = pd.read_csv(IN_PATH)
    coords = pd.read_csv(COORDS_CSV)

    if "flag_exclude" in calib.columns:
        flag_col = "flag_exclude"
    elif "flag_low_flux" in calib.columns:
        flag_col = "flag_low_flux"
        print("[WARN] no flag_exclude column -- this catalogue predates "
              "16_flag_unreliable_colours.py. Falling back to flag_low_flux, "
              "which applies only one criterion. The calibration-quality and "
              "manual criteria will NOT be reflected below.")
    else:
        flag_col = None
        print("[WARN] no exclusion flag at all -- reporting on the full sample.")

    df = calib.merge(coords[["object", "ra_deg", "dec_deg"]],
                     on="object", how="left")

    n_no_coord = int(df["ra_deg"].isna().sum())
    if n_no_coord:
        print(f"[warn] {n_no_coord} objects have no resolved coordinate. Their "
              f"dereddened colour will be NaN rather than silently left "
              f"uncorrected.")

    sfd = load_dust_query()

    has_coord = df["ra_deg"].notna() & df["dec_deg"].notna()
    df["ebv_sfd"] = np.nan
    if has_coord.any():
        df.loc[has_coord, "ebv_sfd"] = sfd.ebv(
            df.loc[has_coord, "ra_deg"].values,
            df.loc[has_coord, "dec_deg"].values)

    df["A_B"] = COEFF_B * df["ebv_sfd"]
    df["A_V"] = COEFF_V * df["ebv_sfd"]

    # A_B - A_V is the colour excess, NOT E(B-V). It equals
    # (3.626 - 2.742) x E(B-V)_SFD = 0.884 x E(B-V)_SFD.
    df["colour_excess_BV"] = df["A_B"] - df["A_V"]
    df["ebv_applied"] = df["colour_excess_BV"]   # legacy name, kept

    df["mag_B_dered"] = df["mag_B"] - df["A_B"]
    df["mag_V_dered"] = df["mag_V"] - df["A_V"]
    df["B_minus_V_dered"] = df["B_minus_V"] - df["colour_excess_BV"]

    ebv_err = EBV_FRAC_ERR * df["colour_excess_BV"].abs()
    df["B_minus_V_dered_err"] = np.sqrt(df["B_minus_V_err"] ** 2 + ebv_err ** 2)

    df["gal_b_deg"] = np.nan
    if has_coord.any():
        gal = SkyCoord(df.loc[has_coord, "ra_deg"].values * u.deg,
                       df.loc[has_coord, "dec_deg"].values * u.deg,
                       frame="icrs").galactic
        df.loc[has_coord, "gal_b_deg"] = gal.b.deg

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    backup(OUT_PATH)
    df.to_csv(OUT_PATH, index=False)

    # ----------------------------------------------------------------
    keep = (~df[flag_col].astype(bool) if flag_col
            else pd.Series(True, index=df.index))
    clean = df[keep & df["B_minus_V_dered"].notna()]
    obs, ded = clean["B_minus_V"], clean["B_minus_V_dered"]

    print(f"\nWrote {len(df)} objects -> {OUT_PATH.name}")
    print("\n" + "=" * 74)
    print("GALACTIC EXTINCTION CORRECTION")
    print("=" * 74)
    print(f"Exclusion flag used  : {flag_col}")
    print(f"Clean sample         : {len(clean)} objects")
    print(f"|b| range            : "
          f"{np.nanmin(np.abs(clean['gal_b_deg'])):.1f} - "
          f"{np.nanmax(np.abs(clean['gal_b_deg'])):.1f} deg")
    print()
    print(f"E(B-V)_SFD  median   : {clean['ebv_sfd'].median():.4f} mag")
    print(f"            16-84th  : {clean['ebv_sfd'].quantile(0.16):.4f} - "
          f"{clean['ebv_sfd'].quantile(0.84):.4f} mag")
    print(f"            maximum  : {clean['ebv_sfd'].max():.4f} mag "
          f"({clean.loc[clean['ebv_sfd'].idxmax(), 'object']})")
    print(f"Colour excess applied, median : "
          f"{clean['colour_excess_BV'].median():.4f} mag")
    print(f"   = ({COEFF_B:.3f} - {COEFF_V:.3f}) x E(B-V)_SFD "
          f"= {COEFF_B - COEFF_V:.3f} x E(B-V)_SFD")
    print("   The SF11 recalibration enters through these coefficients;")
    print("   there is deliberately no separate 0.86 factor.")
    print()
    print(f"{'':32s}{'observed':>12s}{'dereddened':>13s}{'change':>10s}")
    for label, fo, fd in [("median B-V (mag)", obs.median(), ded.median()),
                          ("mean B-V (mag)", obs.mean(), ded.mean()),
                          ("scatter, sigma_MAD (mag)", mad(obs), mad(ded)),
                          ("std dev (mag)", obs.std(), ded.std()),
                          ("16th percentile", obs.quantile(.16), ded.quantile(.16)),
                          ("84th percentile", obs.quantile(.84), ded.quantile(.84))]:
        print(f"  {label:30s}{fo:12.4f}{fd:13.4f}{fd - fo:+10.4f}")

    var = mad(obs) ** 2 - mad(ded) ** 2
    if var > 0:
        print(f"\n  implied foreground scatter term : {np.sqrt(var):.4f} mag "
              f"(removed in quadrature)")
    else:
        print(f"\n  scatter did not decrease; the foreground term is not "
              f"resolvable in this sample")

    # ----------------------------------------------------------------
    # The test that the map was queried correctly: any correlation
    # between dust column and measured colour should be removed by the
    # correction. A surviving correlation means the map is being read at
    # the wrong coordinates, or something else in the sample tracks
    # Galactic latitude -- field crowding at low |b| is the usual
    # culprit.
    # ----------------------------------------------------------------
    try:
        from scipy import stats
        r_o, p_o = stats.spearmanr(clean["ebv_sfd"], clean["B_minus_V"])
        r_d, p_d = stats.spearmanr(clean["ebv_sfd"], clean["B_minus_V_dered"])
        print(f"\n  Spearman rho(E(B-V), colour)")
        print(f"    before correction : {r_o:+.3f}  (p = {p_o:.4f})")
        print(f"    after correction  : {r_d:+.3f}  (p = {p_d:.4f})")
        print("    The residual should be consistent with zero.")

        hi_b = clean[np.abs(clean["gal_b_deg"]) >= 20]
        n_low = len(clean) - len(hi_b)
        if n_low > 5 and len(hi_b) > 5:
            r_hi, p_hi = stats.spearmanr(hi_b["ebv_sfd"],
                                         hi_b["B_minus_V_dered"])
            print(f"    excluding |b| < 20 deg ({n_low} objects): "
                  f"rho = {r_hi:+.3f} (p = {p_hi:.4f})")
    except ImportError:
        pass

    print("\n  This is the final catalogue. B_minus_V_dered is the measured")
    print("  local host colour within 1 kpc of the supernova position,")
    print("  corrected for Milky Way reddening.")
    print("=" * 74)


if __name__ == "__main__":
    main()