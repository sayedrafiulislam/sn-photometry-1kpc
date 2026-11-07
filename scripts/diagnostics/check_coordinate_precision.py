"""
check_coordinate_precision.py

Audit script, not production.

Question: how precisely does NED specify each supernova's position, and
is that precision sufficient for a 1 kpc aperture?

Why it matters here specifically
--------------------------------
A position quoted to three decimal places in degrees is quantised at
0.001 deg = 3.6 arcsec. At the sample median z = 0.031 one arcsec
subtends roughly 0.6 kpc, so such a position could be mis-centred by
more than twice the radius being measured.

The same coordinate would be harmless in Kelsey et al. (2021), who work
at 4 kpc and z ~ 0.3. Their apertures are large in angular terms and
their centring tolerance is loose. Ours is the opposite case: an order
of magnitude closer, and a quarter of the aperture radius. Astrometric
precision is a first-order error term for this project and a negligible
one for theirs.

Method
------
Decimal places are counted from the CSV text rather than from the
in-memory float, because the number of digits actually written is the
number NED supplied. The implied quantisation is then converted to a
physical scale at each object's OWN redshift, rather than compared
against a single absolute threshold in arcsec -- a fixed angular cut
would be far stricter for a nearby object than a distant one, which is
backwards.

Cosmology matches Kelsey et al. 2021: flat LCDM, H0 = 70, Om = 0.3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

REPO = Path(__file__).resolve().parent.parent
COORDS = REPO / "results" / "sn_coordinates.csv"

APERTURE_RADIUS_KPC = 1.0
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def decimals_as_written(series: pd.Series) -> np.ndarray:
    """Count decimal places in the text of the CSV, not the float."""
    out = []
    for value in series.astype(str):
        value = value.strip()
        out.append(len(value.split(".")[1]) if "." in value else 0)
    return np.array(out)


def main() -> None:
    if not COORDS.exists():
        raise SystemExit(f"FATAL: {COORDS} not found. Run step 10 first.")

    # Read coordinates as text to preserve the written precision.
    coords = pd.read_csv(COORDS, dtype={"ra_deg": str, "dec_deg": str})

    z_col = next((c for c in ("z_catalog", "redshift", "ned_redshift")
                  if c in coords.columns), None)
    if z_col is None:
        raise SystemExit(f"FATAL: no redshift column in {COORDS.name}. "
                         f"Found: {list(coords.columns)}")
    print(f"Using '{z_col}' as the redshift column.\n")

    df = pd.DataFrame({
        "object": coords["object"],
        "redshift": pd.to_numeric(coords[z_col], errors="coerce"),
        "ra_deg": pd.to_numeric(coords["ra_deg"], errors="coerce"),
        "dec_deg": pd.to_numeric(coords["dec_deg"], errors="coerce"),
    })

    # Precision is set by the coarser of the two coordinates.
    df["decimals"] = np.minimum(decimals_as_written(coords["ra_deg"]),
                                decimals_as_written(coords["dec_deg"]))

    # Quantisation step in arcsec. A value rounded to N decimals sits
    # within half a step of the truth, so half is the typical error and
    # the full step is the worst case. The worst case is reported.
    df["quantisation_arcsec"] = 10.0 ** (-df["decimals"]) * 3600.0

    kpc_per_arcsec = (COSMO.kpc_proper_per_arcmin(df["redshift"].values)
                      .to(u.kpc / u.arcsec).value)
    df["kpc_per_arcsec"] = kpc_per_arcsec
    df["centring_error_kpc"] = df["quantisation_arcsec"] * kpc_per_arcsec
    df["error_over_radius"] = df["centring_error_kpc"] / APERTURE_RADIUS_KPC

    print("=" * 74)
    print("DECIMAL PLACES WRITTEN (the coarser of RA and Dec)")
    print("=" * 74)
    for n, count in df["decimals"].value_counts().sort_index().items():
        step_arcsec = 10.0 ** (-n) * 3600.0
        print(f"  {n} decimals -> {step_arcsec:8.3f} arcsec : {count:4d} objects")

    print("\n" + "=" * 74)
    print(f"IMPLIED CENTRING ERROR, RELATIVE TO THE {APERTURE_RADIUS_KPC:.0f} kpc "
          f"APERTURE RADIUS")
    print("=" * 74)
    for frac in (0.10, 0.25, 0.50, 1.00):
        n = int((df["error_over_radius"] > frac).sum())
        print(f"  worst-case offset > {frac * 100:5.0f}% of the radius "
              f"({frac * APERTURE_RADIUS_KPC:.2f} kpc): {n:4d} / {len(df)} objects")

    good = df[df["error_over_radius"] <= 0.10]
    print(f"\n  median centring error, whole sample : "
          f"{df['centring_error_kpc'].median():.4f} kpc")
    print(f"  median for the {len(good)} well-specified objects : "
          f"{good['centring_error_kpc'].median():.4f} kpc")

    print("\n" + "=" * 74)
    print("WORST 15 OBJECTS")
    print("=" * 74)
    worst = df.nlargest(15, "centring_error_kpc")[
        ["object", "redshift", "ra_deg", "dec_deg", "decimals",
         "quantisation_arcsec", "centring_error_kpc", "error_over_radius"]]
    print(worst.to_string(index=False,
                          float_format=lambda v: f"{v:.4f}"))

    out = REPO / "results" / "coordinate_precision_audit.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out.name} ({len(df)} rows).")

    n_bad = int((df["error_over_radius"] > 0.25).sum())
    if n_bad:
        print(f"\n{n_bad} object(s) have a worst-case centring error above a "
              f"quarter of the aperture radius.")
        print("For these, positions should be taken from the CSP data-release "
              "papers -- Krisciunas et al. (2017, AJ 154, 211) for CSP-I and "
              "Phillips et al. (2019, PASP 131, 014001) for CSP-II -- which "
              "publish coordinates to 0.01 s in RA.")
    else:
        print("\nNo object exceeds a quarter of the aperture radius. NED "
              "precision is adequate for this measurement; note it in "
              "Limitations and proceed.")


if __name__ == "__main__":
    main()