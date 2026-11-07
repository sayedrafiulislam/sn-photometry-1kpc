"""
check_dr3_crossmatch.py

Audit script, not production.

Cross-match the NED positions from step 10 against the CSP-I DR3 object
catalogue of Krisciunas et al. (2017, AJ 154, 211), served by VizieR as
J/AJ/154/211/catalog.

Two purposes, and the second matters more than the first
--------------------------------------------------------
(1) Supply better positions for the ~20 objects whose NED coordinates
    are quoted only to 3.6 arcsec. DR3 gives RA to 0.01 s and Dec to
    0.1 arcsec, roughly 0.1 arcsec overall, or under 0.1 kpc at the
    sample median redshift.

(2) Measure the NED-to-DR3 separation for objects where BOTH sources are
    precise. This is the only available test of astrometric ACCURACY as
    opposed to quotation precision. It cannot be done on the images: the
    CSP template stacks contain no supernova light, so there is nothing
    at the explosion site to check a coordinate against. If independently
    derived positions agree at the sub-arcsecond level, the NED positions
    are demonstrably accurate and not merely finely written.

DR3 covers CSP-I only. Survey-internal designations (ASAS, LSQ, KISS,
PS1, CSP) are CSP-II and are expected to be absent; their absence is a
fact about the catalogue, not a failure of the match.

Name convention: VizieR drops the SN prefix, so SN04dt -> 2004dt and
SN2011im -> 2011im. The two-digit expansion is the same one used in
step 10 and is unambiguous because CSP spans 2004-2015.

Nothing is modified. This writes one new file.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
COORDS = RESULTS / "sn_coordinates.csv"
PRECISION = RESULTS / "coordinate_precision_audit.csv"
OUT = RESULTS / "dr3_crossmatch.csv"

CATALOG = "J/AJ/154/211/catalog"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Objects written to 5 decimals are quantised at 0.036 arcsec and are
# treated as precise; anything coarser is a candidate for replacement.
PRECISE_DECIMALS = 5


def vizier_name(object_name: str) -> str | None:
    """Convert a catalogue object name to the DR3 'SN' column form."""
    m = re.match(r"^SN(\d{4})([A-Za-z]+)$", object_name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = re.match(r"^SN(\d{2})([A-Za-z]+)$", object_name)
    if m:
        return f"20{m.group(1)}{m.group(2)}"
    return None          # survey designation: CSP-II, not in DR3


def main() -> None:
    for path in (COORDS, PRECISION):
        if not path.exists():
            raise SystemExit(f"FATAL: {path.name} not found.")

    coords = pd.read_csv(COORDS)
    precision = pd.read_csv(PRECISION)[["object", "decimals"]]
    df = coords.merge(precision, on="object", how="left")

    z_col = "z_catalog" if "z_catalog" in df.columns else "redshift"

    print(f"Downloading {CATALOG} ...")
    viz = Vizier(columns=["SN", "RAJ2000", "DEJ2000", "z"])
    viz.ROW_LIMIT = -1
    dr3 = viz.get_catalogs(CATALOG)[0].to_pandas()
    dr3["SN"] = dr3["SN"].astype(str).str.strip()
    print(f"  {len(dr3)} objects in DR3.\n")

    dr3_coord = SkyCoord(dr3["RAJ2000"].values, dr3["DEJ2000"].values,
                         unit=(u.hourangle, u.deg))

    df["vizier_name"] = df["object"].apply(vizier_name)
    lookup = {n: i for i, n in enumerate(dr3["SN"])}

    rows = []
    for rec in df.itertuples(index=False):
        vname = rec.vizier_name
        idx = lookup.get(vname) if vname else None

        row = {"object": rec.object, "vizier_name": vname or "",
               "in_dr3": idx is not None,
               "ned_decimals": rec.decimals,
               "redshift": getattr(rec, z_col),
               "ned_ra": rec.ra_deg, "ned_dec": rec.dec_deg,
               "dr3_ra": np.nan, "dr3_dec": np.nan,
               "separation_arcsec": np.nan, "separation_kpc": np.nan,
               "dr3_z": np.nan, "dz": np.nan}

        if idx is not None:
            ned = SkyCoord(rec.ra_deg * u.deg, rec.dec_deg * u.deg)
            sep = ned.separation(dr3_coord[idx]).arcsec
            kpc_per_as = (COSMO.kpc_proper_per_arcmin(row["redshift"])
                          .to(u.kpc / u.arcsec).value)

            row["dr3_ra"] = dr3_coord[idx].ra.deg
            row["dr3_dec"] = dr3_coord[idx].dec.deg
            row["separation_arcsec"] = sep
            row["separation_kpc"] = sep * kpc_per_as
            dr3_z = pd.to_numeric(dr3["z"].iloc[idx], errors="coerce")
            row["dr3_z"] = dr3_z
            row["dz"] = abs(dr3_z - row["redshift"])

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    matched = out[out["in_dr3"]]
    print("=" * 72)
    print("COVERAGE")
    print("=" * 72)
    print(f"  matched to DR3        : {len(matched)} / {len(out)}")
    print(f"  not in DR3 (CSP-II)   : {len(out) - len(matched)}")

    coarse = out[out["ned_decimals"] < PRECISE_DECIMALS]
    coarse_fixed = coarse[coarse["in_dr3"]]
    print(f"\n  coarse NED positions          : {len(coarse)}")
    print(f"    of which DR3 can replace    : {len(coarse_fixed)}")
    print(f"    still needing another source: {len(coarse) - len(coarse_fixed)}")

    # ----------------------------------------------------------------
    # The accuracy test: both sources precise, independently derived
    # ----------------------------------------------------------------
    clean = matched[(matched["ned_decimals"] >= PRECISE_DECIMALS)
                    & matched["separation_arcsec"].notna()]

    print("\n" + "=" * 72)
    print("ASTROMETRIC ACCURACY -- NED vs DR3, precise objects only")
    print("=" * 72)
    if len(clean) == 0:
        print("  no objects qualify.")
    else:
        s = clean["separation_arcsec"]
        print(f"  n = {len(clean)}")
        print(f"  median separation : {s.median():.3f} arcsec "
              f"({clean['separation_kpc'].median():.4f} kpc)")
        print(f"  90th percentile   : {s.quantile(0.90):.3f} arcsec")
        print(f"  maximum           : {s.max():.3f} arcsec")
        for thr in (0.5, 1.0, 2.0):
            print(f"  separation > {thr:.1f} arcsec : {int((s > thr).sum())}")

        worst = clean.nlargest(8, "separation_arcsec")[
            ["object", "redshift", "separation_arcsec", "separation_kpc", "dz"]]
        print("\n  largest separations:")
        print(worst.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))

    # Redshift agreement -- a second, independent consistency check
    dz = matched["dz"].dropna()
    if len(dz):
        print("\n" + "=" * 72)
        print("REDSHIFT AGREEMENT -- NED vs DR3")
        print("=" * 72)
        print(f"  n = {len(dz)}, median |dz| = {dz.median():.5f}, "
              f"max = {dz.max():.5f}")
        bad = matched[matched["dz"] > 0.002][["object", "redshift", "dr3_z", "dz"]]
        if len(bad):
            print(f"\n  {len(bad)} object(s) disagree by more than 0.002:")
            print(bad.to_string(index=False,
                                float_format=lambda v: f"{v:.5f}"))
        else:
            print("  no object disagrees by more than 0.002.")

    print(f"\nWrote {OUT.name}.")


if __name__ == "__main__":
    main()