"""
check_ned_position_uncertainty.py

Audit script, not production. Follows check_coordinate_precision.py.

Why this exists
---------------
The digit-counting audit inferred astrometric precision from the number
of decimal places written to CSV. That inference is unreliable: pandas
strips trailing zeros, so a position genuinely specified to five
decimals is indistinguishable from a coarser one whenever its last
digit happens to be zero. Roughly one in ten five-decimal positions is
misread as four-decimal, and one in a hundred as three-decimal, purely
by chance.

Rather than refine the inference, this script replaces it. NED holds a
separate 'positions' table for each object listing every catalogued
measurement with its uncertainty ellipse and originating reference.
That is a stated error bar rather than a guess, and it is citable.

Scope
-----
Only objects whose written precision was 4 decimals or coarser are
queried -- the objects where the digit count leaves any doubt. Positions
written to 5 decimals are quantised at 0.036 arcsec, which is far below
any plausible astrometric error and below a fifth of a du Pont pixel,
so they need no further checking.

Output is deliberately raw. The NED positions table has changed column
names across astroquery versions and between object types, so this
script prints what it actually receives rather than assuming a schema,
and writes everything to CSV for inspection before any cut is made.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astroquery.ipac.ned import Ned

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
AUDIT = RESULTS / "coordinate_precision_audit.csv"
OUT = RESULTS / "ned_position_uncertainty.csv"
RAW_DIR = RESULTS / "ned_position_tables"

DECIMAL_THRESHOLD = 4          # query anything this coarse or worse
APERTURE_RADIUS_KPC = 1.0
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

Ned.TIMEOUT = 180
POLITE_DELAY_S = 1.5
MAX_ATTEMPTS = 3


def find_column(colnames, *fragments):
    """First column whose name contains all the given fragments."""
    for c in colnames:
        low = c.lower()
        if all(f.lower() in low for f in fragments):
            return c
    return None


def fetch_positions(name):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return Ned.get_table(name, table="positions"), None
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None, f"{type(exc).__name__}: {str(exc)[:90]}"
    return None, "unknown"


def main() -> None:
    if not AUDIT.exists():
        raise SystemExit(f"FATAL: {AUDIT.name} not found. "
                         f"Run check_coordinate_precision.py first.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT)

    targets = (audit[audit["decimals"] <= DECIMAL_THRESHOLD]
               .sort_values("decimals")
               .reset_index(drop=True))

    print(f"{len(targets)} of {len(audit)} objects written to "
          f"{DECIMAL_THRESHOLD} decimals or coarser.")
    print("Querying NED's positions table for each.\n")

    schema_printed = False
    rows = []

    for i, rec in enumerate(targets.itertuples(index=False), 1):
        table, error = fetch_positions(rec.object)
        time.sleep(POLITE_DELAY_S)

        if table is None or len(table) == 0:
            print(f"[{i}/{len(targets)}] {rec.object:<12} -> FAILED "
                  f"({error or 'empty table'})")
            rows.append({"object": rec.object, "written_decimals": rec.decimals,
                         "redshift": rec.redshift, "n_positions": 0,
                         "best_unc_arcsec": np.nan, "best_refcode": "",
                         "error": error or "empty"})
            continue

        if not schema_printed:
            print("NED positions table columns:")
            for c in table.colnames:
                print(f"    {c}")
            print()
            schema_printed = True

        df = table.to_pandas()
        df.to_csv(RAW_DIR / f"{rec.object}_positions.csv", index=False)

        unc_col = (find_column(df.columns, "semi-major")
                   or find_column(df.columns, "uncertainty"))
        ref_col = find_column(df.columns, "refcode") or find_column(df.columns, "ref")

        best_unc, best_ref = np.nan, ""
        if unc_col is not None:
            unc = pd.to_numeric(df[unc_col], errors="coerce")
            if unc.notna().any():
                idx = unc.idxmin()
                best_unc = float(unc.loc[idx])
                best_ref = str(df.loc[idx, ref_col]) if ref_col else ""

        kpc_per_as = (COSMO.kpc_proper_per_arcmin(rec.redshift)
                      .to(u.kpc / u.arcsec).value)
        best_kpc = best_unc * kpc_per_as if np.isfinite(best_unc) else np.nan

        shown = "n/a" if not np.isfinite(best_unc) else f"{best_unc:.3f}\""
        shown_kpc = "n/a" if not np.isfinite(best_kpc) else f"{best_kpc:.3f} kpc"
        print(f"[{i}/{len(targets)}] {rec.object:<12} "
              f"{len(df):3d} positions, best {shown:>8} = {shown_kpc:>10}  "
              f"{best_ref}")

        rows.append({"object": rec.object, "written_decimals": rec.decimals,
                     "redshift": rec.redshift, "n_positions": len(df),
                     "best_unc_arcsec": best_unc, "best_unc_kpc": best_kpc,
                     "best_refcode": best_ref, "error": ""})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if "best_unc_kpc" in out.columns:
        have = out["best_unc_kpc"].notna()
        print(f"  usable uncertainty found for {int(have.sum())}/{len(out)}")
        if have.any():
            over = out.loc[have, "best_unc_kpc"] > 0.25 * APERTURE_RADIUS_KPC
            print(f"  best available position still exceeds "
                  f"{0.25 * APERTURE_RADIUS_KPC:.2f} kpc for {int(over.sum())} object(s)")
    print(f"\n  per-object tables -> {RAW_DIR.name}\\")
    print(f"  summary           -> {OUT.name}")


if __name__ == "__main__":
    main()