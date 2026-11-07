"""
05_get_positions_and_epochs.py

Purpose
-------
Resolves two open questions before we can trust any aperture photometry:

  1. WHERE exactly is the SN, in sky coordinates? We pull this from NED in
     the same query we already use for redshift (no extra network cost),
     rather than relying on manual DS9 inspection, which doesn't scale and
     isn't reproducible for a sample this size.

  2. WHEN was each stack taken, relative to when the SN would have faded?
     Type Ia SNe fade from peak brightness at a well-characterised rate: by
     ~100 days post-maximum they have dropped several magnitudes below peak
     in B and V (see e.g. Contreras et al. 2010, AJ, 139, 519, fig. 6, for
     the CSP-I light-curve templates), and by ~1-2 yr they are far below the
     detection limit of a 1m-2.5m-class telescope stack. We can't determine
     from a header alone whether a *specific* stack still contains SN light
     without the SN's date of maximum brightness, which we don't have here
     -- but we CAN extract each stack's observation date, which is the other
     half of the comparison your professor's answer will complete.

This script does not resolve the contamination question by itself -- it
prepares the evidence (position + observation epoch) so that, combined with
your professor's answer about how these stacks were constructed, we know
exactly how to interpret every aperture measurement that follows.

Usage
-----
    python 05_get_positions_and_epochs.py --data-dir "D:\\Thesis\\pd\\CSPAll" --out-csv results\\positions_epochs.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

from astropy.io import fits

FILENAME_PATTERN = re.compile(
    r"^(?P<object>.+?)_(?P<filter>[A-Za-z]+)_comb_(?P<telescope>dup|swo)\.fits$",
    re.IGNORECASE,
)
TELESCOPE_NAMES = {"dup": "duPont", "swo": "Swope"}

DATE_KEYS = ["DATE-OBS", "DATE_OBS", "DATEOBS"]
MJD_KEYS = ["MJD-OBS", "MJD_OBS", "MJD"]


def parse_filename(path: Path):
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    tel_code = m.group("telescope").lower()
    return {
        "file": str(path),
        "object": m.group("object"),
        "filter": m.group("filter").upper(),
        "telescope": TELESCOPE_NAMES.get(tel_code, tel_code),
    }


def get_obs_date(path: Path):
    """Extract an observation date/MJD from a FITS header, checking common
    keyword variants and both the primary and first data-bearing HDU."""
    try:
        with fits.open(path) as hdul:
            headers_to_check = [hdul[0].header]
            for hdu in hdul:
                if hdu.data is not None and hdu.header is not headers_to_check[0]:
                    headers_to_check.append(hdu.header)
                    break
            for header in headers_to_check:
                for k in DATE_KEYS:
                    if k in header:
                        return "date", header[k]
                for k in MJD_KEYS:
                    if k in header:
                        return "mjd", header[k]
    except Exception as e:  # noqa: BLE001
        return "error", str(e)
    return None, None


def normalise_sn_name(name: str):
    variants = [name]
    m = re.match(r"^SN(\d{2})([A-Za-z]+)$", name)
    if m:
        yy, suffix = m.groups()
        full = f"20{yy}{suffix}"
        variants += [f"SN{full}", f"SN {full}"]
    m = re.match(r"^ASAS(\d{2})([A-Za-z]+)$", name)
    if m:
        yy, suffix = m.groups()
        full = f"20{yy}{suffix}"
        variants += [f"SN{full}", f"SN {full}"]
    return variants


def query_ned_position_and_redshift(name_variants, max_attempts=3, base_delay=3):
    """Query NED for RA, Dec, and redshift together (one query serves both
    needs -- avoids double-querying the same service)."""
    from astroquery.ipac.ned import Ned

    last_error = ""
    for variant in name_variants:
        for attempt in range(max_attempts):
            try:
                result_table = Ned.query_object(variant)
                if result_table is not None and len(result_table) > 0:
                    row = result_table[0]
                    ra = float(row["RA"]) if "RA" in result_table.colnames else None
                    dec = float(row["DEC"]) if "DEC" in result_table.colnames else None
                    z = row["Redshift"] if "Redshift" in result_table.colnames else None
                    z = float(z) if z is not None and str(z) not in ("--", "masked", "") else None
                    if ra is not None and dec is not None:
                        return ra, dec, z, variant, ""
                last_error = f"{variant}: no RA/Dec in NED record"
                break
            except Exception as e:  # noqa: BLE001
                last_error = f"{variant}: {e}"
                if "timed out" in str(e).lower() and attempt < max_attempts - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                break
        time.sleep(1)
    return None, None, None, None, last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--skip-ned", action="store_true",
                         help="Only extract header dates; skip NED position lookup.")
    args = parser.parse_args()

    fits_files = sorted(
        list(args.data_dir.rglob("*.fits")) + list(args.data_dir.rglob("*.fits.fz"))
    )
    parsed = [parse_filename(p) for p in fits_files]
    parsed = [row for row in parsed if row is not None]

    print(f"Extracting observation dates from {len(parsed)} files...")
    for row in parsed:
        kind, val = get_obs_date(Path(row["file"]))
        row["date_kind"] = kind or ""
        row["date_val"] = val if val is not None else ""

    n_with_date = sum(1 for row in parsed if row["date_val"] != "")
    print(f"  {n_with_date}/{len(parsed)} files have a usable date/MJD keyword.")

    position_map = {}
    if not args.skip_ned:
        objects = sorted(set(row["object"] for row in parsed))
        print(f"\nQuerying NED for positions (+ redshift) for {len(objects)} objects...")
        for obj in objects:
            variants = normalise_sn_name(obj)
            ra, dec, z, resolved_name, err = query_ned_position_and_redshift(variants)
            position_map[obj] = (ra, dec, z, resolved_name, err)
            status = f"RA={ra}, Dec={dec}, z={z}" if ra is not None else f"FAILED ({err})"
            print(f"  {obj:15s} -> {status}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "object", "filter", "telescope",
                          "date_kind", "date_val", "ra_deg", "dec_deg",
                          "redshift", "ned_resolved_name", "ned_error"])
        for row in parsed:
            ra, dec, z, resolved_name, err = position_map.get(
                row["object"], (None, None, None, None, "")
            )
            writer.writerow([row["file"], row["object"], row["filter"], row["telescope"],
                              row["date_kind"], row["date_val"], ra, dec, z,
                              resolved_name, err])

    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()