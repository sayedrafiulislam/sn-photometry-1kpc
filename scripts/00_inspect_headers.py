"""
00_inspect_headers.py

Purpose
-------
First reconnaissance step for a new FITS dataset. Before we can design a
photometry pipeline (Section 2.2.3-style local aperture photometry, following
Kelsey et al. 2021), we need to know exactly what metadata each file carries:

    - Is there a WCS solution (for converting SN sky coordinates -> pixels)?
    - Is there a redshift keyword anywhere in the header?
    - Is there an SN position keyword, or do we need a separate source list?
    - What filter/band is each file (we were told "B-V files")?
    - What are the pixel scale and image dimensions?

This script does NOT assume any of that -- it just opens every FITS file it
finds under a directory, dumps the header, and produces a summary CSV so we
can decide the next step with real information instead of guessing.

Change log
----------
v2: Removed the --limit option. The original run used --limit and, because the
    file listing is sorted alphabetically, it happened to sample only du Pont
    (_dup) frames. The plate scale of 0.23 arcsec/pixel was therefore verified
    for du Pont only, yet is hard-coded downstream in 06_measure_psf_fwhm.py and
    10_curve_of_growth.py and applied to Swope (_swo) frames as well. This
    version inspects every file and prints the measured pixel scale grouped by
    telescope, so that assumption is either confirmed or caught.

Usage
-----
    python 00_inspect_headers.py --data-dir D:\\Thesis\\pd\\CSPAll --out-csv results\\header_summary_full.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS


# Keywords we will actively look for. FITS headers are not standardized
# across pipelines, so we check a list of common aliases for each physical
# quantity rather than a single hard-coded key.
REDSHIFT_KEYS = ["Z", "REDSHIFT", "SN_Z", "ZSN", "ZCMB", "ZHEL"]
RA_KEYS = ["RA", "SN_RA", "RA_SN", "CRVAL1"]
DEC_KEYS = ["DEC", "SN_DEC", "DEC_SN", "CRVAL2"]
FILTER_KEYS = ["FILTER", "BAND", "FILTNAME"]
OBJECT_KEYS = ["OBJECT", "SN_NAME", "TARGET", "SNID"]

# Filename convention: <object>_<filter>_comb_<telescope>.fits
# These tags are the only record of which telescope took a given frame,
# since the headers carry no telescope keyword.
TELESCOPE_TAGS = {"dup": "du Pont", "swo": "Swope"}


def telescope_from_filename(path: Path) -> str:
    """Infer the telescope from the filename tag. Returns 'unknown' if absent."""
    stem = path.stem.lower()
    for tag, name in TELESCOPE_TAGS.items():
        if stem.endswith("_" + tag) or ("_" + tag + "_") in stem:
            return name
    return "unknown"


def filter_from_filename(path: Path) -> str:
    """Infer the filter from the filename convention. Returns '' if absent."""
    parts = path.stem.split("_")
    # Expect <object>_<filter>_comb_<telescope>
    if len(parts) >= 2:
        return parts[1]
    return ""


def find_first_present(header: fits.Header, keys: list[str]):
    """Return (key, value) for the first key in `keys` present in header."""
    for k in keys:
        if k in header:
            return k, header[k]
    return None, None


def inspect_file(path: Path) -> dict:
    """Open one FITS file and extract everything relevant into a flat dict."""
    row = {
        "file": str(path),
        "readable": True,
        "error": "",
        "wcs_error": "",
        "telescope_from_name": telescope_from_filename(path),
        "filter_from_name": filter_from_filename(path),
    }

    try:
        with fits.open(path) as hdul:
            # Some pipelines put WCS/metadata in the primary HDU, others in
            # extension 1 (common for .fits.fz compressed images). We check
            # both and record which one had a usable header.
            header = hdul[0].header
            data_shape = None
            for hdu in hdul:
                if hdu.data is not None:
                    data_shape = hdu.data.shape
                    if hdu.header:
                        header = hdu.header
                    break

            row["n_hdus"] = len(hdul)
            row["data_shape"] = str(data_shape)

            z_key, z_val = find_first_present(header, REDSHIFT_KEYS)
            ra_key, ra_val = find_first_present(header, RA_KEYS)
            dec_key, dec_val = find_first_present(header, DEC_KEYS)
            filt_key, filt_val = find_first_present(header, FILTER_KEYS)
            obj_key, obj_val = find_first_present(header, OBJECT_KEYS)

            row["redshift_key"] = z_key or ""
            row["redshift_val"] = z_val if z_val is not None else ""
            row["ra_key"] = ra_key or ""
            row["ra_val"] = ra_val if ra_val is not None else ""
            row["dec_key"] = dec_key or ""
            row["dec_val"] = dec_val if dec_val is not None else ""
            row["filter_key"] = filt_key or ""
            row["filter_val"] = filt_val if filt_val is not None else ""
            row["object_key"] = obj_key or ""
            row["object_val"] = obj_val if obj_val is not None else ""

            # Check WCS validity -- a header can have CRVAL/CRPIX keywords
            # that don't actually form a usable World Coordinate System.
            try:
                wcs = WCS(header)
                row["has_wcs"] = wcs.has_celestial
                if wcs.has_celestial:
                    scale = wcs.proj_plane_pixel_scales()
                    row["pixscale_arcsec"] = float(
                        (scale[0].to("arcsec").value + scale[1].to("arcsec").value) / 2
                    )
                else:
                    row["pixscale_arcsec"] = ""
            except Exception as e:  # noqa: BLE001
                row["has_wcs"] = False
                row["pixscale_arcsec"] = ""
                row["wcs_error"] = str(e)

    except Exception as e:  # noqa: BLE001
        row["readable"] = False
        row["error"] = str(e)

    return row


def summarise_pixel_scales(rows: list[dict]) -> None:
    """
    Print the measured pixel scale grouped by telescope.

    This is the check that motivated dropping --limit: the value 0.23
    arcsec/pixel is hard-coded in scripts 04 and 10 and applied to every
    frame, but was originally verified on du Pont frames only. If Swope
    frames report a different scale, every Swope aperture radius in kpc and
    every Swope FWHM in arcsec is wrong -- and since the Swope frames are
    V-band only, that would show up as a systematic B-V colour error in
    part of the sample rather than a harmless offset.
    """
    by_tel = defaultdict(list)
    for r in rows:
        ps = r.get("pixscale_arcsec", "")
        if ps not in ("", None):
            by_tel[r.get("telescope_from_name", "unknown")].append(float(ps))

    print("\nPixel scale by telescope (arcsec/pixel):")
    if not by_tel:
        print("  No valid WCS found in any file -- nothing to summarise.")
        return

    for tel in sorted(by_tel):
        vals = by_tel[tel]
        # Round before taking the unique set, otherwise floating-point noise
        # (0.23000000000000403 vs 0.2300000000000012) reports spurious values.
        uniq = sorted({round(v, 4) for v in vals})
        print(f"  {tel:>8s}: n={len(vals):4d}  min={min(vals):.5f}  "
              f"max={max(vals):.5f}  distinct(4dp)={uniq}")

    all_uniq = sorted({round(v, 4) for vals in by_tel.values() for v in vals})
    if len(all_uniq) == 1:
        print(f"\n  -> Single plate scale across the whole data set: {all_uniq[0]} arcsec/pixel.")
        print("     The hard-coded value in scripts 04 and 10 is justified for all frames.")
    else:
        print(f"\n  -> WARNING: {len(all_uniq)} distinct plate scales found: {all_uniq}")
        print("     Scripts 04 and 10 hard-code a single value and must be corrected")
        print("     to read the scale from each file's own WCS.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing FITS files (searched recursively).")
    parser.add_argument("--out-csv", required=True, type=Path,
                        help="Where to write the summary CSV.")
    args = parser.parse_args()

    fits_files = sorted(
        list(args.data_dir.rglob("*.fits")) + list(args.data_dir.rglob("*.fits.fz"))
    )

    if not fits_files:
        print(f"No .fits or .fits.fz files found under {args.data_dir}")
        return

    print(f"Inspecting {len(fits_files)} files (no limit applied)...")
    rows = []
    for i, p in enumerate(fits_files, start=1):
        rows.append(inspect_file(p))
        # Progress ticker -- a full run over ~700 frames is not instantaneous,
        # and silence for several minutes is indistinguishable from a hang.
        if i % 50 == 0 or i == len(fits_files):
            print(f"  ...{i}/{len(fits_files)}")

    fieldnames = sorted(set(k for row in rows for k in row.keys()))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote summary to {args.out_csv}")

    # Quick console summary so you get an immediate answer, not just a file.
    n_with_z = sum(1 for r in rows if r.get("redshift_val") not in ("", None))
    n_with_wcs = sum(1 for r in rows if r.get("has_wcs"))
    n_with_filt = sum(1 for r in rows if r.get("filter_val") not in ("", None))
    n_unreadable = sum(1 for r in rows if not r.get("readable"))

    print(f"Files inspected:                      {len(rows)}")
    print(f"Files unreadable:                     {n_unreadable}")
    print(f"Files with a redshift keyword found:  {n_with_z}/{len(rows)}")
    print(f"Files with a filter keyword found:    {n_with_filt}/{len(rows)}")
    print(f"Files with valid celestial WCS:       {n_with_wcs}/{len(rows)}")

    # Frame counts per telescope/filter as recovered from the filename,
    # which is the only place that information exists.
    counts = defaultdict(int)
    for r in rows:
        counts[(r.get("telescope_from_name", "unknown"),
                r.get("filter_from_name", ""))] += 1
    print("\nFrame counts by telescope and filter (from filename):")
    for (tel, filt), n in sorted(counts.items()):
        print(f"  {tel:>8s}  {filt or '(none)':>6s}: {n}")

    summarise_pixel_scales(rows)


if __name__ == "__main__":
    main()