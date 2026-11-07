"""
04_apply_redshift_cut.py

Purpose
-------
Splits the NED-queried catalog (output of 02_build_catalog.py) into:

  (1) sn_catalog_final.csv   -- objects with a resolved redshift; this is
      the working sample for all downstream steps (PSF measurement,
      aperture photometry, curve-of-growth analysis).

  (2) excluded_objects_log.csv -- objects with no resolved redshift, kept
      as an explicit, auditable record of what was excluded and why,
      rather than silently dropping them. This file is what the
      methods-section selection-cut statement (Table 2-style) is drawn
      from, and it is what would let a future student pick this sample
      back up and resolve the stragglers via a CSP-II target list or TNS.

Usage
-----
    python 04_apply_redshift_cut.py --in-csv results\\sn_catalog_v2.csv --out-dir results\\
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with open(args.in_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by object, since each object has multiple rows (B/V filters,
    # possibly both telescopes)
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object"]].append(row)

    n_total = len(by_object)
    resolved_objects = {
        obj: obj_rows for obj, obj_rows in by_object.items()
        if obj_rows[0]["redshift"] not in (None, "", "None")
    }
    excluded_objects = {
        obj: obj_rows for obj, obj_rows in by_object.items()
        if obj not in resolved_objects
    }

    n_resolved = len(resolved_objects)
    n_excluded = len(excluded_objects)

    print(f"Total unique objects: {n_total}")
    print(f"Resolved (retained for analysis): {n_resolved}")
    print(f"Excluded (no redshift): {n_excluded}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write the final working catalog -- one row per FITS file, only for
    # resolved objects
    final_path = args.out_dir / "sn_catalog_final.csv"
    with open(final_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for obj_rows in resolved_objects.values():
            writer.writerows(obj_rows)

    # Write the exclusion log -- one row per excluded object, with reason
    log_path = args.out_dir / "excluded_objects_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "n_files_affected", "ned_error", "reason_category"])
        for obj, obj_rows in excluded_objects.items():
            err = obj_rows[0].get("ned_error", "")
            if "EGRET" in err:
                reason = "NED name-checker server error"
            elif obj.startswith("CSP"):
                reason = "CSP-internal designator, no IAU name"
            elif obj.startswith("PTF"):
                reason = "PTF-internal designator, no IAU name"
            elif obj.startswith("LSQ"):
                reason = "LSQ-internal designator, no IAU name"
            else:
                reason = "not recognized by NED name interpreter"
            writer.writerow([obj, len(obj_rows), err, reason])

    print(f"\nWrote working catalog ({n_resolved} objects) to {final_path}")
    print(f"Wrote exclusion log ({n_excluded} objects) to {log_path}")


if __name__ == "__main__":
    main()