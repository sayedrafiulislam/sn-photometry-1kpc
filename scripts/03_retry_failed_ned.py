"""
03_retry_failed_ned.py

Purpose
-------
Re-query NED for the objects that 02_build_catalog.py failed to resolve, in
order to separate two things that the original run conflated:

    (a) objects NED genuinely cannot resolve, because the name is a
        survey-internal designator that was never assigned an IAU name
        (CSP12A, PTF13ebh, LSQ12fxd, ...), and
    (b) objects that failed only because the NED service returned an error
        or timed out at that moment.

Why this is needed
------------------
02_build_catalog.py retries a query only when the exception message contains
"timeout" or "timed out":

    is_timeout = "timed out" in str(e).lower() or "timeout" in str(e).lower()

A server-side failure from NED arrives as "The remote service returned the
following error message...", which contains neither string. Those queries were
therefore abandoned after a single attempt and the object was logged as
excluded. In excluded_objects_log.csv, 9 objects (21 files) carry the reason
category "NED name-checker server error" -- these were never given a second
chance, and we currently cannot say whether their exclusion was real.

This script does NOT modify sn_catalog_v2.csv, sn_catalog_final.csv, or
excluded_objects_log.csv. It writes one new file containing a verdict per
object, so the decision about what to do with the result stays explicit.

Outcomes reported
-----------------
    RESOLVED         NED returned a usable redshift. The original exclusion
                     was a false negative and the object can be reinstated.
    NO_REDSHIFT      The name resolved, but NED holds no redshift for it.
                     A genuine exclusion, but for a different stated reason
                     than the one currently logged.
    NOT_RECOGNIZED   NED's name interpreter definitively rejected every name
                     tried. A genuine, correctly-logged exclusion.
    STILL_ERRORING   The service still failed after all retries. Inconclusive
                     -- re-run later rather than drawing a conclusion.

Usage
-----
    # the 9 server-error objects only (the default, and the point of the script)
    python 03_retry_failed_ned.py --log-csv results\\excluded_objects_log.csv \\
                                   --out-csv results\\ned_retry_results.csv

    # all 72 excluded objects, as a completeness check
    python 03_retry_failed_ned.py --log-csv results\\excluded_objects_log.csv \\
                                   --out-csv results\\ned_retry_all.csv --all
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

# The reason category written by the exclusion step for transient NED-side
# failures. These are the objects this script exists to re-test.
TRANSIENT_CATEGORY = "NED name-checker server error"

# More generous than script 01. These queries are few (9 objects, not 338),
# so we can afford to be patient and remove any doubt about whether the
# failure was transient.
MAX_RETRIES = 5
BASE_BACKOFF_S = 3
POLITE_DELAY_S = 2


def is_transient_error(exc: Exception) -> bool:
    """
    Decide whether a failed NED query is worth retrying.

    This is the corrected version of the test in 02_build_catalog.py, which
    only recognised timeouts. A transient failure is one where the service
    misbehaved; a definitive failure is one where the service answered
    clearly and told us the name is not recognised.
    """
    msg = str(exc).lower()

    # Definitive answers -- retrying cannot change these.
    if "not currently recognized" in msg or "not recognized" in msg:
        return False
    if "no object" in msg or "not found" in msg:
        return False

    # Transient conditions.
    transient_markers = (
        "timed out", "timeout",
        "remote service returned",     # NED-side error -- the case script 01 missed
        "connection", "connectionreset", "connectionerror",
        "temporarily", "unavailable", "try again",
        "500", "502", "503", "504",
    )
    return any(m in msg for m in transient_markers)


def generate_alt_names(object_name: str) -> list[str]:
    """
    Candidate IAU-style designations for a CSP object name.

    Identical in behaviour to the function in 02_build_catalog.py, reproduced
    here so this diagnostic runs standalone. Two rules:

      1. ASAS-SN internal name  -> IAU name       (ASAS14ad  -> SN2014ad)
      2. Two-digit-year SN name -> four-digit year (SN04dt   -> SN2004dt)

    The YY -> 20YY expansion is only unambiguous because CSP-I/II observations
    span 2004-2015; there is no 19YY object in this data set.
    """
    alts: list[str] = []

    m_asas = re.match(r"ASAS(\d{2})([a-zA-Z]+)", object_name)
    if m_asas:
        alts.append(f"SN20{m_asas.group(1)}{m_asas.group(2)}")

    m_short_year = re.match(r"^SN(\d{2})([A-Za-z]+)$", object_name)
    if m_short_year:
        alts.append(f"SN20{m_short_year.group(1)}{m_short_year.group(2)}")

    # A spaced variant is also accepted by NED's interpreter for IAU names
    # and costs nothing to try once the unspaced form has failed.
    for a in list(alts):
        m = re.match(r"^SN(\d{4})([A-Za-z]+)$", a)
        if m:
            alts.append(f"SN {m.group(1)}{m.group(2)}")

    return alts


def query_one(object_name: str) -> dict:
    """
    Query NED for a single object, trying its own name and any alternates.

    Returns a dict with the outcome, the name that worked (if any), the
    redshift, and the last error seen. The distinction between a definitive
    rejection and a service failure is preserved, because that distinction is
    the entire point of this script.
    """
    from astroquery.ipac.ned import Ned

    names_to_try = [object_name] + generate_alt_names(object_name)
    last_error = ""
    saw_definitive_rejection = False
    saw_resolution_without_redshift = False

    for name in names_to_try:
        for attempt in range(MAX_RETRIES):
            try:
                table = Ned.query_object(name)
                if table is not None and len(table) > 0:
                    z = table["Redshift"][0]
                    if z is not None and str(z) not in ("--", "masked", ""):
                        return {
                            "outcome": "RESOLVED",
                            "resolved_name": name,
                            "redshift": float(z),
                            "names_tried": ";".join(names_to_try),
                            "error": "",
                        }
                    # Name resolved but carries no redshift -- a real result,
                    # not a failure. Record it and stop retrying this name.
                    saw_resolution_without_redshift = True
                break

            except Exception as e:  # noqa: BLE001
                last_error = f"{name}: {e}"
                if not is_transient_error(e):
                    saw_definitive_rejection = True
                    break
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF_S * (2 ** attempt)
                    print(f"      transient failure, retrying in {wait}s "
                          f"(attempt {attempt + 2}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                break

        time.sleep(POLITE_DELAY_S)  # be courteous to the NED service

    if saw_resolution_without_redshift:
        outcome = "NO_REDSHIFT"
    elif saw_definitive_rejection:
        outcome = "NOT_RECOGNIZED"
    else:
        outcome = "STILL_ERRORING"

    return {
        "outcome": outcome,
        "resolved_name": "",
        "redshift": None,
        "names_tried": ";".join(names_to_try),
        "error": last_error,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-csv", required=True, type=Path,
                    help="excluded_objects_log.csv from the exclusion step.")
    ap.add_argument("--out-csv", required=True, type=Path,
                    help="Where to write the retry verdicts. Nothing else is modified.")
    ap.add_argument("--all", action="store_true",
                    help="Retry all excluded objects, not just the server-error ones.")
    args = ap.parse_args()

    try:
        from astroquery.ipac.ned import Ned  # noqa: F401
    except ImportError as e:
        raise SystemExit("astroquery is not installed. Run: pip install astroquery") from e

    log = pd.read_csv(args.log_csv)
    if "reason_category" not in log.columns or "object" not in log.columns:
        raise SystemExit(f"{args.log_csv} does not look like the exclusion log "
                         f"(columns found: {list(log.columns)})")

    if args.all:
        targets = log.copy()
        print(f"Retrying ALL {len(targets)} excluded objects.")
    else:
        targets = log[log["reason_category"] == TRANSIENT_CATEGORY].copy()
        print(f"Retrying {len(targets)} objects logged as "
              f"'{TRANSIENT_CATEGORY}'.")
        if targets.empty:
            print("Nothing to do -- no objects carry that reason category.")
            return

    files_at_stake = int(targets.get("n_files_affected", pd.Series(dtype=int)).sum())
    print(f"Files at stake if these resolve: {files_at_stake}\n")

    rows = []
    for i, rec in enumerate(targets.itertuples(index=False), start=1):
        obj = rec.object
        print(f"[{i}/{len(targets)}] {obj}")
        result = query_one(obj)
        print(f"      -> {result['outcome']}"
              + (f"  z={result['redshift']:.5f} as '{result['resolved_name']}'"
                 if result["outcome"] == "RESOLVED" else ""))
        rows.append({
            "object": obj,
            "n_files_affected": getattr(rec, "n_files_affected", ""),
            "original_reason": getattr(rec, "reason_category", ""),
            **result,
        })

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 66)
    print("Verdict")
    print("=" * 66)
    for outcome, n in out["outcome"].value_counts().items():
        n_files = int(pd.to_numeric(
            out.loc[out["outcome"] == outcome, "n_files_affected"],
            errors="coerce").fillna(0).sum())
        print(f"  {outcome:>15s}: {n:3d} objects, {n_files:3d} files")

    n_resolved = int((out["outcome"] == "RESOLVED").sum())
    n_stuck = int((out["outcome"] == "STILL_ERRORING").sum())

    print()
    if n_resolved:
        print(f"  {n_resolved} object(s) were excluded in error and can be reinstated.")
        print("  Their exclusion was caused by the retry policy in 02_build_catalog.py")
        print("  treating a NED server error as a definitive name failure.")
    else:
        print("  No object was recovered. The original exclusions stand, but the")
        print("  reason category should be corrected: these failed because NED")
        print("  cannot resolve the name, not because the service errored.")
    if n_stuck:
        print(f"  {n_stuck} object(s) still returned service errors -- inconclusive.")
        print("  Re-run later before treating these as genuine failures.")

    print(f"\nWrote {args.out_csv}. No existing file was modified.")


if __name__ == "__main__":
    main()