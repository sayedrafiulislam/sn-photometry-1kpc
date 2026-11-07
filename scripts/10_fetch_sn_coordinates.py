"""
10_fetch_sn_coordinates.py

Fetch a sky position (RA/Dec, degrees, J2000) for every object surviving
the redshift cut, so that curve-of-growth apertures at step 13 can be
centred on the supernova site rather than placed by hand.

WHY THE SN SITE AND NOT THE NUCLEUS
-----------------------------------
The claim of this project concerns the *local* host environment. Galaxy
nuclei are systematically older and redder than their discs, so an
aperture centred on a nucleus measures the nucleus, with a bias that
does not cancel between B and V. The centre is a scientific choice.

WHY THIS SCRIPT RESOLVES NAMES ITSELF
-------------------------------------
sn_catalog_final.csv holds 266 objects but a ned_resolved_name for only
100 of them. Step 03b queried NED successfully for the remaining 166 --
their redshifts are in the catalogue and demonstrably came from NED --
but wrote only the redshift to ned_progress.csv and discarded the name
that produced it. The names were never lost to a bug; they were never
recorded. This script regenerates them.

Two rules only, taken unchanged from 03_retry_failed_ned.py:
    ASAS14hp -> SN2014hp        (survey internal -> IAU)
    SN04dt   -> SN2004dt        (two-digit year -> four-digit)
The YY -> 20YY expansion is unambiguous because CSP spans 2004-2015.

THE SAFEGUARD -- read this before trusting any output
-----------------------------------------------------
Alias regeneration is exactly what produced the ASAS14ad error, where a
survey-to-IAU mapping returned an object at z = 0.0264 instead of
z = 0.0060: a different galaxy entirely, delivered with no error message.
A name is therefore NOT accepted merely because NED answered to it. NED's
redshift is compared against the redshift already in the catalogue, and
a name that disagrees is rejected however confident the reply looked.
Objects failing this test are flagged Z_MISMATCH and written to the
failure log for inspection. They are never silently kept.

NED's object type is also recorded. Type 'SN' means the position is the
explosion site as reported to the IAU/TNS. Anything else may be a host
nucleus and must not reach photometry unexamined.

FAILURE HANDLING
----------------
In step 02, 231 of 237 recorded failures were network errors on this
machine rather than NED reporting missing data, because the retry test
looked for the word "timeout" and nothing else. The default here is
inverted: every error is retried unless NED explicitly says the name is
not recognised. Progress is written after every object, so a dropped
connection costs one query.

No --limit option, deliberately. An alphabetical --limit is what caused
the plate-scale error that propagated through six scripts.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from astroquery.ipac.ned import Ned

# --------------------------------------------------------------------
# Paths -- from this file's location, never hard-coded
# --------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

CATALOG_CSV = RESULTS / "sn_catalog_final.csv"
OUT_PATH = RESULTS / "sn_coordinates.csv"
FAILED_LOG_PATH = RESULTS / "sn_coordinates_failed.csv"
PROGRESS_PATH = RESULTS / "sn_coordinates_progress_v2.csv"

# --------------------------------------------------------------------
# Network
# --------------------------------------------------------------------
NED_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = [5, 15, 45, 90]
POLITE_DELAY_SECONDS = 1.0

Ned.TIMEOUT = NED_TIMEOUT_SECONDS

# --------------------------------------------------------------------
# Redshift agreement tolerance
#
# The catalogue redshift came from NED for the same physical object, so
# a correct name should reproduce it to the printed precision. Allow a
# small absolute floor for rounding in the stored value, plus a 1%
# relative term. A wrong-object match -- the ASAS14ad case -- misses by
# tens of percent and is nowhere near this.
# --------------------------------------------------------------------
Z_ABS_TOL = 0.0005
Z_REL_TOL = 0.01

PERMANENT_MARKERS = (
    "not currently recognized", "not recognized",
    "not found", "no object", "no matching", "obj_not_found",
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_if_exists(path: Path) -> None:
    if path.exists():
        target = path.with_suffix(path.suffix + f".bak_{timestamp()}")
        path.rename(target)
        print(f"  preserved existing file -> {target.name}")


def is_permanent(message: str) -> bool:
    low = message.lower()
    return any(m in low for m in PERMANENT_MARKERS)


def candidate_names(object_name: str, stored_name) -> list[str]:
    """
    Names to try, best first.

    A name already recorded in the catalogue is tried first because it is
    known to have worked. Then the object's own name, then the two
    expansions from step 03. Duplicates removed, order preserved.
    """
    names: list[str] = []

    if isinstance(stored_name, str) and stored_name.strip():
        names.append(stored_name.strip())

    names.append(object_name)

    m = re.match(r"^ASAS(\d{2})([A-Za-z]+)$", object_name)
    if m:
        names.append(f"SN20{m.group(1)}{m.group(2)}")

    m = re.match(r"^SN(\d{2})([A-Za-z]+)$", object_name)
    if m:
        names.append(f"SN20{m.group(1)}{m.group(2)}")

    # NED's interpreter also accepts a spaced IAU form; costs one query.
    for n in list(names):
        m = re.match(r"^SN(\d{4})([A-Za-z]+)$", n)
        if m:
            names.append(f"SN {m.group(1)}{m.group(2)}")

    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def parse_row(table):
    """Pull position, NED name, type and redshift out of a result table."""
    row, cols = table[0], table.colnames

    ra = dec = None
    for ra_col, dec_col in (("RA", "DEC"), ("RA(deg)", "DEC(deg)")):
        if ra_col in cols and dec_col in cols:
            ra, dec = float(row[ra_col]), float(row[dec_col])
            break
    if ra is None:
        raise KeyError(f"no recognised RA/Dec columns in {cols}")

    ned_name = str(row["Object Name"]).strip() if "Object Name" in cols else ""
    ned_type = str(row["Type"]).strip() if "Type" in cols else ""

    z = None
    if "Redshift" in cols:
        raw = row["Redshift"]
        try:
            if raw is not None and str(raw) not in ("--", "masked", "", "nan"):
                z = float(raw)
        except (TypeError, ValueError):
            z = None

    return ra, dec, ned_name, ned_type, z


def z_agrees(z_ned, z_catalog) -> bool:
    if z_ned is None or pd.isna(z_catalog):
        return False
    tol = max(Z_ABS_TOL, Z_REL_TOL * abs(float(z_catalog)))
    return abs(float(z_ned) - float(z_catalog)) <= tol


def query_name(name: str) -> tuple[dict | None, str, bool]:
    """
    Query one name. Returns (parsed, error_message, retryable).
    parsed is None on failure.
    """
    last_error, retryable = "unknown", True

    for attempt in range(MAX_ATTEMPTS):
        try:
            table = Ned.query_object(name)
            if table is None or len(table) == 0:
                return None, "no_ned_entry", False

            ra, dec, ned_name, ned_type, z = parse_row(table)
            return ({"ra_deg": ra, "dec_deg": dec,
                     "ned_catalog_name": ned_name,
                     "ned_object_type": ned_type,
                     "ned_redshift": z}, "", False)

        except KeyError as exc:
            return None, f"parse_error:{exc}", False

        except (requests.exceptions.RequestException, OSError) as exc:
            last_error, retryable = f"network:{type(exc).__name__}", True

        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if is_permanent(message):
                return None, f"not_recognised:{message[:70]}", False
            last_error, retryable = f"{type(exc).__name__}:{message[:70]}", True

        if attempt < MAX_ATTEMPTS - 1:
            wait = BACKOFF_SECONDS[attempt]
            print(f"        {name}: {last_error}; waiting {wait}s")
            time.sleep(wait)

    return None, last_error, retryable


def resolve_object(object_name: str, stored_name, z_catalog) -> dict:
    """
    Try each candidate name until one returns a position whose redshift
    agrees with the catalogue. A disagreeing name is recorded, not used.
    """
    names = candidate_names(object_name, stored_name)
    mismatches, last_error, any_retryable = [], "", False

    for name in names:
        parsed, error, retryable = query_name(name)
        time.sleep(POLITE_DELAY_SECONDS)

        if parsed is None:
            last_error = f"{name}:{error}"
            any_retryable = any_retryable or retryable
            continue

        if z_agrees(parsed["ned_redshift"], z_catalog):
            return {"name_used": name, "status": "OK", "retryable": False,
                    "error": None, "z_catalog": z_catalog, **parsed}

        z_shown = ("none" if parsed["ned_redshift"] is None
                   else f"{parsed['ned_redshift']:.5f}")
        mismatches.append(f"{name}(z={z_shown})")

    if mismatches:
        return {"name_used": "", "status": "Z_MISMATCH", "retryable": False,
                "error": "redshift disagrees: " + "; ".join(mismatches),
                "z_catalog": z_catalog, "ra_deg": None, "dec_deg": None,
                "ned_catalog_name": "", "ned_object_type": "",
                "ned_redshift": None}

    return {"name_used": "", "status": "UNRESOLVED", "retryable": any_retryable,
            "error": last_error or "no candidate name resolved",
            "z_catalog": z_catalog, "ra_deg": None, "dec_deg": None,
            "ned_catalog_name": "", "ned_object_type": "",
            "ned_redshift": None}


def load_catalog() -> pd.DataFrame:
    if not CATALOG_CSV.exists():
        sys.exit(f"FATAL: input catalogue not found: {CATALOG_CSV}")

    catalog = pd.read_csv(CATALOG_CSV)

    required = {"object", "ned_resolved_name", "redshift"}
    missing = required - set(catalog.columns)
    if missing:
        sys.exit(f"FATAL: {CATALOG_CSV.name} missing columns {sorted(missing)}. "
                 f"Present: {sorted(catalog.columns)}")

    # One row per object. Sort so that rows carrying a name come first,
    # then keep the first -- this way an object whose name sits on its
    # second frame does not lose it. (dropna-then-dedup would discard
    # objects with no name at all; here they are kept and resolved.)
    per_object = (catalog[["object", "ned_resolved_name", "redshift"]]
                  .assign(_has_name=lambda d: d["ned_resolved_name"].notna())
                  .sort_values(["object", "_has_name"],
                               ascending=[True, False])
                  .drop_duplicates(subset="object", keep="first")
                  .drop(columns="_has_name")
                  .dropna(subset=["redshift"])
                  .reset_index(drop=True))

    n_named = int(per_object["ned_resolved_name"].notna().sum())
    print(f"Catalogue: {len(per_object)} objects, {n_named} with a stored "
          f"NED name, {len(per_object) - n_named} to be re-resolved.")

    if len(per_object) == 0:
        sys.exit("FATAL: no objects with a redshift. Nothing to do.")

    return per_object


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {}
    prog = pd.read_csv(PROGRESS_PATH)
    keep = {r["object"]: r for r in prog.to_dict("records")
            if pd.notna(r.get("ra_deg")) or not bool(r.get("retryable", False))}
    print(f"Progress file: {len(prog)} rows, {len(keep)} settled, "
          f"{len(prog) - len(keep)} network failures queued for retry.")
    return keep


def save_progress(records: dict) -> None:
    pd.DataFrame(list(records.values())).to_csv(PROGRESS_PATH, index=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    objects = load_catalog()
    done = load_progress()
    todo = [r for r in objects.itertuples(index=False) if r.object not in done]

    print(f"{len(todo)} objects to query. NED timeout {NED_TIMEOUT_SECONDS}s, "
          f"up to {MAX_ATTEMPTS} attempts per name.\n")

    try:
        for i, row in enumerate(todo, 1):
            outcome = resolve_object(row.object, row.ned_resolved_name,
                                     row.redshift)
            record = {"object": row.object, **outcome}
            done[row.object] = record
            save_progress(done)

            if outcome["status"] == "OK":
                status = (f"{outcome['ra_deg']:.6f} {outcome['dec_deg']:+.6f}  "
                          f"[{outcome['ned_object_type'] or '?'}]  "
                          f"as '{outcome['name_used']}'")
            else:
                status = f"{outcome['status']} ({outcome['error']})"
            print(f"[{i}/{len(todo)}] {row.object} -> {status}")

    except KeyboardInterrupt:
        save_progress(done)
        sys.exit(f"\nInterrupted. {len(done)} objects saved. Re-run to continue.")

    results = pd.DataFrame(list(done.values()))
    ok = results[results["status"] == "OK"].copy()
    bad = results[results["status"] != "OK"].copy()

    backup_if_exists(OUT_PATH)
    backup_if_exists(FAILED_LOG_PATH)
    ok.drop(columns=["error", "retryable", "status"]).to_csv(OUT_PATH, index=False)
    bad.to_csv(FAILED_LOG_PATH, index=False)

    print("\n" + "=" * 70)
    print(f"Resolved {len(ok)}/{len(objects)} objects -> {OUT_PATH.name}")

    if len(bad):
        print(f"\n{len(bad)} not resolved -> {FAILED_LOG_PATH.name}")
        for status, n in bad["status"].value_counts().items():
            print(f"  {status:<14} {n}")
        n_mismatch = int((bad["status"] == "Z_MISMATCH").sum())
        if n_mismatch:
            print(f"\n  {n_mismatch} object(s) resolved to a name whose NED "
                  f"redshift disagrees with the catalogue.")
            print("  These are wrong-object matches, the ASAS14ad failure mode.")
            print("  Inspect individually; do not reinstate without a "
                  "coordinate cross-match.")
        n_retry = int(bad["retryable"].sum())
        if n_retry:
            print(f"\n  {n_retry} are network failures, not missing entries. "
                  f"Re-run this script to retry them.")

    if len(ok):
        print("\nNED object type of each accepted position:")
        for t, n in ok["ned_object_type"].fillna("(blank)").value_counts().items():
            print(f"  {t or '(blank)':<12} {n}")
        n_sn = int((ok["ned_object_type"] == "SN").sum())
        print(f"\n  {n_sn}/{len(ok)} are SN entries (explosion site).")
        if n_sn < len(ok):
            print(f"  {len(ok) - n_sn} are NOT -- these may be host nuclei "
                  f"and must be checked at step 11 before photometry.")

    print("\nNEXT: step 11 projects these through each frame's WCS and "
          "confirms the aperture lands on the SN site.")
    print("=" * 70)


if __name__ == "__main__":
    main()