"""
check_coarse_object_survival.py

Audit script, not production.

Question: of the 25 CSP-II objects with coarse NED coordinates and no
DR3 replacement, how many will actually survive the rest of the
pipeline? Objects already excluded by the 1 kpc seeing floor do not
need their coordinates improved.

Rationale: coarse positions and high redshift are expected to correlate,
because both reflect how faint and poorly localised the transient was.
The seeing floor at step 09 already removed 11 objects, all at
z = 0.075-0.137. If the coarse tail overlaps that population, the
manual-extraction workload shrinks.

Reads only. Writes one summary file.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

CROSSMATCH = RESULTS / "dr3_crossmatch.csv"
FLOOR = RESULTS / "phase4_aperture" / "aperture_floor_per_object_corrected.csv"
OUT = RESULTS / "coarse_objects_needing_positions.csv"

PRECISE_DECIMALS = 5
APERTURE_RADIUS_KPC = 1.0


def main() -> None:
    if not CROSSMATCH.exists():
        raise SystemExit(f"FATAL: {CROSSMATCH.name} not found.")

    xm = pd.read_csv(CROSSMATCH)
    coarse = xm[(xm["ned_decimals"] < PRECISE_DECIMALS) & (~xm["in_dr3"])].copy()
    print(f"{len(coarse)} coarse objects with no DR3 replacement.\n")

    if not FLOOR.exists():
        print(f"WARNING: {FLOOR} not found -- cannot check seeing survival.")
        print("Listing the coarse objects without it:\n")
        print(coarse[["object", "redshift", "ned_decimals"]]
              .sort_values("redshift", ascending=False)
              .to_string(index=False))
        coarse.to_csv(OUT, index=False)
        return

    floor = pd.read_csv(FLOOR)
    print(f"Seeing-floor file columns: {list(floor.columns)}\n")

    # Find the per-object worst-case floor column without assuming a name.
    floor_col = next((c for c in floor.columns
                      if "floor" in c.lower() and "kpc" in c.lower()), None)
    if floor_col is None:
        print("Could not identify the floor column. Columns above.")
        coarse.to_csv(OUT, index=False)
        return

    worst = (floor.groupby("object")[floor_col].max()
             .rename("worst_floor_kpc").reset_index())

    merged = coarse.merge(worst, on="object", how="left")
    merged["passes_1kpc"] = merged["worst_floor_kpc"] < APERTURE_RADIUS_KPC

    n_pass = int(merged["passes_1kpc"].sum())
    n_unknown = int(merged["worst_floor_kpc"].isna().sum())

    print("=" * 70)
    print("SURVIVAL OF THE COARSE-COORDINATE OBJECTS")
    print("=" * 70)
    print(f"  clear the 1 kpc floor        : {n_pass}")
    print(f"  already excluded by seeing   : {len(merged) - n_pass - n_unknown}")
    print(f"  no floor record found        : {n_unknown}")
    print(f"\n  ==> positions actually needed: {n_pass}")

    print("\n" + "=" * 70)
    print("OBJECTS STILL NEEDING A POSITION")
    print("=" * 70)
    need = merged[merged["passes_1kpc"]].sort_values("redshift", ascending=False)
    print(need[["object", "redshift", "ned_decimals", "worst_floor_kpc",
                "ned_ra", "ned_dec"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    merged.to_csv(OUT, index=False)
    print(f"\nWrote {OUT.name}.")


if __name__ == "__main__":
    main()