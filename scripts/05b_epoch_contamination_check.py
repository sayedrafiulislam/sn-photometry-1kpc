"""
05b_epoch_contamination_check.py

Could any stack contain supernova light?

A Type Ia at peak can briefly outshine its entire host, and it sits at exactly
the position being measured. Any exposure taken while the supernova was still
bright would put that light inside the aperture, with no way to separate it
afterwards and no error message to warn you.

Type Ia fade on a well-characterised timescale, driven by the radioactive
decay chain Ni-56 -> Co-56 -> Fe-56. By roughly 100 days past maximum they are
several magnitudes below peak in B and V (Contreras et al. 2010, AJ 139, 519),
and after one to two years they are far below the host surface brightness in a
1-2.5 m class stack.

The discovery year is encoded in each object's name (SN2012fr, ASAS14ad,
LSQ13dhj -> 2012, 2014, 2013). Comparing that year with each stack's earliest
observation date gives the separation. This is coarse -- a year, not a day --
but it is enough to identify which objects need closer inspection.

READS ONLY.
"""

from pathlib import Path
import re
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

ep = pd.read_csv(RESULTS / "positions_epochs.csv")
cat = pd.read_csv(RESULTS / "sn_catalog_final.csv")
keep = set(cat["object"])

ep = ep[ep["object"].isin(keep)].copy()

# The extractor writes the value in `date_val` and records which keyword it
# came from in `date_kind`. That kind matters: DATE-OBS is a calendar string,
# MJD-OBS is a floating-point day count from 1858-11-17. They cannot be parsed
# the same way, so each is converted separately.
if not {"date_val", "date_kind"}.issubset(ep.columns):
    raise SystemExit(f"Expected date_val/date_kind. Columns: {list(ep.columns)}")

print("  Date keyword used, by kind:")
for k, n in ep["date_kind"].value_counts().items():
    print(f"    {k:<12s} {n:4d} frames")
print()

kind = ep["date_kind"].astype(str).str.upper()
is_mjd = kind.str.contains("MJD")

obs_year = pd.Series(pd.NA, index=ep.index, dtype="Float64")

# Calendar strings.
cal = pd.to_datetime(ep.loc[~is_mjd, "date_val"], errors="coerce")
obs_year.loc[~is_mjd] = cal.dt.year

# MJD: day count since 1858-11-17.
mjd = pd.to_numeric(ep.loc[is_mjd, "date_val"], errors="coerce")
obs_year.loc[is_mjd] = (
    pd.to_datetime(mjd, unit="D", origin="1858-11-17", errors="coerce").dt.year
)

ep["obs_year"] = obs_year


def discovery_year(name: str):
    """Year encoded in the object name. CSP spans 2004-2015."""
    m = re.search(r"(?:SN|ASAS|LSQ|PTF|CSP|KISS|PS1?-?)(\d{2,4})", str(name))
    if not m:
        return None
    y = int(m.group(1))
    if y < 100:
        return 2000 + y
    return y


ep["sn_year"] = ep["object"].apply(discovery_year)
ep = ep[ep["obs_year"].notna() & ep["sn_year"].notna()]
ep["gap_years"] = ep["obs_year"] - ep["sn_year"]

earliest = ep.groupby("object")["gap_years"].min().reset_index()

print("=" * 70)
print("EPOCH SEPARATION: earliest frame minus supernova year")
print("=" * 70)
print(f"  objects checked : {len(earliest)}")
print(f"  frames checked  : {len(ep)}\n")

print("  Distribution of the smallest gap per object:")
for lo, hi, label in ((-99, 0, "frame BEFORE or SAME year as SN"),
                      (1, 1, "1 year after"),
                      (2, 4, "2-4 years after"),
                      (5, 99, "5+ years after")):
    n = int(((earliest["gap_years"] >= lo) &
             (earliest["gap_years"] <= hi)).sum())
    print(f"    {label:<34s} {n:4d}")

risky = earliest[earliest["gap_years"] <= 1].sort_values("gap_years")
print(f"\n  Objects with any frame within one year ({len(risky)}):")
if len(risky):
    for _, r in risky.iterrows():
        print(f"    {r['object']:<14s} gap = {int(r['gap_years'])} yr")
    print("\n    These warrant direct surface-brightness inspection: compare")
    print("    the flux at the SN position with a surrounding ring. A residual")
    print("    point source would make the centre BRIGHTER than the ring.")
else:
    print("    none")

print("\n  Note: the year is coarse. An object discovered in December and")
print("  imaged the following January shows a gap of 1 but is separated by")
print("  weeks. Treat a gap of 1 as 'inspect', not as 'contaminated'.")
print("=" * 70)