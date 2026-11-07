"""
05d_epoch_check_corrected.py

Which stacks could plausibly contain supernova light?

Corrects two errors in 05b. First, frames were filtered against the working
catalogue before being counted, so the loss was invisible. Second, and more
importantly, the risk threshold was set at "within one year", which flagged
180 objects. Those are not a warning: CSP template imaging was taken in the
season AFTER the supernova, so a one-year separation is the campaign working
as designed.

Type Ia supernovae fade on a well-characterised timescale set by the decay
chain Ni-56 -> Co-56 -> Fe-56. By ~100 days past maximum they are several
magnitudes below peak in B and V (Contreras et al. 2010, AJ 139, 519), and by
one year they are far below host surface brightness in a 1-2.5 m class stack.

So a gap of one year or more is safe. Only a gap of ZERO or NEGATIVE -- a
frame taken in the same year as the supernova, or before it -- identifies a
stack that could still carry supernova light and needs direct inspection.

Dates are Modified Julian Dates: days from 1858-11-17. The archive spans
2005-11-03 to 2018-05-18, consistent with post-campaign template imaging.

READS ONLY.
"""

from pathlib import Path
import re
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

ep = pd.read_csv(RESULTS / "positions_epochs.csv")
cat = pd.read_csv(RESULTS / "sn_catalog_final.csv")

print("=" * 70)
print("EPOCH CHECK")
print("=" * 70)
print(f"  frames in archive          : {len(ep)}")

keep = set(cat["object"])
ep = ep[ep["object"].isin(keep)].copy()
print(f"  frames for catalogue objects: {len(ep)}")
print(f"  objects in catalogue        : {cat['object'].nunique()}")
print(f"  objects with frames         : {ep['object'].nunique()}\n")

ep["obs_date"] = pd.to_datetime(pd.to_numeric(ep["date_val"]),
                                unit="D", origin="1858-11-17")
ep["obs_year"] = ep["obs_date"].dt.year
print(f"  observation dates: {ep['obs_date'].min().date()} "
      f"to {ep['obs_date'].max().date()}")


def sn_year(name):
    """Discovery year from the object name. CSP spans 2004-2015."""
    m = re.search(r"(?:SN|ASAS|LSQ|PTF|CSP|KISS|PS1?-?)(\d{2,4})", str(name))
    if not m:
        return None
    y = int(m.group(1))
    return 2000 + y if y < 100 else y


ep["sn_year"] = ep["object"].apply(sn_year)
unnamed = ep[ep["sn_year"].isna()]["object"].nunique()
if unnamed:
    print(f"  WARNING: {unnamed} objects have no parseable year in their name")

ep = ep[ep["sn_year"].notna()]
ep["gap"] = ep["obs_year"] - ep["sn_year"]

earliest = ep.groupby("object")["gap"].min().reset_index()

print(f"\n  Smallest gap per object ({len(earliest)} objects):")
for lo, hi, label in ((-99, -1, "frame taken BEFORE the SN year"),
                      (0, 0, "same year as the SN"),
                      (1, 1, "1 year after"),
                      (2, 4, "2-4 years after"),
                      (5, 99, "5 or more years after")):
    n = int(((earliest["gap"] >= lo) & (earliest["gap"] <= hi)).sum())
    flag = "  <-- inspect" if hi <= 0 else ""
    print(f"    {label:<32s} {n:4d}{flag}")

risky = earliest[earliest["gap"] <= 0].sort_values("gap")
print(f"\n  Objects needing direct inspection ({len(risky)}):")
for _, r in risky.iterrows():
    d = ep[ep["object"] == r["object"]]
    print(f"    {r['object']:<12s} SN {int(d['sn_year'].iloc[0])}   "
          f"earliest frame {d['obs_date'].min().date()}   gap {int(r['gap'])}")

print("\n  For these, compare the flux at the SN position against a surrounding")
print("  ring. A residual point source makes the centre BRIGHTER than the ring.")
print("=" * 70)