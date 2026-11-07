"""
05c_inspect_dates.py

What is actually in the date column?

The epoch check produced an implausible result: 180 of 267 objects showing a
gap of exactly one year between the supernova and the earliest frame. That
clustering is the signature of a conversion error, not a property of the
observing campaign.

Before any epoch conclusion is drawn, the raw values must be examined: how
many kinds are present, what range they span, and whether they are Modified
Julian Dates (days from 1858-11-17), full Julian Dates (from 4713 BC, about
2.4 million larger), or calendar strings.

READS ONLY. Interprets nothing.
"""

from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ep = pd.read_csv(REPO / "results" / "positions_epochs.csv")

print("=" * 70)
print(f"positions_epochs.csv   rows: {len(ep)}")
print("=" * 70)

print("\n  date_kind values:")
print(ep["date_kind"].value_counts(dropna=False).to_string())

print("\n  First 15 rows of date_kind / date_val:")
print(ep[["object", "date_kind", "date_val"]].head(15).to_string(index=False))

num = pd.to_numeric(ep["date_val"], errors="coerce")
print(f"\n  numeric date_val : {int(num.notna().sum())} of {len(ep)}")

if num.notna().any():
    print(f"    min {num.min():.4f}   max {num.max():.4f}")
    print("\n  Interpreted as MJD (origin 1858-11-17):")
    as_mjd = pd.to_datetime(num, unit="D", origin="1858-11-17", errors="coerce")
    print(f"    {as_mjd.min()}  to  {as_mjd.max()}")
    print("\n  Interpreted as full JD (MJD = JD - 2400000.5):")
    as_jd = pd.to_datetime(num - 2400000.5, unit="D",
                           origin="1858-11-17", errors="coerce")
    print(f"    {as_jd.min()}  to  {as_jd.max()}")
    print("\n  CSP observing ran roughly 2004-2015. Whichever interpretation")
    print("  lands in that window is the correct one.")

non_num = ep[num.isna()]
print(f"\n  non-numeric date_val : {len(non_num)}")
if len(non_num):
    print(non_num[["object", "date_kind", "date_val"]].head(10).to_string(index=False))

print("=" * 70)