"""
check_thesis_counts.py  (diagnostics, read-only)

Reconciles numbers that disagree between sections of the thesis draft:
  - how many objects fail each quality criterion (Sec 4.7 vs Table 4.1)
  - how many fail ONLY that criterion (the "unique" count)
  - whether any "implausible" colours are really NaN
  - which median colour uncertainty is 0.107 and which is 0.111

Writes nothing. Place in scripts/diagnostics/.
"""
from pathlib import Path
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
CAL_DIR = REPO / "results" / "phase5_calibration"
FLAGGED = CAL_DIR / "calibrated_colour_1kpc_flagged.csv"
DERED = CAL_DIR / "calibrated_colour_1kpc_dered.csv"

for p in (FLAGGED, DERED):
    if not p.exists():
        raise SystemExit(f"STOP: missing input {p}")

df = pd.read_csv(FLAGGED)
crit = ["flag_high_bkg", "flag_large_err", "flag_low_flux", "flag_manual"]
need = crit + ["flag_exclude", "flag_implausible_bv", "B_minus_V",
               "B_minus_V_err", "object"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"STOP: columns missing from {FLAGGED.name}: {missing}")
for c in crit + ["flag_exclude", "flag_implausible_bv"]:
    df[c] = df[c].astype(bool)

print(f"Rows in flagged catalogue : {len(df)}")
print(f"Rows with NaN B-V         : {int(df['B_minus_V'].isna().sum())}")
print(df.loc[df["B_minus_V"].isna(), "object"].tolist())

print("\nCriterion      total  only-this-one")
n_fail = df[crit].sum(axis=1)
for c in crit:
    total = int(df[c].sum())
    only = int((df[c] & (n_fail == 1)).sum())
    print(f"  {c:16s} {total:4d}  {only:4d}")
print(f"  excluded (any)   {int(df['flag_exclude'].sum()):4d}")
print(f"  kept             {int((~df['flag_exclude']).sum()):4d}")

print("\nTable 4.1 check: flux floor vs plausibility")
print(pd.crosstab(df["flag_low_flux"].rename("below floor"),
                  df["flag_implausible_bv"].rename("implausible")))

print("\nImplausible objects and every flag they carry:")
cols = ["object", "B_minus_V", "B_minus_V_err"] + crit
print(df.loc[df["flag_implausible_bv"], cols].to_string(index=False))

d = pd.read_csv(DERED)
if "B_minus_V_dered_err" not in d.columns:
    raise SystemExit(f"STOP: B_minus_V_dered_err missing from {DERED.name}")
kept = d[~d["flag_exclude"].astype(bool)]
print("\nMedian colour uncertainty")
print(f"  flagged file, all rows : {df['B_minus_V_err'].median():.4f}")
print(f"  dered file, all rows   : {d['B_minus_V_dered_err'].median():.4f}")
print(f"  dered file, kept rows  : {kept['B_minus_V_dered_err'].median():.4f}"
      f"  (n={len(kept)})")