"""
03e_audit_alias_resolution.py

How many redshifts were obtained via a GUESSED alternate name?

The catalogue builder tries the filename's object name first, then falls back
to generated IAU-style alternates. Two rules exist:

    SN04dt   -> SN2004dt    two-digit year expanded to four
    ASAS14ad -> SN2014ad    survey designation mapped to an IAU name

The first rule is safe: it rewrites the same designation in full form, and CSP
spans 2004-2015 so "YY" is unambiguous.

The SECOND rule is an assumption, and it may not hold. ASAS-SN assigns internal
designations in its own discovery order; the IAU assigns letters in order of
reported discovery across all surveys. The suffixes need not correspond, so
ASAS14ad and SN2014ad may be different objects. Attaching one supernova's
redshift to another's images would give a wrong distance, a wrong angular
diameter distance, and therefore a wrong physical aperture size.

This script lists every object whose redshift came from an alias rather than
from its own name, so each can be verified.

READS ONLY.
"""

from pathlib import Path
import re
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
cat = pd.read_csv(REPO / "results" / "sn_catalog_v2.csv")

obj = cat.drop_duplicates("object").copy()
obj = obj[obj["redshift"].notna()]

has_name = "ned_resolved_name" in obj.columns
if has_name:
    obj["via_alias"] = (obj["ned_resolved_name"].notna() &
                        (obj["ned_resolved_name"] != obj["object"]))
else:
    obj["via_alias"] = False

print("=" * 72)
print("HOW WAS EACH REDSHIFT OBTAINED?")
print("=" * 72)
print(f"  objects with a redshift : {len(obj)}")
if has_name:
    print(f"  resolved by own name    : {int((~obj['via_alias']).sum())}")
    print(f"  resolved via an alias   : {int(obj['via_alias'].sum())}")
else:
    print("  (ned_resolved_name column absent -- rerun 02 to record it)")

# Split the alias cases by which rule produced them.
year_rule, survey_rule = [], []
for _, r in obj[obj["via_alias"]].iterrows():
    if re.match(r"^SN\d{2}[A-Za-z]+$", str(r["object"])):
        year_rule.append((r["object"], r["ned_resolved_name"], r["redshift"]))
    else:
        survey_rule.append((r["object"], r["ned_resolved_name"], r["redshift"]))

print(f"\n  SAFE  two-digit-year expansion ({len(year_rule)}):")
print("        same designation written in full; no ambiguity.")

print(f"\n  CHECK survey-name mapping ({len(survey_rule)}):")
if survey_rule:
    print(f"        {'filename':<14s} {'NED entry':<14s} redshift")
    for a, b, z in sorted(survey_rule):
        print(f"        {a:<14s} {b:<14s} {z}")
    print("\n        Each of these assumed a survey designation and an IAU")
    print("        designation refer to the same object. Verify individually")
    print("        against the CSP-II target list or TNS before use.")
else:
    print("        none")

# Objects NED knew by their own survey name -- these are the safe comparison.
own = obj[(~obj["via_alias"]) & obj["object"].astype(str).str.startswith("ASAS")]
print(f"\n  For contrast, NED knew {len(own)} ASAS names directly,")
print("  so the survey-name mapping is not needed for those.")
print("=" * 72)