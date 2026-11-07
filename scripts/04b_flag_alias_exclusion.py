"""
04b_flag_alias_exclusion.py

Exclude ASAS14ad: redshift obtained via an unverified name alias.

The catalogue builder falls back to generated IAU-style alternates when NED
does not recognise a filename's object name. Two rules exist. Expanding a
two-digit year (SN04dt -> SN2004dt) is safe: it rewrites the same designation
in full, and CSP spans 2004-2015 so "YY" is unambiguous. Mapping a survey
designation to an IAU name (ASAS14ad -> SN2014ad) is an assumption, because
ASAS-SN numbers objects in its own discovery order while the IAU numbers them
across all surveys; the suffixes need not correspond.

That second rule fired exactly once in this sample. NED recognised 36 of the
37 ASAS designations directly, so no alias was needed for those.

For ASAS14ad there is positive evidence of a problem: two runs of this
pipeline returned z = 0.0264 and z = 0.0060 for the same file, differing by a
factor of 4.4. At least one is wrong and neither can be verified without an
independent coordinate cross-match. Since a wrong redshift gives a wrong
angular diameter distance and therefore a wrong physical aperture, the object
is excluded rather than carried forward unverified.

To recover it, cross-match by COORDINATES against the CSP-II data release
(Phillips et al. 2019, PASP 131, 014001), not by name.
"""

from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

TARGET = "ASAS14ad"
REASON = ("redshift via unverified survey-to-IAU name alias; "
          "two runs disagreed by a factor of 4.4")

final = pd.read_csv(RESULTS / "sn_catalog_final.csv")
excl = pd.read_csv(RESULTS / "excluded_objects_log.csv")

if TARGET not in set(final["object"]):
    raise SystemExit(f"{TARGET} is not in the working catalogue -- nothing to do.")

moved = final[final["object"] == TARGET].copy()
final = final[final["object"] != TARGET]

moved["exclusion_reason"] = REASON
if "exclusion_reason" not in excl.columns:
    excl["exclusion_reason"] = "no NED redshift"

excl = pd.concat([excl, moved], ignore_index=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for p in (RESULTS / "sn_catalog_final.csv", RESULTS / "excluded_objects_log.csv"):
    shutil.copy2(p, p.with_suffix(f".bak_{stamp}.csv"))

final.to_csv(RESULTS / "sn_catalog_final.csv", index=False)
excl.to_csv(RESULTS / "excluded_objects_log.csv", index=False)

print("=" * 66)
print(f"  excluded : {TARGET}")
print(f"  reason   : {REASON}")
print(f"\n  working catalogue : {final['object'].nunique()} objects")
print(f"  exclusion log     : {excl['object'].nunique()} objects")
print(f"  backups stamped   : {stamp}")
print("=" * 66)