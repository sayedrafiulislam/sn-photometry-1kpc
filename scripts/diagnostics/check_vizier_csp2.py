"""
check_vizier_csp2.py

Audit script. Look for the CSP-II data release (Phillips et al. 2019,
PASP 131, 014001) on VizieR under several phrasings, since the earlier
photometry search did not surface it.

Needed for 25 objects whose NED coordinates are quoted only to 3.6
arcsec and which are CSP-II, hence absent from CSP-I DR3.

Note on the previous version: it formatted cat.description with a width
specifier, which raises TypeError when VizieR returns a catalogue with
no description. The exception fired while printing, so queries that had
in fact returned results were reported as failures. Descriptions are now
coerced to str before formatting, and the exception handler prints the
traceback type rather than swallowing the query.
"""

import warnings
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

QUERIES = [
    "Carnegie Supernova Project II",
    "CSP-II supernovae",
    "Phillips 2019 supernovae",
    "low-z Type Ia supernovae spectroscopy CSP",
    "Carnegie Supernova Project natural system photometry",
    "CSP-II Phillips",
    "Type Ia supernovae du Pont Swope",
]


def describe(cat) -> str:
    for attr in ("description", "title", "name"):
        value = getattr(cat, attr, None)
        if value:
            return str(value)
    return "(no description)"


for q in QUERIES:
    print("=" * 76)
    print(f"QUERY: {q}")
    print("=" * 76)
    try:
        found = Vizier.find_catalogs(q)
        if not found:
            print("  nothing returned.\n")
            continue
        print(f"  {len(found)} catalogue(s):")
        for key, cat in found.items():
            print(f"    {str(key):30s} {describe(cat)[:76]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  query itself failed: {type(exc).__name__}: {exc}")
    print()