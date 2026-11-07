"""
check_vizier_dr3_structure.py

Audit script, not production.

Inspect the table structure of Krisciunas et al. (2017, AJ 154, 211),
the CSP-I DR3 photometry release, as served by VizieR under
J/AJ/154/211. We are looking for the table that lists objects with
J2000 coordinates -- in a photometry catalogue this is usually a small
object-summary table, separate from the much larger light-curve tables.

Nothing is downloaded in bulk: ROW_LIMIT is set to a handful of rows so
this costs one small query regardless of how large the catalogue is.
"""

import warnings

from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

CATALOG = "J/AJ/154/211"

viz = Vizier()
viz.ROW_LIMIT = 5

print(f"Fetching table structure for {CATALOG} ...\n")
catalogs = viz.get_catalogs(CATALOG)
print(f"{len(catalogs)} table(s) returned.\n")

for table in catalogs:
    name = table.meta.get("name", "(unnamed)")
    print("=" * 72)
    print(f"TABLE: {name}")
    print("=" * 72)
    print(f"  columns: {table.colnames}")

    coord_cols = [c for c in table.colnames
                  if any(k in c.upper() for k in
                         ("RAJ", "DEJ", "_RA", "_DE", "RAB", "DEB"))]
    name_cols = [c for c in table.colnames
                 if any(k in c.lower() for k in ("sn", "name", "obj"))]

    if coord_cols:
        print(f"  >>> coordinate columns: {coord_cols}")
    if name_cols:
        print(f"  >>> name columns      : {name_cols}")
    if coord_cols and name_cols:
        print("  >>> THIS LOOKS LIKE THE OBJECT TABLE")

    print("\n  first rows:")
    print(table[:5].to_pandas().to_string(index=False))
    print()