"""
03b_retry_ned_resumable.py

Retry NED lookups for objects that failed on a network error.

The step-02 run was executed while the connection was unstable: 231 of 237
failures were DNS, timeout or dropped-connection errors on this machine, not
NED reporting a missing redshift. Only 6 objects genuinely have no redshift.

Two changes from the inherited retry logic:

  1. DNS failures are retryable. The original checked only for "timeout" in
     the error text, so a NameResolutionError broke out immediately with no
     backoff -- a momentary DNS blip became a permanent recorded failure.

  2. Progress is saved after every object. If the connection drops mid-run,
     re-running resumes from where it stopped instead of starting over.

Objects that already have a redshift are skipped, so this is safe to run
repeatedly and is polite to a shared public service.
"""

from pathlib import Path
import re
import time
import pandas as pd
from astroquery.ipac.ned import Ned

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "results" / "sn_catalog_v2.csv"
PROGRESS = REPO / "results" / "ned_progress.csv"

MAX_ATTEMPTS = 4
PAUSE_BETWEEN_OBJECTS = 1.0

NETWORK_MARKERS = ("getaddrinfo", "nameresolution", "timed out", "timeout",
                   "connection aborted", "newconnection", "remotedisconnected",
                   "max retries exceeded")


def is_network_error(msg: str) -> bool:
    low = str(msg).lower()
    return any(m in low for m in NETWORK_MARKERS)


def alt_names(name: str) -> list[str]:
    """IAU-style alternates. CSP spans 2004-2015, so YY means 20YY."""
    out = []
    m = re.match(r"ASAS(\d{2})([a-zA-Z]+)$", name)
    if m:
        out.append(f"SN20{m.group(1)}{m.group(2)}")
    m = re.match(r"^SN(\d{2})([A-Za-z]+)$", name)
    if m:
        out.append(f"SN20{m.group(1)}{m.group(2)}")
    return out


def lookup(name: str):
    """Return (z, resolved_name, error). Retries network errors with backoff."""
    last = ""
    for candidate in [name] + alt_names(name):
        for attempt in range(MAX_ATTEMPTS):
            try:
                tbl = Ned.query_object(candidate)
                if tbl is not None and len(tbl) > 0:
                    z = tbl["Redshift"][0]
                    if z is not None and str(z) not in ("--", "masked", ""):
                        return float(z), candidate, ""
                break  # answered, but no usable redshift -- try next name
            except Exception as e:
                last = f"{candidate}: {e}"
                if is_network_error(e) and attempt < MAX_ATTEMPTS - 1:
                    wait = 3 * (2 ** attempt)  # 3s, 6s, 12s
                    print(f"      network error, waiting {wait}s "
                          f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
                    time.sleep(wait)
                    continue
                break
        time.sleep(PAUSE_BETWEEN_OBJECTS)
    return None, None, last


df = pd.read_csv(CATALOG)

# Resume from saved progress if a previous run was interrupted.
resolved = {}
if PROGRESS.exists():
    prev = pd.read_csv(PROGRESS)
    resolved = dict(zip(prev["object"], prev["redshift"]))
    print(f"Resuming: {len(resolved)} objects already recorded.\n")

known = df[df["redshift"].notna()].drop_duplicates("object")
for _, r in known.iterrows():
    resolved.setdefault(r["object"], r["redshift"])

todo = [o for o in df["object"].unique() if o not in resolved]

print("=" * 66)
print(f"  objects total    : {df['object'].nunique()}")
print(f"  already resolved : {len(resolved)}")
print(f"  to attempt       : {len(todo)}")
print("=" * 66 + "\n")

for i, obj in enumerate(todo, 1):
    z, name, err = lookup(obj)
    if z is not None:
        print(f"  [{i:3d}/{len(todo)}] {obj:<12s} z = {z:.5f}  ({name})")
        resolved[obj] = z
    else:
        tag = "NETWORK" if is_network_error(err) else "no redshift"
        print(f"  [{i:3d}/{len(todo)}] {obj:<12s} FAILED  {tag}")
        if not is_network_error(err):
            resolved[obj] = None  # genuine miss -- do not retry next time

    # Save after every object so a dropout costs nothing.
    (pd.DataFrame({"object": list(resolved.keys()),
                   "redshift": list(resolved.values())})
       .to_csv(PROGRESS, index=False))

got = sum(1 for v in resolved.values() if pd.notna(v))
print(f"\n  resolved with redshift : {got}")
print(f"  still missing          : {df['object'].nunique() - got}")
print(f"\n  Progress saved to {PROGRESS}")
print("  Re-run this script to continue if the connection dropped.")