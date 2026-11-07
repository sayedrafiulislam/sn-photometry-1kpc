"""
02b_ned_failure_audit.py

Separate NED network failures from genuine missing redshifts.

The step-02 run was executed while the network was unstable. Most recorded
failures are DNS or timeout errors on this machine, not NED reporting that it
has no redshift for an object. Those two things must never be conflated: one
is a transient infrastructure problem, the other is a real sample cut.

READS ONLY.
"""

from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO / "results" / "sn_catalog_v2.csv")

obj = df.drop_duplicates("object").copy()
err = obj["ned_error"].fillna("")

def classify(e):
    if not e.strip():
        return "no error - NED answered"
    low = e.lower()
    if "getaddrinfo" in low or "nameresolution" in low:
        return "DNS failure (network)"
    if "timed out" in low or "timeout" in low:
        return "timeout (network)"
    if "connection aborted" in low or "newconnection" in low:
        return "connection dropped (network)"
    return "other"

obj["kind"] = err.apply(classify)
obj["has_z"] = obj["redshift"].notna()

print("=" * 66)
print(f"NED RUN AUDIT   objects: {len(obj)}")
print("=" * 66)
print(f"  with redshift    : {int(obj['has_z'].sum())}")
print(f"  without redshift : {int((~obj['has_z']).sum())}\n")

print("  Failures by cause:")
for k, n in obj[~obj["has_z"]]["kind"].value_counts().items():
    print(f"    {k:<28s} {n:4d}")

genuine = obj[(~obj["has_z"]) & (obj["kind"] == "no error - NED answered")]
print(f"\n  Genuine missing redshifts (NED answered, no z): {len(genuine)}")
if len(genuine):
    print("    " + ", ".join(sorted(genuine["object"])))

retryable = int((~obj["has_z"] & obj["kind"].str.contains("network")).sum())
print(f"\n  Retryable network failures: {retryable}")
print("=" * 66)