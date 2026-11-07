"""
12_verify_sn_positions.py

Project every object's coordinate through every frame's own WCS and
check that the geometry required by the measurement fits on the
detector.

SOURCE OF TRUTH
---------------
Frames are taken from sn_catalog_final.csv, which lists each file with
its filter and telescope as established in step 02. They are NOT
discovered by globbing the archive: filenames in this data set are not
uniformly structured (see RENAME_LOG.md), so a filename pattern silently
misses files and makes a present observation look like an absent one.
An earlier version of this script globbed for {object}_{filter}_comb_
{telescope}.fits and reported 45 objects as lacking a du Pont B/V pair,
all of them showing a single frame -- an artefact of the pattern, not a
property of the data.

WHAT IS CHECKED, PER FRAME
--------------------------
  1. The file named in the catalogue exists on disk.
  2. The WCS parses, and the coordinate lands on the detector.
  3. A 1 kpc photometric aperture fits entirely within the frame.
  4. The 20-30 kpc background annulus fits, or by how much it is
     truncated.

Point 4 is a warning, not an exclusion. At low redshift 30 kpc subtends
several arcminutes -- larger than a du Pont field -- so nearby hosts
will inevitably have truncated annuli. Where the usable threshold lies
is a decision for step 13, once the distribution can be seen.

Plate scale is read per frame from the WCS, never from the telescope
keyword. Nothing is modified; two new files are written.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

REPO = Path(__file__).resolve().parent.parent
FITS_DIR = Path(r"E:\CSPAll")
RESULTS = REPO / "results"

COORDS_CSV = RESULTS / "sn_coordinates.csv"
CATALOG_CSV = RESULTS / "sn_catalog_final.csv"
OUT_FRAMES = RESULTS / "position_verification_per_frame.csv"
OUT_OBJECTS = RESULTS / "position_verification_per_object.csv"

APERTURE_KPC = 1.0
ANNULUS_OUTER_KPC = 30.0
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def plate_scale_arcsec(wcs: WCS) -> float:
    return float(np.mean(np.abs(np.diag(wcs.pixel_scale_matrix)) * 3600.0))


def annulus_fraction_inside(x, y, r_out, nx, ny, n=720) -> float:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = x + r_out * np.cos(theta), y + r_out * np.sin(theta)
    return float(((xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)).mean())


def is_dupont(value: str) -> bool:
    return "dup" in str(value).lower()


def main() -> None:
    for path in (COORDS_CSV, CATALOG_CSV):
        if not path.exists():
            raise SystemExit(f"FATAL: {path.name} not found.")
    if not FITS_DIR.exists():
        raise SystemExit(f"FATAL: FITS archive not found at {FITS_DIR}")

    coords = pd.read_csv(COORDS_CSV)
    catalog = pd.read_csv(CATALOG_CSV)

    z_col = "z_catalog" if "z_catalog" in coords.columns else "redshift"
    pos = coords.set_index("object")[["ra_deg", "dec_deg", z_col]]

    frames = catalog[catalog["object"].isin(pos.index)].copy()

    required = {"object", "file", "filter", "telescope"}
    missing_cols = required - set(frames.columns)
    if missing_cols:
        raise SystemExit(f"FATAL: catalogue missing columns {sorted(missing_cols)}. "
                         f"Present: {sorted(frames.columns)}")

    required = {"object", "file", "filter", "telescope"}
    missing_cols = required - set(frames.columns)
    if missing_cols:
        raise SystemExit(f"FATAL: catalogue missing columns {sorted(missing_cols)}. "
                         f"Present: {sorted(frames.columns)}")
    print(f"{len(frames)} frames listed for {frames['object'].nunique()} "
          f"objects in the catalogue.\n")

    rows = []
    for i, (_, rec) in enumerate(frames.iterrows(), 1):
        obj = rec["object"]
        ra, dec, z = pos.loc[obj, "ra_deg"], pos.loc[obj, "dec_deg"], pos.loc[obj, z_col]

        # Indexed access, not attribute access: itertuples renames any
        # column colliding with a Python builtin, and 'filter' is one.
        row = {"object": obj, "redshift": float(z), "file": rec["file"],
               "filter": rec["filter"], "telescope": rec["telescope"]}

        path = FITS_DIR / str(rec["file"])
        if not path.exists():
            matches = list(FITS_DIR.glob(f"{Path(str(rec['file'])).stem}*"))
            path = matches[0] if matches else None

        if path is None:
            row["status"] = "FILE_NOT_FOUND"
            rows.append(row)
            continue

        try:
            with fits.open(path) as hdul:
                ny, nx = hdul[0].data.shape
                wcs = WCS(hdul[0].header)

            scale = plate_scale_arcsec(wcs)
            kpc_per_pix = scale * (COSMO.kpc_proper_per_arcmin(float(z))
                                   .to(u.kpc / u.arcsec).value)
            x, y = (float(v) for v in wcs.all_world2pix(float(ra), float(dec), 0))

            on_image = (0 <= x < nx) and (0 <= y < ny)
            edge = min(x, y, nx - 1 - x, ny - 1 - y) if on_image else np.nan
            r_ap = APERTURE_KPC / kpc_per_pix
            r_out = ANNULUS_OUTER_KPC / kpc_per_pix

            row.update({
                "nx": nx, "ny": ny,
                "plate_scale_arcsec": scale, "kpc_per_pixel": kpc_per_pix,
                "x_pix": x, "y_pix": y, "on_image": on_image,
                "edge_dist_pix": edge, "aperture_radius_pix": r_ap,
                "aperture_fits": bool(on_image and edge >= r_ap),
                "annulus_frac_on_chip": (
                    annulus_fraction_inside(x, y, r_out, nx, ny)
                    if on_image else 0.0),
                "status": "OK" if on_image else "OFF_IMAGE",
            })
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"ERROR:{type(exc).__name__}:{str(exc)[:60]}"

        rows.append(row)
        if i % 100 == 0:
            print(f"  ... {i}/{len(frames)} frames")

    frames_df = pd.DataFrame(rows)
    frames_df.to_csv(OUT_FRAMES, index=False)

    summary = []
    for obj, grp in frames_df.groupby("object"):
        good = grp[grp["status"] == "OK"]

        # Guard the empty case explicitly. An empty Series from .apply()
        # carries dtype object, and indexing a DataFrame with an empty
        # non-boolean Series is read as a LABEL selector rather than a
        # mask -- it returns a frame with no columns, so the next lookup
        # raises KeyError. Any object whose frames all failed hits this.
        if len(good) == 0:
            has_b = has_v = False
        else:
            mask = good["telescope"].apply(is_dupont).astype(bool)
            dup = good[mask]
            band = dup["filter"].astype(str).str.strip().str.upper()
            has_b = bool((band == "B").any())
            has_v = bool((band == "V").any())
        summary.append({
            "object": obj, "redshift": grp["redshift"].iloc[0],
            "n_frames": len(grp), "n_ok": len(good),
            "has_dupont_B": bool(has_b), "has_dupont_V": bool(has_v),
            "has_dupont_pair": bool(has_b and has_v),
            "all_apertures_fit": bool(good["aperture_fits"].all()) if len(good) else False,
            "min_annulus_frac": float(good["annulus_frac_on_chip"].min()) if len(good) else 0.0,
            "n_bad_frames": int((grp["status"] != "OK").sum()),
        })

    objects_df = pd.DataFrame(summary)
    objects_df.to_csv(OUT_OBJECTS, index=False)

    print("\n" + "=" * 72)
    print("PER-FRAME")
    print("=" * 72)
    for status, n in frames_df["status"].value_counts().items():
        print(f"  {status:<30} {n}")

    n = len(objects_df)
    n_pair = int(objects_df["has_dupont_pair"].sum())
    print("\n" + "=" * 72)
    print("PER-OBJECT")
    print("=" * 72)
    print(f"  objects                          : {n}")
    print(f"  with a du Pont B/V pair          : {n_pair}")
    print(f"  WITHOUT a du Pont pair           : {n - n_pair}")
    print(f"  1 kpc aperture fits on all frames: "
          f"{int(objects_df['all_apertures_fit'].sum())}")

    missing = objects_df[~objects_df["has_dupont_pair"]]
    if len(missing):
        print("\n  Objects lacking a du Pont B/V pair:")
        print(missing[["object", "redshift", "n_frames", "n_ok",
                       "has_dupont_B", "has_dupont_V"]].to_string(index=False))

    print("\n" + "=" * 72)
    print("BACKGROUND ANNULUS TRUNCATION (warning only, no cut applied)")
    print("=" * 72)
    for thr in (1.0, 0.9, 0.5, 0.25):
        print(f"  >= {thr * 100:5.0f}% of the 30 kpc annulus on chip: "
              f"{int((objects_df['min_annulus_frac'] >= thr).sum()):3d}/{n}")
    print("\n  most truncated:")
    print(objects_df.nsmallest(10, "min_annulus_frac")[
        ["object", "redshift", "min_annulus_frac"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nWrote {OUT_FRAMES.name} and {OUT_OBJECTS.name}.")


if __name__ == "__main__":
    main()