"""
13_curve_of_growth.py

Measure background-subtracted flux in a series of circular apertures
centred on each supernova position, for every du Pont B and V frame of
the 221-object colour sample.

WHY A CURVE RATHER THAN A SINGLE APERTURE
-----------------------------------------
The measurement this project reports is a colour inside one physical
radius, 1 kpc. Measuring only that radius would leave three questions
unanswerable:

  1. Is 1 kpc resolved, or is it inside the seeing disc? Below the PSF
     width the enclosed flux stops responding to radius, because the
     aperture is sampling the point-spread function rather than the
     galaxy. The radius at which the curve stops flattening is a direct,
     data-driven measurement of the seeing floor derived analytically at
     step 09.

  2. Is the background correct? Over-subtraction makes the curve turn
     over and decline at large radii; under-subtraction makes it climb
     without converging. Neither is visible from a single radius.

  3. Is 1 kpc a stable choice? If flux is still rising steeply there,
     the measurement is sensitive to small centring errors. The local
     logarithmic slope quantifies that sensitivity.

APERTURE GRID
-------------
0.25 to 10 kpc, sampled every 0.25 kpc below 3 kpc and every 0.5 kpc
above. The upstream 5 kpc version began at 1.0 kpc, which is this
project's target radius; the grid must extend well below it for the
seeing turnover to be visible at all.

The smallest radii are deliberately below the seeing floor. They are
expected to be unusable and are measured anyway, because a floor that
is asserted is weaker evidence than a floor that is seen.

GUARDS -- FLAGGED, NOT MASKED
-----------------------------
photutils sums whichever pixels exist without complaint, while
aperture.area returns the full geometric pi*r^2. Subtracting
bkg_per_pixel * area from a partial sum removes background for pixels
that were never summed, and the deficit grows as r^2. Non-finite pixels
are the same problem differently dressed: CSP combined stacks carry NaN
in the dither margins, and one NaN inside an aperture propagates to the
whole sum silently.

Both are therefore measured per aperture and recorded, never patched.
Masking would replace a visibly missing measurement with a plausible
looking partial one, which is the failure being guarded against.

Downstream must filter on aperture_ok AND annulus_ok.

PLATE SCALE
-----------
Taken per frame from the frame's own WCS, as verified at step 12. Three
scales are present in this data set and the telescope keyword does not
determine which. Colours are du Pont only, so the scale is 0.230 in
practice, but the value is read rather than assumed.

INPUTS
------
  position_verification_per_frame.csv   verified pixel positions, scales
  position_verification_per_object.csv  the du Pont pair flag
  aperture_floor_per_object_corrected.csv   sigma_min_kpc per frame

OUTPUTS
-------
  curve_of_growth_ann10-15.csv
  curve_of_growth_ann15-25.csv
  curve_of_growth_ann20-30.csv     <- the setting the catalogue uses
  annulus_setting_comparison.csv

Existing files are renamed to *.bak_YYYYmmdd_HHMMSS, never deleted.
"""

from __future__ import annotations

import datetime
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.wcs import WCS, FITSFixedWarning
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry)

warnings.simplefilter("ignore", FITSFixedWarning)

# --------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
FITS_DIR = Path(r"E:\CSPAll")
RESULTS = REPO / "results"
PHASE4 = RESULTS / "phase4_aperture"

FRAMES_CSV = RESULTS / "position_verification_per_frame.csv"
OBJECTS_CSV = RESULTS / "position_verification_per_object.csv"
FLOOR_CSV = PHASE4 / "aperture_floor_per_object_corrected.csv"

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

APERTURE_GRID_KPC = np.concatenate([
    np.arange(0.25, 3.0, 0.25),
    np.arange(3.0, 10.5, 0.5),
])

ANNULUS_SETTINGS = [(10.0, 15.0), (15.0, 25.0), (20.0, 30.0)]

MIN_ANNULUS_FRAC_ON_CHIP = 0.80
MIN_ANNULUS_FRAC_PIXELS = 0.30
MIN_APERTURE_FRAC_ON_CHIP = 0.98
SIGMA_CLIP = SigmaClip(sigma=3.0, maxiters=5)


def kpc_to_pixels(radius_kpc, z, plate_scale):
    """Physical radius -> angular size -> pixels, at redshift z."""
    d_a_mpc = COSMO.angular_diameter_distance(z).value
    theta_arcsec = np.rad2deg((radius_kpc / 1000.0) / d_a_mpc) * 3600.0
    return theta_arcsec / plate_scale, theta_arcsec


def mask_values(data, aperture):
    """
    Finite on-detector pixel values inside an aperture, with the counts
    needed to judge whether the measurement can be trusted.
    Returns (values, n_finite, n_nonfinite, frac_on_chip).

    The pixel values (used for the background statistics) come from a
    centre-method mask, so each pixel is counted whole. The on-chip
    fraction and the non-finite count use the exact geometric overlap,
    the same weighting aperture_photometry uses for the flux sum. A
    pixel-count fraction is unreliable for apertures only a few pixels
    across and rejected apertures that lay wholly on the detector.
    """
    mask = aperture.to_mask(method="center")
    try:
        raw = mask.get_values(data)
    except AttributeError:
        cutout = mask.multiply(data)
        raw = np.array([]) if cutout is None else cutout[mask.data > 0]
    raw = np.asarray(raw, dtype=float)
    finite = np.isfinite(raw)

    exact = aperture.to_mask(method="exact")
    slc_large, slc_small = exact.get_overlap_slices(data.shape)
    if slc_large is None:
        return raw[finite], int(finite.sum()), 0, 0.0
    weights = exact.data[slc_small]
    pixels = data[slc_large]
    expected = aperture.area
    frac = float(weights.sum() / expected) if expected > 0 else 0.0
    n_bad = int((~np.isfinite(pixels[weights > 0])).sum())

    return (raw[finite], int(finite.sum()), n_bad,
            float(np.clip(frac, 0.0, 1.0)))


def background_from_annulus(data, x, y, r_in, r_out):
    """Sigma-clipped median background per pixel, with diagnostics."""
    ann = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    values, n_pix, _, frac = mask_values(data, ann)

    expected = ann.area
    frac_expected = n_pix / expected if expected > 0 else 0.0
    if n_pix == 0:
        return np.nan, 0, 0.0, 0.0, False

    clipped = SIGMA_CLIP(values, masked=False, copy=True)
    clipped = clipped[np.isfinite(clipped)]
    bkg = float(np.median(clipped)) if len(clipped) else np.nan

    ok = (frac >= MIN_ANNULUS_FRAC_ON_CHIP
          and frac_expected >= MIN_ANNULUS_FRAC_PIXELS
          and np.isfinite(bkg))
    return bkg, n_pix, frac, float(frac_expected), bool(ok)


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = path.with_suffix(path.suffix + f".bak_{stamp}")
        shutil.move(str(path), str(dest))
        print(f"  [backup] {path.name} -> {dest.name}")


def resolve_path(recorded) -> Path | None:
    """The catalogue stores full paths; fall back to the archive folder."""
    p = Path(str(recorded))
    if p.is_absolute() and p.exists():
        return p
    candidate = FITS_DIR / p.name
    return candidate if candidate.exists() else None


def main() -> None:
    for path in (FRAMES_CSV, OBJECTS_CSV):
        if not path.exists():
            raise SystemExit(f"FATAL: {path.name} not found. Run step 12 first.")

    frames = pd.read_csv(FRAMES_CSV)
    objects = pd.read_csv(OBJECTS_CSV)

    paired = set(objects.loc[objects["has_dupont_pair"], "object"])
    use = frames[(frames["status"] == "OK")
                 & frames["object"].isin(paired)
                 & frames["telescope"].astype(str).str.lower().str.contains("dup")
                 & frames["filter"].astype(str).str.upper().isin(["B", "V"])].copy()

    # sigma_min_kpc for the seeing_safe flag, if step 09's output is present
    if FLOOR_CSV.exists():
        floor = pd.read_csv(FLOOR_CSV)
        floor["_name"] = floor["file"].apply(lambda v: Path(str(v)).name)
        use["_name"] = use["file"].apply(lambda v: Path(str(v)).name)
        use = use.merge(floor[["_name", "sigma_min_kpc"]], on="_name", how="left")
    else:
        print(f"[warn] {FLOOR_CSV.name} not found; seeing_safe will be null.")
        use["sigma_min_kpc"] = np.nan

    print("=" * 74)
    print("INPUT")
    print("=" * 74)
    print(f"  frames verified at step 12 : {len(frames)}")
    print(f"  objects with du Pont pair  : {len(paired)}")
    print(f"  du Pont B/V frames to use  : {len(use)}")
    print(f"  radii per frame            : {len(APERTURE_GRID_KPC)}")
    print(f"  annulus settings           : {len(ANNULUS_SETTINGS)}")
    print(f"  seeing floor available for : "
          f"{int(use['sigma_min_kpc'].notna().sum())} frames")
    print("=" * 74 + "\n")

    results = {s: [] for s in ANNULUS_SETTINGS}
    n_total = len(use)
    n_missing, n_nowcs = 0, 0

    for i, (_, rec) in enumerate(use.iterrows(), 1):
        path = resolve_path(rec["file"])
        if path is None:
            n_missing += 1
            print(f"[{i}/{n_total}] [skip] file not found: {rec['file']}")
            continue

        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            try:
                wcs = WCS(hdul[0].header)
            except Exception:
                wcs = None

        if wcs is None or not wcs.has_celestial:
            n_nowcs += 1
            print(f"[{i}/{n_total}] [skip] no usable WCS: {path.name}")
            continue

        z = float(rec["redshift"])
        scale = float(rec["plate_scale_arcsec"])
        x, y = float(rec["x_pix"]), float(rec["y_pix"])

        radii = [kpc_to_pixels(r, z, scale) for r in APERTURE_GRID_KPC]
        radii_pix = [r[0] for r in radii]
        radii_arcsec = [r[1] for r in radii]

        # Raw sums and aperture guards: background-independent, so
        # computed once per frame and reused across annulus settings.
        raw_sums, areas, ap_frac, ap_nan, ap_ok = [], [], [], [], []
        for r_pix in radii_pix:
            ap = CircularAperture((x, y), r=r_pix)
            phot = aperture_photometry(data, ap)
            raw_sums.append(float(phot["aperture_sum"][0]))
            areas.append(float(ap.area))

            _, _, n_bad, frac = mask_values(data, ap)
            ap_frac.append(frac)
            ap_nan.append(n_bad)
            ap_ok.append(bool(frac >= MIN_APERTURE_FRAC_ON_CHIP and n_bad == 0))

        raw_sums = np.array(raw_sums)
        areas = np.array(areas)
        sigma_min = rec.get("sigma_min_kpc", np.nan)

        for (ann_in, ann_out) in ANNULUS_SETTINGS:
            r_in_pix, _ = kpc_to_pixels(ann_in, z, scale)
            r_out_pix, _ = kpc_to_pixels(ann_out, z, scale)

            bkg, n_pix, frac_chip, frac_exp, ann_ok = background_from_annulus(
                data, x, y, r_in_pix, r_out_pix)

            fluxes = raw_sums - bkg * areas

            for j, r_kpc in enumerate(APERTURE_GRID_KPC):
                results[(ann_in, ann_out)].append({
                    "object": rec["object"],
                    "file": path.name,
                    "telescope": rec["telescope"],
                    "filter": rec["filter"],
                    "z": z,
                    "radius_kpc": float(r_kpc),
                    "radius_arcsec": radii_arcsec[j],
                    "radius_pix": radii_pix[j],
                    "flux_bkgsub": fluxes[j],
                    "aperture_sum_raw": raw_sums[j],
                    "aperture_area_pix": areas[j],
                    "seeing_safe": (bool(r_kpc >= sigma_min)
                                    if np.isfinite(sigma_min) else None),
                    "sigma_min_kpc": sigma_min,
                    "annulus_inner_kpc": ann_in,
                    "annulus_outer_kpc": ann_out,
                    "bkg_per_pixel": bkg,
                    "annulus_n_pix": n_pix,
                    "annulus_frac_on_chip": frac_chip,
                    "annulus_frac_expected_pix": frac_exp,
                    "annulus_ok": ann_ok,
                    "plate_scale_used": scale,
                    "aperture_frac_on_chip": ap_frac[j],
                    "aperture_n_nonfinite": ap_nan[j],
                    "aperture_ok": ap_ok[j],
                })

        if i % 25 == 0 or i == n_total:
            print(f"[{i}/{n_total}] frames processed")

    # ----------------------------------------------------------------
    PHASE4.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for (ann_in, ann_out), rows in results.items():
        df = pd.DataFrame(rows)
        tag = f"ann{ann_in:g}-{ann_out:g}"
        out = PHASE4 / f"curve_of_growth_{tag}.csv"
        backup(out)
        df.to_csv(out, index=False)

        per_frame = df.drop_duplicates(subset=["file"])
        good = df[df["annulus_ok"] & df["aperture_ok"]]

        # Behaviour at the target radius
        at1 = good[np.isclose(good["radius_kpc"], 1.0)]
        summary_rows.append({
            "annulus_inner_kpc": ann_in,
            "annulus_outer_kpc": ann_out,
            "n_frames": len(per_frame),
            "n_annulus_ok": int(per_frame["annulus_ok"].sum()),
            "n_rows_total": len(df),
            "n_rows_both_guards_ok": len(good),
            "median_bkg_per_pixel": float(np.nanmedian(per_frame["bkg_per_pixel"])),
            "n_frames_usable_at_1kpc": int(at1["file"].nunique()),
            "median_flux_at_1kpc": float(np.nanmedian(at1["flux_bkgsub"]))
                                   if len(at1) else np.nan,
            "n_negative_at_1kpc": int((at1["flux_bkgsub"] < 0).sum()),
            "output_file": out.name,
        })

        print(f"\nWrote {len(df)} rows ({len(per_frame)} frames) -> {out.name}")
        print(f"  annulus usable : {int(per_frame['annulus_ok'].sum())}"
              f"/{len(per_frame)} frames")
        print(f"  rows both ok   : {len(good)}/{len(df)}")

    summary = pd.DataFrame(summary_rows)
    summary_path = PHASE4 / "annulus_setting_comparison.csv"
    backup(summary_path)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 74)
    print("READ ERRORS")
    print("=" * 74)
    print(f"  file not found : {n_missing}")
    print(f"  no usable WCS  : {n_nowcs}")

    print("\n" + "=" * 74)
    print("ANNULUS SETTING COMPARISON")
    print("=" * 74)
    print(summary.to_string(index=False))

    print("""
How to read this:

  median_bkg_per_pixel should FALL as the annulus moves outward if the
  inner annulus was sitting in host light. Flat across settings means
  the 10-15 kpc annulus was already clean.

  median_flux_at_1kpc should RISE as the background falls, since less
  is being subtracted.

  n_negative_at_1kpc counts frames where background-subtracted flux at
  the target radius is below zero. Some are expected: where the local
  surface brightness is at sky level the measurement is noise, and
  noise is negative half the time. A large count means the background
  is over-subtracted.

CAUTION: the frames passing the guard differ between settings, so these
lines are not computed on identical samples. The controlled comparison
is step 22.
""")


if __name__ == "__main__":
    main()