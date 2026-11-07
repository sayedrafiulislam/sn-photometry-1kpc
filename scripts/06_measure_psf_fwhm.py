"""
06_measure_psf_fwhm.py

Measure the point-spread function (PSF) full width at half maximum (FWHM)
for every frame in the CSP archive, grouped by telescope and filter.

--------------------------------------------------------------------------
Motivation (Kelsey et al. 2021, MNRAS 501, 4861, Section 2.2.3)
--------------------------------------------------------------------------
The local aperture used to measure host-galaxy photometry cannot be made
arbitrarily small. Below some radius set by the PSF, the aperture no longer
encloses a well-defined fraction of the source light and photometric
uncertainties blow up. Kelsey et al. impose a single maximum PSF cut
(FWHM = 1.3 arcsec) across their seeing-optimised stacks, and convert this
into a minimum usable aperture radius via the Gaussian relation

    FWHM = 2 * sqrt(2 * ln 2) * sigma ~= 2.355 * sigma

giving sigma_min ~= 0.55 arcsec (their Section 2.2.3).

CSP images are individual combined frames per object rather than a uniformly
seeing-cut stack, taken with two telescopes (du Pont 2.5 m, Swope 1 m) of
different aperture and therefore different delivered image quality. A single
fixed PSF cannot be assumed. FWHM is measured empirically per frame, which
establishes the seeing floor that governs the smallest defensible aperture.

--------------------------------------------------------------------------
Method
--------------------------------------------------------------------------
1. Detect point sources with DAOStarFinder.
2. Reject non-stellar, blended, edge and saturated detections using
   sharpness, roundness, isolation and border cuts.
3. Optionally mask a region around the known SN position, so a still-visible
   supernova cannot masquerade as a calibration star.
4. Fit each surviving star with a 2D Gaussian to obtain FWHM_x and FWHM_y,
   averaged to a single FWHM in pixels.
5. Convert to arcsec using THAT FRAME'S OWN plate scale, read from its WCS.
6. Aggregate to a median and scatter per telescope and filter.

--------------------------------------------------------------------------
Correction applied in this version: per-frame plate scale
--------------------------------------------------------------------------
The original version hard-coded 0.23 arcsec/pixel with the comment
"confirmed from WCS headers". The step-00 and step-01 audits showed this is
false for part of the archive. Three plate scales are present:

    0.159 arcsec/pixel :  12 frames
    0.230 arcsec/pixel : 585 frames
    0.430 arcsec/pixel : 119 frames, all Swope

so 131 of 716 frames were being converted with the wrong scale. For a
0.430 arcsec/pixel frame the FWHM in arcsec came out a factor of 1.87 too
small. That error does not stay local: FWHM in arcsec feeds sigma_min, which
feeds the seeing floor in kpc, which is the basis of the entire resolution
argument for a 1 kpc aperture.

Nor can the scale be inferred from the telescope tag in the filename. The
audit found 25 frames tagged as Swope carrying du Pont-like scales, so the
tag is not reliable evidence of the optics. The scale is therefore read from
each frame's own WCS, with the historical constant retained only as a
fallback for a frame whose WCS cannot supply one.

--------------------------------------------------------------------------
Runtime
--------------------------------------------------------------------------
Opens all 716 frames and fits up to 25 stars in each. Expect one to three
hours. Occasional "no sources detected" warnings are normal for shallow or
crowded frames; a continuous stream of them indicates a problem.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.detection import DAOStarFinder

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# Fallback only. The real scale is read per frame from its own WCS; see the
# correction note in the module docstring. This value is used solely when a
# frame has no usable celestial WCS, and every such use is printed.
PLATE_SCALE_FALLBACK = 0.23

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 1 / 2.355

CUTOUT_HALF_SIZE = 15          # pixels, half-width of the fit box around each star
MIN_SEPARATION_PIX = 25        # reject stars closer together than this (blending)
SN_EXCLUSION_RADIUS_PIX = 20   # mask radius around the SN position, if known
MAX_STARS_PER_IMAGE = 25       # cap for speed; DAOStarFinder often over-detects
DAOFIND_FWHM_GUESS = 4.0       # rough initial guess in pixels, detection only
DAOFIND_THRESHOLD_NSIGMA = 8.0  # detection threshold in units of background sigma
BORDER_MARGIN_PIX = 60         # exclude detections this close to any edge
                               # (overscan strips and edge glow can otherwise be
                               # misidentified as point sources -- confirmed
                               # visually in ASAS14lq_V_comb_swo.fits, where a
                               # row of false detections hugged the y=0 edge)

FILENAME_RE = re.compile(r"^(?P<obj>.+?)_(?P<filt>[BV])_comb_(?P<tel>dup|swo)\.fits$")


def parse_filename(path: Path):
    """Extract object name, filter, and telescope from a CSP filename."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("obj"), m.group("filt"), m.group("tel")


def read_plate_scale(header, name=""):
    """
    Return this frame's plate scale in arcsec/pixel, from its own WCS.

    Falls back to PLATE_SCALE_FALLBACK only if no usable celestial WCS is
    present, and says so, because a silent fallback would reintroduce exactly
    the single-scale assumption this correction exists to remove.
    """
    try:
        w = WCS(header)
        if w.has_celestial:
            scales_deg = proj_plane_pixel_scales(w.celestial)
            return float(np.mean(scales_deg) * 3600.0), True
    except Exception:
        pass
    print(f"  [warn] no usable WCS scale, using fallback {PLATE_SCALE_FALLBACK}: {name}")
    return PLATE_SCALE_FALLBACK, False


def detect_stars(data, bkg_mean, bkg_std):
    """Run DAOStarFinder and return a source table sorted by brightness."""
    daofind = DAOStarFinder(
        fwhm=DAOFIND_FWHM_GUESS,
        threshold=DAOFIND_THRESHOLD_NSIGMA * bkg_std,
        sharpness_range=(0.2, 1.0),   # reject cosmic rays (sharp) / galaxies (flat)
        roundness_range=(-0.5, 0.5),  # reject elongated / blended sources
    )
    sources = daofind(data - bkg_mean)
    if sources is None or len(sources) == 0:
        return None
    sources.sort("flux")
    sources.reverse()
    return sources


def filter_edge_sources(sources, shape, margin=BORDER_MARGIN_PIX):
    """Drop detections within `margin` pixels of any image edge."""
    ny, nx = shape
    x, y = sources["x_centroid"], sources["y_centroid"]
    keep = (x > margin) & (x < nx - margin) & (y > margin) & (y < ny - margin)
    return sources[keep]


def filter_isolated(sources, min_sep=MIN_SEPARATION_PIX):
    """Keep only stars with no neighbour within min_sep pixels (avoid blends)."""
    keep = np.ones(len(sources), dtype=bool)
    xs, ys = sources["x_centroid"], sources["y_centroid"]
    for i in range(len(sources)):
        if not keep[i]:
            continue
        d = np.hypot(xs - xs[i], ys - ys[i])
        d[i] = np.inf
        if np.any(d < min_sep):
            keep[i] = False
            keep[np.argmin(d)] = False
    return sources[keep]


def mask_sn_position(sources, sn_xy, radius=SN_EXCLUSION_RADIUS_PIX):
    """Drop any detection within `radius` pixels of the known SN pixel position."""
    if sn_xy is None:
        return sources
    sn_x, sn_y = sn_xy
    d = np.hypot(sources["x_centroid"] - sn_x, sources["y_centroid"] - sn_y)
    return sources[d > radius]


def fit_gaussian_fwhm(data, x0, y0, half_size=CUTOUT_HALF_SIZE):
    """
    Fit a 2D Gaussian to a cutout centred on (x0, y0).
    Returns (fwhm_x_pix, fwhm_y_pix), or None if the fit fails or is unphysical.
    """
    try:
        cutout = Cutout2D(data, (x0, y0), size=2 * half_size + 1, mode="partial",
                          fill_value=np.nan)
    except Exception:
        return None

    cdata = cutout.data
    if np.any(~np.isfinite(cdata)) or cdata.size == 0:
        cdata = np.nan_to_num(cdata, nan=np.nanmedian(cdata))

    yy, xx = np.mgrid[0:cdata.shape[0], 0:cdata.shape[1]]
    amp_guess = np.nanmax(cdata) - np.nanmedian(cdata)
    if amp_guess <= 0:
        return None

    g_init = models.Gaussian2D(
        amplitude=amp_guess,
        x_mean=cdata.shape[1] / 2, y_mean=cdata.shape[0] / 2,
        x_stddev=2.0, y_stddev=2.0,
    ) + models.Const2D(amplitude=np.nanmedian(cdata))

    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fitted = fitter(g_init, xx, yy, cdata)
        except Exception:
            return None

    g = fitted[0]  # the Gaussian2D component
    sigma_x, sigma_y = abs(g.x_stddev.value), abs(g.y_stddev.value)

    # Reject clearly failed or unphysical fits.
    if not (0.5 < sigma_x < half_size) or not (0.5 < sigma_y < half_size):
        return None

    fwhm_x = sigma_x / FWHM_TO_SIGMA
    fwhm_y = sigma_y / FWHM_TO_SIGMA
    return fwhm_x, fwhm_y


def process_image(path: Path, sn_positions: dict | None = None):
    """Measure per-star FWHM values for a single FITS image."""
    parsed = parse_filename(path)
    if parsed is None:
        print(f"  [skip] filename does not match CSP convention: {path.name}")
        return []
    obj, filt, tel = parsed

    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header

    # This frame's own scale, not a sample-wide assumption.
    plate_scale, wcs_ok = read_plate_scale(header, path.name)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    sources = detect_stars(data, median, std)
    if sources is None:
        print(f"  [warn] no sources detected: {path.name}")
        return []

    sources = filter_edge_sources(sources, data.shape)
    if len(sources) == 0:
        print(f"  [warn] all detections were within the border margin: {path.name}")
        return []

    sources = filter_isolated(sources)

    sn_xy = sn_positions.get(obj) if sn_positions else None
    sources = mask_sn_position(sources, sn_xy)

    if len(sources) == 0:
        print(f"  [warn] no isolated non-SN stars survived cuts: {path.name}")
        return []

    sources = sources[:MAX_STARS_PER_IMAGE]

    rows = []
    for row in sources:
        result = fit_gaussian_fwhm(data, row["x_centroid"], row["y_centroid"])
        if result is None:
            continue
        fwhm_x_pix, fwhm_y_pix = result
        fwhm_avg_pix = 0.5 * (fwhm_x_pix + fwhm_y_pix)
        rows.append({
            "object": obj,
            "filter": filt,
            "telescope": tel,
            "file": path.name,
            "x": row["x_centroid"],
            "y": row["y_centroid"],
            "flux": row["flux"],
            "fwhm_x_pix": fwhm_x_pix,
            "fwhm_y_pix": fwhm_y_pix,
            "fwhm_avg_pix": fwhm_avg_pix,
            "plate_scale": plate_scale,
            "plate_scale_from_wcs": wcs_ok,
            "fwhm_avg_arcsec": fwhm_avg_pix * plate_scale,
        })
    return rows


def run(fits_dir: str, out_dir: str, sn_positions: dict | None = None,
        max_images: int | None = None):
    fits_dir = Path(fits_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(fits_dir.glob("*.fits"))
    if max_images is not None:
        files = files[:max_images]

    all_rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        all_rows.extend(process_image(path, sn_positions))

    if not all_rows:
        raise RuntimeError("No star FWHM measurements were obtained. "
                           "Check detection/fit thresholds and file paths.")

    per_star = pd.DataFrame(all_rows)
    per_star_path = out_dir / "psf_fwhm_per_star.csv"
    per_star.to_csv(per_star_path, index=False)
    print(f"\nSaved per-star measurements -> {per_star_path}")

    summary = (
        per_star.groupby(["telescope", "filter"])["fwhm_avg_arcsec"]
        .agg(n_stars="count", median="median", mean="mean", std="std",
             p16=lambda s: np.percentile(s, 16),
             p84=lambda s: np.percentile(s, 84))
        .reset_index()
    )
    summary_path = out_dir / "psf_fwhm_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved per-group summary      -> {summary_path}\n")
    print(summary.to_string(index=False))

    # Plate-scale audit. If the archive really were single-scale this prints
    # one row; the audits say it will print three, and the count per scale is
    # the check that the per-frame reading is working.
    print("\nPlate scales actually used (arcsec/pixel):")
    scale_counts = (per_star.drop_duplicates("file")["plate_scale"]
                    .round(4).value_counts().sort_index())
    for scale, n in scale_counts.items():
        print(f"  {scale:.4f} : {n:4d} frames")

    n_fallback = int((~per_star.drop_duplicates("file")["plate_scale_from_wcs"]).sum())
    if n_fallback:
        print(f"\n  WARNING: {n_fallback} frames fell back to "
              f"{PLATE_SCALE_FALLBACK} arcsec/pixel (no usable WCS).")
    else:
        print("\n  Every frame supplied its own scale from its WCS.")

    return per_star, summary


if __name__ == "__main__":
    # Paths are derived from this script's own location, so the pipeline
    # survives a change of machine or drive letter without edits. Twelve
    # scripts in the original repository carried absolute D:\ paths and all
    # broke when the work moved to another computer.
    REPO = Path(__file__).resolve().parent.parent

    FITS_DIR = r"E:\CSPAll"
    OUT_DIR = REPO / "results" / "phase1_psf"

    # SN pixel positions, as {object_name: (x_pix, y_pix)}, to stop a
    # still-visible supernova being fitted as a calibration star. Coordinates
    # come from step 10, so this stays None on the first pass. Ten objects
    # have a frame from the same calendar year as their supernova and are
    # logged in results/pending_checks.md for direct inspection.
    SN_POSITIONS = None

    # Set to a small number (40-60) for a trial run spanning both telescopes
    # and both filters; None runs the full archive.
    MAX_IMAGES = None

    run(FITS_DIR, str(OUT_DIR), sn_positions=SN_POSITIONS, max_images=MAX_IMAGES)