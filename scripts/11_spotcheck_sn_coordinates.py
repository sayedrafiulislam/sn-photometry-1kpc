"""
11_spotcheck_sn_coordinates.py

Visual check on a handful of objects: project the NED RA/Dec through
each frame's own WCS and plot the resulting pixel position on the image.

WHAT THIS CAN AND CANNOT ESTABLISH
----------------------------------
It cannot confirm the position lands on the explosion site. The CSP
template stacks contain no supernova light, so there is nothing at the
SN position to see. Astrometric accuracy was established instead by
comparison against CSP's own positions in Krisciunas et al. (2017),
which agree with NED to 0.014 arcsec median.

What it can catch are gross failures, which are the ones that would
silently ruin a measurement:
  - a position falling outside the image, or so near an edge that a
    1 kpc aperture would run off the detector
  - a position landing on the host nucleus rather than offset from it
  - a frame whose WCS is wrong, which no amount of coordinate accuracy
    would fix

Objects chosen
--------------
  SN2012fr  bright well-resolved host (NGC 1365). If the marker sits on
            the nucleus, NED is giving galaxy centres. If offset, it is
            giving SN sites.
  SN2006D   ordinary object, no reason to expect trouble -- the baseline.
  KISS13v   fails the 1 kpc seeing floor at z = 0.080 from poor seeing
            (1.885 arcsec). Worth seeing what a marginal object looks like.
  LSQ13ry   the control. Its NED right ascension is quoted to one decimal
            degree while its declination is given to five, which is not
            consistent with any real astrometric solution. Its marker
            should land somewhere visibly wrong. If it does not, the
            diagnostic is not sensitive enough to trust on the others.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

REPO = Path(__file__).resolve().parent.parent
FITS_DIR = Path(r"E:\CSPAll")
COORDS_CSV = REPO / "results" / "sn_coordinates.csv"
OUT_DIR = REPO / "results" / "phase4_aperture" / "spotchecks"

SPOTCHECK_OBJECTS = ["SN2012fr", "SN06D", "KISS13v", "LSQ13ry"]
ZOOM_PIX = 300


def find_image(obj: str):
    """Prefer du Pont V, else any du Pont B/V frame. Never the r-band."""
    preferred = list(FITS_DIR.glob(f"{obj}_V_comb_dup.fits"))
    if preferred:
        return preferred[0]
    others = [p for p in FITS_DIR.glob(f"{obj}_*_comb_*.fits")
              if "_r_comb_" not in p.name]
    return others[0] if others else None


def main() -> None:
    if not FITS_DIR.exists():
        raise SystemExit(f"FATAL: FITS archive not found at {FITS_DIR}")
    if not COORDS_CSV.exists():
        raise SystemExit(f"FATAL: {COORDS_CSV} not found. Run step 10 first.")

    coords = pd.read_csv(COORDS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for obj in SPOTCHECK_OBJECTS:
        row = coords[coords["object"] == obj]
        if row.empty:
            print(f"[skip] {obj} is not in sn_coordinates.csv")
            continue

        ra = float(row.iloc[0]["ra_deg"])
        dec = float(row.iloc[0]["dec_deg"])

        path = find_image(obj)
        if path is None:
            print(f"[skip] no FITS frame found for {obj}")
            continue

        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            wcs = WCS(hdul[0].header)

        x, y = wcs.all_world2pix(ra, dec, 0)
        ny, nx = data.shape
        on_image = (0 <= x < nx) and (0 <= y < ny)

        edge = min(x, y, nx - 1 - x, ny - 1 - y) if on_image else np.nan
        verdict = ("ON IMAGE" if on_image else "*** OFF IMAGE ***")
        print(f"{obj:<10} {path.name:<32} pixel ({x:8.1f}, {y:8.1f}) "
              f"of ({nx}, {ny})  {verdict}"
              + (f"  edge dist {edge:.0f} px" if on_image else ""))

        if not on_image:
            print(f"           -> a coordinate this far off cannot be "
                  f"photometered; flag {obj} for exclusion.")
            continue

        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(data, origin="lower", cmap="gray",
                  norm=LogNorm(vmin=max(median, 1), vmax=median + 50 * std))
        ax.scatter([x], [y], s=300, facecolors="none",
                   edgecolors="red", linewidths=2)
        ax.scatter([x], [y], s=20, color="red", marker="+")
        ax.set_title(f"{obj}   RA={ra:.5f}  Dec={dec:.5f}\n{path.name}")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_xlim(max(0, x - ZOOM_PIX), min(nx, x + ZOOM_PIX))
        ax.set_ylim(max(0, y - ZOOM_PIX), min(ny, y + ZOOM_PIX))

        out = OUT_DIR / f"spotcheck_{obj}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"           -> {out.name}")

    print(f"\nPNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()