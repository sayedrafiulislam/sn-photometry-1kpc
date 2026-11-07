"""
check_lsq14foj.py

Audit script. LSQ14foj survives every quality criterion with
B-V = -1.01, an impossible colour: its B flux (296 counts) exceeds its V
flux (148), whereas every other object in the catalogue has V brighter
than B by a factor of two to three. Both fluxes are well above the
measurement floor, the background fractions are unremarkable and the
quoted zero-point error is small, so this is not a signal-to-noise
failure.

Something specific is wrong. This displays both frames at the aperture
so the cause can be identified rather than the threshold adjusted until
the object disappears.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
import warnings

warnings.simplefilter("ignore", FITSFixedWarning)

# Walk up until the results directory is found, so this works whether
# the script sits in scripts/ or scripts/diagnostics/.
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "results").is_dir())
RESULTS = REPO / "results"
OUT_DIR = RESULTS / "phase4_aperture" / "spotchecks"

OBJECT = "LSQ14foj"
ZOOM = 120


def main() -> None:
    frames = pd.read_csv(RESULTS / "position_verification_per_frame.csv")
    rows = frames[(frames["object"] == OBJECT) & (frames["status"] == "OK")]
    if rows.empty:
        raise SystemExit(f"No verified frames for {OBJECT}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 7))
    axes = np.atleast_1d(axes)

    for ax, (_, rec) in zip(axes, rows.iterrows()):
        path = Path(str(rec["file"]))
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)

        x, y = float(rec["x_pix"]), float(rec["y_pix"])
        r_ap = float(rec["aperture_radius_pix"])

        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        ax.imshow(data, origin="lower", cmap="gray",
                  norm=LogNorm(vmin=max(median, 1), vmax=median + 30 * std))
        circ = plt.Circle((x, y), r_ap, fill=False, color="red", lw=2)
        ax.add_patch(circ)
        ax.set_xlim(x - ZOOM, x + ZOOM)
        ax.set_ylim(y - ZOOM, y + ZOOM)
        ax.set_title(f"{OBJECT}  {rec['filter']}  "
                     f"r_ap = {r_ap:.1f} px\n{path.name}")

        # Pixel statistics inside the aperture
        yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= r_ap ** 2
        vals = data[inside]
        vals = vals[np.isfinite(vals)]
        print(f"{rec['filter']}: n_pix={len(vals)}  min={vals.min():.1f}  "
              f"median={np.median(vals):.1f}  max={vals.max():.1f}  "
              f"sum={vals.sum():.1f}  frame_median={median:.1f}")

    fig.tight_layout()
    out = OUT_DIR / f"spotcheck_{OBJECT}.png"
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()