# PIPELINE_1KPC.md

**Local host-galaxy photometry at a 1.0 kpc aperture radius**
Sayed Rafiul Islam — Department of Physics, Shahjalal University of Science and Technology

Repository: `E:\My Work\sn-local-photometry-1kpc`
Upstream repository: `E:\My Work\sn-local-photometry`

---

## 0. What this pipeline is, in one paragraph

The upstream pipeline measured aperture photometry on Carnegie Supernova Project
imaging of Type Ia supernova host galaxies, at nineteen aperture radii from 1.0
to 10.0 kpc in 0.5 kpc steps, and built a calibrated catalogue at a 5.0 kpc
fiducial radius. This pipeline re-extracts the **1.0 kpc** point from those same
measurements and carries it through calibration to a final catalogue of local
B, V and B−V magnitudes.

**No photometry is re-measured.** 1.0 kpc was the first point on the upstream
aperture grid, so every flux this pipeline needs already exists, with its
background subtraction, per-frame plate scale and aperture guards already
applied. This is a re-extraction and re-calibration, which is also why the 1 kpc
and 5 kpc results are guaranteed to rest on identical photometry.

---

## 1. Upstream — already run, not repeated here

These steps live in the main repository and produced the six files this pipeline
consumes. They are listed because the paper's Methods section must describe
them, not because they need re-running.

| # | script | reads | writes | why it matters here |
|---|---|---|---|---|
| 00 | `00_inspect_headers.py` | 716 FITS frames | `header_summary_full.csv` | Established that no filter label and no redshift are stored in the headers. Both had to come from elsewhere. |
| 01 | `01_audit_plate_scales.py` | header summary | `plate_scale_status.csv` | Found **three** plate scales, not one: 0.230″ (585 frames), 0.430″ (119, all Swope V), 0.159″ (12). Every aperture radius depends on this. |
| 02 | `02_build_catalog.py` | FITS filenames, NED | `sn_catalog_v2.csv` | Filter and object name parsed from filenames; redshift queried from NED. |
| 03 | `03_retry_failed_ned.py` | exclusion log, NED | `ned_retry_results.csv` | Recovered objects lost to transient NED name-resolution failures. |
| 04 | `04_apply_redshift_cut.py` | `sn_catalog_v2.csv` | **`sn_catalog_final.csv`**, `excluded_objects_log.csv` | 338 objects in, 266 out. Redshift is required to convert kpc to arcsec. |
| 05 | `05_get_positions_and_epochs.py` | FITS archive | `positions_epochs.csv` | Observation dates — the evidence used to address the supernova-light question. |
| 06 | `06_measure_psf_fwhm.py` | FITS frames | `psf_fwhm_per_star.csv` | Seeing measured per frame from stars in that frame. |
| 07 | `07_flag_image_quality.py` | per-star PSF table | `image_quality_flags_corrected.csv` | Rejects unusable frames and spurious sharp detections. |
| 08 | `08_summarise_psf.py` | per-star PSF table | PSF summary | Diagnostic only. |
| 09 | `09_aperture_floor_per_object.py` | PSF table, flags, redshifts | **`aperture_floor_per_object_corrected.csv`** | Converts each frame's seeing FWHM into σ, then into kpc at that object's redshift. **This is the file the 1 kpc resolution analysis depends on.** |
| 10 | `10_fetch_sn_coordinates.py` | NED | **`sn_coordinates.csv`** | Supernova sky positions — the aperture centres, and the lines of sight for Galactic extinction. |
| 11 | `11_spotcheck_sn_coordinates.py` | FITS, coordinates | inspection images | Visual confirmation that positions land on the SN site, not the host nucleus. |
| 12 | `12_verify_sn_positions.py` | FITS, coordinates | verification table | Systematic check of the above. |
| 13 | `13_curve_of_growth.py` | FITS, aperture floor, coordinates | **`curve_of_growth_ann20-30.csv`** | The photometry itself. 19 radii × 2 filters × every object, with a 20–30 kpc sigma-clipped background annulus. Writes `seeing_safe`, `aperture_ok`, `annulus_ok` per row. |

Steps 14–22 of the upstream repository build and analyse the **5.0 kpc**
catalogue. They are not inputs to this pipeline. Steps 15, 16, 21 and 22 are
across-radius analyses and have no meaning at a single radius.

---

## 2. Inputs to this pipeline

Copied once into `inputs/`, set read-only, hashes recorded in
`inputs/PROVENANCE.md`. Nothing in this pipeline writes to the upstream
repository.

| file | source | what it supplies |
|---|---|---|
| `curve_of_growth_ann20-30.csv` | upstream step 13 | background-subtracted flux at every radius, including 1.0 kpc |
| `aperture_floor_per_object_corrected.csv` | upstream step 09 | `fwhm_arcsec`, `sigma_min_arcsec`, `sigma_min_kpc`, `plate_scale`, `z` |
| `sn_catalog_final.csv` | upstream step 04 | redshifts |
| `sn_coordinates.csv` | upstream step 10 | RA/Dec for the SFD extinction lookup |
| `B_ZP_dup.dat` | supervisor | du Pont B zero points: object, ZP, ZP error, number of reference stars |
| `V_ZP_dup.dat` | supervisor | du Pont V zero points, same format |

---

## 3. This pipeline, step by step

Scripts are numbered in true execution order. The number is the run order.

### Stage A — Is 1 kpc a defensible radius for this data?

| # | script | reads | writes | question answered |
|---|---|---|---|---|
| 00 | `00_seeing_audit_1kpc.py` | `curve_of_growth_ann20-30.csv` | *(prints only)* | Does the inherited seeing floor pass at 1 kpc, and which objects fail? |
| 01 | `01_resolution_check_1kpc.py` | `aperture_floor_per_object_corrected.csv` | *(prints only)* | How many seeing discs wide is the aperture, object by object? |
| 02 | `02_resolution_cut_options.py` | `aperture_floor_per_object_corrected.csv` | *(prints only)* | What does each candidate resolution criterion cost in objects and in redshift reach? |

**Results so far.** 445 measurements at 1.0 kpc across 225 objects; 216 objects
have both B and V above the inherited floor. Median aperture diameter is 3.04
seeing discs; median seeing FWHM is 0.657 kpc physical. About 2.2 per cent of
measurements pass the inherited floor while having an aperture narrower than the
PSF — the reason a stricter criterion is under consideration.

These three scripts read only and write nothing. They can be re-run freely.

### Stage B — Colour

| # | script | reads | writes | what it does |
|---|---|---|---|---|
| 03 | `03_local_colour_1kpc.py` | `curve_of_growth_ann20-30.csv` | `results/local_colour_1kpc.csv` | Selects radius = 1.0 kpc, du Pont only; applies `annulus_ok`, `aperture_ok` and the adopted resolution criterion; pairs B against V per object; forms the instrumental colour. |

Instrumental colour:

```
instrumental (B−V) = −2.5 log10( F_B / F_V )
```

Requires both fluxes positive. Non-positive flux is the measurement floor, not
an error: such objects are counted and logged, never silently dropped.

### Stage C — Calibration

| # | script | reads | writes | what it does |
|---|---|---|---|---|
| 04 | `04_apply_zero_points_1kpc.py` | `local_colour_1kpc.csv`, `B_ZP_dup.dat`, `V_ZP_dup.dat` | `results/calibrated_color_1kpc.csv`, `..._exclusions.csv` | Converts flux to calibrated magnitude in each band and forms the calibrated colour. |

```
m   = ZP − 2.5 log10( flux )
B−V = ( ZP_B − ZP_V ) + instrumental (B−V)
```

The zero points enter the colour as a single additive constant per object.
Anything compared *within* one object is unaffected by this step; anything
compared *between* objects, or against the standard system, depends on it
entirely.

**Expect the largest cut in the pipeline here.** The supplied zero-point files
were derived for CSP-II. Splitting by the year in the object name gives 90
CSP-I objects with no zero point against 0 with one, and 111 CSP-II objects with
one against 1 without. The published catalogue is therefore a **CSP-II sample**,
which reshapes its redshift distribution and host population rather than
thinning it evenly. This must be stated in the paper.

### Stage D — Quality and extinction

| # | script | reads | writes | what it does |
|---|---|---|---|---|
| 05 | `05_flag_unreliable_1kpc.py` | `calibrated_color_1kpc.csv` | `results/calibrated_color_1kpc_flagged.csv` | Flags measurements with excessive background contribution or excessive quoted colour uncertainty. Flags, never drops. |
| 06 | `06_apply_extinction_1kpc.py` | flagged catalogue, `sn_coordinates.csv`, SFD dust maps | `results/calibrated_color_1kpc_dered.csv` | Removes reddening by dust in our own Galaxy along each line of sight. |

Galactic extinction reddens everything behind it, so it must be removed before
the measured colour can be attributed to the host. E(B−V) comes from the SFD98
dust maps at each supernova's coordinates, with A_B and A_V derived from it via
standard coefficients.

**Note on quality thresholds.** At 1 kpc the aperture area is twenty-five times
smaller than at 5 kpc, so enclosed counts fall by roughly that factor while the
background *fraction* rises correspondingly. Thresholds calibrated on 5 kpc
fluxes cannot be carried across unchanged. Every cut is expressed relative to
something the object itself supplies.

### Stage E — Final result

| # | script | reads | writes | what it does |
|---|---|---|---|---|
| 07 | `07_final_catalogue_1kpc.py` | `calibrated_color_1kpc_dered.csv` | `results/final_catalogue_1kpc.csv`, figures | The deliverable: local B, V and B−V per object, with uncertainties, plus the distribution figure and the full sample cascade. |

---

## 4. Conventions carried over from the upstream audit

- **Never overwrite.** Existing outputs are renamed `*.bak_YYYYmmdd_HHMMSS`.
- **Flag, never drop.** Every exclusion is logged with a reason and counted in
  the sample cascade.
- **No absolute thresholds.** Every cut is expressed relative to something the
  object itself supplies.
- **No hard-coded drive letters.** Paths are derived from the script's own
  location, so the pipeline survives a change of machine.
- **Provenance travels with the data.** Input hashes are recorded; outputs carry
  the annulus tag.

`ANNULUS_TAG = "ann20-30"` throughout. Any script reading a tagged file verifies
the tag and refuses to run on a mismatch.

---

## 5. Facts about the dataset that constrain everything

1. **There is no Swope B imaging.** Swope contributes 103 V frames and zero B
   frames. A colour requires both bands from the same telescope, so all colour
   work is **du Pont only**. `B_ZP_swo.dat` exists and covers 121 objects, which
   indicates Swope B observations were taken during the campaign but are not
   part of the delivered subset. The supervisor has confirmed the delivered
   dataset is final.
2. **The zero-point files cover CSP-II only.** The break falls at 2010, between
   the two campaigns.
3. **Three plate scales exist in the archive.** Handled per frame from each
   frame's own WCS, with the audited value preferred.
4. **The combined stacks are free of supernova light.** Confirmed by the
   supervisor. Independently supported by the observation epochs and by direct
   surface-brightness inspection of the two most exposed objects.
5. **The background annulus sits at 20–30 kpc** because host light extends well
   past 10 kpc for most of the sample. A closer annulus would subtract host
   light along with sky.

---

## 6. Running it

```powershell
cd "E:\My Work\sn-local-photometry-1kpc"
.\.venv\Scripts\Activate.ps1

python scripts\00_seeing_audit_1kpc.py
python scripts\01_resolution_check_1kpc.py
python scripts\02_resolution_cut_options.py
python scripts\03_local_colour_1kpc.py
python scripts\04_apply_zero_points_1kpc.py
python scripts\05_flag_unreliable_1kpc.py
python scripts\06_apply_extinction_1kpc.py
python scripts\07_final_catalogue_1kpc.py
```

Environment: Python 3.12.7, numpy 2.5.2, pandas 3.0.5, astropy 8.0.1, scipy,
matplotlib. Exact versions recorded in `requirements-lock.txt`.

**Note.** The upstream pipeline was developed under pandas 2.x; this one runs
under pandas 3.x, which changed some `pivot_table` and missing-value defaults.
Before the 1 kpc numbers are compared with the 5 kpc numbers, one upstream step
should be re-run under the current environment to confirm the software stack is
not itself a variable.
