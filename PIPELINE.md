# Sky Camera Pipeline — Technical Documentation

Complete reference for the `src/skycamera/` package.
Read this to understand the data flow, every public function, and design decisions.

---

## Table of Contents

1. [Data Flow Overview](#1-data-flow-overview)
2. [config.py — Constants & Paths](#2-configpy)
3. [io.py — Loading, Indexing, Dataset Building](#3-iopy)
4. [preprocessing.py — Masking & Reprojection](#4-preprocessingpy)
5. [threshold.py — R/B Ratio Method](#5-thresholdpy)
6. [cnn.py — U-Net Segmentation](#6-cnnpy)
7. [sun.py — Sun Position & Ignore Mask](#7-sunpy)
8. [process_roboflow_labels.py — Roboflow COCO Converter](#8-process_roboflow_labelspy)
9. [Label Format Reference](#9-label-format-reference)
10. [Known Limitations](#10-known-limitations)
11. [Legacy / Retired Code](#11-legacy--retired-code)

---

## 1. Data Flow Overview

```
RAW IMAGES (data/full_raw/ — 30-min intervals, full 2024 year)
        │
        ▼
io.build_image_index(apply_daytime_filter=False)
  → image_index.csv  (path, timestamp, month, hour, sun_elevation_deg, is_daytime)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                          ┌───────────┴───────────┐
threshold.run_on_index()           │  GT MASK SOURCE       │
  [daytime rows only]              │                       │
  → cf_rb_threshold.csv            │  Roboflow COCO export │
                                   │  → process_roboflow_  │
cnn.run_cnn_on_index()             │    labels.py          │
  [all rows incl. night]           │                       │
  → cf_cnn.csv                     └───────────┬───────────┘
                                               │
                             data/masks_manual/*_GT.png
                             (+ default_ignore.png applied,
                              + sun-disk IGNORE from sun.py)
                             (CF logged with zenith weighting)
                                               │
                         io.build_combined_dataset()
                           ← data/acs_wsi/  (77 ACS_WSI pairs)
                           → masks_combined/dataset_index.csv
                                               │
                            ┌──────────────────┴────────────┐
                            ▼                               ▼
                      cnn.train_cnn()             03b_simpler_models
                  (ResNet-34 U-Net)           (MobileNetV2 + Random Forest)
                → cnn_sky.pt                 → cnn_mobilenet.pt
                → cf_cnn.csv                → cf_mobilenet.csv / cf_rf.csv

All method CSVs → 06_comparison.ipynb → comparison_summary.csv
All method CSVs + IMGW + ERA5 → 08_imgw_comparison.ipynb
Sun mask visual QA → 09_sun_mask_check.ipynb
```

---

## 2. `config.py`

Single source of truth for all constants. Import from here — never hardcode paths elsewhere.

### Camera intrinsics
```python
CX = 1438    # optical centre x (px)
CY = 928     # optical centre y (px)
R  = 938     # dome radius (px) — maps to θ = 90° in equidistant model
```

### Key paths (all relative to project ROOT)
```python
ROOT               # …/skycamera/  (project root)
RAW_DIR            # data/raw/           — original 12-day pilot (reference only)
FULL_RAW_DIR       # data/full_raw/      — full 2024 year, 30-min intervals (PRIMARY)
ACS_WSI_DIR        # data/acs_wsi/       — ACS_WSI dataset (77 pairs)
MASKS_MANUAL_DIR   # data/masks_manual/  — GT masks from Roboflow pipeline
MASKS_COMBINED_DIR # data/masks_combined/
CSV_DIR            # outputs/csv/
MASKS_PRED_DIR     # outputs/masks_pred/
PLOTS_DIR          # outputs/plots/
MODEL_DIR          # outputs/models/
```

### Other constants
```python
DAYTIME_MIN_ELEVATION_DEG = -6.0   # civil twilight threshold for is_daytime()
CF_MAX_ZENITH_DEG = 70.0           # zenith cutoff for area-weighted CF
CAMERA_NORTH_OFFSET_DEG = -114.0   # clockwise rotation from North to image top
CAMERA_PROJECTION_SCALE = 1.12     # radial stretch factor vs ideal equidistant
SUN_IGNORE_RADIUS_DEG = 10.0       # angular radius of sun disk ignore circle
OUT_H = OUT_W = 512                # equirectangular output size (visualisation only)
FILENAME_PATTERN                   # regex for YYYY_MM_DD__HH_MM_SS.jpg
```

---

## 3. `io.py`

### `parse_timestamp(filepath) → Optional[datetime]`
Parses `2024_01_15__12_04_51.jpg` → `datetime(2024, 1, 15, 12, 4, 51, tzinfo=UTC)`.
Returns `None` if filename doesn't match the pattern.

### `load_image(filepath) → np.ndarray`
Loads any JPEG/PNG via OpenCV, returns RGB uint8 `(H, W, 3)`.

### `build_image_index(root_dir=FULL_RAW_DIR, apply_daytime_filter=False, ...) → pd.DataFrame`
Recursively finds all `.jpg` files matching the timestamp pattern.
Returns DataFrame sorted by timestamp:
`path`, `timestamp`, `month`, `hour`, `sun_elevation_deg`, `is_daytime`.

**No image loading required** — daytime flag computed from timestamps only. Fast.

**Note:** CNN notebooks use `apply_daytime_filter=False` to include all images.
R/B threshold is applied only to rows where `is_daytime=True` inside `run_on_index`.

### `build_combined_dataset(acs_root, manual_masks_dir, ...) → pd.DataFrame`
Merges ACS_WSI pairs with all `*_GT.png` files in `manual_masks_dir`.
Computes **area-weighted** `cf_measured` for every pair using `weighted_cf()`.
Prints class-balance summary (clear/partial/overcast).

**Current dataset (stale — retrain pending):** 627 Warsaw GT masks + 77 ACS_WSI = 704 total pairs.
735 Roboflow images are annotated; run `process_roboflow_labels.py --overwrite` and retrain to update.

**Output columns:** `image_path`, `mask_path`, `source` ("acs_wsi"/"manual"),
`cf_level`, `cf_approx`, `cf_measured`.

**Search order:** searches `full_raw/` first, then `raw/` as fallback (fixed io.py:437).

### ACS_WSI GT mask encoding (verified by pixel inspection)

| Colour | RGB | Label |
|--------|-----|-------|
| Black | [0, 0, 0] | Outside dome |
| Blue | [R≈0, G≈0, B≈87] | Sky |
| Violet | [R≈30, G≈30, B≈190] | Cloud |

Binarisation: `red_channel > 15` → cloud.

---

## 4. `preprocessing.py`

### `build_circular_mask(h, w, cx, cy, r) → np.ndarray`
Boolean mask, True inside the dome circle. Build once at native resolution and reuse.

### `build_zenith_weight_map(h, w, cx, cy, r, max_zenith_deg=70.0) → np.ndarray`
Float32 array: each dome pixel has weight `cos(θ)`, where `θ` is the zenith angle
from the equidistant fisheye model (`ρ = R · θ / (π/2)`).
Pixels outside dome or beyond `max_zenith_deg` → weight 0.

Pass to `weighted_cf()` for area-weighted CF (WMO hemisphere definition).

### `weighted_cf(binary_mask, weights) → float`
`Σ(cloud · w) / Σ(valid · w)`. `binary_mask` values: 0=sky, 1=cloud, 255=ignore.
Returns `nan` when no valid weighted pixels exist.

### `fisheye_to_equirectangular(...)` — visualisation only
Not used in the CF pipeline. All CF computation and CNN training operate on native
fisheye images to avoid train/inference geometry mismatch.

---

## 5. `threshold.py`

### `cloud_fraction_rb_threshold(img, mask, threshold=0.6, weights=None)`
R/B ≥ threshold → cloud. Pixels with B=0 excluded. Returns `(cf, debug_overlay)`.

**Threshold selection:** Warsaw-tuned (~0.55 from GT masks) is always preferred.
ACS_WSI fallback (~0.85) used only when fewer than 5 Warsaw GT masks exist.
Cameras differ substantially — do not cross-apply thresholds.

### `run_on_index(df_index, dome_mask, threshold, daytime_only=True, weights=None)`
Applies R/B threshold to index DataFrame. `daytime_only=True` skips night rows —
R/B is physically meaningless at night (no Rayleigh scattering).
Applies `mask_sun_pixels()` per image before CF computation.
Output: `timestamp`, `cloud_fraction`, `month`, `hour`.

---

## 6. `cnn.py`

**Architecture:** U-Net with ResNet-34 encoder (ImageNet pretrained).
**Task:** Binary segmentation (cloud/sky), 512×512 input.

### `predict_mask(model, img, dome_mask, img_size, threshold=0.5, weights=None, image_path=None)`
1. Resize → normalise (ImageNet stats) → forward pass → sigmoid → threshold
2. Resize prediction back to original dimensions
3. If `image_path` provided: apply `mask_sun_pixels()` — exclude sun disk from CF
4. CF = `weighted_cf(pred, zenith_weights)`

Always pass `image_path` at inference — ensures sun mask is applied consistently
with how sun pixels are annotated as `LABEL_IGNORE` in GT masks during training.

### `run_cnn_on_index(df_index, model, dome_mask, ..., weights=None)`
Batch inference over **all** rows (day + night). CNN works on any image.
`save_masks=False` in notebook 03 — avoids writing thousands of PNG files to disk.

### `train_cnn(df_train, df_val, ...)`
- Loss: `BCEWithLogitsLoss`, weighted per-pixel
- Optimizer: Adam lr=1e-4, `ReduceLROnPlateau(patience=3, factor=0.5)`
- Early stopping: restores best weights when val loss stagnates for `patience=7` epochs

**Training data (stale — retrain pending):** 627 Warsaw masks + 77 ACS_WSI → ~500 train, ~200 val, ~126 test.
Test set = 20% of Warsaw masks only (never ACS_WSI in test — different camera).
After re-run with 735 masks these numbers will update.

---

## 7. `sun.py`

Computes sun altitude/azimuth from filename timestamps and projects onto fisheye pixel grid.

**Applied consistently at all stages:**
- GT masks: `LABEL_IGNORE` written by `process_roboflow_labels.py`
- Inference: `mask_sun_pixels()` in `threshold.run_on_index`, `cnn.predict_mask`, RF/MobileNet cells

### Key functions

**`sun_altaz(dt) → (altitude_deg, azimuth_deg)`**
Altitude = degrees above horizon (negative = below). Azimuth = clockwise from North.

**`altaz_to_pixel(altitude_deg, azimuth_deg) → Optional[(px, py)]`**
Equidistant projection with `CAMERA_NORTH_OFFSET_DEG` and `CAMERA_PROJECTION_SCALE`.
Returns `None` if sun is below horizon.

**`sun_ignore_mask(image_path, img_shape) → Optional[np.ndarray]`**
Full pipeline: filename → timestamp → altitude/azimuth → pixel → boolean circle.
Radius = `SUN_IGNORE_RADIUS_DEG` converted to pixels via projection scale.
Returns `None` when sun below horizon or timestamp not parseable.

**`mask_sun_pixels(dome_mask, image_path) → np.ndarray`**
Returns copy of `dome_mask` with sun pixels set False. Safe to call on every image.

### Calibration constants
```python
CAMERA_NORTH_OFFSET_DEG = -114.0   # tune if circle is rotationally offset from sun
CAMERA_PROJECTION_SCALE =   1.12   # tune if circle is too close/far from dome centre
SUN_IGNORE_RADIUS_DEG   =   10.0   # tune if glare extends further than circle
```

Validate visually with `09_sun_mask_check.ipynb`. After any change, re-run
`process_roboflow_labels.py --overwrite` so GT masks get the updated sun ignore region.

---

## 8. `process_roboflow_labels.py`

Converts Roboflow COCO-segmentation export to GT masks in `data/masks_manual/`.
This is the **only active path** for creating GT masks — manual and SAM 2 labelling
tools exist in `src/skycamera/` but are no longer used (see section 11).

### Processing steps per image
1. Rasterise COCO polygon/RLE annotations → label array
2. Fill complement class (cloud-only → fill sky, sky-only → fill cloud, mixed → leave)
3. Apply `default_ignore.png` (antenna, cables)
4. Apply `sun_ignore_mask()` (sun disk → LABEL_IGNORE)
5. Skip if ≥ `skip_fraction` (default 0.50) of dome pixels are IGNORE
6. Save `{stem}_GT.png` + append to `labelling_log.csv`

### Entry points
```python
process_all()                    # skip existing masks
process_all(overwrite=True)      # re-process everything (use after config changes)
process_all(skip_fraction=0.4)   # stricter corrupted-image filter
```

### Adding new Roboflow labels
1. Label in Roboflow (cloud / sky / ignore classes) → re-export as COCO Segmentation
2. Place export in `Cloud segmentation nibqv.coco-segmentation/train/`
3. `python -m skycamera.process_roboflow_labels --overwrite`
4. Delete stale CSVs and model checkpoints; retrain in notebooks 03 and 03b

---

## 9. Label Format Reference

```
File:    {image_stem}_GT.png
Format:  PNG, RGB, same resolution as source image

Pixel colours:
  [30,  120, 200] → LABEL_SKY        = 0
  [220, 220, 220] → LABEL_CLOUD      = 1
  [255, 140,   0] → LABEL_IGNORE     = 2
  [0,   0,   0  ] → LABEL_UNLABELLED = 255 (default)

Log:  labelling_log.csv
  filename | timestamp_labelled | cf_estimate | notes
  notes values: "roboflow_coco" (active), "sam2_interactive"/"sam2_batch"/"" (legacy)
```

---

## 10. Known Limitations & Methodological Notes

| Area | Issue | Status |
|------|-------|--------|
| R/B threshold | Daytime only — ~9,276 daytime rows vs 16,585 for CNN. Always inner-join on timestamp for pairwise comparisons. | By design |
| R/B threshold | Warsaw threshold (0.55) vs ACS_WSI (0.85) — cameras differ, never cross-apply | Documented |
| R/B threshold | Struggles with thin cirrus and haze | Use CNN for these images |
| CNN | ACS_WSI training uses inferred dome geometry — zenith weights approximate for those 77 pairs | Minor; exact intrinsics unavailable |
| CNN | Training biased toward clear/overcast; partial-cloud under-represented | Add `WeightedRandomSampler` — see TODO |
| CNN | IoU ~0.43 lower than CF MAE ~0.065 suggests — edge boundaries imprecise, CF accurate | Expected; document if publishing |
| IMGW | Okta→CF linear conversion introduces ~0.06 CF quantisation uncertainty | Standard conversion; document |
| IMGW | Hourly mean of two 30-min camera readings (HH:00 + HH:30); no intra-hour stability filter | See notebook 08 |
| ERA5 | ~30 km grid resolution — nearest grid point may not exactly match camera location | Check Δlat/Δlon in notebook 08 output |
| ERA5 loading | ERA5 files from CDS use a `points` dimension (not lat/lon grid) — `.sel(lat=..., lon=...)` fails | Use numpy argmin on latitude/longitude arrays to find nearest point index |
| Sun calibration | North offset and projection scale calibrated visually | Validate with notebook 09 |
| build_image_index | Astronomical daytime flag — corrupted images still get `is_daytime=True` | Load validation happens downstream |
| process_roboflow_labels | Compressed RLE requires `pycocotools` | `pip install pycocotools` |
| Kim 2023 comparison | Closest paper (full-year, synoptic validation, day+night CNN) but uses image-level classification not pixel segmentation — different task | Our pixel segmentation + area-weighted CF + multi-method comparison + ERA5 are the differentiators |

---

## 11. Legacy / Retired Code

These modules and notebooks remain in the repository but are **no longer part of the
active pipeline**. The code is correct; the tools were superseded by Roboflow's
professional labelling interface which is faster and more consistent.

| Item | Location | Reason retired |
|------|----------|----------------|
| `notebooks/legacy/04_gemma_vlm.ipynb` | Gemma 3 zero-shot CF via Ollama | ~10 min/image on CPU — computationally infeasible for full dataset |
| `notebooks/legacy/07_sam2_labelling.ipynb` | SAM 2 interactive + batch pseudo-labelling | Superseded by Roboflow; Roboflow gives faster, more consistent results |
| `src/skycamera/vlm.py` | Gemma 3 / Ollama client | Supporting code for retired notebook 04 |
| `src/skycamera/labelling.py` | Manual brush labelling tool | Superseded by Roboflow |
| `src/skycamera/sam_labelling.py` | SAM 2 assisted labelling tool | Superseded by Roboflow |

The `labelling.py` label constants (`LABEL_SKY`, `LABEL_CLOUD`, `LABEL_IGNORE`,
`LABEL_UNLABELLED`) are still imported by `process_roboflow_labels.py` and `io.py` —
do not delete `labelling.py` even though the `LabellingTool` class is unused.
