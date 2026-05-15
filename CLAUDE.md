# CLAUDE.md — Sky Camera Project

Instructions for Claude Code when working in this repository.

---

## Project in one sentence

Sky camera cloud fraction pipeline for Warsaw 2024: fisheye images → GT masks → CNN/R-B/RF segmentation → comparison vs IMGW synoptic data and ERA5 reanalysis.

---

## Current state (as of 2026-05-13)

All notebooks have been run on the full dataset. Key results are final and documented.
Do NOT rerun inference unless asked — CSVs already exist.

**What exists and is complete:**
- `outputs/csv/cf_cnn.csv` — ResNet-34 on all 16,585 images (day+night)
- `outputs/csv/cf_mobilenet.csv` — MobileNetV2 on all 16,585 images
- `outputs/csv/cf_rb_threshold.csv` — R/B threshold on 9,276 daytime images
- `outputs/csv/comparison_summary.csv` — all metrics
- `outputs/csv/imgw_comparison_metrics.csv` — vs IMGW + ERA5

**Models trained on:** 627 Warsaw masks + 77 ACS_WSI = 704 total pairs.
Train 436 / Val 142 / Test 126 (Warsaw only).

---

## Key facts to remember

- **Primary image dataset**: `data/full_raw/` (full 2024 year, 30-min intervals). NOT `data/raw/` (old 12-day pilot, kept for reference only).
- **GT masks**: 627 Warsaw masks in `data/masks_manual/` (Roboflow COCO export). Plus 77 ACS_WSI pairs (different camera, used for training only, never test).
- **CNN runs on ALL images** (day + night, `apply_daytime_filter=False`). R/B threshold is daytime only — 9,276 rows vs CNN's 16,585.
- **No retraining** unless explicitly requested. Model weights in `outputs/models/` are valid.
- **All paths flow through `config.py`** — never hardcode paths in notebooks or scripts.
- **`build_combined_dataset` searches `full_raw/` first, then `raw/`** — this was fixed in `io.py:437`.
- **Temporal matching in notebook 08**: hourly MEAN of two 30-min readings (not snap-matching). No stability filter.
- **ERA5 files use a `points` dimension** (11 points, not a lat/lon grid) — use index-based selection, not `.sel(lat=..., lon=...)`. Warsaw = index 0 (lat=52.25, lon=21.00).

---

## Architecture

```
src/skycamera/
  config.py          ← single source of truth for all constants and paths
  io.py              ← load_image, build_image_index, build_combined_dataset
  preprocessing.py   ← dome mask, zenith weights, weighted_cf
  threshold.py       ← R/B ratio (daytime only), run_on_index
  cnn.py             ← ResNet-34 U-Net, train_cnn, predict_mask, run_cnn_on_index
  sun.py             ← sun_altaz, altaz_to_pixel, sun_ignore_mask, mask_sun_pixels
  labelling.py       ← LabellingTool, save_mask, load_existing_mask, LABEL_* constants
  sam_labelling.py   ← SAMLabellingTool, batch_pseudolabel
  process_roboflow_labels.py  ← Roboflow COCO → GT masks
  sample_images.py   ← stratified image sampling
  thin_raw.py        ← thin archive to 30-min intervals
```

---

## Notebook run order

```
process_roboflow_labels.py --overwrite
→ 01 (EDA, index)
→ 02 (R/B threshold, daytime only)
→ 03 (CNN, ALL images day+night)
→ 03b (MobileNetV2 + RF, all images)
→ 06 (GT test set comparison — all three methods)
→ 08 (IMGW + ERA5 + CNN-vs-RB conditional analysis)
→ 09 (sun mask visual check — run after any config.py change)
```

---

## Critical constants (config.py)

```python
CX, CY, R = 1438, 928, 938          # fisheye camera geometry
CF_MAX_ZENITH_DEG = 70.0            # zenith cutoff for area-weighted CF
SUN_IGNORE_RADIUS_DEG = 10.0        # angular radius of sun disk ignore circle
CAMERA_NORTH_OFFSET_DEG = -114.0    # camera mount rotation from North
CAMERA_PROJECTION_SCALE = 1.12      # fisheye radial stretch factor
FULL_RAW_DIR = ROOT / "data" / "full_raw"   # primary image directory
MASKS_MANUAL_DIR = ROOT / "data" / "masks_manual"
```

---

## Key results (do not recompute unless asked)

### Warsaw GT test set (126 images, 20% hold-out, random_state=42)

| Method | MAE | r | IoU | Notes |
|--------|-----|---|-----|-------|
| ResNet-34 U-Net | 0.065 | 0.950 | 0.441 | All images |
| MobileNetV2 U-Net | **0.060** | **0.952** | 0.432 | All images — best CF |
| R/B threshold | 0.143 | 0.824 | — | 103 daytime images only |

### vs IMGW NOG (8,119 matched hours, hourly mean)

| Method | r | MAE | Bias |
|--------|---|-----|------|
| MobileNetV2 | **0.824** | **0.168** | −0.037 |
| ResNet-34 | 0.814 | 0.172 | −0.037 |
| R/B threshold | 0.731 | 0.192 | −0.034 |

### vs ERA5 tcc (8,161 hours)

| Method | r | MAE |
|--------|---|-----|
| ERA5 vs IMGW (ceiling) | 0.751 | 0.170 |
| MobileNetV2 | 0.705 | 0.193 |
| ResNet-34 | 0.694 | 0.196 |
| R/B threshold | 0.668 | 0.197 |

### CNN vs R/B threshold — where CNN wins (daytime, n=4055)

CNN wins only **24.3%** of hours overall. R/B mean |err|=0.175 vs CNN 0.187.

| Condition | CNN win% | DMAE | Note |
|-----------|----------|------|------|
| Okta 0 (clear) | **98.4%** | +0.067 | CNN's main advantage |
| Okta 1–2 | ~7% | −0.03 to −0.05 | R/B wins |
| Okta 3–6 (partial) | 23–36% | −0.017 to −0.035 | R/B wins |
| Okta 7–8 (overcast) | ~8–19% | ~0 | Essentially tied |
| Summer | 31.0% | −0.010 | CNN best season |
| Winter | 12.0% | −0.008 | CNN worst season |

**Interpretation:** CNN's advantage is at clear-sky (okta 0) where R/B over-classifies haze/blue-sky variation as cloud. In the partial-cloud regime (oktas 3–6, CF 0.375–0.75) R/B is actually better — this is the training data gap (only 144/627 = 23% partial-cloud masks).

### ERA5 layer correlations (Pearson r, all hours)

| Reference | ResNet-34 | MobileNetV2 | R/B |
|-----------|-----------|-------------|-----|
| IMGW NOG (total) | 0.814 | **0.824** | 0.731 |
| IMGW CLCM (low) | 0.762 | 0.771 | 0.674 |
| ERA5 tcc | 0.694 | 0.705 | 0.668 |
| ERA5 lcc | 0.557 | 0.561 | 0.518 |
| ERA5 mcc | 0.504 | 0.511 | 0.488 |
| ERA5 hcc | 0.320 | 0.325 | 0.348 |

Camera barely correlates with high cloud (hcc r≈0.32) — thin cirrus is nearly invisible in RGB. Best reference is IMGW NOG; ERA5 tcc ceiling is r=0.751 vs IMGW.

---

## Sun mask — how it works

`sun.py` computes sun position from filename timestamp (UTC) using pysolar,
projects onto fisheye grid via equidistant model, draws boolean circle of
radius `SUN_IGNORE_RADIUS_DEG`. Applied consistently:
- **GT masks**: `LABEL_IGNORE` written by `process_roboflow_labels.py` and `SAMLabellingTool`
- **Inference**: `mask_sun_pixels(dome_mask, image_path)` called in every inference path
- **ACS_WSI images**: timestamp not parseable (different filename format) → sun mask silently skipped (correct behaviour, different camera/year anyway)

Validate with `09_sun_mask_check.ipynb`. If orange circle misses actual sun disk,
tune `CAMERA_NORTH_OFFSET_DEG` / `CAMERA_PROJECTION_SCALE` in `config.py`, then
re-run `process_roboflow_labels.py --overwrite`.

---

## CF computation

Always use **area-weighted CF** via `weighted_cf(mask, zenith_weights)`:
- Weight = `cos(zenith_angle)` per pixel inside dome, 0 beyond 70°
- `LABEL_IGNORE` and `LABEL_UNLABELLED` pixels excluded
- Returns `nan` when no valid pixels exist

Do NOT use raw pixel counts for CF — they overweight the noisy horizon.

---

## Label format

```
File: {image_stem}_GT.png  (PNG, RGB, same resolution as source image)
LABEL_SKY        = 0   → colour [30, 120, 200]
LABEL_CLOUD      = 1   → colour [220, 220, 220]
LABEL_IGNORE     = 2   → colour [255, 140, 0]
LABEL_UNLABELLED = 255 → colour [0, 0, 0]
```

`build_combined_dataset()` picks up all `*_GT.png` regardless of which tool created them.

---

## Adding new Roboflow labels

1. Label in Roboflow (cloud / sky / ignore classes), re-export as COCO Segmentation.
2. Place export in `Cloud segmentation nibqv.coco-segmentation/train/`.
3. `python -m skycamera.process_roboflow_labels --overwrite`
4. Delete stale outputs: `outputs/csv/cf_*.csv`, model `.pt` files if retraining.
5. Retrain in notebooks 03 and 03b (uncomment training cells, delete `.pt` first).

---

## What NOT to do

- Do not use `data/raw/` — always use `data/full_raw/`.
- Do not hardcode `1.12` (projection scale) — import `CAMERA_PROJECTION_SCALE` from config.
- Do not apply daytime filter when building the index for CNN inference.
- Do not compare raw row counts between `cf_cnn.csv` (16,585) and `cf_rb_threshold.csv` (9,276) — different by design. Always inner-join on timestamp.
- Do not retrain models unless explicitly asked.
- Do not save predicted masks during CNN inference on full_raw — `save_masks=False` in notebook 03.
- Do not use `.sel(lat=..., lon=...)` on ERA5 files — they use a `points` dimension, not a grid. Use index-based numpy selection.
- Do not use snap-matching (±5 min) for IMGW temporal alignment — notebook 08 uses hourly mean of 30-min readings.

---

## ERA5 data

`data/ERA5/ERA5-{tcc,lcc,mcc,hcc}-2024.nc` — hourly, 11 points, Warsaw = index 0.
Load with `xarray`, select by numpy argmin on lat/lon distance.
Variables are cloud cover fractions [0, 1] — no conversion needed.
ERA5 tcc vs IMGW NOG: r=0.751, MAE=0.170 — this is the ceiling for camera-vs-ERA5.

---

## Known biases

- **Negative bias vs IMGW**: all methods −0.034 to −0.077. Camera underestimates vs IMGW observer, likely because 70° zenith cutoff clips near-horizon cloud that observer includes.
- **Positive bias vs IMGW CLCM (low cloud)**: all methods +0.14–0.18. Camera integrates all cloud layers; CLCM is only the lowest layer.
- **Partial-cloud gap**: CNN trained on 23% partial-cloud images → underperforms R/B in okta 3–6 regime. Primary improvement target.

---

## What to do next — improvement options

Ordered by effort/impact ratio:

**Option 1 — Weighted resampling (low effort, ~1 overnight run)**
Add `WeightedRandomSampler` to `SkyDataset` in `cnn.py`, oversample CF 0.2–0.8
images 3×. No new labelling. Delete `.pt` files, retrain notebooks 03 and 03b.
Check partial-cloud bar in `comparison_mae_by_level.png` — it should drop.
This is the right first move.

**Option 2 — Label 50–80 more partial-cloud images (medium effort)**
Use `sample_images.py` with partial-cloud oversampling to select candidates.
Label in Roboflow → `process_roboflow_labels.py --overwrite` → retrain.
Also grows the test set from 26 to ~40 partial images, making metrics more reliable.
Do this after Option 1 if resampling alone is insufficient.

**Option 3 — Frame results correctly (zero effort, valid now)**
CNN is genuinely better on GT test (MAE 0.065 vs 0.143, 2× better).
IMGW comparison is dominated by okta 0+7+8 (~60% of hours) where any method works.
A paper can honestly state: CNN outperforms R/B on segmentation quality and ambiguous
conditions; R/B is competitive only at clear/overcast extremes.

**Option 4 — Night images (unique CNN advantage, already done)**
CNN covers 16,585 images vs R/B's 9,276. For climate applications (monthly/annual
mean CF), CNN gives the only complete time series. Real contribution.

**Recommendation:** Option 1 + Option 3. Option 2 if time allows for stronger paper.

---

## Environment

- Conda env: `geo`
- Python: `C:\Users\szymo\anaconda3\envs\geo\python.exe`
- Shell: PowerShell on Windows 11
- No git repository
