# Sky Camera Cloud Fraction Pipeline

Automated cloud fraction estimation from fish-eye sky camera images.
Warsaw, Poland (~52°N), 5-minute temporal resolution.
Full dataset: 2024 full year, thinned to 30-min intervals (`data/full_raw/`).

---

## Project layout

```
skycamera/
├── Cloud segmentation nibqv.coco-segmentation/
│   └── train/              ← Roboflow COCO export: ~627 images + _annotations.coco.json
├── data/
│   ├── raw/                ← original 12-day pilot (kept for reference)
│   ├── full_raw/           ← full 2024 year, thinned to 30-min intervals (primary dataset)
│   ├── IMGW/               ← IMGW Warsaw-Okęcie hourly synoptic data (s_t_375_2024.csv)
│   ├── ERA5/               ← ERA5 reanalysis NetCDF: tcc, lcc, mcc, hcc (2024)
│   ├── acs_wsi/            ← ACS_WSI dataset (Ye et al. 2022), 77 labelled pairs
│   ├── masks_manual/       ← GT masks (*_GT.png) + labelling_log.csv
│   │   └── default_ignore.png   ← static antenna/cable mask (applied to every image)
│   ├── masks_combined/     ← merged dataset index CSV for CNN training
│   └── sam2_checkpoints/   ← sam2.1_hiera_small.pt (184 MB)
├── notebooks/
│   ├── 01_preprocessing.ipynb       ← EDA, dome mask, image index, ACS_WSI loader
│   ├── 02_rb_threshold.ipynb        ← R/B ratio cloud detection, threshold tuning (daytime only)
│   ├── 03_cnn_segmentation.ipynb    ← U-Net fine-tuning, evaluation, full run (all images)
│   ├── 04_gemma_vlm.ipynb           ← Gemma 3 zero-shot via Ollama (slow — see notes)
│   ├── 05_labelling_tool.ipynb      ← Manual brush labelling tool
│   ├── 06_comparison.ipynb          ← Head-to-head comparison of all methods
│   ├── 07_sam2_labelling.ipynb      ← SAM 2 assisted + batch pseudo-labelling
│   ├── 08_imgw_comparison.ipynb     ← Camera vs IMGW station + ERA5 reanalysis
│   └── 09_sun_mask_check.ipynb      ← Visual validation of sun disk ignore region
├── outputs/
│   ├── csv/                ← cf_rb_threshold.csv, cf_cnn.csv, cf_mobilenet.csv, image_index.csv, ...
│   ├── masks_pred/         ← predicted mask PNGs from CNN
│   ├── models/             ← cnn_sky.pt, cnn_mobilenet.pt
│   └── plots/              ← all figures
├── src/skycamera/
│   ├── config.py                      ← all constants and paths
│   ├── io.py                          ← image loading, index, ACS_WSI loader, combined dataset
│   ├── preprocessing.py               ← dome mask, zenith weighting, fisheye reprojection
│   ├── threshold.py                   ← R/B ratio method (daytime only)
│   ├── cnn.py                         ← U-Net training and inference
│   ├── vlm.py                         ← Gemma 3 via Ollama REST API
│   ├── labelling.py                   ← manual brush labelling tool + _compute_cf
│   ├── sam_labelling.py               ← SAM 2 assisted + batch pseudo-labelling
│   ├── sun.py                         ← sun position → pixel projection (pysolar)
│   ├── thin_raw.py                    ← thins raw image archive to 30-min intervals
│   ├── sample_images.py               ← stratified image sampling for labelling
│   └── process_roboflow_labels.py     ← convert Roboflow COCO export → GT masks
├── PIPELINE.md             ← full technical reference for the src/skycamera package
├── CLAUDE.md               ← instructions for Claude Code (includes key results + improvement options)
├── LITERATURE.md           ← literature review, paper comparison table, citation counts, publication strategy
├── TODO.md                 ← task list and known gaps
└── requirements.txt
```

---

## Camera parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 3096 × 2080 px |
| Optical centre | CX=1438, CY=928 |
| Dome radius | R=938 px |
| Projection | Equidistant fisheye |
| Location | Warsaw, Poland (52.24°N, 21.03°E) |
| Timestamps | UTC |
| Interval | 30 min (thinned full_raw) / 5 min (raw pilot) |
| Dataset | 2024 full year (full_raw, primary) |

---

## Setup

```bash
# Activate the geo conda environment
conda activate geo

# Install packages (first time only)
pip install segmentation-models-pytorch torchvision sam2 pycocotools xarray netcdf4
```

### Running scripts directly

```powershell
# Run process_roboflow_labels
$env:PYTHONPATH = "src"; python -m skycamera.process_roboflow_labels --overwrite
```

### Ollama (for VLM notebook)
```bash
ollama serve           # start server if not running
# model already pulled: gemma3:4b (3.3 GB)
```

**Warning:** Gemma 3 inference takes ~10 min/image on CPU — ~30 hours for 180 images.
See the VLM section below for faster alternatives.

---

## Run order

```
process_roboflow_labels.py → [07] → 01 → 02 → 03 → 03b → 06 → 08 → 09
```

| Step | Notebook / Script | Output | Notes |
|------|------------------|--------|-------|
| 0 | `process_roboflow_labels.py --overwrite` | GT masks in `masks_manual/` | Run after each new Roboflow export |
| 1 | `01_preprocessing.ipynb` | `image_index.csv`, plots | EDA only, fast |
| 2 | `02_rb_threshold.ipynb` | `cf_rb_threshold.csv` | Daytime only; re-tunes threshold on 627 masks |
| 3 | `03_cnn_segmentation.ipynb` | `cf_cnn.csv` | All images (day+night); loads existing weights |
| 4 | `03b_simpler_models.ipynb` | `cf_mobilenet.csv` | All images; MobileNetV2 + Random Forest |
| 5 | `06_comparison.ipynb` | `comparison_summary.csv`, plots | Instant — reads CSVs |
| 6 | `08_imgw_comparison.ipynb` | plots, `imgw_comparison_metrics.csv` | IMGW + ERA5 comparison |
| 7 | `09_sun_mask_check.ipynb` | `09_sun_mask_check.png` | Visual QA — run after any config change |
| — | `07_sam2_labelling.ipynb` | GT masks | Optional extra labels |
| — | `04_gemma_vlm.ipynb` | `cf_gemma.csv` | Optional, very slow on CPU |

---

## Cloud detection methods

### Method 1 — R/B Ratio Threshold (`threshold.py`)
Classical statistical method (Long et al. 2006). R/B ≥ threshold → cloud.
- **Daytime only** — R/B is physically meaningless at night (no Rayleigh scattering)
- Threshold tuned on Warsaw GT masks (627 masks); ACS_WSI is reference/fallback only
- Optimal Warsaw threshold: ~0.55 (ACS_WSI would give 0.85 — cameras differ substantially)
- Fast: < 1 ms/image; suitable for full time-series

### Method 2 — ResNet-34 U-Net (`cnn.py`, notebook 03)
Large pretrained encoder, fine-tuned on ACS_WSI + manual/Roboflow labels.
- Works on **all images** including night (set `apply_daytime_filter=False`)
- Training: 627 Warsaw masks + 77 ACS_WSI; test set: 20% held-out Warsaw images
- Input: 512×512, ~50 ms/image CPU inference

### Method 3 — MobileNetV2 U-Net (`03b_simpler_models.ipynb`)
Same U-Net framework, lighter encoder (~6.6M vs 24M params).
- ~3× faster training and inference than ResNet-34
- Input: 256×256; checkpoint: `outputs/models/cnn_mobilenet.pt`

### Method 4 — Random Forest pixel classifier (`03b_simpler_models.ipynb`)
No neural network. Per-pixel features: R/B, R/G, HSV, zenith angle, patch stats.
- Trains in seconds; interpretable via feature importances

### Method 5 — Gemma 3 VLM (`vlm.py`, notebook 04) — optional
Zero-shot via `gemma3:4b` via Ollama. ~10 min/image on CPU — excluded from notebook 06.

---

## GT masks — current state

All masks live in `data/masks_manual/`. Format: `{stem}_GT.png` colour PNG.

| Source | Count | Notes |
|--------|-------|-------|
| Roboflow COCO export | ~627 | Primary batch source — run `process_roboflow_labels.py` |
| SAM 2 interactive | varies | Near pixel-perfect, ~30 sec/image |
| SAM 2 batch auto | varies | ~10 sec/image, good for bootstrapping |
| Manual brush | varies | Slowest but most precise |

`build_combined_dataset()` automatically picks up every `*_GT.png` in the directory
regardless of which tool created it. All tools log CF to `labelling_log.csv`.

**CF values in `labelling_log.csv` are area-weighted** using the cosine-zenith weight map
(`CF_MAX_ZENITH_DEG=70°`) — consistent with how notebooks compute CF.

### Default ignore mask

`data/masks_manual/default_ignore.png` marks static structures (antenna mast, cables).
Applied automatically by all three labelling routes. Created once in notebook 07.

---

## Cloud fraction computation

All CF values are **area-weighted** using a cosine-zenith weight map:

- Each dome pixel is weighted by `cos(θ)`, where `θ` is the zenith angle
- Pixels beyond `CF_MAX_ZENITH_DEG = 70°` get weight 0 (horizon exclusion)
- This matches the WMO hemisphere definition and removes noisy near-horizon pixels
- Implemented in `preprocessing.build_zenith_weight_map()` and `preprocessing.weighted_cf()`

**Sun-disk pixels are excluded at all stages** via `sun.mask_sun_pixels()`:
- GT mask creation: sun pixels written as `LABEL_IGNORE`
- Inference (R/B, CNN, RF, MobileNetV2): sun pixels removed from the active dome mask before CF computation

Use `09_sun_mask_check.ipynb` to visually validate that the sun disk ignore region
is landing correctly. Tune `CAMERA_NORTH_OFFSET_DEG` and `CAMERA_PROJECTION_SCALE`
in `config.py` if it is off, then re-run `process_roboflow_labels.py --overwrite`.

---

## Converting Roboflow labels to GT masks

```powershell
# Convert all images (skip existing)
& C:/Users/szymo/anaconda3/envs/geo/python.exe src/skycamera/process_roboflow_labels.py

# Overwrite all existing masks (e.g. after config change or new export with updated labels)
& C:/Users/szymo/anaconda3/envs/geo/python.exe src/skycamera/process_roboflow_labels.py --overwrite
```

Per image the script: rasterises polygons → fills complement class → applies default_ignore →
applies sun-disk ignore → skips if too many IGNORE pixels → saves `_GT.png` + logs to CSV.

---

## ACS_WSI dataset

77 labelled pairs (7 per CF level 0–10), 501×501 px, different camera than Warsaw pilot.
Stored at `data/acs_wsi/`. Source: Ye et al. 2022, DOI 10.1029/2022EA002220.

Used for CNN training (train+val only, never test) and R/B threshold tuning (reference/fallback).
The Warsaw-tuned threshold (from 627 GT masks) is always preferred over the ACS_WSI fallback.

---

## Key results (2024, full_raw dataset)

### vs. held-out GT masks (Warsaw test set, 126 images = 20% of 627)

| Method | MAE | Pearson r | Mean IoU | Notes |
|--------|-----|-----------|----------|-------|
| MobileNetV2 U-Net | **0.060** | **0.952** | 0.432 | All images incl. night |
| ResNet-34 U-Net | 0.065 | 0.950 | **0.441** | All images incl. night |
| R/B threshold | 0.143 | 0.824 | — | 103 daytime images only |

CNN is 2× better than R/B on GT test. MobileNetV2 edges ResNet-34 on CF accuracy.

### vs. IMGW Warsaw-Okęcie station (8,119 matched hours, hourly mean of 30-min readings)

| Method | r | MAE | Bias |
|--------|---|-----|------|
| MobileNetV2 U-Net | **0.824** | **0.168** | −0.037 |
| ResNet-34 U-Net | 0.814 | 0.172 | −0.037 |
| R/B threshold | 0.731 | 0.192 | −0.034 |

CNN advantage vs IMGW is modest overall because the comparison is dominated by clear
(okta 0) and overcast (okta 7–8) hours where all methods work. CNN wins decisively
only at okta 0 (clear sky, win rate 98.4%). Primary gap: partial-cloud (okta 3–6)
where training data is under-represented (23% of masks).

### vs. ERA5 reanalysis (8,161 hours)

ERA5 tcc vs IMGW: r=0.751, MAE=0.170 — this is the upper bound for camera-vs-ERA5.
Camera vs ERA5 tcc: MobileNetV2 r=0.705, ResNet-34 r=0.694 (within 6% of ceiling).
Camera barely correlates with ERA5 hcc (r≈0.32) — thin cirrus invisible in RGB.

Negative bias all methods (−0.034 to −0.077): 70° zenith cutoff clips near-horizon
cloud that IMGW observer includes. Positive bias vs IMGW CLCM (+0.14): camera
integrates all cloud layers; CLCM records only the lowest.

---

## License

This repository uses a dual license:

| Scope | License |
|-------|---------|
| Code (`src/`, `notebooks/`, scripts) | [MIT](LICENSE-MIT) |
| Results, figures, model weights, GT masks (`outputs/`, `data/masks_manual/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |

If you use results or figures in a publication, please cite accordingly.

---

## Data Attribution

Sky camera images were provided by the Institute of Geophysics, Faculty of Physics, University of Warsaw via the Poland AOD server. Data access was granted for research purposes. The data is not publicly available and is not included in this repository.

---

## References

- Long et al. (2006) — R/B ratio threshold method
- Ye et al. (2022) — ACS_WSI dataset, DOI 10.1029/2022EA002220
- Ravi et al. (2024) — SAM 2, Meta AI
- AMT (2018) — Cloud fraction from all-sky cameras (70° zenith cutoff)
