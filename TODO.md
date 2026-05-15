# TODO — Sky Camera Cloud Fraction Pipeline

Last updated: 2026-05-13. All notebooks have been run on full_raw. Results are final.

---

## 1. High priority — improve partial-cloud performance

The core gap: CNN trained on 23% partial-cloud images (144/627) underperforms R/B
in the okta 3–6 regime (CF 0.2–0.8). Two options, ordered by effort:

- [ ] **Option A — Weighted resampling (1 overnight run, no new data)**
  Add `WeightedRandomSampler` to `SkyDataset` in `cnn.py`, oversample CF 0.2–0.8
  images 3×. Delete `cnn_sky.pt` and `cnn_mobilenet.pt`, retrain both notebooks 03
  and 03b. Check MAE-by-level chart in notebook 06 — partial-cloud bar should drop.
  Current partial MAE (CNN): unknown (only 26 test images). Target: beat R/B in okta 4–6.

- [ ] **Option B — Label 50–80 more partial-cloud images (1–2 days)**
  Use `sample_images.py` with partial-cloud oversampling to select candidates from
  `full_raw/`. Label in Roboflow (cloud/sky/ignore). Re-export COCO →
  `process_roboflow_labels.py --overwrite` → retrain. This also increases the test set
  from 26 to ~40 partial images, making partial-cloud metrics more reliable.
  Do Option A first to see if resampling alone is sufficient.

---

## 2. Validation — remaining checks

- [ ] **Sun mask visual check (notebook 09)**
  Run `09_sun_mask_check.ipynb` and confirm orange circle lands on actual sun disk.
  If offset: tune `CAMERA_NORTH_OFFSET_DEG` / `CAMERA_PROJECTION_SCALE` in config.py,
  then re-run `process_roboflow_labels.py --overwrite`.

- [ ] **2025-01-01 anomaly**
  In `flagged_disagreements.csv`, R/B=0.997 vs CNN=0.435 on 2025-01-01.
  Camera may have been covered or images corrupted. Inspect manually.

- [ ] **Negative bias investigation**
  All methods show Bias ≈ −0.037 to −0.077 vs IMGW NOG. Most likely cause: 70°
  zenith cutoff clips near-horizon cloud that IMGW observer includes. Test: reduce
  `CF_MAX_ZENITH_DEG` from 70° to 60°, recompute cf_cnn.csv, check if bias improves.
  (Fast — just rerun inference cell with modified config, no retraining needed.)

- [ ] **SAM2 batch mask quality check**
  Batch pseudo-labels not yet validated against manual GT. Inspect ≥5 batch masks
  before next retrain.

---

## 3. Paper / reporting framing

Key messages to communicate clearly:

- CNN is 2× better than R/B on GT test MAE (0.065 vs 0.143) — the headline result.
- CNN's IMGW advantage is modest overall (r=0.814 vs 0.731) because IMGW comparison
  is dominated by clear (okta 0, ~5%) and overcast (okta 7–8, ~47%) hours where any
  method works. In the partial-cloud regime (oktas 3–6) R/B is currently better.
- CNN's unique advantage: complete 24/7 time series (16,585 images vs R/B's 9,276).
  R/B cannot run at night. For climate/annual CF means, CNN is the only option.
- ERA5 tcc vs IMGW r=0.751 is the upper bound for camera-vs-ERA5. Camera r=0.705
  is within 6% of that ceiling — reasonable for a point vs gridded comparison.
- Low cloud bias (+0.14): camera overestimates vs IMGW CLCM because camera integrates
  all layers; CLCM records only the lowest. Structural limitation, not model error.

---

## 4. Optional improvements

- [ ] **R/B threshold seasonal tuning**
  Threshold 0.55 tuned on random mix. Test if seasonal thresholds (e.g. 0.50 winter,
  0.58 summer) reduce the ~10% disagreement months. Check notebook 02 sweep by month.

- [ ] **MobileNetV2 at 512×512**
  Current input is 256×256. MobileNetV2 MAE (0.060) already beats ResNet-34 (0.065).
  Retraining at 512×512 may improve further — likely eliminates the resolution gap.

- [ ] **Probabilistic CF**
  MC dropout (enable dropout at inference, run 20 forward passes) gives uncertainty
  estimate per image. Useful for flagging ambiguous predictions. ~1 day of work.

---

## 5. Infrastructure

- [ ] **pip-installable package**
  Add `pyproject.toml` so `pip install -e .` works — eliminates `sys.path.insert` hacks.

- [ ] **Regenerate requirements.txt**
  `conda activate geo && pip freeze > requirements.txt`

- [ ] **Operational integration**
  Link `cf_cnn.csv` to Warsaw PM2.5 forecasting pipeline (low CF + low BLH = smog signal).
