# Literature Review — Sky Camera Cloud Fraction Estimation

Compiled 2026-05-13. Covers ground-based all-sky camera cloud detection/segmentation
papers with focus on: deep learning methods, mask creation, external validation,
and relevance to the Warsaw sky camera pipeline.

---

## Summary comparison table

Citations retrieved from Crossref/ADS as of 2026-05-13. Papers published in 2025 have naturally low citation counts.

| Paper | Year | Journal | Citations | Method | n images | External validation | Key metric |
|-------|------|---------|-----------|--------|----------|--------------------|-|
| SegCloud (Xie et al.) | 2020 | AMT | **67** (Crossref), ~45 (ADS) | CNN VGG-16 U-Net | 400 | Human observer, 1 month | r=0.84 |
| Fabel et al. | 2022 | AMT | **41** (Crossref) | U-Net ResNet34 self-supervised | 770 labelled | None | IoU=80.5% |
| Kim et al. | 2023 | AMT | ~10 (views only on AMT) | 7-layer CNN, 11-class | 10,349 | Synoptic + satellite + ceilometer | r=0.95 |
| Luo et al. | 2024 | AMT | **7** (Crossref) | YOLOv8 + k-means | 4,000 | None | F1>95% |
| Hernández-López et al. | 2024 | Sensors | — | EfficientNetV2 | 4,500 | None | acc=98% |
| Sarangi et al. | 2025 | AMT | 0–1 (new) | Random Forest pixel | ~2,000 | TSI outputs only | IoU>0.75 |
| Rivonirina et al. | 2025 | ANGEO | — | RF + threshold | "thousands of px" | MSG satellite | r=0.82 |
| Buntin et al. | 2025 | RASTI | — | Difference imaging | 619,421 | Skycam photometry | 65% forecast |
| UCloudNet | 2025 | arXiv | — | Residual U-Net | 6,768 | None | F=0.93 |
| **This work** | **2026** | **—** | — | **ResNet-34+MobileNetV2+R/B+RF** | **627** | **IMGW synoptic + ERA5 (full year)** | **MAE=0.060, r=0.952** |

---

## Detailed paper summaries

---

### 1. SegCloud (Xie et al., 2020)
**Full title:** SegCloud: a novel cloud image segmentation model using a deep convolutional neural network for ground-based all-sky-view camera observation
**Journal:** Atmospheric Measurement Techniques | **DOI:** 10.5194/amt-13-1953-2020
**Location:** Hefei, China | **Camera:** ASC100 fisheye, 2000×1944 px

**GT masks:** Manually annotated using unspecified photo editing software. 3 classes: cloud, sky, background (sun + structures). 400 images total (340 train / 60 test). Test set deliberately balanced: 10 clear, 10 overcast, 40 partial.

**Method:** Symmetric encoder-decoder CNN based on VGG-16. Three-class Softmax output.

**Validation:** Correlation with human cloud cover observations from July 2018 (279 paired observations, 1 month only). No reanalysis comparison.

**Key results:** r=0.84 vs human observer. Error within ±1 okta: 75.3%. Error within ±2 oktas: 90.9%. Outperforms R/B threshold and Otsu, especially near circumsolar region.

**Relevance to this work:** Direct predecessor — same task (CNN segmentation, fisheye, CF extraction). Our dataset is larger (627 vs 400), our validation period is an entire year vs 1 month, and we add ERA5 comparison. Their correlation r=0.84 vs observer is lower than our r=0.95 on GT test.

**Their limitations:** Only 400 images. 1-month validation. No ERA5 or reanalysis. Thin cloud remains challenging.

---

### 2. Self-supervised cloud segmentation (Fabel et al., 2022)
**Full title:** Applying self-supervised learning for semantic cloud segmentation of all-sky images
**Journal:** Atmospheric Measurement Techniques | **DOI:** 10.5194/amt-15-797-2022
**Location:** Plataforma Solar de Almeria, Spain | **Camera:** Mobotix Q25 fisheye surveillance camera

**GT masks:** 770 manually labelled images from a dataset of 286,500. 4 classes: clear sky, low-layer, mid-layer, high-layer cloud. Self-supervised pretraining (inpainting + super-resolution) reduces need for large labelled set.

**Method:** U-Net with ResNet34 encoder (same as our architecture). Self-supervised pretraining on 286,500 unlabelled images → fine-tuned on 770 labelled.

**Validation:** Internal only — no synoptic station, no reanalysis. Pixel accuracy and IoU on held-out set.

**Key results:** Mean IoU=80.5%, pixel accuracy=85.8% (binary: 95.2%). ResNet34 U-Net directly comparable to our architecture.

**Relevance:** Uses identical architecture (ResNet34 U-Net). Key difference: they use self-supervised pretraining with 286,500 images; we use ImageNet pretraining + 627 labelled. They separate cloud layers (4 classes); we don't. Their validation is internal only — no external reference. Our paper adds the external synoptic + ERA5 validation angle they lack entirely.

**Their limitations:** No external validation. Single location. Adjacent cloud layer confusion.

---

### 3. 24h continuous cloud cover CNN (Kim et al., 2023) ⭐ Most similar to ours on validation
**Full title:** Estimation of 24 h continuous cloud cover using a ground-based imager with a convolutional neural network
**Journal:** Atmospheric Measurement Techniques | **DOI:** 10.5194/amt-16-5403-2023
**Location:** Daejeon, South Korea | **Camera:** ACOS fisheye (160° effective FOV, 80° zenith cutoff)

**GT masks:** Human synoptic observer labels (0–10 tenths) used directly as training targets — no pixel-level segmentation masks. Classification approach (11 classes), not segmentation.

**Method:** Custom 7-layer CNN (not U-Net). 128×128 input. Day+night operation.

**Validation:** Full-year test set (2020, 4,742 images). Multi-source: synoptic station observers + GeoKOMPSAT-2A satellite + Vaisala CL31 ceilometer. CNN outperforms satellite and ceilometer.

**Key results:** r=0.95, RMSE=1.40 tenths (~0.156 CF), bias=−0.13 tenths, acc=0.92. Best during daytime; degraded at sunrise/sunset.

**Relevance:** **Closest paper to ours on validation strategy.** Both use synoptic station as primary reference, both cover a full year, both include day+night. Key differences: (a) they use classification (11 classes), we use pixel segmentation → our approach provides spatial mask; (b) they validate against ceilometer and satellite too; (c) we compare CNN vs R/B vs MobileNetV2 head-to-head; (d) we add ERA5 layer correlation analysis; (e) their CNN is custom 7-layer, ours is ResNet34 U-Net with ImageNet pretraining.

**Their limitations:** No pixel-level segmentation. No ERA5/reanalysis comparison. No method comparison (R/B, RF). Single site. Sunrise/sunset degradation.

---

### 4. YOLOv8 cloud classification (Luo et al., 2024)
**Full title:** Innovative cloud quantification: deep learning classification and finite-sector clustering for ground-based all-sky imaging
**Journal:** Atmospheric Measurement Techniques | **DOI:** 10.5194/amt-17-3765-2024
**Location:** Yangbajing Observatory, Tibet, 4300m | **Camera:** CMOS fisheye, 4288×2848 px

**GT masks:** Expert classification into 4 cloud types (cirrus, clear sky, cumulus, stratus). 4,000 images (1,000 per type). No pixel-level segmentation — image-level labels only.

**Method:** YOLOv8 for cloud type classification + k-means clustering (k=5) for finite-sector CF computation. Adaptive image enhancement via dark-channel prior.

**Validation:** Internal cross-validation + public TCI dataset. No synoptic station. No reanalysis.

**Key results:** Precision/recall/F1 >95%. Cross-validation on TCI: 98.31% accuracy.

**Relevance:** Different task (cloud type classification, not CF segmentation). High-altitude single site. No external validation. Less comparable to our work.

---

### 5. Transfer learning classification (Hernández-López et al., 2024)
**Full title:** Sky Image Classification Based on Transfer Learning Approaches
**Journal:** Sensors (MDPI) | **DOI:** 10.3390/s24123726
**Location:** Gran Canaria, Spain | **Camera:** Vivotek FE8391-EHV fisheye IP camera

**GT masks:** 3-class image-level labels (clear / partly cloudy / cloudy). 4,500 images. No pixel segmentation.

**Method:** EfficientNetV2 and ResNet transfer learning from ImageNet. Image classification only.

**Validation:** 5-fold cross-validation only. No synoptic station or reanalysis.

**Key results:** Best accuracy 98.09% (EfficientNetV2-B1/B2). Image-level classification, not pixel CF.

**Relevance:** Image-level classification (not CF extraction or pixel segmentation). No external validation. Limited direct overlap.

---

### 6. Random Forest pixel classifier (Sarangi et al., 2025) — AMT
**Full title:** Cloud fraction estimation using random forest classifier on sky images
**Journal:** Atmospheric Measurement Techniques | **DOI:** 10.5194/amt-18-5637-2025
**Location:** 5 sites: Germany, USA, Australia, India (×2) | **Camera:** TSI (CCD dome mirror) + Prede CMOS (Japan)

**GT masks:** MATLAB Image Labeller app. 3 domain experts annotate; overlap taken as ground truth. 4 classes: sky (0), sun (1), clouds (2), occlusions (3). ~300 images per site, ~2,000 total.

**Method:** Random Forest pixel classifier. Features: RGB, HSV, R/B ratio, log variants, RAS composite parameter.

**Validation:** Against TSI algorithm outputs. **No independent synoptic station validation.** Multi-site generalization tested.

**Key results:** Accuracy >85% all sites. IoU >0.75–0.79. RMSE=0.05, R²=0.98 at Merak. Outperforms TSI in high-pollution environments.

**Relevance:** RF is our secondary method in notebook 03b (MAE=0.100 on GT test — worse than CNN). Key difference: they use RF as primary method and show it outperforms TSI; we show RF is the weakest of four methods. No synoptic or ERA5 validation — our biggest differentiation. Their multi-site approach (5 locations) vs our single-site full-year depth.

**Their limitations:** No synoptic station validation. No CNN comparison. Cirrus and sun-glare remain challenging (mean error 0.12–0.14 for those cases).

---

### 7. Sky camera vs MSG satellite (Rivonirina et al., 2025)
**Full title:** Cloudiness retrieved from All-Sky camera and MSG satellite over Reunion Island and Antananarivo Madagascar
**Journal:** Annales Geophysicae | **DOI:** 10.5194/angeo-43-651-2025
**Location:** Reunion Island + Antananarivo, Madagascar | **Camera:** Sky Cam Vision (Reuniwatt), 2048×2048 px

**GT masks:** Thousands of pixels manually labelled for RF training. 4 classes: clear sky, thin cloud, thick cloud, sun. No count given.

**Method:** Two algorithms: Reuniwatt (RF-based) and Elifan (threshold-based RBR). MSG/SEVIRI satellite cross-validation.

**Validation:** Camera vs MSG satellite (no synoptic station). r=0.82 (Saint-Denis), 0.78 (Antananarivo) vs satellite. 3-year dataset.

**Key results:** Reuniwatt vs Elifan: r=0.99, RMSE=6.48%. Camera vs satellite RMSE=25–28%.

**Relevance:** Uses satellite (not synoptic station) as reference — our work adds the IMGW point-observation validation, which is more directly comparable to camera CF. Their r=0.82 vs satellite aligns with our ERA5 r≈0.70 (ERA5 is reanalysis, satellite is more precise → their numbers are higher).

---

### 8. Nighttime cloud detection (Buntin et al., 2025)
**Full title:** Nighttime cloud detection, tracking and prediction with All-Sky cameras
**Journal:** RAS Techniques and Instruments | **DOI:** 10.1093/rasti/rzaf034
**Location:** La Palma + Emleben, Germany | **Camera:** Starlight Xpress Oculus + Trius (astronomical)

**GT masks:** None — unsupervised difference imaging approach.

**Method:** Temporal incoherence + Otsu thresholding + Kalman filter for tracking/prediction.

**Validation:** Comparison with existing algorithms; photometric validation. 619,421 images.

**Relevance:** Nighttime-focused (astronomical use case). Very different domain — no daytime, no CF extraction, no synoptic validation. Our work handles nighttime as part of a daytime+night pipeline.

---

### 9. UCloudNet (2025, arXiv:2501.06440)
**Full title:** UCloudNet: A Residual U-Net with Deep Supervision for Cloud Image Segmentation
**Year:** 2025 | **Venue:** arXiv preprint
**Location:** Singapore (SWINySEG dataset) | **Camera:** Ground-based

**GT masks:** SWINySEG public dataset — binary (cloud/sky). 6,768 images (6,078 day + 690 night).

**Method:** Residual U-Net with deep supervision at 1/2 and 1/4 scales. Auxiliary loss branches.

**Validation:** Internal only. F-measure, error-rate. No external reference.

**Key results:** F-measure=0.93, error-rate=0.06. Daytime+nighttime combined.

**Relevance:** Binary segmentation (no CF extraction). Public dataset (not custom). No external validation.

---

## What is NOT in the literature (your differentiators)

Going through all 9 papers above:

| Feature | Found in literature? | Notes |
|---------|---------------------|-------|
| ResNet-34 U-Net for fisheye CF segmentation | Yes (Fabel 2022 uses same arch) | Not novel alone |
| R/B threshold as baseline | Yes (SegCloud, Rivonirina) | Standard baseline |
| RF pixel classifier | Yes (Sarangi 2025, Rivonirina 2025) | Not novel |
| Full-year continuous time series | Only Kim 2023 | They use classification, not segmentation |
| Validation vs synoptic station (oktas) | Only SegCloud (1 month) and Kim 2023 | Kim has no pixel segmentation or ERA5 |
| Validation vs ERA5 reanalysis | **Nobody** | **Unique to this work** |
| ERA5 tcc/lcc/mcc/hcc layer correlation | **Nobody** | **Unique to this work** |
| CNN vs R/B vs MobileNetV2 vs RF head-to-head | **Nobody** | **Unique to this work** |
| Okta-conditional CNN vs R/B breakdown | **Nobody** | **Unique to this work** |
| SAM2-assisted labelling pipeline | **Nobody** | **Novel methodology** |
| Day + night unified pipeline with CF | Kim 2023 partially | They don't do pixel segmentation |
| Central Europe / Poland location | **Nobody** | Geographic gap |

---

## Publication recommendation

**Target journal: Atmospheric Measurement Techniques (AMT)**
Open access, EGU, impact factor ~4.0. All the closest competitors published here.
Article type: Research article (~8,000–12,000 words).

**Narrative structure:**
1. Motivation: existing papers either do segmentation without external validation (Fabel, UCloudNet) or external validation without segmentation (Kim), or RF only (Sarangi). No paper does all three: pixel segmentation + multi-method comparison + synoptic+ERA5 dual validation over a full year.
2. Methodology: ResNet-34 U-Net, MobileNetV2, R/B, RF; SAM2-assisted labelling; 627 Warsaw GT masks; area-weighted CF.
3. Results: CNN 2× better on GT test; okta-conditional analysis (CNN wins at okta 0, tied/loses oktas 1–8); ERA5 layer correlation; full-year CF time series.
4. Discussion: framing negative bias (zenith cutoff), CLCM positive bias (layer integration), ERA5 ceiling (r=0.751), partial-cloud gap as future work.

**Alternative journals:**
- Remote Sensing (MDPI) — faster, broader scope, open access
- Journal of Atmospheric and Oceanic Technology (JAOT/AMS) — good for instrument/method papers

**Estimated time to submission:** 1–2 months if writing starts now.
Retrain with weighted sampling first (1 overnight) — strengthens the partial-cloud section.

---

## Key papers to cite (BibTeX-ready DOIs)

```
SegCloud:         10.5194/amt-13-1953-2020
Fabel 2022:       10.5194/amt-15-797-2022
Kim 2023:         10.5194/amt-16-5403-2023
Luo 2024:         10.5194/amt-17-3765-2024
Sarangi 2025:     10.5194/amt-18-5637-2025
Rivonirina 2025:  10.5194/angeo-43-651-2025
Buntin 2025:      10.1093/rasti/rzaf034
Long 2006:        (R/B ratio method — find DOI)
Ye 2022:          10.1029/2022EA002220  (ACS_WSI dataset)
ERA5:             10.24381/cds.adc8027c (Copernicus CDS)
```
