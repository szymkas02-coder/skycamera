"""Central configuration — all constants and paths for the sky camera pipeline."""
from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # …/skycamera/

# ── Camera intrinsics ─────────────────────────────────────────────────
CX = 1438        # optical centre x (px)
CY = 928         # optical centre y (px)
R  = 938         # dome radius (px)

# ── Output geometry ───────────────────────────────────────────────────
OUT_H = 512      # equirectangular output height (px)
OUT_W = 512      # equirectangular output width  (px)

# ── Daytime filter ───────────────────────────────────────────────────
# Solar elevation threshold for daytime classification (degrees).
# -6° = civil twilight — includes usable dawn/dusk images.
#  0° = sun strictly above horizon only.
# Computed from image UTC timestamp + Warsaw coordinates via pysolar.
# The old brightness threshold (BRIGHTNESS_THRESHOLD) is kept for reference
# but is no longer used by build_image_index.
DAYTIME_MIN_ELEVATION_DEG = -6.0
BRIGHTNESS_THRESHOLD = 20.0   # legacy brightness threshold (kept for reference)

# ── Cloud fraction weighting ──────────────────────────────────────────
# Pixels beyond this zenith angle are excluded from CF computation.
# 70° is a common WMO-aligned cutoff that removes noisy horizon pixels.
CF_MAX_ZENITH_DEG = 70.0

# ── Data paths ────────────────────────────────────────────────────────
# Pilot sky camera images — monthly subdirectories YYYY-MM-DD
# data/raw/       — original 12-day pilot dataset (kept for reference)
# data/full_raw/  — full year download, thinned to 30-min intervals by thin_raw.py
RAW_DIR          = ROOT / "data" / "raw"
FULL_RAW_DIR     = ROOT / "data" / "full_raw"

# ACS_WSI labelled dataset (Ye et al. 2022) — copied into project at data/acs_wsi/
ACS_WSI_DIR      = ROOT / "data" / "acs_wsi"

# Manually labelled masks produced by the labelling tool
MASKS_MANUAL_DIR = ROOT / "data" / "masks_manual"

# Merged ACS_WSI + manual labels used for CNN training
MASKS_COMBINED_DIR = ROOT / "data" / "masks_combined"

# ── Output paths ──────────────────────────────────────────────────────
CSV_DIR        = ROOT / "outputs" / "csv"
MASKS_PRED_DIR = ROOT / "outputs" / "masks_pred"
PLOTS_DIR      = ROOT / "outputs" / "plots"
MODEL_DIR      = ROOT / "outputs" / "models"

# ── Camera location (Copernicus Science Centre, Warsaw) ───────────────
CAMERA_LAT =  52.2411   # degrees North
CAMERA_LON =  21.0327   # degrees East
CAMERA_ALT =  20.0      # metres above sea level (approx.)

# ── Sun disk ignore region ────────────────────────────────────────────
SUN_IGNORE_RADIUS_DEG = 10.0   # angular radius around sun centre to mark as IGNORE

# Rotation of the camera mount — degrees clockwise from geographic North to
# image "up". Set to 0 if the camera top points exactly North. Adjust until
# the sun disk lands on the actual sun in a test image.
CAMERA_NORTH_OFFSET_DEG = -114.0

# Fisheye projection scale — radial_px = R * zenith_deg / 90 * CAMERA_PROJECTION_SCALE
# Increase above 1.0 if sun appears too close to centre, decrease if too far out.
# Tune visually using the calibration cell in notebook 07.
CAMERA_PROJECTION_SCALE = 1.12

# ── Filename pattern ──────────────────────────────────────────────────
# Pilot images: 2024_01_15__12_00_32.jpg
FILENAME_PATTERN = r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})\.jpg"
