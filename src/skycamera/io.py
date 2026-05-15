"""Image loading, timestamp parsing, daytime filtering, and index building."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    RAW_DIR, FILENAME_PATTERN, BRIGHTNESS_THRESHOLD, CX, CY, R,
    CAMERA_LAT, CAMERA_LON, CAMERA_ALT, DAYTIME_MIN_ELEVATION_DEG,
)


# ── Timestamp parsing ─────────────────────────────────────────────────

def parse_timestamp(filepath: Path) -> Optional[datetime]:
    """Parse a UTC datetime from a sky camera filename.

    Expected format: ``YYYY_MM_DD__HH_MM_SS.jpg``

    Args:
        filepath: Path to an image file.

    Returns:
        ``datetime`` (UTC, timezone-naive) or ``None`` if the filename
        does not match the expected pattern.

    Example:
        >>> parse_timestamp(Path("2024_01_15__12_00_32.jpg"))
        datetime.datetime(2024, 1, 15, 12, 0, 32)
    """
    m = re.search(FILENAME_PATTERN, filepath.name)
    if m is None:
        return None
    year, month, day, hour, minute, second = (int(x) for x in m.groups())
    return datetime(year, month, day, hour, minute, second)


# ── Image loading ─────────────────────────────────────────────────────

def load_image(filepath: Path) -> np.ndarray:
    """Load an image from disk and return it as an RGB NumPy array.

    Args:
        filepath: Path to a JPEG or PNG image file.

    Returns:
        ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, in RGB order.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot decode the image.

    Example:
        >>> img = load_image(Path("2024_01_15__12_00_32.jpg"))
        >>> img.shape
        (2080, 3096, 3)
    """
    import cv2

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")
    bgr = cv2.imread(str(filepath))
    if bgr is None:
        raise RuntimeError(f"OpenCV could not decode: {filepath}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Daytime check ─────────────────────────────────────────────────────

def sun_elevation(dt: "datetime") -> float:
    """Return solar elevation angle (degrees) for Warsaw at the given UTC datetime.

    Uses pysolar. Returns -90.0 if pysolar is not installed.

    Args:
        dt: UTC datetime (timezone-naive or UTC-aware).

    Returns:
        Solar elevation in degrees; positive = above horizon.
    """
    from datetime import timezone
    try:
        from pysolar.solar import get_altitude
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(get_altitude(CAMERA_LAT, CAMERA_LON, dt,
                                  elevation=CAMERA_ALT))
    except ImportError:
        return -90.0
    except Exception:
        return -90.0


def is_daytime(
    dt: "datetime",
    min_elevation_deg: float = -6.0,
) -> bool:
    """Return True if the sun is above the civil-twilight threshold at Warsaw.

    Uses the image timestamp and Warsaw coordinates to compute solar elevation
    via pysolar — no image loading required. This is more reliable than
    brightness thresholding, which misclassifies dark overcast days as night
    and bright full-moon nights as day.

    Civil twilight (default -6°) includes dawn/dusk where the sky is still
    usable for cloud detection. Use 0° for strict astronomical daytime only.

    Args:
        dt:                UTC datetime parsed from the image filename.
        min_elevation_deg: Minimum solar elevation to be considered daytime.
                           Default -6° (civil twilight). Use 0° for sun-above-horizon only.

    Returns:
        ``True`` if solar elevation ≥ min_elevation_deg.

    Example:
        >>> from datetime import datetime
        >>> is_daytime(datetime(2024, 6, 15, 12, 0, 0))
        True
        >>> is_daytime(datetime(2024, 6, 15, 0, 0, 0))
        False
    """
    return sun_elevation(dt) >= min_elevation_deg


def is_daytime_brightness(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = BRIGHTNESS_THRESHOLD,
) -> bool:
    """Brightness-based daytime check (legacy — prefer is_daytime()).

    Returns True if mean pixel brightness inside the dome mask is above threshold.
    Less reliable than the astronomical check — kept for reference and comparison.
    """
    gray = img.mean(axis=2)
    pixels = gray[mask] if mask is not None else gray.ravel()
    return float(pixels.mean()) >= threshold


# ── Image index ───────────────────────────────────────────────────────

def build_image_index(
    root_dir: Path = RAW_DIR,
    apply_daytime_filter: bool = True,
    min_elevation_deg: float = DAYTIME_MIN_ELEVATION_DEG,
) -> pd.DataFrame:
    """Scan a directory tree of sky camera images and build a metadata index.

    Daytime classification uses the solar elevation angle computed from each
    image's UTC timestamp and Warsaw coordinates (via pysolar) — no image
    loading required for the daytime flag. This is ~10× faster than the
    old brightness-based approach and more accurate.

    Args:
        root_dir: Root directory containing date subdirectories
            (e.g. ``2024-01-15/``).
        apply_daytime_filter: If True, compute ``is_daytime`` from solar
            elevation. If False, all images are flagged ``is_daytime=True``.
        min_elevation_deg: Solar elevation threshold for daytime.
            Default -6° (civil twilight — includes usable dawn/dusk).
            Use 0° for strict sun-above-horizon only.

    Returns:
        ``pd.DataFrame`` with columns:
            - ``path``:              absolute ``Path`` to the image
            - ``timestamp``:         ``datetime`` (UTC)
            - ``month``:             int 1–12
            - ``hour``:              int 0–23
            - ``sun_elevation_deg``: float solar elevation at image time
            - ``is_daytime``:        bool — True if sun_elevation_deg ≥ min_elevation_deg

    Example:
        >>> df = build_image_index(Path("data/raw"))
        >>> df[df['is_daytime']].shape
        (3228, 6)
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {root_dir}")

    jpg_files = sorted(root_dir.rglob("*.jpg"))
    rows = []

    for fpath in jpg_files:
        ts = parse_timestamp(fpath)
        if ts is None:
            continue

        if apply_daytime_filter:
            elev = sun_elevation(ts)
            daytime = elev >= min_elevation_deg
        else:
            elev    = float("nan")
            daytime = True

        rows.append({
            "path":              fpath,
            "timestamp":         ts,
            "month":             ts.month,
            "hour":              ts.hour,
            "sun_elevation_deg": round(elev, 2),
            "is_daytime":        daytime,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── ACS_WSI dataset ───────────────────────────────────────────────────

# GT mask colour encoding (verified by inspection, 2024-Q4):
#   Black  [R=0,   G=0,   B=0  ] — outside dome / no-data region
#   Blue   [R=0,   G=0,   B≈87 ] — sky (clear)
#   Violet [R≈30,  G≈30,  B≈190] — cloud
#
# JPEG compression creates blurred transitions between the three pure colours,
# so binarisation is done by thresholding on the red channel:
#   red < 20  AND  blue > 50  -> sky   (label 0)
#   red > 20  OR   blue < 50  -> cloud (label 1)  [if not outside dome]
#   black [0,0,0]             -> outside dome, excluded from CF calculation
#
# Cloud fraction = cloud_pixels / (cloud_pixels + sky_pixels)

# CF level midpoint lookup (level 0..10 → approximate cloud fraction)
_CF_MIDPOINTS = {
    0: 0.00,   # < 5 %
    1: 0.10,   # 5–15 %
    2: 0.20,   # 15–25 %
    3: 0.30,   # 25–35 %
    4: 0.40,   # 35–45 %
    5: 0.50,   # 45–55 %
    6: 0.60,   # 55–65 %
    7: 0.70,   # 65–75 %
    8: 0.80,   # 75–85 %
    9: 0.90,   # 85–95 %
    10: 1.00,  # > 95 %
}


def load_acs_wsi_dataset(acs_root: Path) -> pd.DataFrame:
    """Scan the ACS_WSI directory tree and return a metadata DataFrame.

    Each row is one image/mask pair with the GT cloud fraction label.

    Args:
        acs_root: Root of the ACS_WSI dataset (contains subdirs ``0``–``10``).

    Returns:
        ``pd.DataFrame`` with columns:
            - ``image_path``:  Path to the raw sky image
            - ``mask_path``:   Path to the GT mask JPEG
            - ``cf_level``:    Integer CF level (0–10)
            - ``cf_approx``:   Float approximate cloud fraction (midpoint of level range)

    Example:
        >>> df = load_acs_wsi_dataset(Path('D:/MOJE/DATA_SCIENCE/SKYCAMERA/ACS_WSI-v1.0.0'))
        >>> len(df)
        77
    """
    acs_root = Path(acs_root)
    rows = []
    for folder in sorted(acs_root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        level = int(folder.name)
        images = sorted(f for f in folder.glob("*.jpg") if "_GT" not in f.name)
        for img_path in images:
            mask_path = img_path.with_name(img_path.stem + "_GT.jpg")
            if not mask_path.exists():
                continue
            rows.append({
                "image_path": img_path,
                "mask_path":  mask_path,
                "cf_level":   level,
                "cf_approx":  _CF_MIDPOINTS[level],
            })

    df = pd.DataFrame(rows).sort_values(["cf_level", "image_path"]).reset_index(drop=True)
    return df


def load_acs_wsi_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ACS_WSI image/mask pair and return normalised arrays.

    GT mask encoding (verified by pixel inspection):
        - Outside dome: pure black [0, 0, 0]
        - Sky (clear) : blue-ish   [R≈0,  G≈0,  B≈87 ]
        - Cloud       : violet     [R≈30, G≈30, B≈190 ]

    Binarisation uses the red channel as discriminator — clouds have
    R > 20 while the sky region has R ≈ 0. Outside-dome pixels (all
    channels = 0) are excluded from both classes.

    Args:
        image_path: Path to the raw sky image JPEG.
        mask_path:  Path to the GT mask JPEG.

    Returns:
        Tuple of:
            img:  ``np.ndarray`` shape ``(H, W, 3)`` uint8 RGB — raw sky image.
            mask: ``np.ndarray`` shape ``(H, W)``    uint8 — binary mask:
                    1 = cloud, 0 = sky, 255 = outside dome (ignore).

    Example:
        >>> img, mask = load_acs_wsi_pair(img_path, mask_path)
        >>> (mask == 1).sum() / (mask != 255).sum()   # cloud fraction
        0.47
    """
    import cv2

    img = load_image(image_path)   # RGB uint8

    bgr_gt = cv2.imread(str(mask_path))
    if bgr_gt is None:
        raise RuntimeError(f"Cannot read GT mask: {mask_path}")
    rgb_gt = cv2.cvtColor(bgr_gt, cv2.COLOR_BGR2RGB)

    r = rgb_gt[:, :, 0].astype(np.int16)
    g = rgb_gt[:, :, 1].astype(np.int16)
    b = rgb_gt[:, :, 2].astype(np.int16)

    # Outside dome: all channels near zero
    outside = (r < 10) & (g < 10) & (b < 10)

    # Cloud: high red channel relative to green (violet hue)
    # Sky:   low red, moderate-high blue (pure blue hue)
    # Threshold chosen empirically from cluster analysis (r≈0 sky, r≈30 cloud)
    cloud = (~outside) & (r > 15)
    sky   = (~outside) & (r <= 15)

    binary = np.full(rgb_gt.shape[:2], 255, dtype=np.uint8)  # default = outside
    binary[sky]   = 0
    binary[cloud] = 1

    return img, binary


# ── Combined dataset ──────────────────────────────────────────────────

def build_combined_dataset(
    acs_root: Path,
    manual_masks_dir: Path,
    combined_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge ACS_WSI labelled pairs with manually annotated pilot images.

    Sources:
        - ACS_WSI: pairs discovered by :func:`load_acs_wsi_dataset`.
        - Manual:  every ``*_GT.png`` found in *manual_masks_dir* whose
          corresponding raw image can be located under *manual_masks_dir*'s
          parent tree or the same directory.

    The returned DataFrame has a unified schema so both sources can be fed
    directly into the CNN training pipeline.

    Args:
        acs_root: Root of the ACS_WSI dataset (contains subdirs ``0``–``10``).
        manual_masks_dir: Directory containing manually labelled masks
            (``*_GT.png``), produced by :class:`~skycamera.labelling.LabellingTool`.
        combined_dir: If provided, the combined index CSV is saved here as
            ``dataset_index.csv``.

    Returns:
        ``pd.DataFrame`` with columns:

            - ``image_path``:  Path to raw sky image
            - ``mask_path``:   Path to GT mask (JPEG for ACS_WSI, PNG for manual)
            - ``source``:      ``"acs_wsi"`` or ``"manual"``
            - ``cf_level``:    int level (0–10 for ACS_WSI, -1 for manual)
            - ``cf_approx``:   float label midpoint (NaN for manual — use GT mask)
            - ``cf_measured``: float cloud fraction computed from GT mask pixels

    Example:
        >>> df = build_combined_dataset(ACS_ROOT, MASKS_MANUAL_DIR)
        >>> df.groupby('source').size()
        source
        acs_wsi    77
        manual     12
    """
    from .labelling import (
        LABEL_CLOUD, LABEL_SKY, load_existing_mask,
    )
    from .preprocessing import (
        build_zenith_weight_map, weighted_cf, _infer_dome_params,
    )
    from .config import CF_MAX_ZENITH_DEG

    rows: list[dict] = []

    # ── ACS_WSI source ────────────────────────────────────────────────
    df_acs = load_acs_wsi_dataset(acs_root)
    for _, r in df_acs.iterrows():
        try:
            _, mask = load_acs_wsi_pair(r["image_path"], r["mask_path"])
            cx, cy, radius = _infer_dome_params(mask)
            w_map = build_zenith_weight_map(
                mask.shape[0], mask.shape[1], cx, cy, radius, CF_MAX_ZENITH_DEG
            )
            cf = weighted_cf(mask, w_map)
        except Exception:
            cf = float("nan")
        rows.append({
            "image_path":  r["image_path"],
            "mask_path":   r["mask_path"],
            "source":      "acs_wsi",
            "cf_level":    int(r["cf_level"]),
            "cf_approx":   float(r["cf_approx"]),
            "cf_measured": cf,
        })

    n_acs = len(df_acs)

    # ── Manual source ─────────────────────────────────────────────────
    manual_masks_dir = Path(manual_masks_dir)
    n_manual = 0

    if manual_masks_dir.exists():
        gt_pngs = sorted(manual_masks_dir.glob("*_GT.png"))
        for mask_path in gt_pngs:
            stem = mask_path.stem.replace("_GT", "")
            # Search for matching raw image: full_raw first (primary dataset),
            # then raw (12-day pilot fallback).
            # E.g. skycamera/data/masks_manual -> skycamera/data/full_raw
            img_candidates: list[Path] = []
            data_dir = manual_masks_dir.parent
            # Search full_raw first (primary), then raw (pilot fallback)
            for raw_dirname in ("full_raw", "raw"):
                candidate_root = data_dir / raw_dirname
                if candidate_root.exists():
                    img_candidates = list(candidate_root.rglob(stem + ".jpg"))
                    if img_candidates:
                        break
            # Fallback: same directory as mask
            if not img_candidates:
                local = manual_masks_dir / (stem + ".jpg")
                if local.exists():
                    img_candidates = [local]

            if not img_candidates:
                # Still record the mask even without a found raw image
                img_path = manual_masks_dir / (stem + ".jpg")
            else:
                img_path = img_candidates[0]

            # Load mask to compute CF — shape is read from PNG itself
            try:
                from .config import CX, CY, R
                mask = load_existing_mask(mask_path)
                # Convert labelling-tool labels to binary (1=cloud, 0=sky, 255=ignore)
                binary = np.full(mask.shape, 255, dtype=np.uint8)
                binary[mask == LABEL_SKY]   = 0
                binary[mask == LABEL_CLOUD] = 1
                w_map = build_zenith_weight_map(
                    mask.shape[0], mask.shape[1], CX, CY, R, CF_MAX_ZENITH_DEG
                )
                cf = weighted_cf(binary, w_map)
            except Exception:
                cf = float("nan")

            rows.append({
                "image_path":  img_path,
                "mask_path":   mask_path,
                "source":      "manual",
                "cf_level":    -1,
                "cf_approx":   float("nan"),
                "cf_measured": cf,
            })
            n_manual += 1

    df = pd.DataFrame(rows).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 50)
    print("Combined dataset summary")
    print("=" * 50)
    print(f"  ACS_WSI pairs  : {n_acs}")
    print(f"  Manual pairs   : {n_manual}")
    print(f"  Total          : {len(df)}")
    print()

    cf_all = df["cf_measured"].dropna()
    print(f"  CF measured — mean={cf_all.mean():.3f}  std={cf_all.std():.3f}  "
          f"min={cf_all.min():.3f}  max={cf_all.max():.3f}")
    print()

    # Class balance: fraction of images that are majority-cloud vs majority-sky
    clear_n   = (df["cf_measured"] < 0.2).sum()
    partial_n = ((df["cf_measured"] >= 0.2) & (df["cf_measured"] < 0.8)).sum()
    cloudy_n  = (df["cf_measured"] >= 0.8).sum()
    total_n   = cf_all.shape[0]
    print(f"  Class balance (by measured CF):")
    print(f"    Clear   (<0.2) : {clear_n:3d}  ({clear_n/total_n*100:.1f}%)")
    print(f"    Partial (0.2–0.8): {partial_n:3d}  ({partial_n/total_n*100:.1f}%)")
    print(f"    Cloudy  (>0.8) : {cloudy_n:3d}  ({cloudy_n/total_n*100:.1f}%)")
    print("=" * 50)

    # ── Save index ────────────────────────────────────────────────────
    if combined_dir is not None:
        combined_dir = Path(combined_dir)
        combined_dir.mkdir(parents=True, exist_ok=True)
        out_csv = combined_dir / "dataset_index.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved -> {out_csv}")

    return df
