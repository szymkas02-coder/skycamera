"""sample_images.py — Build a stratified random sample from data/full_raw/.

Copies images to data/random_sample_full/, stratified so that partial-cloud
conditions (CF 0.2–0.8) are over-represented relative to clear/overcast.

CF is estimated quickly using the R/B threshold method (no model needed).
Images where R/B CF estimation fails (night, bad file) are skipped.

Existing images in data/masks_manual/ are always included in the output list
so you know which ones are already labelled (they are NOT copied — already present).

Usage:
    python -m skycamera.sample_images                  # default: 300 images
    python -m skycamera.sample_images --n 500
    python -m skycamera.sample_images --n 200 --no-rb  # skip R/B, pure random stratified by month/hour

Options:
    --n          Total images to copy (default 300)
    --no-rb      Skip R/B CF estimation; sample purely by month × hour stratum
    --raw-dir    Override source directory (default: config.FULL_RAW_DIR)
    --out-dir    Override output directory (default: data/random_sample_full)
    --seed       Random seed (default 42)

# uses R/B to estimate CF and oversample partial-cloud 3×
& C:/Users/szymo/anaconda3/envs/geo/python.exe -m skycamera.sample_images --n 300

# if R/B is too slow (full year = many images), skip it:
& C:/Users/szymo/anaconda3/envs/geo/python.exe -m skycamera.sample_images --n 300 --no-rb
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

FILENAME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})\.jpg", re.IGNORECASE)

# Partial-cloud oversampling weight relative to clear/overcast
# e.g. 3 means partial-cloud images are 3× more likely to be selected
PARTIAL_WEIGHT = 3


def parse_ts(path: Path) -> datetime | None:
    m = FILENAME_RE.search(path.name)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def is_daytime(dt: datetime, lat: float, lon: float, alt: float) -> bool:
    try:
        from pysolar.solar import get_altitude
        return get_altitude(lat, lon, dt, elevation=alt) > 0
    except Exception:
        return False


def estimate_cf_rb(img_path: Path, dome_mask, zenith_weights, threshold: float = 0.55):
    """Fast R/B CF estimate. Returns float or None on failure."""
    try:
        import cv2
        import numpy as np
        from skycamera.preprocessing import weighted_cf
        from skycamera.sun import mask_sun_pixels

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        active_mask = mask_sun_pixels(dome_mask, img_path)
        r = img[:, :, 0].astype(float)
        b = img[:, :, 2].astype(float)

        with_b = active_mask & (b > 0)
        ratio = np.where(with_b, r / np.where(b > 0, b, 1), 0)
        cloud_mask = np.where(with_b, (ratio >= threshold).astype(float), 255)
        cf = weighted_cf(cloud_mask.astype('uint8'), zenith_weights)
        return float(cf) if cf == cf else None  # nan check
    except Exception:
        return None


def cf_stratum(cf: float | None) -> str:
    if cf is None:
        return "unknown"
    if cf < 0.2:
        return "clear"
    if cf <= 0.8:
        return "partial"
    return "overcast"


def build_sample(
    raw_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    n_total: int,
    use_rb: bool,
    seed: int,
    partial_only: bool = False,
) -> None:
    random.seed(seed)

    try:
        from skycamera.config import CAMERA_LAT, CAMERA_LON, CAMERA_ALT, CX, CY, R, CF_MAX_ZENITH_DEG
        from skycamera.preprocessing import build_circular_mask, build_zenith_weight_map
        dome_mask = build_circular_mask(2080, 3096, CX, CY, R)
        zenith_weights = build_zenith_weight_map(2080, 3096, CX, CY, R, CF_MAX_ZENITH_DEG)
        lat, lon, alt = CAMERA_LAT, CAMERA_LON, CAMERA_ALT
    except ImportError:
        dome_mask = None
        zenith_weights = None
        lat, lon, alt = 52.2411, 21.0327, 20.0
        use_rb = False
        print("Warning: skycamera package not found — falling back to month×hour stratification.")

    # Already-labelled stems — report but don't copy
    labelled_stems = {p.stem.replace("_GT", "") for p in masks_dir.glob("*_GT.png")}
    print(f"Already labelled images: {len(labelled_stems)}")

    # Collect all daytime images from full_raw
    all_images = sorted(raw_dir.rglob("*.jpg"))
    print(f"Total images in {raw_dir.name}: {len(all_images):,}  — filtering daytime...")

    daytime_images = []
    for img in all_images:
        ts = parse_ts(img)
        if ts is None:
            continue
        if not is_daytime(ts, lat, lon, alt):
            continue
        daytime_images.append((img, ts))

    print(f"Daytime images: {len(daytime_images):,}")

    if not daytime_images:
        print("No daytime images found — check raw_dir path.", file=sys.stderr)
        sys.exit(1)

    # Estimate CF via R/B (or assign None for month×hour fallback)
    print(f"Estimating CF {'via R/B threshold' if use_rb else '(skipped — using month×hour strata)'}...")
    records = []
    for i, (img, ts) in enumerate(daytime_images):
        if use_rb and dome_mask is not None:
            cf = estimate_cf_rb(img, dome_mask, zenith_weights)
        else:
            cf = None
        records.append({
            "path": img,
            "ts": ts,
            "month": ts.month,
            "hour": ts.hour,
            "cf": cf,
            "stratum": cf_stratum(cf),
            "labelled": img.stem in labelled_stems,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1:,}/{len(daytime_images):,}...")

    # Print CF distribution
    from collections import Counter
    strata_counts = Counter(r["stratum"] for r in records)
    print(f"\nCF strata: {dict(strata_counts)}")

    # Weighted sampling: partial-cloud images get PARTIAL_WEIGHT times the weight
    # If partial_only=True, clear and overcast get weight 0 (excluded entirely)
    weights = []
    for r in records:
        if r["labelled"]:
            weights.append(0)  # already labelled — exclude from sampling pool
        elif r["stratum"] == "partial":
            weights.append(PARTIAL_WEIGHT)
        elif partial_only:
            weights.append(0)  # exclude clear and overcast
        else:
            weights.append(1)

    if partial_only:
        n_partial_available = sum(1 for r, w in zip(records, weights) if w > 0)
        print(f"Partial-only mode: {n_partial_available} partial-cloud images available (CF 0.2–0.8)")

    pool = [r for r, w in zip(records, weights) if w > 0]
    pool_weights = [w for r, w in zip(records, weights) if w > 0]

    if len(pool) < n_total:
        print(f"Warning: pool ({len(pool)}) smaller than requested n={n_total}. Using all.")
        n_total = len(pool)

    # Weighted random sample without replacement
    # Python's random.choices allows replacement — use manual approach
    import numpy as np
    rng = np.random.default_rng(seed)
    total_w = sum(pool_weights)
    probs = [w / total_w for w in pool_weights]
    chosen_idx = rng.choice(len(pool), size=n_total, replace=False, p=probs)
    chosen = [pool[idx] for idx in chosen_idx]

    # Copy to output dir
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    strata_copied = Counter()
    for r in chosen:
        dest = out_dir / r["path"].name
        if not dest.exists():
            shutil.copy2(r["path"], dest)
        copied += 1
        strata_copied[r["stratum"]] += 1

    print(f"\nCopied {copied} images to {out_dir}")
    print(f"  clear:    {strata_copied['clear']:3d}")
    print(f"  partial:  {strata_copied['partial']:3d}  (oversampled {PARTIAL_WEIGHT}×)")
    print(f"  overcast: {strata_copied['overcast']:3d}")
    print(f"  unknown:  {strata_copied['unknown']:3d}")
    print(f"\nAlready-labelled images (in masks_manual, not copied): {len(labelled_stems)}")
    print("Prioritise labelling the partial-cloud images first.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="Total images to sample (default 300)")
    parser.add_argument("--no-rb", action="store_true", help="Skip R/B CF estimation")
    parser.add_argument("--partial-only", action="store_true", help="Sample only partial-cloud images (CF 0.2-0.8), skip clear and overcast")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from skycamera.config import FULL_RAW_DIR, MASKS_MANUAL_DIR, ROOT
        raw_dir = args.raw_dir or FULL_RAW_DIR
        masks_dir = args.masks_dir if hasattr(args, 'masks_dir') else MASKS_MANUAL_DIR
        out_dir = args.out_dir or (ROOT / "data" / "random_sample_full")
    except ImportError:
        raw_dir = args.raw_dir or Path("data/full_raw")
        masks_dir = Path("data/masks_manual")
        out_dir = args.out_dir or Path("data/random_sample_full")

    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    build_sample(
        raw_dir=raw_dir,
        masks_dir=masks_dir,
        out_dir=out_dir,
        n_total=args.n,
        use_rb=not args.no_rb,
        seed=args.seed,
        partial_only=args.partial_only,
    )


if __name__ == "__main__":
    main()
