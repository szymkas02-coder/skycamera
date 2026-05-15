"""Convert Roboflow COCO-segmentation export to project GT masks.

For each image in the COCO JSON:
  1. Rasterise all polygon annotations (cloud / sky / ignore).
  2. Determine labelling mode:
       - "cloud mode"  — only cloud polygons drawn → fill remainder as sky
       - "sky mode"    — only sky polygons drawn   → fill remainder as cloud
       - "mixed mode"  — both present              → keep as-is (unlabelled stays unlabelled)
  3. Apply default_ignore.png (static antenna / cable mask).
  4. Apply per-image sun-disk ignore region derived from the filename timestamp.
  5. Skip and warn if ≥ IGNORE_SKIP_FRACTION of dome pixels are marked IGNORE
     after all masks are applied.
  6. Save as  data/masks_manual/{stem}_GT.png  using the project colour encoding.
  7. Append one row to labelling_log.csv.

Usage (from repo root):
    python -m skycamera.process_roboflow_labels

Or import and call process_all() from a notebook.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Support both `python script.py` and `python -m skycamera.process_roboflow_labels`
if __name__ == "__main__":
    import sys
    _src = str(Path(__file__).resolve().parents[1])
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from skycamera.config import CX, CY, R, MASKS_MANUAL_DIR, CF_MAX_ZENITH_DEG
    from skycamera.labelling import (
        LABEL_SKY, LABEL_CLOUD, LABEL_IGNORE, LABEL_UNLABELLED,
        save_mask, load_existing_mask, append_log, _compute_cf,
    )
    from skycamera.preprocessing import build_circular_mask, build_zenith_weight_map
    from skycamera.sun import sun_ignore_mask
else:
    from .config import CX, CY, R, MASKS_MANUAL_DIR, CF_MAX_ZENITH_DEG
    from .labelling import (
        LABEL_SKY, LABEL_CLOUD, LABEL_IGNORE, LABEL_UNLABELLED,
        save_mask, load_existing_mask, append_log, _compute_cf,
    )
    from .preprocessing import build_circular_mask, build_zenith_weight_map
    from .sun import sun_ignore_mask

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
COCO_DIR = Path(__file__).resolve().parents[2] / "Cloud segmentation nibqv.coco-segmentation" / "train"
COCO_JSON = COCO_DIR / "_annotations.coco.json"

DEFAULT_IGNORE_PATH = MASKS_MANUAL_DIR / "default_ignore.png"
IGNORE_SKIP_FRACTION = 0.50   # skip image if this fraction of dome is IGNORE

# COCO category_id → project label
_CATEGORY_MAP = {
    1: LABEL_CLOUD,
    2: LABEL_IGNORE,
    3: LABEL_SKY,
}


# ── Core processing ───────────────────────────────────────────────────

def _rle_to_mask(rle: dict, h: int, w: int) -> np.ndarray:
    """Decode a COCO RLE segmentation dict to a boolean mask."""
    counts = rle["counts"]
    # Compressed RLE is a byte string; uncompressed is a list of ints
    if isinstance(counts, str):
        # pycocotools-style compressed RLE — decode if available, else skip
        try:
            from pycocotools import mask as coco_mask
            return coco_mask.decode({"counts": counts.encode(), "size": [h, w]}).astype(bool)
        except ImportError:
            log.warning("pycocotools not installed — skipping compressed RLE annotation")
            return np.zeros((h, w), dtype=bool)
    # Uncompressed RLE: alternating background/foreground run lengths, column-major
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, run in enumerate(counts):
        if i % 2 == 1:   # odd index = foreground
            flat[pos:pos + run] = 1
        pos += run
    # COCO RLE is column-major (Fortran order)
    return flat.reshape(h, w, order="F").astype(bool)


def _rasterise_annotations(
    annotations: list[dict],
    h: int,
    w: int,
) -> np.ndarray:
    """Rasterise COCO polygon or RLE annotations onto a label array.

    Annotations are drawn in order; later ones overwrite earlier ones.
    Pixels not covered by any annotation remain LABEL_UNLABELLED.
    """
    mask = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)
    for ann in annotations:
        label = _CATEGORY_MAP.get(ann["category_id"])
        if label is None:
            continue
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            # RLE format
            region = _rle_to_mask(seg, h, w)
            mask[region] = int(label)
        else:
            # Polygon format: list of [x1,y1,x2,y2,...] lists
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts_int = pts.round().astype(np.int32)
                cv2.fillPoly(mask, [pts_int], int(label))
    return mask


def _fill_complement(mask: np.ndarray, dome_mask: np.ndarray) -> np.ndarray:
    """Fill unlabelled dome pixels with the complementary class.

    Rules:
      - cloud-only image  → fill unlabelled dome pixels with SKY
      - sky-only image    → fill unlabelled dome pixels with CLOUD
      - mixed / neither   → leave unlabelled pixels as UNLABELLED

    Outside the dome pixels remain LABEL_UNLABELLED (treated as outside by
    the CNN dataset loader and weighted_cf).
    """
    has_cloud = np.any(mask == LABEL_CLOUD)
    has_sky   = np.any(mask == LABEL_SKY)

    unlabelled_dome = (mask == LABEL_UNLABELLED) & dome_mask

    if has_cloud and not has_sky:
        mask[unlabelled_dome] = LABEL_SKY
    elif has_sky and not has_cloud:
        mask[unlabelled_dome] = LABEL_CLOUD
    # mixed or neither: leave as-is

    return mask


def process_image(
    image_info: dict,
    annotations: list[dict],
    dome_mask: np.ndarray,
    default_ignore_mask: Optional[np.ndarray],
    out_dir: Path,
    log_path: Path,
    skip_fraction: float = IGNORE_SKIP_FRACTION,
    overwrite: bool = False,
    zenith_weights: Optional[np.ndarray] = None,
) -> bool:
    """Process one image entry from the COCO JSON.

    Args:
        image_info:          COCO image dict (id, file_name, height, width, extra).
        annotations:         List of annotation dicts for this image.
        dome_mask:           Boolean (H, W) — True inside the fisheye dome.
        default_ignore_mask: Boolean (H, W) or None — static antenna/cable regions.
        out_dir:             Directory to write GT masks.
        log_path:            Path to labelling_log.csv.
        skip_fraction:       Skip if IGNORE fraction of dome exceeds this.
        overwrite:           If False, skip images that already have a GT mask.

    Returns:
        True if a mask was saved, False if skipped.
    """
    # Recover original stem from the Roboflow-renamed filename or extra.name
    extra = image_info.get("extra") or {}
    original_name = extra.get("name") or image_info["file_name"]
    stem = Path(original_name).stem   # e.g. 2024_07_15__12_32_56

    out_path = out_dir / f"{stem}_GT.png"
    if out_path.exists() and not overwrite:
        log.debug("Skip (exists): %s", out_path.name)
        return False

    h, w = image_info["height"], image_info["width"]

    # 1. Rasterise polygons
    mask = _rasterise_annotations(annotations, h, w)

    # 2. Fill complement (cloud-only → fill sky, sky-only → fill cloud)
    mask = _fill_complement(mask, dome_mask)

    # 3. Apply default ignore (antennas / cables)
    if default_ignore_mask is not None:
        mask[default_ignore_mask] = LABEL_IGNORE

    # 4. Apply per-image sun-disk ignore (use original filename for timestamp parsing)
    sun_mask = sun_ignore_mask(Path(original_name), (h, w))
    if sun_mask is not None:
        mask[sun_mask] = LABEL_IGNORE

    # 5. Skip if too many dome pixels are IGNORE
    dome_pixels = dome_mask.sum()
    if dome_pixels > 0:
        ignore_fraction = (
            ((mask == LABEL_IGNORE) & dome_mask).sum() / dome_pixels
        )
        if ignore_fraction >= skip_fraction:
            log.warning(
                "SKIP %s — %.0f%% of dome is IGNORE (threshold %.0f%%)",
                stem, ignore_fraction * 100, skip_fraction * 100,
            )
            return False

    # 6. Save
    save_mask(mask, out_path)
    cf = _compute_cf(mask, zenith_weights)
    append_log(log_path, stem + ".jpg", cf, notes="roboflow_coco")
    log.info("Saved %s  CF=%.3f", out_path.name, cf if not np.isnan(cf) else float("nan"))
    return True


def process_all(
    coco_json: Path = COCO_JSON,
    out_dir: Path = MASKS_MANUAL_DIR,
    skip_fraction: float = IGNORE_SKIP_FRACTION,
    overwrite: bool = False,
) -> dict:
    """Process all images in the Roboflow COCO export.

    Args:
        coco_json:     Path to _annotations.coco.json.
        out_dir:       Output directory for GT masks (default: data/masks_manual/).
        skip_fraction: Skip images where ≥ this fraction of dome is IGNORE.
        overwrite:     Overwrite existing GT masks if True.

    Returns:
        dict with keys "saved", "skipped", "failed".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "labelling_log.csv"

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Index annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Build dome mask and zenith weight map once at native resolution
    sample_img = coco["images"][0]
    h, w = sample_img["height"], sample_img["width"]
    dome_mask = build_circular_mask(h, w, CX, CY, R)
    zenith_weights = build_zenith_weight_map(h, w, CX, CY, R, CF_MAX_ZENITH_DEG)

    # Load default ignore mask
    default_ignore_mask: Optional[np.ndarray] = None
    if DEFAULT_IGNORE_PATH.exists():
        raw = load_existing_mask(DEFAULT_IGNORE_PATH)
        default_ignore_mask = (raw == LABEL_IGNORE)
        log.info("Loaded default ignore mask: %s", DEFAULT_IGNORE_PATH.name)
    else:
        log.warning("Default ignore mask not found at %s", DEFAULT_IGNORE_PATH)

    saved = skipped = failed = 0

    for img_info in coco["images"]:
        annotations = anns_by_image.get(img_info["id"], [])
        try:
            ok = process_image(
                img_info, annotations, dome_mask, default_ignore_mask,
                out_dir, log_path, skip_fraction, overwrite, zenith_weights,
            )
            if ok:
                saved += 1
            else:
                skipped += 1
        except Exception:
            log.exception("Failed to process image %s", img_info.get("file_name"))
            failed += 1

    log.info("Done — saved=%d  skipped=%d  failed=%d", saved, skipped, failed)
    print(f"Done — saved={saved}  skipped={skipped}  failed={failed}")
    return {"saved": saved, "skipped": skipped, "failed": failed}


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO labels to GT masks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks.")
    parser.add_argument(
        "--skip-fraction", type=float, default=IGNORE_SKIP_FRACTION,
        help=f"Skip images with ≥ this fraction of dome as IGNORE (default {IGNORE_SKIP_FRACTION}).",
    )
    args = parser.parse_args()
    process_all(overwrite=args.overwrite, skip_fraction=args.skip_fraction)
