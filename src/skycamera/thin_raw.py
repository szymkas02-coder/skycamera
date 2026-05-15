"""thin_raw.py — Thin data/raw/ to one image per 30-minute slot.

Keeps the image whose timestamp is closest to each :00 or :30 mark within
every day folder.  Images that have a corresponding GT mask in
data/masks_manual/ are ALWAYS kept, regardless of timing.

Usage (dry-run — prints what would be deleted, touches nothing):
    python -m skycamera.thin_raw

Usage (actually delete):
    python -m skycamera.thin_raw --delete

Options:
    --interval  Minutes between kept slots (default 30, use 60 for hourly-only)
    --raw-dir   Override raw image directory
    --masks-dir Override masks directory
    --delete    Perform deletion (default is dry-run)

& C:/Users/szymo/anaconda3/envs/geo/python.exe -m skycamera.thin_raw
# check output, then:
& C:/Users/szymo/anaconda3/envs/geo/python.exe -m skycamera.thin_raw --delete

"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

FILENAME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})\.jpg", re.IGNORECASE)


def parse_ts(path: Path) -> datetime | None:
    m = FILENAME_RE.search(path.name)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s)


def slot_targets(date: datetime.date, interval_min: int) -> list[datetime]:
    """Return all slot datetimes for a given date at `interval_min` spacing."""
    slots = []
    t = datetime(date.year, date.month, date.day, 0, 0, 0)
    end = t + timedelta(days=1)
    while t < end:
        slots.append(t)
        t += timedelta(minutes=interval_min)
    return slots


def thin_directory(
    raw_dir: Path,
    masks_dir: Path,
    interval_min: int,
    delete: bool,
) -> tuple[int, int]:
    """
    Returns (n_kept, n_deleted).
    """
    # Build set of stems that have GT masks — these are ALWAYS kept
    protected_stems: set[str] = {
        p.stem.replace("_GT", "") for p in masks_dir.glob("*_GT.png")
    }
    print(f"Protected stems (have GT mask): {len(protected_stems)}")

    # Collect all day folders
    day_dirs = sorted(p for p in raw_dir.rglob("*") if p.is_dir())
    # Also include raw_dir itself in case images are directly there
    day_dirs = [raw_dir] + day_dirs

    total_kept = 0
    total_deleted = 0
    total_protected_saved = 0

    for day_dir in day_dirs:
        images = sorted(day_dir.glob("*.jpg"))
        if not images:
            continue

        # Parse timestamps
        ts_map: dict[Path, datetime] = {}
        for img in images:
            ts = parse_ts(img)
            if ts is not None:
                ts_map[img] = ts

        if not ts_map:
            continue

        # Group images by calendar date (a day folder may theoretically span midnight)
        by_date: dict[datetime.date, list[Path]] = defaultdict(list)
        for img, ts in ts_map.items():
            by_date[ts.date()].append(img)

        for date, imgs in by_date.items():
            slots = slot_targets(date, interval_min)

            # For each slot pick the closest image (within half the interval)
            max_gap = timedelta(minutes=interval_min / 2)
            keep: set[Path] = set()

            for slot in slots:
                candidates = [
                    (abs((ts_map[img] - slot).total_seconds()), img)
                    for img in imgs
                    if img in ts_map
                ]
                if not candidates:
                    continue
                best_gap_sec, best_img = min(candidates)
                if best_gap_sec <= max_gap.total_seconds():
                    keep.add(best_img)

            # Always keep protected images (have GT mask)
            for img in imgs:
                if img.stem in protected_stems:
                    keep.add(img)
                    if img not in keep:  # was not already a slot winner
                        total_protected_saved += 1

            to_delete = [img for img in imgs if img not in keep]

            total_kept += len(keep)
            total_deleted += len(to_delete)

            if to_delete:
                if delete:
                    for img in to_delete:
                        img.unlink()
                else:
                    # dry-run: just show a sample
                    sample = to_delete[:3]
                    rest = len(to_delete) - len(sample)
                    names = ", ".join(p.name for p in sample)
                    suffix = f"  … +{rest} more" if rest > 0 else ""
                    print(f"  [{date}] would delete {len(to_delete):3d}:  {names}{suffix}")

    return total_kept, total_deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=30,
                        help="Minutes between kept slots (default 30)")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Raw image directory (default: config.RAW_DIR)")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="GT masks directory (default: config.MASKS_MANUAL_DIR)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files (default is dry-run)")
    args = parser.parse_args()

    # Resolve paths — try config first, fall back to relative paths
    if args.raw_dir:
        raw_dir = args.raw_dir
    else:
        try:
            from skycamera.config import FULL_RAW_DIR
            raw_dir = FULL_RAW_DIR
        except ImportError:
            raw_dir = Path("data/full_raw")

    if args.masks_dir:
        masks_dir = args.masks_dir
    else:
        try:
            from skycamera.config import MASKS_MANUAL_DIR
            masks_dir = MASKS_MANUAL_DIR
        except ImportError:
            masks_dir = Path("data/masks_manual")

    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"{'='*60}")
    print(f"thin_raw.py  [{mode}]")
    print(f"  raw dir   : {raw_dir}")
    print(f"  masks dir : {masks_dir}")
    print(f"  interval  : {args.interval} min  (slots at :{args.interval} spacing)")
    print(f"{'='*60}\n")

    if not args.delete:
        print("Dry-run — showing sample of files that WOULD be deleted.\n"
              "Re-run with --delete to actually remove them.\n")

    kept, deleted = thin_directory(raw_dir, masks_dir, args.interval, args.delete)

    print(f"\n{'='*60}")
    print(f"  Images kept   : {kept:,}")
    print(f"  Images {'deleted' if args.delete else 'to delete'} : {deleted:,}")
    print(f"  Total found   : {kept + deleted:,}")
    if not args.delete:
        print(f"\nRe-run with --delete to remove the {deleted:,} files listed above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
