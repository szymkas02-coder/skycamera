"""Interactive matplotlib-based sky image labelling tool backend."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MASKS_MANUAL_DIR, CX, CY, R, CF_MAX_ZENITH_DEG

log = logging.getLogger(__name__)

# ── Label constants ───────────────────────────────────────────────────
LABEL_SKY    = 0    # painted with right-click
LABEL_CLOUD  = 1    # painted with left-click
LABEL_IGNORE = 2    # painted with middle-click (cables, glare, artifacts)
LABEL_UNLABELLED = 255  # default — not yet painted

# Colours shown while painting (RGB)
COLOUR_SKY    = np.array([30,  120, 200], dtype=np.uint8)
COLOUR_CLOUD  = np.array([230, 230, 230], dtype=np.uint8)
COLOUR_IGNORE = np.array([255, 140,   0], dtype=np.uint8)
COLOUR_UNLABELLED = np.array([0, 0, 0],   dtype=np.uint8)

LABEL_LOG_CSV = "labelling_log.csv"


# ── Utility ───────────────────────────────────────────────────────────

def _label_colour(label_val: int) -> np.ndarray:
    return {
        LABEL_SKY:    COLOUR_SKY,
        LABEL_CLOUD:  COLOUR_CLOUD,
        LABEL_IGNORE: COLOUR_IGNORE,
    }.get(label_val, COLOUR_UNLABELLED)


def _compute_cf(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Cloud fraction weighted by zenith angle map (or simple pixel ratio if no weights).

    Args:
        mask:    Label mask (uint8) using LABEL_* constants.
        weights: Float32 zenith weight map from build_zenith_weight_map. Pixels with
                 weight=0 (beyond CF_MAX_ZENITH_DEG) are excluded. If None, falls back
                 to unweighted pixel count (all dome pixels treated equally).
    """
    if weights is not None:
        cloud_w = float(weights[mask == LABEL_CLOUD].sum())
        sky_w   = float(weights[mask == LABEL_SKY].sum())
        total   = cloud_w + sky_w
        return cloud_w / total if total > 0 else float("nan")
    cloud = int((mask == LABEL_CLOUD).sum())
    sky   = int((mask == LABEL_SKY).sum())
    return float(cloud / (cloud + sky)) if (cloud + sky) > 0 else float("nan")


def _mask_to_png_array(mask: np.ndarray) -> np.ndarray:
    """Convert label mask to a saveable RGB PNG array."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val, colour in [
        (LABEL_SKY,    COLOUR_SKY),
        (LABEL_CLOUD,  COLOUR_CLOUD),
        (LABEL_IGNORE, COLOUR_IGNORE),
    ]:
        rgb[mask == val] = colour
    return rgb


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Save a label mask as a colour PNG.

    Args:
        mask: 2-D uint8 array with values LABEL_SKY/CLOUD/IGNORE/UNLABELLED.
        out_path: Destination path (will be created if parent dirs missing).
    """
    import cv2
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _mask_to_png_array(mask)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)
    log.info("Mask saved: %s", out_path)


def load_existing_mask(
    mask_path: Path,
    shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load a previously saved colour-PNG mask back to a label array.

    Shape is always read from the PNG itself. The ``shape`` parameter is
    retained for API compatibility but is ignored when the file exists — the
    PNG's own dimensions are authoritative.  If the file does not exist and
    ``shape`` is provided, an all-UNLABELLED array of that shape is returned.

    Args:
        mask_path: Path to a colour PNG saved by :func:`save_mask`.
        shape: Fallback ``(H, W)`` used only when the file does not exist.

    Returns:
        uint8 label array whose shape matches the PNG dimensions.

    Example:
        >>> mask = load_existing_mask(Path('image_GT.png'))
        >>> mask.shape
        (2080, 3096)
    """
    import cv2
    bgr = cv2.imread(str(mask_path))
    if bgr is None:
        if shape is None:
            raise FileNotFoundError(
                f"Mask file not found and no fallback shape provided: {mask_path}"
            )
        return np.full(shape, LABEL_UNLABELLED, dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Use the PNG's own shape — never trust the caller's shape parameter
    mask = np.full(rgb.shape[:2], LABEL_UNLABELLED, dtype=np.uint8)
    for val, colour in [
        (LABEL_SKY,    COLOUR_SKY),
        (LABEL_CLOUD,  COLOUR_CLOUD),
        (LABEL_IGNORE, COLOUR_IGNORE),
    ]:
        match = np.all(np.abs(rgb.astype(int) - colour.astype(int)) < 30, axis=2)
        mask[match] = val
    return mask


def append_log(
    log_path: Path,
    filename: str,
    cf_estimate: float,
    notes: str = "",
) -> None:
    """Append one row to the labelling log CSV.

    Args:
        log_path: Path to labelling_log.csv.
        filename: Name of the labelled image file.
        cf_estimate: Cloud fraction at time of saving.
        notes: Optional free-text notes.
    """
    log_path = Path(log_path)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "timestamp_labelled",
                                               "cf_estimate", "notes"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "filename":           filename,
            "timestamp_labelled": datetime.utcnow().isoformat(timespec="seconds"),
            "cf_estimate":        f"{cf_estimate:.4f}" if not np.isnan(cf_estimate) else "nan",
            "notes":              notes,
        })


# ── Main labelling tool ───────────────────────────────────────────────

class LabellingTool:
    """Interactive sky image labelling tool using matplotlib.

    Controls:
        Left-click  + drag → paint CLOUD  (light grey)
        Right-click + drag → paint SKY    (blue)
        Middle-click+ drag → paint IGNORE (orange) — cables, glare, artifacts

        s → save current mask
        n → next image
        p → previous image
        z → undo last stroke
        r → reset current mask to all UNLABELLED
        q → quit

    Args:
        image_paths: Ordered list of image paths to label.
        masks_dir: Directory where masks are saved.
        brush_radius: Painting brush radius in pixels (image-space).
        dome_mask: Optional boolean array to restrict painting inside dome.
        log_path: Path to labelling_log.csv.

    Example:
        >>> tool = LabellingTool(image_paths, masks_dir=Path('data/masks_manual'))
        >>> tool.run()
    """

    def __init__(
        self,
        image_paths: list[Path],
        masks_dir: Path = MASKS_MANUAL_DIR,
        brush_radius: int = 15,
        dome_mask: Optional[np.ndarray] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.image_paths  = [Path(p) for p in image_paths]
        self.masks_dir    = Path(masks_dir)
        self.brush_radius = brush_radius
        self.dome_mask    = dome_mask
        self.log_path     = Path(log_path) if log_path else self.masks_dir / LABEL_LOG_CSV

        self._idx       = 0           # current image index
        self._img       : Optional[np.ndarray] = None
        self._mask      : Optional[np.ndarray] = None
        self._undo_stack: list[np.ndarray]     = []
        self._painting  : Optional[int]        = None   # active label while dragging
        self._zenith_weights: Optional[np.ndarray] = None
        self._fig       = None
        self._ax_img    = None
        self._ax_info   = None
        self._im_handle = None   # AxesImage for the overlay
        self._title_handle = None

    # ── Internal helpers ──────────────────────────────────────────────

    def _mask_path_for(self, img_path: Path) -> Path:
        return self.masks_dir / (img_path.stem + "_GT.png")

    def _load_current(self) -> None:
        from .io import load_image
        from .preprocessing import build_zenith_weight_map
        img_path = self.image_paths[self._idx]
        self._img  = load_image(img_path)
        h, w = self._img.shape[:2]
        if self._zenith_weights is None:
            self._zenith_weights = build_zenith_weight_map(h, w, CX, CY, R, CF_MAX_ZENITH_DEG)
        mp = self._mask_path_for(img_path)
        if mp.exists():
            self._mask = load_existing_mask(mp, (h, w))
        else:
            self._mask = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)
        self._undo_stack.clear()

    def _overlay(self) -> np.ndarray:
        """Blend original image with label colours (50/50 where painted)."""
        overlay = self._img.copy().astype(np.float32)
        painted = self._mask != LABEL_UNLABELLED
        for val, colour in [
            (LABEL_SKY,    COLOUR_SKY),
            (LABEL_CLOUD,  COLOUR_CLOUD),
            (LABEL_IGNORE, COLOUR_IGNORE),
        ]:
            m = self._mask == val
            overlay[m] = overlay[m] * 0.45 + colour.astype(np.float32) * 0.55
        return overlay.clip(0, 255).astype(np.uint8)

    def _update_display(self) -> None:
        self._im_handle.set_data(self._overlay())
        cf = _compute_cf(self._mask, self._zenith_weights)
        n = len(self.image_paths)
        name = self.image_paths[self._idx].name
        cf_str = f"{cf:.3f}" if not np.isnan(cf) else "n/a"
        unlabelled_pct = (self._mask == LABEL_UNLABELLED).mean() * 100
        self._ax_img.set_title(
            f"[{self._idx + 1}/{n}]  {name}\n"
            f"CF={cf_str}   unlabelled={unlabelled_pct:.1f}%   "
            f"brush={self.brush_radius}px\n"
            "L=cloud  R=sky  M=ignore  |  s=save  n=next  p=prev  z=undo  r=reset  q=quit",
            fontsize=8, loc="left",
        )
        self._fig.canvas.draw_idle()

    def _paint(self, x: float, y: float, label: int) -> None:
        """Paint a circular brush stroke at (x, y) in image coordinates."""
        if self._mask is None:
            return
        h, w = self._mask.shape
        xi, yi = int(round(x)), int(round(y))
        r = self.brush_radius
        yy, xx = np.ogrid[max(0, yi - r):min(h, yi + r + 1),
                          max(0, xi - r):min(w, xi + r + 1)]
        circle = ((xx - xi) ** 2 + (yy - yi) ** 2) <= r ** 2

        # Restrict to dome if mask provided
        if self.dome_mask is not None:
            sub_dome = self.dome_mask[
                max(0, yi - r):min(h, yi + r + 1),
                max(0, xi - r):min(w, xi + r + 1),
            ]
            circle = circle & sub_dome

        self._mask[
            max(0, yi - r):min(h, yi + r + 1),
            max(0, xi - r):min(w, xi + r + 1),
        ][circle] = label

    # ── Event handlers ────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax_img or event.xdata is None:
            return
        # Save undo snapshot before first stroke
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > 30:   # keep last 30 strokes
            self._undo_stack.pop(0)

        label = {1: LABEL_CLOUD, 3: LABEL_SKY, 2: LABEL_IGNORE}.get(event.button)
        if label is None:
            return
        self._painting = label
        self._paint(event.xdata, event.ydata, label)
        self._update_display()

    def _on_motion(self, event) -> None:
        if self._painting is None or event.inaxes is not self._ax_img or event.xdata is None:
            return
        self._paint(event.xdata, event.ydata, self._painting)
        self._update_display()

    def _on_release(self, event) -> None:
        self._painting = None

    def _on_key(self, event) -> None:
        key = event.key

        if key == "s":
            self._do_save()

        elif key == "n":
            self._do_save(silent=True)
            self._idx = min(self._idx + 1, len(self.image_paths) - 1)
            self._load_current()
            self._update_display()

        elif key == "p":
            self._do_save(silent=True)
            self._idx = max(self._idx - 1, 0)
            self._load_current()
            self._update_display()

        elif key == "z":
            if self._undo_stack:
                self._mask = self._undo_stack.pop()
                self._update_display()

        elif key == "r":
            h, w = self._img.shape[:2]
            self._undo_stack.append(self._mask.copy())
            self._mask = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)
            self._update_display()

        elif key == "q":
            self._do_save(silent=True)
            import matplotlib.pyplot as plt
            plt.close(self._fig)

        elif key in ("+", "="):
            self.brush_radius = min(self.brush_radius + 5, 100)
            self._update_display()

        elif key == "-":
            self.brush_radius = max(self.brush_radius - 5, 2)
            self._update_display()

    def _do_save(self, silent: bool = False) -> None:
        img_path = self.image_paths[self._idx]
        mp = self._mask_path_for(img_path)
        save_mask(self._mask, mp)
        cf = _compute_cf(self._mask, self._zenith_weights)
        append_log(self.log_path, img_path.name, cf)
        if not silent:
            print(f"Saved: {mp}  |  CF={cf:.3f}")

    # ── Public interface ──────────────────────────────────────────────

    def run(self) -> None:
        """Launch the interactive labelling GUI (blocking call).

        The window blocks until the user presses ``q`` or closes the figure.

        Example:
            >>> tool = LabellingTool(image_paths, masks_dir=Path('data/masks_manual'))
            >>> tool.run()
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if not self.image_paths:
            raise ValueError("No images provided to LabellingTool.")

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self._load_current()

        self._fig = plt.figure(figsize=(14, 9))
        self._fig.patch.set_facecolor("#1a1a1a")
        gs = gridspec.GridSpec(1, 1)
        self._ax_img = self._fig.add_subplot(gs[0, 0])
        self._ax_img.set_facecolor("#1a1a1a")

        self._im_handle = self._ax_img.imshow(self._overlay())
        self._ax_img.axis("off")
        self._update_display()

        # Connect events
        self._fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("key_press_event",      self._on_key)

        self._fig.tight_layout()
        plt.show()
