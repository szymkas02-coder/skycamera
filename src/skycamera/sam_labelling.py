"""SAM 2 assisted labelling — interactive and batch modes.

Two usage modes
---------------
Interactive (SAM-assisted labelling tool)
    User clicks a point on a cloud or sky region.
    SAM 2 instantly segments the clicked region.
    User can accept, correct with the manual brush, or reject.
    Controls:
        Left-click   -> SAM segments as CLOUD (applied immediately)
        Right-click  -> SAM segments as SKY (applied immediately)
        Middle-click -> SAM segments as IGNORE (cables, sun disk, glare)
        i            -> toggle manual brush mode (left-click+drag paints IGNORE pixels)
        s -> save
        n/p -> next/prev image (auto-saves)
        z -> undo  r -> reset  q -> quit

Batch (automatic pseudo-labelling)
    SAM 2 is run with a regular grid of point prompts across the dome.
    Each point either selects cloud (if it landed on a bright region)
    or sky. The union of all cloud segments becomes the cloud mask.
    Produces pseudo-labels for CNN training with no human interaction.
    Quality is lower than interactive but sufficient for bootstrapping.

Checkpoint
    Default: sam2.1_hiera_small.pt (184 MB, good balance of speed/quality)
    Config:  configs/sam2.1/sam2.1_hiera_s.yaml (bundled with the sam2 package)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MASKS_MANUAL_DIR, CX, CY, R, CF_MAX_ZENITH_DEG
from .labelling import (
    LABEL_SKY, LABEL_CLOUD, LABEL_IGNORE, LABEL_UNLABELLED,
    save_mask, load_existing_mask, append_log, _compute_cf,
)

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
SAM2_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "sam2_checkpoints"
SAM2_CHECKPOINT     = SAM2_CHECKPOINT_DIR / "sam2.1_hiera_small.pt"
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Grid density for batch mode (points per axis across the dome)
BATCH_GRID_POINTS = 10


# ── Model loading ─────────────────────────────────────────────────────

def load_sam2(
    checkpoint: Path = SAM2_CHECKPOINT,
    config: str = SAM2_CONFIG,
    device: str = "cpu",
):
    """Load SAM 2 predictor.

    Args:
        checkpoint: Path to ``.pt`` checkpoint file.
        config: Config name relative to the sam2 package directory.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ``SAM2ImagePredictor`` instance ready for ``set_image`` calls.

    Example:
        >>> predictor = load_sam2()
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            f"Download it with:\n"
            f"  python -c \"import urllib.request; "
            f"urllib.request.urlretrieve("
            f"'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', "
            f"'{checkpoint}')\""
        )
    model = build_sam2(config, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    log.info("SAM2 loaded: %s on %s", checkpoint.name, device)
    return predictor


# ── Core segmentation ─────────────────────────────────────────────────

def segment_point(
    predictor,
    point_xy: tuple[int, int],
    label: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Run SAM 2 with a single point prompt, returning all 3 mask candidates.

    ``predictor.set_image(img)`` must be called before this function.

    Args:
        predictor: ``SAM2ImagePredictor`` with image already set.
        point_xy: (x, y) pixel coordinate of the click in image space.
        label: ``LABEL_CLOUD``, ``LABEL_SKY``, or ``LABEL_IGNORE``.

    Returns:
        Tuple of (masks, scores) where masks is a list of 3 boolean arrays
        sorted best-first by SAM confidence score.

    Example:
        >>> masks, scores = segment_point(predictor, (800, 400), LABEL_CLOUD)
        >>> best_mask = masks[0]
    """
    coords = np.array([[point_xy[0], point_xy[1]]])
    labels = np.array([1])   # SAM always gets a foreground point
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    # Sort best-first by confidence
    order = np.argsort(scores)[::-1]
    sorted_masks  = [masks[i].astype(bool) for i in order]
    sorted_scores = [float(scores[i]) for i in order]
    return sorted_masks, sorted_scores


# ── Batch pseudo-labelling ────────────────────────────────────────────

def batch_pseudolabel(
    img: np.ndarray,
    predictor,
    dome_mask: np.ndarray,
    grid_n: int = BATCH_GRID_POINTS,
    brightness_threshold: float = 0.55,
) -> np.ndarray:
    """Generate an automatic cloud mask using SAM 2 with a grid of point prompts.

    A regular grid of ``grid_n × grid_n`` points is placed across the dome.
    For each point:
        - If mean normalised brightness of the pixel neighbourhood > threshold
          -> it likely landed on a cloud -> run SAM, add to cloud mask
        - Else -> sky, skip

    This produces a pseudo-label with no human interaction. Quality is lower
    than clicked SAM masks but sufficient for bootstrapping the CNN training set.

    Args:
        img: RGB image array ``(H, W, 3)`` uint8.
        predictor: ``SAM2ImagePredictor`` (image already set via set_image).
        dome_mask: Boolean dome mask ``(H, W)``.
        grid_n: Number of grid points per axis inside the dome bounding box.
        brightness_threshold: Normalised brightness (0–1) above which a point
            is classified as cloud. 0.55 works well for daylight images.

    Returns:
        uint8 label array ``(H, W)`` with values LABEL_CLOUD, LABEL_SKY,
        LABEL_UNLABELLED (for pixels outside the dome).

    Example:
        >>> predictor.set_image(img)
        >>> mask = batch_pseudolabel(img, predictor, dome_mask)
    """
    H, W = img.shape[:2]
    result = np.full((H, W), LABEL_UNLABELLED, dtype=np.uint8)
    # Default all dome pixels to sky; cloud segments will overwrite
    result[dome_mask] = LABEL_SKY

    # Bounding box of dome
    ys, xs = np.where(dome_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    grid_y = np.linspace(y0, y1, grid_n, dtype=int)
    grid_x = np.linspace(x0, x1, grid_n, dtype=int)

    img_norm = img.astype(np.float32) / 255.0

    cloud_union = np.zeros((H, W), dtype=bool)

    for gy in grid_y:
        for gx in grid_x:
            if not dome_mask[gy, gx]:
                continue

            # Local brightness in a small patch
            r = 10
            patch = img_norm[
                max(0, gy-r):min(H, gy+r),
                max(0, gx-r):min(W, gx+r),
            ]
            brightness = float(patch.mean())

            if brightness >= brightness_threshold:
                try:
                    masks, _ = segment_point(predictor, (gx, gy), LABEL_CLOUD)
                    cloud_union |= (masks[0] & dome_mask)
                except Exception:
                    pass

    result[cloud_union] = LABEL_CLOUD
    return result


def run_batch_pseudolabel(
    image_paths: list[Path],
    masks_dir: Path,
    dome_mask: np.ndarray,
    checkpoint: Path = SAM2_CHECKPOINT,
    config: str = SAM2_CONFIG,
    grid_n: int = BATCH_GRID_POINTS,
    brightness_threshold: float = 0.55,
    overwrite: bool = False,
) -> list[Path]:
    """Run batch pseudo-labelling on a list of images.

    Loads SAM 2 once, then processes each image in sequence.
    Skips images that already have a mask unless ``overwrite=True``.

    Args:
        image_paths: List of raw image paths.
        masks_dir: Directory to save generated masks.
        dome_mask: Boolean dome mask at image native resolution.
        checkpoint: SAM2 checkpoint path.
        config: SAM2 config name.
        grid_n: Grid density for batch prompting.
        brightness_threshold: Cloud brightness threshold.
        overwrite: If True, overwrite existing masks.

    Returns:
        List of saved mask paths.

    Example:
        >>> saved = run_batch_pseudolabel(image_paths, masks_dir, dome_mask)
    """
    from .io import load_image
    from .preprocessing import build_zenith_weight_map

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_sam2(checkpoint, config)
    saved_paths = []
    _zenith_weights = None

    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        mask_out = masks_dir / (img_path.stem + "_GT.png")

        if mask_out.exists() and not overwrite:
            log.info("[%d/%d] Skipping %s (already exists)", i+1, len(image_paths), img_path.name)
            continue

        try:
            img = load_image(img_path)
            if _zenith_weights is None:
                h, w = img.shape[:2]
                _zenith_weights = build_zenith_weight_map(h, w, CX, CY, R, CF_MAX_ZENITH_DEG)
            predictor.set_image(img)
            mask = batch_pseudolabel(img, predictor, dome_mask, grid_n, brightness_threshold)
            save_mask(mask, mask_out)
            cf = _compute_cf(mask, _zenith_weights)
            append_log(masks_dir / "labelling_log.csv", img_path.name, cf,
                       notes="sam2_batch")
            log.info("[%d/%d] %s -> CF=%.3f saved %s",
                     i+1, len(image_paths), img_path.name, cf, mask_out.name)
            saved_paths.append(mask_out)
        except Exception as e:
            log.error("[%d/%d] Failed %s: %s", i+1, len(image_paths), img_path.name, e)

    log.info("Batch pseudo-labelling complete: %d masks saved", len(saved_paths))
    return saved_paths


# ── Default ignore mask creator ──────────────────────────────────────

DEFAULT_IGNORE_FILENAME = "default_ignore.png"


def create_default_ignore_mask(
    reference_image_path: Path,
    masks_dir: Path,
    predictor=None,
    checkpoint: Path = SAM2_CHECKPOINT,
    dome_mask: Optional[np.ndarray] = None,
    brush_radius: int = 15,
) -> Path:
    """SAM-assisted tool to paint fixed ignore regions (antennas, cables).

    Left-click  -> SAM segments the clicked structure as IGNORE.
    Right-drag  -> erase brush (remove wrongly marked pixels).

    The result is saved as ``masks_dir/default_ignore.png`` and passed to
    :class:`SAMLabellingTool` via ``default_ignore_path`` so every new image
    starts with those regions pre-marked. Run once; re-run to refine.

    Args:
        reference_image_path: Any representative sky image.
        masks_dir: Directory where ``default_ignore.png`` will be saved.
        predictor: Pre-loaded SAM2ImagePredictor. If None, loaded automatically.
        checkpoint: SAM2 checkpoint path (used only when predictor is None).
        dome_mask: Optional dome mask to restrict painting.
        brush_radius: Erase brush size in pixels.

    Returns:
        Path to the saved ``default_ignore.png``.
    """
    from .io import load_image
    import matplotlib.pyplot as plt

    if predictor is None:
        predictor = load_sam2(checkpoint)

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / DEFAULT_IGNORE_FILENAME

    img = load_image(reference_image_path)
    h, w = img.shape[:2]
    predictor.set_image(img)

    # Load existing default if present so it can be refined
    if out_path.exists():
        mask = load_existing_mask(out_path)
    else:
        mask = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)

    state = {"mask": mask, "erasing": False, "undo_stack": [], "brush_radius": brush_radius}

    def _overlay():
        ov = img.copy().astype(np.float32)
        m = state["mask"] == LABEL_IGNORE
        ov[m] = ov[m] * 0.3 + np.array([255, 140, 0], np.float32) * 0.7
        return ov.clip(0, 255).astype(np.uint8)

    def _erase(x, y):
        xi, yi = int(round(x)), int(round(y))
        r = state["brush_radius"]
        yy, xx = np.ogrid[max(0, yi-r):min(h, yi+r+1), max(0, xi-r):min(w, xi+r+1)]
        circle = ((xx-xi)**2 + (yy-yi)**2) <= r**2
        state["mask"][max(0,yi-r):min(h,yi+r+1), max(0,xi-r):min(w,xi+r+1)][circle] = LABEL_UNLABELLED

    def _update():
        im.set_data(_overlay())
        n_px = (state["mask"] == LABEL_IGNORE).sum()
        ax.set_title(
            f"Default ignore mask  |  {n_px} ignore pixels\n"
            "L=SAM ignore  R-drag=erase  z=undo  r=reset  +/-=brush size  s=save & quit",
            fontsize=9, loc="left",
        )
        fig.canvas.draw_idle()

    def on_press(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        state["undo_stack"].append(state["mask"].copy())
        if len(state["undo_stack"]) > 30:
            state["undo_stack"].pop(0)
        if ev.button == 1:   # left -> SAM ignore
            try:
                masks, _ = segment_point(predictor, (int(round(ev.xdata)), int(round(ev.ydata))), LABEL_IGNORE)
                seg = masks[0]
                if dome_mask is not None:
                    seg = seg & dome_mask
                state["mask"][seg] = LABEL_IGNORE
            except Exception as e:
                log.warning("SAM click failed: %s", e)
        elif ev.button == 3:  # right -> erase brush
            state["erasing"] = True
            _erase(ev.xdata, ev.ydata)
        _update()

    def on_motion(ev):
        if not state["erasing"] or ev.inaxes is not ax or ev.xdata is None:
            return
        _erase(ev.xdata, ev.ydata)
        _update()

    def on_release(ev):
        state["erasing"] = False

    def on_key(ev):
        if ev.key == "s":
            save_mask(state["mask"], out_path)
            print(f"Default ignore mask saved: {out_path}")
            plt.close(fig)
        elif ev.key == "z" and state["undo_stack"]:
            state["mask"] = state["undo_stack"].pop()
            _update()
        elif ev.key == "r":
            state["undo_stack"].append(state["mask"].copy())
            state["mask"] = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)
            _update()
        elif ev.key in ("+", "="):
            state["brush_radius"] = min(state["brush_radius"] + 5, 100)
            _update()
        elif ev.key == "-":
            state["brush_radius"] = max(state["brush_radius"] - 5, 2)
            _update()

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    im = ax.imshow(_overlay())
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event",      on_key)
    _update()
    plt.show()
    return out_path


# ── Ultimate interactive labelling tool ──────────────────────────────

# ── UI constants ──────────────────────────────────────────────────────
_MODE_SAM   = "sam"
_MODE_LASSO = "lasso"
_MODE_BRUSH = "brush"

_LABEL_COLOURS = {
    LABEL_SKY:    np.array([30,  120, 200], np.float32),
    LABEL_CLOUD:  np.array([220, 220, 220], np.float32),
    LABEL_IGNORE: np.array([255, 140,   0], np.float32),
}
_PENDING_COLOUR = np.array([50, 220, 50], np.float32)

# Button definitions: (text, x_frac, width_frac, group)
# group: "mode" | "label" | "nav" | "action"
_BTN_DEFS = [
    ("SAM",       0.01, 0.07, "mode"),
    ("Lasso",     0.09, 0.07, "mode"),
    ("Brush",     0.17, 0.07, "mode"),
    ("☁ Cloud",  0.28, 0.09, "label"),
    ("Sky",       0.38, 0.07, "label"),
    ("Ignore",    0.46, 0.07, "label"),
    ("◀ Prev",   0.58, 0.08, "nav"),
    ("Next ▶",   0.67, 0.08, "nav"),
    ("Skip",      0.76, 0.07, "nav"),
    ("Save",      0.87, 0.07, "action"),
    ("Reset",     0.95, 0.05, "action"),
]

_BTN_COLOURS = {
    "mode":   ("#2d5a8e", "#1a3a5e", "#4a8fd4"),   # base, active, hover
    "label":  ("#2e6b2e", "#1a4a1a", "#4aaa4a"),
    "nav":    ("#555555", "#333333", "#777777"),
    "action": ("#7a3a1a", "#4a2010", "#cc6633"),
}


class SAMLabellingTool:
    """Sky camera labelling tool — SAM-assisted, lasso, and brush modes.

    Modes
    -----
    SAM (default)
        L-click = cloud preview   R-click = sky preview   M-click = ignore preview
        Tab = cycle 3 SAM candidates   a = accept   Esc = cancel
        By default SAM only paints UNLABELLED pixels — existing labels preserved.
        Hold Shift while clicking to overwrite existing labels.

    Lasso  (button or  f  key)
        Hold L + drag = draw outline   release = fill INSIDE with active label
        x = fill OUTSIDE instead   Esc = cancel

    Brush  (button or  b  key)
        L-drag = paint active label   +/- = brush size

    Labels  (buttons or  c / s / i  keys)
        c = Cloud   s = Sky   i = Ignore

    Navigation
        n = next (auto-save)   p = prev (auto-save)
        skip button / Skip key = next WITHOUT saving
        S = save now   z = undo   r = reset   q = quit
    """

    def __init__(
        self,
        image_paths: list[Path],
        masks_dir: Path = MASKS_MANUAL_DIR,
        dome_mask: Optional[np.ndarray] = None,
        predictor=None,
        checkpoint: Path = SAM2_CHECKPOINT,
        brush_radius: int = 15,
        log_path: Optional[Path] = None,
        default_ignore_path: Optional[Path] = None,
    ) -> None:
        self.image_paths  = [Path(p) for p in image_paths]
        self.masks_dir    = Path(masks_dir)
        self.dome_mask    = dome_mask
        self.brush_radius = brush_radius
        self.log_path     = log_path or self.masks_dir / "labelling_log.csv"

        if default_ignore_path is not None and Path(default_ignore_path).exists():
            self._default_ignore = load_existing_mask(Path(default_ignore_path))
        else:
            self._default_ignore = None

        self._predictor = predictor if predictor is not None else load_sam2(checkpoint)

        # Core state
        self._idx           = 0
        self._img           = None
        self._mask          = None
        self._undo_stack    : list[np.ndarray] = []
        self._zenith_weights: Optional[np.ndarray] = None
        self._mode          = _MODE_SAM
        self._active_label  = LABEL_CLOUD
        self._painting      = False
        self._overwrite     = False     # Shift held — overwrite existing labels

        # SAM candidates: (masks, scores, label, candidate_idx)
        self._pending = None

        # Lasso state
        self._lasso_verts : list[tuple[float, float]] = []
        self._lasso_line  = None

        # Overlay cache
        self._overlay_cache: Optional[np.ndarray] = None

        # Figure handles (set in run())
        self._fig       = None
        self._ax_img    = None
        self._ax_info   = None
        self._ax_prog   = None
        self._im_handle = None
        self._btns      = {}    # name → Button widget
        self._btn_axes  = {}    # name → Axes

    # ── Mask helpers ──────────────────────────────────────────────────

    def _mask_path_for(self, img_path: Path) -> Path:
        return self.masks_dir / (img_path.stem + "_GT.png")

    def _fresh_mask(self) -> np.ndarray:
        from .sun import sun_ignore_mask
        h, w = self._img.shape[:2]
        mask = np.full((h, w), LABEL_UNLABELLED, dtype=np.uint8)
        if self._default_ignore is not None:
            mask[self._default_ignore == LABEL_IGNORE] = LABEL_IGNORE
        sun = sun_ignore_mask(self.image_paths[self._idx], (h, w))
        if sun is not None:
            mask[sun] = LABEL_IGNORE
        return mask

    def _load_current(self) -> None:
        from .io import load_image
        from .preprocessing import build_zenith_weight_map
        img_path = self.image_paths[self._idx]
        self._img  = load_image(img_path)
        if self._zenith_weights is None:
            h, w = self._img.shape[:2]
            self._zenith_weights = build_zenith_weight_map(h, w, CX, CY, R, CF_MAX_ZENITH_DEG)
        mp = self._mask_path_for(img_path)
        self._mask = load_existing_mask(mp) if mp.exists() else self._fresh_mask()
        self._undo_stack.clear()
        self._pending       = None
        self._lasso_verts   = []
        self._lasso_line    = None
        self._overlay_cache = None
        self._mode          = _MODE_SAM
        self._active_label  = LABEL_CLOUD
        self._predictor.set_image(self._img)

    # ── Overlay ───────────────────────────────────────────────────────

    def _invalidate_overlay(self) -> None:
        self._overlay_cache = None

    def _overlay(self) -> np.ndarray:
        if self._overlay_cache is not None:
            return self._overlay_cache
        ov = self._img.copy().astype(np.float32)
        for val, colour in _LABEL_COLOURS.items():
            m = self._mask == val
            ov[m] = ov[m] * 0.45 + colour * 0.55
        if self._pending is not None:
            masks, _, _, cidx = self._pending
            ov[masks[cidx]] = ov[masks[cidx]] * 0.3 + _PENDING_COLOUR * 0.7
        self._overlay_cache = ov.clip(0, 255).astype(np.uint8)
        return self._overlay_cache

    # ── Display update ────────────────────────────────────────────────

    def _label_name(self, label: int) -> str:
        return {LABEL_CLOUD: "Cloud", LABEL_SKY: "Sky",
                LABEL_IGNORE: "Ignore"}.get(label, "?")

    def _update_display(self) -> None:
        self._im_handle.set_data(self._overlay())
        self._update_info()
        self._update_buttons()
        self._update_progress()
        self._fig.canvas.draw_idle()

    def _update_info(self) -> None:
        cf    = _compute_cf(self._mask, self._zenith_weights)
        n     = len(self.image_paths)
        name  = self.image_paths[self._idx].name
        cf_str = f"{cf:.3f}" if not np.isnan(cf) else "n/a"
        unlbl  = (self._mask == LABEL_UNLABELLED).mean() * 100
        done   = sum(1 for p in self.image_paths
                     if self._mask_path_for(p).exists())

        if self._mode == _MODE_SAM:
            if self._pending is not None:
                _, scores, _, cidx = self._pending
                hint = (f"SAM preview — candidate {cidx+1}/3  "
                        f"score={scores[cidx]:.2f}  "
                        f"[Tab=cycle  a=accept  Esc=cancel]")
            else:
                ow = "  OVERWRITE ON (Shift)" if self._overwrite else ""
                hint = f"SAM mode{ow}  —  L=cloud  R=sky  M=ignore  |  f=lasso  b=brush"
        elif self._mode == _MODE_LASSO:
            lname = self._label_name(self._active_label)
            x_hint = "  [x=fill OUTSIDE]" if len(self._lasso_verts) >= 3 else ""
            hint = f"Lasso — active: {lname}{x_hint}  |  hold L=draw  release=fill  c/s/i=label"
        else:
            lname = self._label_name(self._active_label)
            hint = f"Brush — active: {lname}  size={self.brush_radius}px  |  c/s/i=label  +/-=size"

        self._ax_info.clear()
        self._ax_info.set_facecolor("#111111")
        self._ax_info.axis("off")
        # Left: image info
        self._ax_info.text(0.01, 0.72, f"{name}",
                           color="#dddddd", fontsize=9, fontweight="bold",
                           transform=self._ax_info.transAxes, va="top")
        self._ax_info.text(0.01, 0.35,
                           f"[{self._idx+1}/{n}]   CF = {cf_str}   "
                           f"unlabelled = {unlbl:.1f}%   "
                           f"session saved: {done}/{n}",
                           color="#aaaaaa", fontsize=8,
                           transform=self._ax_info.transAxes, va="top")
        # Right: mode hint
        self._ax_info.text(0.99, 0.72, hint,
                           color="#88ccff", fontsize=8,
                           transform=self._ax_info.transAxes,
                           va="top", ha="right")
        self._ax_info.text(0.99, 0.35,
                           "S=save  n/p=next/prev  skip=no save  z=undo  r=reset  q=quit",
                           color="#666666", fontsize=7,
                           transform=self._ax_info.transAxes,
                           va="top", ha="right")

    def _update_progress(self) -> None:
        """Draw a thin progress bar showing labelled fraction of the queue."""
        labelled = sum(1 for p in self.image_paths
                       if self._mask_path_for(p).exists())
        frac = labelled / max(len(self.image_paths), 1)
        unlbl_frac = (self._mask == LABEL_UNLABELLED).mean()

        self._ax_prog.clear()
        self._ax_prog.set_facecolor("#222222")
        self._ax_prog.set_xlim(0, 1)
        self._ax_prog.set_ylim(0, 1)
        self._ax_prog.axis("off")
        # Session progress (green)
        self._ax_prog.barh(0.6, frac, height=0.35, color="#2a7a2a", left=0)
        self._ax_prog.barh(0.6, 1 - frac, height=0.35, color="#333333",
                           left=frac)
        # Current image unlabelled (orange)
        self._ax_prog.barh(0.1, 1 - unlbl_frac, height=0.3, color="#cc7700",
                           left=0)
        self._ax_prog.barh(0.1, unlbl_frac, height=0.3, color="#333333",
                           left=1 - unlbl_frac)
        self._ax_prog.text(0.5, 0.95,
                           f"Session: {labelled}/{len(self.image_paths)} labelled   "
                           f"This image: {(1-unlbl_frac)*100:.0f}% covered",
                           color="#aaaaaa", fontsize=7, ha="center", va="top",
                           transform=self._ax_prog.transAxes)

    def _update_buttons(self) -> None:
        """Highlight active mode and active label buttons."""
        mode_map   = {"SAM": _MODE_SAM, "Lasso": _MODE_LASSO, "Brush": _MODE_BRUSH}
        label_map  = {"☁ Cloud": LABEL_CLOUD, "Sky": LABEL_SKY, "Ignore": LABEL_IGNORE}

        for name, btn in self._btns.items():
            _, _, _, group = next(d for d in _BTN_DEFS if d[0] == name)
            base, active, _ = _BTN_COLOURS[group]
            is_active = (
                (name in mode_map  and mode_map[name]  == self._mode) or
                (name in label_map and label_map[name] == self._active_label)
            )
            colour = active if is_active else base
            btn.ax.set_facecolor(colour)
            btn.color        = colour
            btn.hovercolor   = _BTN_COLOURS[group][2]

    # ── SAM helpers ───────────────────────────────────────────────────

    def _commit_pending(self) -> None:
        if self._pending is None:
            return
        masks, _, label, cidx = self._pending
        seg = masks[cidx]
        if self.dome_mask is not None:
            seg = seg & self.dome_mask
        if not self._overwrite:
            seg = seg & (self._mask == LABEL_UNLABELLED)
        self._mask[seg] = label
        self._pending = None
        self._invalidate_overlay()

    def _do_sam_click(self, x: float, y: float, label: int) -> None:
        self._commit_pending()
        try:
            masks, scores = segment_point(
                self._predictor, (int(round(x)), int(round(y))), label)
            if self.dome_mask is not None:
                masks = [m & self.dome_mask for m in masks]
            self._pending = (masks, scores, label, 0)
            self._invalidate_overlay()
        except Exception as e:
            log.warning("SAM click failed: %s", e)

    # ── Lasso helpers ─────────────────────────────────────────────────

    def _lasso_clear_line(self) -> None:
        if self._lasso_line is not None:
            try:
                self._lasso_line.remove()
            except Exception:
                pass
            self._lasso_line = None

    def _lasso_update_line(self) -> None:
        self._lasso_clear_line()
        if len(self._lasso_verts) >= 2:
            xs = [v[0] for v in self._lasso_verts] + [self._lasso_verts[0][0]]
            ys = [v[1] for v in self._lasso_verts] + [self._lasso_verts[0][1]]
            self._lasso_line, = self._ax_img.plot(
                xs, ys, "-", color="#ffff00", linewidth=1.5, zorder=10)

    def _lasso_fill_region(self, invert: bool = False) -> None:
        if len(self._lasso_verts) < 3:
            self._lasso_verts = []
            return
        from matplotlib.path import Path as MplPath
        h, w = self._mask.shape
        verts = np.array(self._lasso_verts)
        path  = MplPath(verts)
        x0 = max(0,   int(verts[:, 0].min()))
        x1 = min(w-1, int(verts[:, 0].max())) + 1
        y0 = max(0,   int(verts[:, 1].min()))
        y1 = min(h-1, int(verts[:, 1].max())) + 1

        if not invert:
            cols, rows = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            pts    = np.column_stack([cols.ravel(), rows.ravel()])
            region = path.contains_points(pts).reshape(rows.shape)
            if self.dome_mask is not None:
                region &= self.dome_mask[y0:y1, x0:x1]
            self._mask[y0:y1, x0:x1][region] = self._active_label
        else:
            # Outside = full dome minus inside
            cols, rows = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            pts    = np.column_stack([cols.ravel(), rows.ravel()])
            inside = path.contains_points(pts).reshape(rows.shape)
            outside = np.zeros((h, w), dtype=bool)
            if self.dome_mask is not None:
                outside[self.dome_mask] = True
            outside[y0:y1, x0:x1] &= ~inside
            self._mask[outside] = self._active_label
            self._lasso_verts = []
            self._lasso_clear_line()

        self._invalidate_overlay()

    # ── Brush helpers ─────────────────────────────────────────────────

    def _brush_paint(self, x: float, y: float) -> None:
        if self._mask is None:
            return
        h, w = self._mask.shape
        xi, yi = int(round(x)), int(round(y))
        r = self.brush_radius
        yy, xx = np.ogrid[max(0, yi-r):min(h, yi+r+1),
                          max(0, xi-r):min(w, xi+r+1)]
        circle = ((xx - xi)**2 + (yy - yi)**2) <= r**2
        if self.dome_mask is not None:
            circle &= self.dome_mask[max(0,yi-r):min(h,yi+r+1),
                                     max(0,xi-r):min(w,xi+r+1)]
        self._mask[max(0,yi-r):min(h,yi+r+1),
                   max(0,xi-r):min(w,xi+r+1)][circle] = self._active_label
        self._invalidate_overlay()

    # ── Save / undo ───────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > 40:
            self._undo_stack.pop(0)

    def _do_save(self, silent: bool = False) -> None:
        only_ignore = np.all(
            (self._mask == LABEL_UNLABELLED) | (self._mask == LABEL_IGNORE))
        if only_ignore:
            return
        img_path = self.image_paths[self._idx]
        mp = self._mask_path_for(img_path)
        save_mask(self._mask, mp)
        cf = _compute_cf(self._mask, self._zenith_weights)
        append_log(self.log_path, img_path.name, cf, notes="sam2_interactive")
        if not silent:
            print(f"Saved: {mp.name}  CF={cf:.3f}")

    # ── Event handlers ────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax_img or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        btn  = event.button

        if self._mode == _MODE_SAM:
            self._push_undo()
            if btn == 1:
                self._do_sam_click(x, y, LABEL_CLOUD)
            elif btn == 3:
                self._do_sam_click(x, y, LABEL_SKY)
            elif btn == 2:
                self._do_sam_click(x, y, LABEL_IGNORE)

        elif self._mode == _MODE_LASSO:
            if btn == 1:
                self._push_undo()
                self._lasso_verts = [(x, y)]
                self._painting = True

        elif self._mode == _MODE_BRUSH:
            if btn == 1:
                self._push_undo()
                self._painting = True
                self._brush_paint(x, y)

        self._update_display()

    def _on_motion(self, event) -> None:
        if not self._painting or event.inaxes is not self._ax_img \
                or event.xdata is None:
            return
        if self._mode == _MODE_LASSO:
            self._lasso_verts.append((event.xdata, event.ydata))
            self._lasso_update_line()
            self._fig.canvas.draw_idle()
        elif self._mode == _MODE_BRUSH:
            self._brush_paint(event.xdata, event.ydata)
            self._update_display()

    def _on_release(self, event) -> None:
        if self._painting and self._mode == _MODE_LASSO:
            self._lasso_clear_line()
            self._lasso_fill_region(invert=False)
            self._update_display()
        self._painting = False

    def _on_key(self, event) -> None:
        import matplotlib.pyplot as plt
        key = event.key

        # Shift modifier for SAM overwrite toggle
        if key in ("shift",):
            return

        # ── Mode switches ──────────────────────────────────────────
        if key == "escape":
            self._pending = None
            self._lasso_verts = []
            self._lasso_clear_line()
            self._painting = False
            self._mode = _MODE_SAM
            self._update_display()
            return

        if key == "f":
            self._commit_pending()
            self._lasso_verts = []
            self._lasso_clear_line()
            self._mode = _MODE_LASSO if self._mode != _MODE_LASSO else _MODE_SAM
            self._update_display()
            return

        if key == "b":
            self._commit_pending()
            self._mode = _MODE_BRUSH if self._mode != _MODE_BRUSH else _MODE_SAM
            self._update_display()
            return

        # ── Overwrite toggle ───────────────────────────────────────
        if key == "o":
            self._overwrite = not self._overwrite
            self._update_display()
            return

        # ── Active label ───────────────────────────────────────────
        if key == "c":
            self._active_label = LABEL_CLOUD;  self._update_display(); return
        if key == "s":
            self._active_label = LABEL_SKY;    self._update_display(); return
        if key == "i":
            self._active_label = LABEL_IGNORE; self._update_display(); return

        # ── SAM candidate cycling ──────────────────────────────────
        if key == "tab" and self._pending is not None:
            masks, scores, label, cidx = self._pending
            self._pending = (masks, scores, label, (cidx + 1) % len(masks))
            self._invalidate_overlay()
            self._update_display()
            return

        if key == "a" and self._pending is not None:
            self._commit_pending()
            self._update_display()
            return

        # ── Lasso exterior fill ────────────────────────────────────
        if key == "x" and self._mode == _MODE_LASSO \
                and len(self._lasso_verts) >= 3:
            self._push_undo()
            self._lasso_fill_region(invert=True)
            self._update_display()
            return

        # ── Navigation & saving ────────────────────────────────────
        if key == "S":
            self._commit_pending()
            self._do_save()
            self._update_display()

        elif key == "n":
            self._commit_pending()
            self._do_save(silent=True)
            self._idx = min(self._idx + 1, len(self.image_paths) - 1)
            self._load_current()
            self._update_display()

        elif key == "N":   # skip without saving
            self._pending = None
            self._lasso_verts = []
            self._idx = min(self._idx + 1, len(self.image_paths) - 1)
            self._load_current()
            self._update_display()

        elif key == "p":
            self._commit_pending()
            self._do_save(silent=True)
            self._idx = max(self._idx - 1, 0)
            self._load_current()
            self._update_display()

        elif key == "z":
            self._pending = None
            if self._undo_stack:
                self._mask = self._undo_stack.pop()
                self._invalidate_overlay()
                self._update_display()

        elif key == "r":
            self._pending = None
            self._lasso_verts = []
            self._lasso_clear_line()
            self._push_undo()
            self._mask = self._fresh_mask()
            self._invalidate_overlay()
            self._update_display()

        elif key == "q":
            self._commit_pending()
            self._do_save(silent=True)
            plt.close(self._fig)

        elif key in ("+", "="):
            self.brush_radius = min(self.brush_radius + 5, 100)
            self._update_display()

        elif key == "-":
            self.brush_radius = max(self.brush_radius - 5, 2)
            self._update_display()

    # ── Button callbacks ──────────────────────────────────────────────

    def _btn_mode(self, mode: str) -> None:
        self._commit_pending()
        self._lasso_verts = []
        self._lasso_clear_line()
        self._mode = mode
        self._update_display()

    def _btn_label(self, label: int) -> None:
        self._active_label = label
        self._update_display()

    def _btn_nav(self, direction: int) -> None:
        self._commit_pending()
        self._do_save(silent=True)
        self._idx = max(0, min(len(self.image_paths) - 1, self._idx + direction))
        self._load_current()
        self._update_display()

    def _btn_skip(self) -> None:
        self._pending = None
        self._lasso_verts = []
        self._idx = min(self._idx + 1, len(self.image_paths) - 1)
        self._load_current()
        self._update_display()

    def _btn_save(self) -> None:
        self._commit_pending()
        self._do_save()
        self._update_display()

    def _btn_reset(self) -> None:
        self._pending = None
        self._lasso_verts = []
        self._lasso_clear_line()
        self._push_undo()
        self._mask = self._fresh_mask()
        self._invalidate_overlay()
        self._update_display()

    # ── Layout builder ────────────────────────────────────────────────

    def _build_layout(self) -> None:
        from matplotlib.widgets import Button

        callbacks = {
            "SAM":      lambda _: self._btn_mode(_MODE_SAM),
            "Lasso":    lambda _: self._btn_mode(_MODE_LASSO),
            "Brush":    lambda _: self._btn_mode(_MODE_BRUSH),
            "☁ Cloud":  lambda _: self._btn_label(LABEL_CLOUD),
            "Sky":      lambda _: self._btn_label(LABEL_SKY),
            "Ignore":   lambda _: self._btn_label(LABEL_IGNORE),
            "◀ Prev":   lambda _: self._btn_nav(-1),
            "Next ▶":   lambda _: self._btn_nav(+1),
            "Skip":     lambda _: self._btn_skip(),
            "Save":     lambda _: self._btn_save(),
            "Reset":    lambda _: self._btn_reset(),
        }

        for name, x, w, group in _BTN_DEFS:
            base, _, hover = _BTN_COLOURS[group]
            ax  = self._fig.add_axes([x, 0.005, w, 0.042])
            btn = Button(ax, name, color=base, hovercolor=hover)
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            btn.on_clicked(callbacks[name])
            self._btns[name]     = btn
            self._btn_axes[name] = ax

    # ── Public interface ──────────────────────────────────────────────

    def run(self) -> None:
        """Launch the interactive labelling GUI (blocking call)."""
        import matplotlib.pyplot as plt

        if not self.image_paths:
            raise ValueError("No images provided.")

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self._load_current()

        self._fig = plt.figure(figsize=(15, 10))
        self._fig.patch.set_facecolor("#111111")

        # Layout: image (top 82%) | info strip (8%) | progress (3%) | buttons (5%)
        self._ax_img  = self._fig.add_axes([0.0, 0.14, 1.0, 0.86])
        self._ax_info = self._fig.add_axes([0.0, 0.07, 1.0, 0.07])
        self._ax_prog = self._fig.add_axes([0.0, 0.05, 1.0, 0.02])

        self._ax_img.set_facecolor("#111111")
        self._ax_img.axis("off")
        self._im_handle = self._ax_img.imshow(self._overlay())

        self._build_layout()
        self._update_display()

        self._fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("key_press_event",      self._on_key)

        plt.show()
