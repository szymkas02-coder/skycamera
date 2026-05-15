"""Circular masking, fisheye reprojection, and image preprocessing."""
from __future__ import annotations

import numpy as np


def build_circular_mask(
    h: int,
    w: int,
    cx: int,
    cy: int,
    r: int,
) -> np.ndarray:
    """Create a boolean circular mask with True inside the dome.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        cx: Optical centre x coordinate.
        cy: Optical centre y coordinate.
        r: Dome radius in pixels.

    Returns:
        Boolean ``np.ndarray`` of shape ``(h, w)`` — True inside the circle.

    Example:
        >>> mask = build_circular_mask(2080, 3096, 1438, 928, 938)
        >>> mask.sum() > 0
        True
    """
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the mask.

    Args:
        img: RGB image array ``(H, W, 3)`` uint8.
        mask: Boolean array ``(H, W)`` — True = keep.

    Returns:
        Masked image; pixels outside the mask set to 0.

    Example:
        >>> masked = apply_mask(img, mask)
    """
    out = img.copy()
    out[~mask] = 0
    return out


def fisheye_to_equirectangular(
    img: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    out_h: int = 512,
    out_w: int = 512,
) -> np.ndarray:
    """Reproject a fisheye image to an equirectangular view using the equidistant model.

    In the equidistant projection, the angle from zenith θ is proportional to
    the radial distance from the optical centre: ρ = f · θ, where f = R / (π/2).

    Args:
        img: RGB image array ``(H, W, 3)`` uint8.
        cx: Optical centre x.
        cy: Optical centre y.
        r: Dome radius in pixels (maps to θ = 90°).
        out_h: Output image height.
        out_w: Output image width.

    Returns:
        Reprojected RGB image ``(out_h, out_w, 3)`` uint8.

    Example:
        >>> reproj = fisheye_to_equirectangular(img, 1438, 928, 938)
        >>> reproj.shape
        (512, 512, 3)
    """
    import cv2

    # Build output grid in azimuth / zenith angle space
    az = np.linspace(0, 2 * np.pi, out_w, endpoint=False)      # 0 → 2π
    ze = np.linspace(0, np.pi / 2, out_h, endpoint=False)       # 0 → π/2 (zenith to horizon)

    az_grid, ze_grid = np.meshgrid(az, ze)

    # Equidistant: ρ = R * (θ / (π/2))
    rho = r * (ze_grid / (np.pi / 2))

    src_x = (cx + rho * np.sin(az_grid)).astype(np.float32)
    src_y = (cy - rho * np.cos(az_grid)).astype(np.float32)    # y increases downward

    return cv2.remap(img, src_x, src_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _infer_dome_params(binary_mask: np.ndarray) -> tuple[float, float, float]:
    """Infer (cx, cy, r) from the valid-pixel region of a binary mask.

    Valid pixels are those != 255. The dome centre is estimated as the
    centroid of valid pixels and the radius as the maximum distance from
    that centroid to any valid pixel.
    """
    ys, xs = np.where(binary_mask != 255)
    if len(xs) == 0:
        h, w = binary_mask.shape
        return float(w / 2), float(h / 2), float(min(h, w) / 2)
    cx = float(xs.mean())
    cy = float(ys.mean())
    r = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
    return cx, cy, r


def build_zenith_weight_map(
    h: int,
    w: int,
    cx: float,
    cy: float,
    r: float,
    max_zenith_deg: float = 70.0,
) -> np.ndarray:
    """Return a float32 cosine-zenith weight map for area-corrected CF.

    Uses the equidistant fisheye model (rho = R * theta / (pi/2)).
    Each pixel inside the dome gets weight cos(zenith_angle).
    Pixels outside the dome or beyond *max_zenith_deg* get weight 0.

    Args:
        h: Image height.
        w: Image width.
        cx: Optical centre x.
        cy: Optical centre y.
        r: Dome radius in pixels (maps to theta = 90 deg).
        max_zenith_deg: Horizon cutoff — pixels beyond this angle are excluded.

    Returns:
        float32 array of shape (h, w).
    """
    ys, xs = np.mgrid[0:h, 0:w]
    rho = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    zenith_rad = rho / r * (np.pi / 2)
    max_rad = np.deg2rad(max_zenith_deg)
    weights = np.where(zenith_rad <= max_rad, np.cos(zenith_rad), 0.0)
    return weights.astype(np.float32)


def weighted_cf(binary_mask: np.ndarray, weights: np.ndarray) -> float:
    """Compute area-weighted cloud fraction from a binary mask.

    Args:
        binary_mask: uint8 array with values 0 (sky), 1 (cloud), 255 (ignore).
        weights: float32 weight map from :func:`build_zenith_weight_map`.

    Returns:
        Weighted cloud fraction in [0, 1], or NaN if no valid pixels.
    """
    valid = (binary_mask != 255) & (weights > 0)
    if valid.sum() == 0:
        return float("nan")
    return float(
        (binary_mask[valid].astype(np.float32) * weights[valid]).sum()
        / weights[valid].sum()
    )
