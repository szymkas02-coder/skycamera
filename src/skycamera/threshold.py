"""R/B ratio threshold cloud segmentation method.

The red-to-blue (R/B) ratio is the classical statistical approach for
sky-image cloud detection (Long et al. 2006; Dev et al. 2016).

Physics:
    Clear sky scatters blue light (Rayleigh scattering) -> low R, high B -> R/B < 1
    Clouds are white/grey (Mie scattering) -> R ≈ B -> R/B ≈ 1

Decision rule:
    R/B >= threshold  ->  cloud
    R/B <  threshold  ->  sky
    Pixels outside the dome mask are excluded before any ratio computation.

Typical threshold range: 0.55 – 0.90. Tune on local GT masks (notebook 02).
Default 0.6 is a literature starting point — tuning on ACS_WSI found 0.85 optimal
but this should be re-tuned on Warsaw GT masks for best results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def cloud_fraction_rb_threshold(
    img: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.6,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Compute cloud fraction using the R/B ratio threshold method.

    Only pixels inside *mask* (dome region) are considered.
    Pixels where the blue channel is zero are excluded to avoid division
    by zero (they typically correspond to saturated or dead pixels).

    Args:
        img: RGB image array ``(H, W, 3)`` uint8.
        mask: Boolean dome mask ``(H, W)`` — True = valid pixel.
        threshold: R/B ratio threshold. Pixels with R/B >= threshold are
            classified as cloud (default 0.6).
        weights: Optional float32 zenith-cosine weight map from
            :func:`~skycamera.preprocessing.build_zenith_weight_map`.
            When provided, CF is area-weighted and pixels with weight 0
            (beyond the horizon cutoff) are excluded. When None, falls
            back to simple unweighted pixel counting.

    Returns:
        Tuple of:
            cf: float cloud fraction in [0, 1].  Returns ``nan`` when no
                valid pixels exist inside the dome.
            debug_img: ``(H, W, 3)`` uint8 debug overlay —
                red tint = classified cloud,
                blue tint = classified sky,
                black = outside dome or excluded pixel.

    Example:
        >>> cf, debug = cloud_fraction_rb_threshold(img, dome_mask, threshold=0.6)
        >>> 0.0 <= cf <= 1.0
        True
    """
    r = img[:, :, 0].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    # Valid = inside dome AND blue channel > 0 (avoid division by zero)
    valid = mask & (b > 0)

    if valid.sum() == 0:
        debug_img = np.zeros_like(img)
        return float("nan"), debug_img

    ratio = np.where(valid, r / np.where(b > 0, b, 1.0), 0.0)

    is_cloud = valid & (ratio >= threshold)
    is_sky   = valid & (ratio <  threshold)

    if weights is not None:
        cloud_mask = np.where(is_cloud, np.uint8(1), np.uint8(0))
        # pixels outside dome/horizon already have weight 0; mark rest as 255 so
        # weighted_cf skips them correctly
        cf_mask = np.where(valid, cloud_mask, np.uint8(255))
        from .preprocessing import weighted_cf
        cf = weighted_cf(cf_mask, weights)
    else:
        n_cloud = int(is_cloud.sum())
        n_sky   = int(is_sky.sum())
        cf = float(n_cloud / (n_cloud + n_sky)) if (n_cloud + n_sky) > 0 else float("nan")

    # Debug overlay: blend original with red (cloud) or blue (sky) tint
    debug_img = img.copy().astype(np.float32)
    debug_img[is_cloud] = debug_img[is_cloud] * 0.4 + np.array([220, 60, 60],  dtype=np.float32) * 0.6
    debug_img[is_sky]   = debug_img[is_sky]   * 0.4 + np.array([60,  60, 220], dtype=np.float32) * 0.6
    debug_img[~mask]    = 0

    return cf, debug_img.clip(0, 255).astype(np.uint8)


def run_on_index(
    df_index,
    dome_mask: np.ndarray,
    threshold: float = 0.6,
    daytime_only: bool = True,
    save_masks: bool = False,
    masks_dir: Optional[Path] = None,
    weights: Optional[np.ndarray] = None,
) -> "pd.DataFrame":
    """Apply R/B threshold to every image in a build_image_index DataFrame.

    Args:
        df_index: DataFrame from :func:`~skycamera.io.build_image_index`
            with at minimum columns ``path``, ``timestamp``, ``month``,
            ``hour``, ``is_daytime``.
        dome_mask: Boolean dome mask applied to every image.
        threshold: R/B threshold passed to :func:`cloud_fraction_rb_threshold`.
        daytime_only: If True (default), skip rows where ``is_daytime`` is False.
        save_masks: If True, save each debug overlay PNG to *masks_dir*.
        masks_dir: Required when *save_masks* is True.
        weights: Optional zenith-cosine weight map from
            :func:`~skycamera.preprocessing.build_zenith_weight_map`.
            Passed through to :func:`cloud_fraction_rb_threshold`.

    Returns:
        ``pd.DataFrame`` with columns:
            ``timestamp``, ``cloud_fraction``, ``month``, ``hour``.
        Rows where CF is NaN (all-black images, corrupted files) are dropped.

    Example:
        >>> df_cf = run_on_index(df_index, dome_mask, threshold=0.6)
        >>> df_cf.to_csv('outputs/csv/cf_rb_threshold.csv', index=False)
    """
    import pandas as pd
    from .io import load_image

    if daytime_only and "is_daytime" in df_index.columns:
        rows = df_index[df_index["is_daytime"]].copy()
    else:
        rows = df_index.copy()

    if save_masks:
        assert masks_dir is not None, "masks_dir required when save_masks=True"
        import cv2
        Path(masks_dir).mkdir(parents=True, exist_ok=True)

    from .sun import mask_sun_pixels

    results = []
    for _, row in rows.iterrows():
        try:
            img = load_image(row["path"])
            active_mask = mask_sun_pixels(dome_mask, Path(row["path"]))
            cf, debug = cloud_fraction_rb_threshold(img, active_mask, threshold, weights)
        except Exception:
            continue

        if np.isnan(cf):
            continue

        if save_masks and masks_dir is not None:
            import cv2
            out_name = Path(row["path"]).stem + "_rb.jpg"
            bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(masks_dir) / out_name), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "timestamp":       row["timestamp"],
            "cloud_fraction":  round(cf, 4),
            "month":           int(row["month"]),
            "hour":            int(row["hour"]),
        })

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values("timestamp").reset_index(drop=True)
    return df_out
