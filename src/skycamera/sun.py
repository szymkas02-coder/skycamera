"""Sun position utilities for sky camera labelling.

Computes sun azimuth/elevation from image filename timestamp and camera
location, then projects the sun disk onto the image pixel grid so it can
be automatically marked as IGNORE in the labelling tool.

Requires: pysolar  (pip install pysolar)
"""
from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as _config
from .config import (
    CAMERA_LAT, CAMERA_LON, CAMERA_ALT,
    SUN_IGNORE_RADIUS_DEG,
    CX, CY, R,
    FILENAME_PATTERN,
)

log = logging.getLogger(__name__)


def parse_timestamp(image_path: Path) -> Optional[datetime]:
    """Extract UTC datetime from an image filename.

    Filename format: ``2024_01_15__12_04_51.jpg``
    Assumes the camera clock is in UTC (or local — adjust tzinfo if needed).

    Returns:
        timezone-aware UTC datetime, or None if filename doesn't match.
    """
    m = re.search(FILENAME_PATTERN, Path(image_path).name)
    if m is None:
        return None
    year, mon, day, hr, minute, sec = (int(x) for x in m.groups())
    return datetime(year, mon, day, hr, minute, sec, tzinfo=timezone.utc)


def sun_altaz(dt: datetime,
              lat: float = CAMERA_LAT,
              lon: float = CAMERA_LON,
              alt: float = CAMERA_ALT) -> tuple[float, float]:
    """Return (altitude_deg, azimuth_deg) for the sun at the given UTC time.

    Azimuth is measured clockwise from North (0–360).
    Altitude is degrees above horizon (negative = below horizon).

    Requires pysolar. Falls back to (nan, nan) if unavailable.
    """
    try:
        from pysolar.solar import get_altitude, get_azimuth
        altitude = get_altitude(lat, lon, dt, elevation=alt)
        # pysolar returns azimuth clockwise from South; convert to clockwise from North
        azimuth  = (get_azimuth(lat, lon, dt, elevation=alt) + 180.0) % 360.0
        return float(altitude), float(azimuth)
    except ImportError:
        log.warning("pysolar not installed — sun position unavailable. "
                    "Install with: pip install pysolar")
        return float("nan"), float("nan")
    except Exception as e:
        log.warning("Sun position calculation failed: %s", e)
        return float("nan"), float("nan")


def altaz_to_pixel(altitude_deg: float, azimuth_deg: float,
                   cx: int = CX, cy: int = CY, r: int = R
                   ) -> Optional[tuple[int, int]]:
    """Project sun (altitude, azimuth) onto the fisheye image pixel grid.

    Uses an equidistant (linear) fisheye projection:
        radial_distance = r * (90 - altitude) / 90

    Azimuth 0° = North = up in the image (negative y direction).

    Returns:
        (px, py) pixel coordinates, or None if sun is below horizon.
    """
    if altitude_deg < 0:
        return None
    zenith_deg = 90.0 - altitude_deg
    radial_px  = r * zenith_deg / 90.0 * _config.CAMERA_PROJECTION_SCALE
    # Apply camera mount rotation then project: North = -y in image coords
    az_rad = np.deg2rad(azimuth_deg - _config.CAMERA_NORTH_OFFSET_DEG)
    px = int(round(cx + radial_px * np.sin(az_rad)))
    py = int(round(cy - radial_px * np.cos(az_rad)))
    return px, py


def mask_sun_pixels(
    dome_mask: np.ndarray,
    image_path: Path,
) -> np.ndarray:
    """Return a copy of dome_mask with sun-disk pixels set to False.

    Used at inference time to exclude sun glare from CF computation and
    segmentation predictions. Consistent with the LABEL_IGNORE annotation
    applied to GT masks during labelling.

    Returns the original dome_mask unchanged if the sun is below the horizon,
    the timestamp cannot be parsed, or pysolar is not installed.

    Args:
        dome_mask:  Boolean (H, W) array — True inside the fisheye dome.
        image_path: Path to the image (UTC timestamp parsed from filename).

    Returns:
        Boolean (H, W) array with sun-disk pixels set to False.

    Example:
        >>> active_mask = mask_sun_pixels(dome_mask, Path('2024_06_15__12_00_31.jpg'))
        >>> cf, debug = cloud_fraction_rb_threshold(img, active_mask, ...)
    """
    sun = sun_ignore_mask(image_path, dome_mask.shape)
    if sun is None:
        return dome_mask
    result = dome_mask.copy()
    result[sun] = False
    return result


def sun_ignore_mask(image_path: Path,
                    img_shape: tuple[int, int],
                    radius_deg: Optional[float] = None,
                    cx: int = CX, cy: int = CY, r: int = R,
                    ) -> Optional[np.ndarray]:
    """Return a boolean mask marking the sun disk region as True.

    Args:
        image_path: Path to the image (timestamp parsed from filename).
        img_shape:  (H, W) of the image array.
        radius_deg: Angular radius of the ignore circle around the sun.
        cx, cy, r:  Fisheye camera centre and dome radius in pixels.

    Returns:
        Boolean (H, W) mask — True where the sun disk is, or None if the
        sun is below the horizon / timestamp cannot be parsed.
    """
    dt = parse_timestamp(image_path)
    if dt is None:
        log.debug("Could not parse timestamp from: %s", image_path.name)
        return None

    altitude, azimuth = sun_altaz(dt)
    if np.isnan(altitude) or altitude < 0:
        return None

    sun_px = altaz_to_pixel(altitude, azimuth, cx, cy, r)
    if sun_px is None:
        return None

    if radius_deg is None:
        radius_deg = _config.SUN_IGNORE_RADIUS_DEG
    # Convert angular radius to pixel radius using the same linear projection
    radius_px = int(round(r * radius_deg / 90.0 * _config.CAMERA_PROJECTION_SCALE))

    H, W = img_shape
    sy, sx = np.ogrid[:H, :W]
    mask = ((sx - sun_px[0])**2 + (sy - sun_px[1])**2) <= radius_px**2

    log.debug("Sun: alt=%.1f° az=%.1f° → pixel=(%d,%d) radius=%dpx",
              altitude, azimuth, sun_px[0], sun_px[1], radius_px)
    return mask
