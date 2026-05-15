"""Gemma 3 zero-shot cloud fraction estimation via Ollama local inference.

Scientific note — the exact prompt used:
    This is documented verbatim in PROMPT_TEMPLATE below and in the notebook.
    The prompt text is a scientifically significant methodological detail:
    it defines what the model is asked to estimate and in what format,
    and must be cited in any publication using these results.

Ollama setup:
    Install: https://ollama.com/download
    Pull model: ollama pull gemma3:4b
    Server starts automatically; REST API at http://localhost:11434

API used: POST /api/chat  (multimodal, image passed as base64)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"

# ── Scientific record: exact prompt ───────────────────────────────────
# This prompt is fixed for reproducibility. Any change invalidates comparability
# with previously generated results and must be logged as a new experiment.
PROMPT_TEMPLATE = """You are an expert meteorologist analysing a fish-eye sky camera image
taken at a weather station in Warsaw, Poland.

Examine the image carefully and estimate:
1. The fraction of the sky dome covered by clouds (0.0 = completely clear, 1.0 = completely overcast).
2. The dominant cloud type visible.
3. Your confidence in the estimate.

Respond ONLY with a valid JSON object in exactly this format, with no other text:
{{
  "cloud_fraction": <float between 0.0 and 1.0>,
  "cloud_type": "<one of: clear / cumulus / stratus / cirrus / cumulonimbus / mixed>",
  "confidence": "<one of: low / medium / high>"
}}"""


# ── Core inference ────────────────────────────────────────────────────

def _encode_image(img: np.ndarray) -> str:
    """Encode a uint8 RGB array to a base64 JPEG string for the Ollama API."""
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _parse_response(text: str) -> dict:
    """Extract the JSON object from the model's response text.

    The model sometimes wraps the JSON in markdown code fences or adds
    leading/trailing prose despite the prompt instruction. This function
    extracts the first valid JSON object found in the text.

    Returns:
        Dict with keys ``cloud_fraction``, ``cloud_type``, ``confidence``.
        Values are set to None if parsing fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        log.warning("No JSON object found in VLM response: %s", text[:200])
        return {"cloud_fraction": None, "cloud_type": None, "confidence": None}

    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s  text: %s", e, text[:200])
        return {"cloud_fraction": None, "cloud_type": None, "confidence": None}

    # Validate and coerce types
    cf = obj.get("cloud_fraction")
    try:
        cf = float(cf)
        cf = max(0.0, min(1.0, cf))
    except (TypeError, ValueError):
        cf = None

    cloud_type = str(obj.get("cloud_type", "")).lower().strip() or None
    confidence = str(obj.get("confidence", "")).lower().strip() or None

    return {"cloud_fraction": cf, "cloud_type": cloud_type, "confidence": confidence}


def query_vlm(
    img: np.ndarray,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: int = 240,
    retries: int = 2,
) -> dict:
    """Send one sky image to the Ollama VLM and return structured estimates.

    The PROMPT_TEMPLATE is fixed for reproducibility — do not modify it
    between experiments without logging the change.

    Args:
        img: RGB image array (H, W, 3) uint8.
        model: Ollama model name (must be pulled, e.g. ``gemma3:4b``).
        ollama_url: Base URL of the Ollama server.
        timeout: HTTP request timeout in seconds.
        retries: Number of retry attempts on failure.

    Returns:
        Dict with keys:
            ``cloud_fraction``  float [0,1] or None if parsing failed
            ``cloud_type``      str or None
            ``confidence``      str (low/medium/high) or None
            ``raw_response``    str — full model text for audit trail
            ``model``           str — model name used
            ``prompt``          str — exact prompt sent (for paper citation)

    Example:
        >>> result = query_vlm(img, model='gemma3:4b')
        >>> result['cloud_fraction']
        0.3
    """
    import urllib.request

    img_b64 = _encode_image(img)

    payload = {
        "model":  model,
        "stream": False,
        "messages": [
            {
                "role":    "user",
                "content": PROMPT_TEMPLATE,
                "images":  [img_b64],
            }
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    url  = f"{ollama_url}/api/chat"

    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            raw_text = body.get("message", {}).get("content", "")
            parsed   = _parse_response(raw_text)
            return {
                **parsed,
                "raw_response": raw_text,
                "model":  model,
                "prompt": PROMPT_TEMPLATE,
            }
        except Exception as e:
            last_error = e
            if attempt < retries:
                log.warning("VLM attempt %d failed (%s), retrying...", attempt + 1, e)
                time.sleep(2)

    log.error("VLM query failed after %d attempts: %s", retries + 1, last_error)
    return {
        "cloud_fraction": None,
        "cloud_type":     None,
        "confidence":     None,
        "raw_response":   str(last_error),
        "model":          model,
        "prompt":         PROMPT_TEMPLATE,
    }


# ── Batch inference ───────────────────────────────────────────────────

def run_vlm_on_index(
    df_index,
    dome_mask: np.ndarray,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    subsample: str = "1D",
    timeout: int = 120,
    save_raw: Optional[Path] = None,
) -> "pd.DataFrame":
    """Run VLM inference on a subsample of the pilot image index.

    One image per calendar day is selected (the image closest to solar noon,
    12:00 UTC). This produces ~one estimate per day — appropriate given the
    slow inference speed of local VLMs.

    Args:
        df_index: DataFrame from :func:`~skycamera.io.build_image_index`.
        dome_mask: Boolean dome mask; applied before sending to the VLM
            (zeros vignette border, which could confuse the model).
        model: Ollama model name.
        ollama_url: Ollama server URL.
        subsample: Pandas offset alias for subsampling frequency.
            ``"1D"`` = one image per day (default).
        timeout: Per-request timeout in seconds.
        save_raw: If provided, save full raw responses (including audit trail)
            to this CSV path after each image — allows resuming interrupted runs.

    Returns:
        ``pd.DataFrame`` with columns:
            ``timestamp``, ``cloud_fraction``, ``cloud_type``,
            ``confidence``, ``model``, ``month``, ``hour``.

    Example:
        >>> df_vlm = run_vlm_on_index(df_index, dome_mask, model='gemma3:4b')
        >>> df_vlm.to_csv('outputs/csv/cf_gemma.csv', index=False)
    """
    import pandas as pd
    from .io import load_image
    from .preprocessing import apply_mask

    # Filter to daytime only
    if "is_daytime" in df_index.columns:
        df_day = df_index[df_index["is_daytime"]].copy()
    else:
        df_day = df_index.copy()

    df_day = df_day.sort_values("timestamp").set_index("timestamp")

    # Select one image per day: closest to 12:00 UTC
    def _pick_noon(group):
        group = group.copy()
        group["dist_noon"] = (group.index.hour - 12).abs()
        return group.sort_values("dist_noon").iloc[[0]]

    df_sub = df_day.groupby(df_day.index.date, group_keys=False).apply(_pick_noon)
    df_sub = df_sub.reset_index()
    log.info("VLM: %d images selected (1 per day)", len(df_sub))

    # Load raw results CSV if resuming
    existing_ts: set = set()
    if save_raw and Path(save_raw).exists():
        try:
            existing = pd.read_csv(save_raw, parse_dates=["timestamp"])
            existing_ts = set(existing["timestamp"].astype(str))
            log.info("Resuming: %d already processed", len(existing_ts))
        except Exception:
            pass

    results = []
    for i, row in df_sub.iterrows():
        ts_str = str(row["timestamp"])
        if ts_str in existing_ts:
            continue

        try:
            img = load_image(row["path"])
            img_masked = apply_mask(img, dome_mask)
        except Exception as e:
            log.warning("Could not load %s: %s", row["path"], e)
            continue

        log.info("[%d/%d] %s querying %s...",
                 i + 1, len(df_sub), row["timestamp"], model)
        result = query_vlm(img_masked, model=model,
                           ollama_url=ollama_url, timeout=timeout)

        record = {
            "timestamp":      row["timestamp"],
            "cloud_fraction": result["cloud_fraction"],
            "cloud_type":     result["cloud_type"],
            "confidence":     result["confidence"],
            "model":          result["model"],
            "month":          int(pd.Timestamp(row["timestamp"]).month),
            "hour":           int(pd.Timestamp(row["timestamp"]).hour),
        }
        results.append(record)

        # Append to raw file immediately so progress is not lost
        if save_raw:
            raw_record = {**record, "raw_response": result["raw_response"],
                          "prompt": result["prompt"]}
            raw_path = Path(save_raw)
            write_header = not raw_path.exists()
            pd.DataFrame([raw_record]).to_csv(
                raw_path, mode="a", header=write_header, index=False
            )

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values("timestamp").reset_index(drop=True)
    log.info("VLM inference complete: %d results", len(df_out))
    return df_out


def check_ollama(
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> dict:
    """Check that the Ollama server is reachable and the model is available.

    Args:
        model: Model name to verify.
        ollama_url: Server URL.

    Returns:
        Dict with keys ``server_ok`` (bool), ``model_available`` (bool),
        ``available_models`` (list of str).

    Example:
        >>> status = check_ollama()
        >>> status['server_ok']
        True
    """
    import urllib.request

    status = {"server_ok": False, "model_available": False, "available_models": []}

    try:
        req = urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5)
        data = json.loads(req.read())
        status["server_ok"] = True
        names = [m["name"] for m in data.get("models", [])]
        status["available_models"] = names
        status["model_available"] = any(
            n == model or n.startswith(model.split(":")[0])
            for n in names
        )
    except Exception as e:
        status["error"] = str(e)

    return status
