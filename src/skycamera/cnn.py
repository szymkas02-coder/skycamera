"""CNN segmentation method — fine-tuned U-Net on sky camera images.

Architecture:  U-Net with ResNet-34 encoder (pretrained on ImageNet).
Task:          Binary semantic segmentation — cloud vs sky.
Input:         512 x 512 RGB crop of the dome region (circular mask applied).
Output:        512 x 512 binary mask (1 = cloud, 0 = sky).

Minimum labels needed for reliable fine-tuning:
    The model relies on the ImageNet-pretrained encoder. In practice
    30-50 labelled images spread across CF levels (clear / partial / overcast)
    produce a model that generalises to the pilot dataset. Below ~20 images
    results become noisy. The ACS_WSI dataset (77 pairs) is sufficient for a
    first fine-tune; adding even 10-20 manually labelled Warsaw images typically
    improves performance on the local camera.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
IMG_SIZE         = (512, 512)
ENCODER          = "resnet34"
ENCODER_WEIGHTS  = "imagenet"
BATCH_SIZE       = 8
LR               = 1e-4
EPOCHS           = 10
PATIENCE         = 7
RANDOM_SEED      = 42


# ── Dataset ───────────────────────────────────────────────────────────

class SkyDataset:
    """PyTorch Dataset for sky camera segmentation.

    Each sample is a (image_tensor, mask_tensor, weight_tensor) triple where:
        image_tensor:  float32 (3, H, W)  normalised to ImageNet mean/std
        mask_tensor:   float32 (1, H, W)  binary 0/1 cloud mask
        weight_tensor: float32 (1, H, W)  1 for sky/cloud pixels, 0 for IGNORE/UNLABELLED

    IGNORE and UNLABELLED pixels have weight 0 so they do not contribute to
    the loss — the model never sees a gradient from antenna or sun-disk pixels.

    Args:
        df: DataFrame with columns ``image_path``, ``mask_path``, ``source``.
            Produced by :func:`~skycamera.io.build_combined_dataset`.
        img_size: (H, W) to resize every image and mask to.
        augment: If True, apply random horizontal/vertical flips.
    """

    def __init__(self, df, img_size: Tuple[int, int] = IMG_SIZE,
                 augment: bool = False) -> None:
        self.records  = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment  = augment

        # ImageNet normalisation constants
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import cv2, torch
        from .io import load_image
        from .labelling import (LABEL_CLOUD, LABEL_SKY,
                                 LABEL_IGNORE, LABEL_UNLABELLED,
                                 load_existing_mask)

        row = self.records.iloc[idx]
        img_path  = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        # Load image
        img = load_image(img_path)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_LINEAR)

        # Load mask — handle both ACS_WSI (.jpg) and manual (.png) formats
        if row["source"] == "acs_wsi":
            from .io import load_acs_wsi_pair
            _, raw_mask = load_acs_wsi_pair(img_path, mask_path)
            # raw_mask: 0=sky, 1=cloud, 255=outside dome
            bin_mask   = (raw_mask == LABEL_CLOUD).astype(np.uint8)
            weight_map = (raw_mask != LABEL_UNLABELLED).astype(np.float32)
        else:
            raw_mask   = load_existing_mask(mask_path)
            bin_mask   = (raw_mask == LABEL_CLOUD).astype(np.uint8)
            # Weight 0 for IGNORE and UNLABELLED — exclude from loss
            weight_map = ((raw_mask == LABEL_CLOUD) | (raw_mask == LABEL_SKY)).astype(np.float32)

        bin_mask   = cv2.resize(bin_mask,   (self.img_size[1], self.img_size[0]),
                                interpolation=cv2.INTER_NEAREST)
        weight_map = cv2.resize(weight_map, (self.img_size[1], self.img_size[0]),
                                interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img        = np.fliplr(img).copy()
                bin_mask   = np.fliplr(bin_mask).copy()
                weight_map = np.fliplr(weight_map).copy()
            if random.random() > 0.5:
                img        = np.flipud(img).copy()
                bin_mask   = np.flipud(bin_mask).copy()
                weight_map = np.flipud(weight_map).copy()

        # Normalise image
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - self._mean) / self._std
        img_t    = torch.from_numpy(img_f.transpose(2, 0, 1))           # (3, H, W)
        mask_t   = torch.from_numpy(bin_mask[None].astype(np.float32))  # (1, H, W)
        weight_t = torch.from_numpy(weight_map[None])                   # (1, H, W)

        return img_t, mask_t, weight_t


# ── Model builder ─────────────────────────────────────────────────────

def build_model() -> "torch.nn.Module":
    """Build a U-Net with ResNet-34 encoder pretrained on ImageNet.

    Returns:
        ``segmentation_models_pytorch.Unet`` instance in eval mode.

    Example:
        >>> model = build_model()
    """
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,   # raw logits — BCEWithLogitsLoss handles sigmoid
    )
    return model


# ── Training ──────────────────────────────────────────────────────────

def train_cnn(
    df_train,
    df_val,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = PATIENCE,
    save_path: Optional[Path] = None,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple["torch.nn.Module", dict]:
    """Fine-tune the U-Net on the combined sky dataset.

    Args:
        df_train: Training split DataFrame (from build_combined_dataset).
        df_val:   Validation split DataFrame.
        epochs:   Maximum training epochs.
        batch_size: Mini-batch size.
        lr:       Adam learning rate.
        patience: Early-stopping patience (epochs without val-loss improvement).
        save_path: Where to save the best model checkpoint (.pt).
        img_size: (H, W) each image is resized to before feeding the network.

    Returns:
        Tuple of (best_model, history_dict) where history has keys
        ``train_loss``, ``val_loss``, ``train_iou``, ``val_iou``.

    Example:
        >>> model, history = train_cnn(df_tr, df_val, save_path=Path('model_cnn.pt'))
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training CNN on %s  |  train=%d  val=%d", device, len(df_train), len(df_val))
    log.info("Epochs=%d  batch=%d  lr=%.1e  patience=%d  img_size=%s",
             epochs, batch_size, lr, patience, img_size)
    log.info("-" * 60)

    ds_train = SkyDataset(df_train, img_size=img_size, augment=True)
    ds_val   = SkyDataset(df_val,   img_size=img_size, augment=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")  # per-pixel loss for masking
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history: dict = {"train_loss": [], "val_loss": [],
                     "train_iou": [],  "val_iou": []}

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────
        model.train()
        t_loss, t_iou = 0.0, 0.0
        for xb, yb, wb in dl_train:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits     = model(xb)
            pixel_loss = criterion(logits, yb)           # (B, 1, H, W)
            loss       = (pixel_loss * wb).sum() / wb.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
            t_iou  += _batch_iou(logits, yb)
        t_loss /= len(ds_train)
        t_iou  /= len(dl_train)

        # ── validate ──────────────────────────────────────────────────
        model.eval()
        v_loss, v_iou = 0.0, 0.0
        with torch.no_grad():
            for xb, yb, wb in dl_val:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits     = model(xb)
                pixel_loss = criterion(logits, yb)
                loss       = (pixel_loss * wb).sum() / wb.sum().clamp(min=1)
                v_loss    += loss.item() * len(xb)
                v_iou     += _batch_iou(logits, yb)
        v_loss /= len(ds_val)
        v_iou  /= len(dl_val)

        scheduler.step(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_iou"].append(t_iou)
        history["val_iou"].append(v_iou)

        improved = v_loss < best_val_loss
        if improved:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " *" if improved else f"  (no improve {no_improve}/{patience})"
        log.info("Epoch %3d/%d  loss=%.4f/%.4f  IoU=%.3f/%.3f  lr=%.2e%s",
                 epoch, epochs, t_loss, v_loss, t_iou, v_iou, current_lr, marker)

        if no_improve >= patience:
            log.info("Early stopping at epoch %d — best val loss=%.5f", epoch, best_val_loss)
            break

    model.load_state_dict(best_state)
    log.info("Training complete — best val loss=%.5f", best_val_loss)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state,
                    "img_size":    img_size,
                    "encoder":     ENCODER}, save_path)
        log.info("Model saved -> %s", save_path)

    return model, history


def _batch_iou(logits: "torch.Tensor", targets: "torch.Tensor",
               threshold: float = 0.5) -> float:
    """Mean IoU over a mini-batch (detached, no grad)."""
    import torch
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= threshold).float()
        inter = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
        iou   = (inter / union.clamp(min=1e-6)).mean().item()
    return iou


# ── Inference ─────────────────────────────────────────────────────────

def load_cnn_model(checkpoint_path: Path) -> "torch.nn.Module":
    """Load a saved U-Net checkpoint.

    Args:
        checkpoint_path: Path to .pt file saved by :func:`train_cnn`.

    Returns:
        Model in eval mode on CPU.

    Example:
        >>> model = load_cnn_model(Path('outputs/models/cnn_sky.pt'))
    """
    import torch
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {checkpoint_path}\n"
            "Run notebook 03_cnn_segmentation.ipynb to train and save the model."
        )
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_mask(
    model: "torch.nn.Module",
    img: np.ndarray,
    dome_mask: np.ndarray,
    img_size: Tuple[int, int] = IMG_SIZE,
    threshold: float = 0.5,
    weights: Optional[np.ndarray] = None,
    image_path: Optional[Path] = None,
) -> Tuple[float, np.ndarray]:
    """Run inference on a single image and return CF + predicted mask.

    The image is resized to *img_size*, segmented, then the predicted mask is
    resized back to the original image shape.  Pixels outside *dome_mask* are
    forced to sky (0) before computing cloud fraction.

    Args:
        model: Fitted U-Net from :func:`load_cnn_model` or :func:`train_cnn`.
        img: RGB image array (H, W, 3) uint8 — original resolution.
        dome_mask: Boolean dome mask at original resolution.
        img_size: (H, W) used during training.
        threshold: Sigmoid threshold for cloud classification.
        weights: Optional zenith-cosine weight map from
            :func:`~skycamera.preprocessing.build_zenith_weight_map`.
            When provided, CF is area-weighted and pixels with weight 0
            (beyond the horizon cutoff) are excluded. When None, falls
            back to simple unweighted pixel counting.
        image_path: Optional path to the source image. When provided, the
            sun-disk region is excluded from the active mask so glare pixels
            do not contribute to CF — consistent with GT mask annotation.

    Returns:
        Tuple of:
            cf:   float cloud fraction (cloud pixels / valid dome pixels).
            pred_mask: uint8 array (H, W) — 1=cloud, 0=sky, original resolution.

    Example:
        >>> cf, mask = predict_mask(model, img, dome_mask, image_path=path)
    """
    import cv2, torch

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    orig_h, orig_w = img.shape[:2]

    resized = cv2.resize(img, (img_size[1], img_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    img_f   = resized.astype(np.float32) / 255.0
    img_f   = (img_f - mean) / std
    img_t   = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logit = model(img_t.to(device))
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

    # Resize prediction back to original resolution
    pred_small = (prob >= threshold).astype(np.uint8)
    pred_full  = cv2.resize(pred_small, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST)

    # Apply dome mask
    pred_full[~dome_mask] = 0

    # Exclude sun-disk pixels from CF computation (same region marked LABEL_IGNORE in GT)
    active_mask = dome_mask
    if image_path is not None:
        from .sun import mask_sun_pixels
        active_mask = mask_sun_pixels(dome_mask, Path(image_path))

    if weights is not None:
        from .preprocessing import weighted_cf
        cf_mask = np.where(active_mask, pred_full, np.uint8(255))
        cf = weighted_cf(cf_mask, weights)
    else:
        n_cloud = int(pred_full[active_mask].sum())
        n_valid = int(active_mask.sum())
        cf = float(n_cloud / n_valid) if n_valid > 0 else float("nan")

    return cf, pred_full


def run_cnn_on_index(
    df_index,
    model: "torch.nn.Module",
    dome_mask: np.ndarray,
    img_size: Tuple[int, int] = IMG_SIZE,
    threshold: float = 0.5,
    save_masks: bool = False,
    masks_dir: Optional[Path] = None,
    weights: Optional[np.ndarray] = None,
) -> "pd.DataFrame":
    """Apply the CNN to every daytime image in an index DataFrame.

    Args:
        df_index: DataFrame from :func:`~skycamera.io.build_image_index`.
        model: Fitted U-Net.
        dome_mask: Boolean dome mask at original image resolution.
        img_size: (H, W) the model was trained on.
        threshold: Sigmoid threshold.
        save_masks: Save predicted mask PNGs to *masks_dir*.
        masks_dir: Required when *save_masks* is True.
        weights: Optional zenith-cosine weight map from
            :func:`~skycamera.preprocessing.build_zenith_weight_map`.
            Passed through to :func:`predict_mask`.

    Returns:
        ``pd.DataFrame`` with columns ``timestamp``, ``cloud_fraction``,
        ``month``, ``hour``.
    """
    import pandas as pd
    from .io import load_image

    rows_out = df_index.copy()

    if save_masks:
        assert masks_dir is not None
        Path(masks_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for _, row in rows_out.iterrows():
        try:
            img = load_image(row["path"])
            cf, pred = predict_mask(
                model, img, dome_mask, img_size, threshold, weights,
                image_path=Path(row["path"]),
            )
        except Exception:
            continue
        if np.isnan(cf):
            continue

        if save_masks and masks_dir is not None:
            import cv2
            out = Path(masks_dir) / (Path(row["path"]).stem + "_cnn.png")
            cv2.imwrite(str(out), pred * 255)

        results.append({
            "timestamp":      row["timestamp"],
            "cloud_fraction": round(cf, 4),
            "month":          int(row["month"]),
            "hour":           int(row["hour"]),
        })

        if len(results) % 24 == 0:
            print(f"[{len(results):>5}] {row['timestamp']}  CF={cf:.4f}")

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values("timestamp").reset_index(drop=True)
    return df_out
