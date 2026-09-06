# -*- coding: utf-8 -*-
"""Forensic Image Preprocessing and Calibration for AICatch.

Provides aspect-ratio preserving padding, CLAHE contrast enhancement,
and standardized transforms for visual and Re-ID embedding pipelines.
Does not overwrite the original raw crop on disk, preserving forensic provenance.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def resize_with_aspect_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resizes an image preserving aspect ratio and pads the remaining space."""
    if image is None or image.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / float(w), target_h / float(h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    pad_w = target_w - nw
    pad_h = target_h - nh

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies CLAHE on the L channel of LAB color space to avoid distorting hue/color."""
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def preprocess_crop(
    crop_bgr: np.ndarray,
    mode: str = "baseline",
    target_size: Optional[Tuple[int, int]] = None,
    clahe_clip_limit: float = 2.0
) -> Image.Image:
    """Transforms a raw BGR crop into a preprocessed PIL image for embedding models.

    Modes supported:
    - 'baseline': direct RGB conversion
    - 'aspect_ratio_padding': aspect-preserving resize with padding
    - 'clahe': CLAHE contrast equalization
    - 'aspect_ratio_padding+clahe': combination of padding and CLAHE
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return Image.new("RGB", target_size or (224, 224), (0, 0, 0))

    img = crop_bgr.copy()

    if "clahe" in mode.lower():
        img = apply_clahe(img, clip_limit=clahe_clip_limit)

    if "padding" in mode.lower() or "aspect" in mode.lower():
        size = target_size or (224, 224)
        img = resize_with_aspect_padding(img, target_size=size)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
