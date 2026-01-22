# src/img_attack_pkg/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch


ImageLike = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class LoadedImage:
    name: str              
    ext: str               
    bgr_u8: np.ndarray     


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(img: ImageLike) -> LoadedImage:
    if isinstance(img, (str, Path)):
        p = Path(img)
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read image: {p}")
        name = p.stem
        ext = p.suffix if p.suffix else ".png"
        return LoadedImage(name=name, ext=ext, bgr_u8=arr)

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            arr = img
        return LoadedImage(name="image", ext=".png", bgr_u8=arr)

    raise TypeError(f"Unsupported image type: {type(img)}")


def load_images_from_dir(image_dir: Union[str, Path]) -> List[LoadedImage]:

    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist or is not a directory: {image_dir}")

    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    return [load_image(p) for p in paths]


def bgr_u8_to_torch_m11(img_bgr_u8: np.ndarray) -> torch.Tensor:

    if img_bgr_u8.ndim == 2:
        chw = img_bgr_u8[None, :, :]
    else:
        chw = np.transpose(img_bgr_u8, (2, 0, 1))

    t = torch.from_numpy(chw).to(torch.float32)  # 0..255
    t01 = (t / 255.0).clamp(0.0, 1.0)
    tm11 = (t01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    return tm11


def torch_m11_to_bgr_u8(x: torch.Tensor) -> np.ndarray:

    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError("Expected batch size 1 for conversion to single image.")
        x = x[0]

    x = x.detach().cpu().clamp(-1.0, 1.0)
    x01 = (x + 1.0) / 2.0
    x255 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)

    arr = x255.numpy()  # CHW
    if arr.shape[0] == 1:
        return arr[0]  # HxW
    return np.transpose(arr, (1, 2, 0))  # HxWxC


def ensure_dir(p: Union[str, Path]) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out
