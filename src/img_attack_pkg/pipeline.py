# src/img_attack_pkg/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import torch

from . import attacks_generic as A
from .io import (
    ensure_dir,
    load_image,
    load_images_from_dir,
    bgr_u8_to_torch_m11,
    torch_m11_to_bgr_u8,
)

# Attack registry
_ATTACKS: Dict[str, Tuple[Callable, str]] = {
    "rotation": (A.rotate_tensor, "angle"),
    "crop": (A.crop, "pct"),
    "cropped": (A.crop, "pct"),
    "scale": (A.scaled, "scale"),
    "scaled": (A.scaled, "scale"),
    "flip": (A.flipping, "mode"),
    "flipping": (A.flipping, "mode"),
    "resize": (A.resized, "pct"),
    "resized": (A.resized, "pct"),
    "jpeg": (A.jpeg_compression, "quality"),
    "jpeg_compression": (A.jpeg_compression, "quality"),
    "jpeg2000": (A.jpeg2000_compression, "quality_layers"),
    "jpeg2000_compression": (A.jpeg2000_compression, "quality_layers"),
    "jpegai": (A.jpegai_compression, "quality"),
    "jpegai_compression": (A.jpegai_compression, "quality"),
    "jpegxl": (A.jpegxl_compression, "quality"),
    "jpegxl_compression": (A.jpegxl_compression, "quality"),
    "gaussian": (A.gaussian_noise, "var"),
    "gaussian_noise": (A.gaussian_noise, "var"),
    "speckle": (A.speckle_noise, "sigma"),
    "speckle_noise": (A.speckle_noise, "sigma"),
    "blur": (A.blurring, "k"),
    "blurring": (A.blurring, "k"),
    "brightness": (A.brightness, "factor"),
    "sharpness": (A.sharpness, "amount"),
    "median": (A.median_filtering, "k"),
    "median_filtering": (A.median_filtering, "k"),
}


@dataclass(frozen=True)
class AttackSpec:
    name: str
    param: Union[int, float, str]


def _canonical_attack_name(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("cropped", "crop"):
        return "crop"
    if r in ("scaled", "scale"):
        return "scale"
    if r in ("flipping", "flip"):
        return "flip"
    if r in ("resized", "resize"):
        return "resize"
    if r in ("jpeg_compression", "jpeg"):
        return "jpeg"
    if r in ("jpeg2000_compression", "jpeg2000", "jp2", "j2k"):
        return "jpeg2000"
    if r in ("jpegai_compression", "jpegai"):
        return "jpegai"
    if r in ("jpegxl_compression", "jpegxl", "jxl"):
        return "jpegxl"
    if r in ("gaussian_noise", "gaussian"):
        return "gaussian"
    if r in ("speckle_noise", "speckle"):
        return "speckle"
    if r in ("blurring", "blur"):
        return "blur"
    if r in ("median_filtering", "median"):
        return "median"
    return r


def parse_attack_tokens(tokens: Sequence[Union[str, int, float]]) -> List[AttackSpec]:
    flat = [str(t) for t in tokens]
    if len(flat) % 2 != 0:
        raise ValueError(f"Expected even number of tokens (name,param,...). Got: {tokens}")

    specs: List[AttackSpec] = []
    for i in range(0, len(flat), 2):
        name = _canonical_attack_name(flat[i])
        raw_param = flat[i + 1]

        if name == "flip":
            param: Union[int, float, str] = raw_param.upper()
        else:
            try:
                param = int(raw_param)
            except ValueError:
                try:
                    param = float(raw_param)
                except ValueError:
                    param = raw_param

        specs.append(AttackSpec(name=name, param=param))

    return specs


def _apply_one_attack(x_m11_chw: torch.Tensor, spec: AttackSpec) -> torch.Tensor:
    lookup_key = spec.name
    if lookup_key not in _ATTACKS:
        if spec.name in _ATTACKS:
            lookup_key = spec.name
        else:
            raise KeyError(f"Unknown attack: {spec.name}. Available: {sorted(set(_ATTACKS.keys()))}")

    fn, _param_name = _ATTACKS[lookup_key]

    if spec.name == "crop":
        return fn(x_m11_chw, float(spec.param))
    if spec.name == "rotation":
        return fn(x_m11_chw, float(spec.param))
    if spec.name in ("scale", "resize"):
        if spec.name == "scale":
            return fn(x_m11_chw, float(spec.param))
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "flip":
        return fn(x_m11_chw, str(spec.param))
    if spec.name == "jpeg":
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "jpeg2000":
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "jpegai":
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "jpegxl":
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "gaussian":
        return fn(x_m11_chw, float(spec.param))
    if spec.name == "speckle":
        return fn(x_m11_chw, float(spec.param))
    if spec.name == "blur":
        return fn(x_m11_chw, int(spec.param))
    if spec.name == "brightness":
        return fn(x_m11_chw, float(spec.param))
    if spec.name == "sharpness":
        return fn(x_m11_chw, float(spec.param))
    if spec.name == "median":
        return fn(x_m11_chw, int(spec.param))

    return fn(x_m11_chw, spec.param)


def _attack_folder(spec: AttackSpec) -> str:
    return f"{spec.name}_{spec.param}"


def _output_filename(base: str, spec: AttackSpec, ext: str = ".png") -> str:
    return f"{base}_{spec.name}_{spec.param}{ext}"


def run_selected_file(
    image,
    attack_tokens: Sequence[Union[str, int, float]],
):
    specs = parse_attack_tokens(attack_tokens)
    li = load_image(image)  # NOTE: this must exist in your codebase

    x = bgr_u8_to_torch_m11(li.bgr_u8)
    outputs = {}

    for spec in specs:
        y = _apply_one_attack(x, spec)
        y_bgr = torch_m11_to_bgr_u8(y)
        outputs[_attack_folder(spec)] = y_bgr

    return outputs


def run_selected_folder(
    image_dir: Union[str, Path],
    attack_tokens: Sequence[Union[str, int, float]],
    out_dir: Union[str, Path],
) -> Dict[str, List[str]]:

    specs = parse_attack_tokens(attack_tokens)
    out_dir = ensure_dir(out_dir)

    loaded_images = load_images_from_dir(image_dir)
    written: Dict[str, List[str]] = {}

    for li in loaded_images:
        x = bgr_u8_to_torch_m11(li.bgr_u8)  # CHW [-1,1]

        for spec in specs:
            y = _apply_one_attack(x, spec)
            y_bgr = torch_m11_to_bgr_u8(y)

            folder = ensure_dir(out_dir / _attack_folder(spec))
            fname = _output_filename(li.name, spec, ext=li.ext or ".png")
            fpath = str(folder / fname)

            if not cv2.imwrite(fpath, y_bgr):
                raise RuntimeError(f"Failed to write {fpath}")

            written.setdefault(_attack_folder(spec), []).append(fpath)

    return written


